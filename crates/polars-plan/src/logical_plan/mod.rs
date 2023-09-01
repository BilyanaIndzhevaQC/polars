use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use arrow::compute::filter::filter_chunk;
#[cfg(feature = "parquet")]
use polars_core::cloud::CloudOptions;
use polars_core::prelude::*;

use crate::logical_plan::LogicalPlan::DataFrameScan;
use crate::prelude::*;
use crate::utils::{expr_to_leaf_column_names, get_single_leaf};

pub(crate) mod aexpr;
pub(crate) mod alp;
pub(crate) mod anonymous_scan;

mod apply;
mod builder;
mod builder_alp;
pub mod builder_functions;
pub(crate) mod conversion;
#[cfg(feature = "debugging")]
pub(crate) mod debug;
mod file_scan;
mod format;
mod functions;
pub(crate) mod iterator;
mod lit;
pub(crate) mod optimizer;
pub(crate) mod options;
pub(crate) mod projection;
mod projection_expr;
#[cfg(feature = "python")]
mod pyarrow;
mod schema;
#[cfg(any(feature = "meta", feature = "cse"))]
pub(crate) mod tree_format;
pub mod visitor;

pub use aexpr::*;
pub use alp::*;
pub use anonymous_scan::*;
pub use apply::*;
pub use builder::*;
pub use builder_alp::*;
pub use conversion::*;
pub use file_scan::*;
pub use functions::*;
pub use iterator::*;
pub use lit::*;
pub use optimizer::*;
pub use schema::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[cfg(any(feature = "ipc", feature = "parquet", feature = "csv", feature = "cse"))]
pub use crate::logical_plan::optimizer::file_caching::{
    collect_fingerprints, find_column_union_and_fingerprints, FileCacher, FileFingerPrint,
};

#[derive(Clone, Copy, Debug)]
pub enum Context {
    /// Any operation that is done on groups
    Aggregation,
    /// Any operation that is done while projection/ selection of data
    Default,
}

#[derive(Debug)]
pub enum ErrorState {
    NotYetEncountered { err: PolarsError },
    AlreadyEncountered { prev_err_msg: String },
}

impl std::fmt::Display for ErrorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorState::NotYetEncountered { err } => write!(f, "NotYetEncountered({err})")?,
            ErrorState::AlreadyEncountered { prev_err_msg } => {
                write!(f, "AlreadyEncountered({prev_err_msg})")?
            },
        };

        Ok(())
    }
}

#[derive(Clone)]
pub struct ErrorStateSync(Arc<Mutex<ErrorState>>);

impl std::ops::Deref for ErrorStateSync {
    type Target = Arc<Mutex<ErrorState>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for ErrorStateSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ErrorStateSync({})", &*self.0.lock().unwrap())
    }
}

impl ErrorStateSync {
    fn take(&self) -> PolarsError {
        let mut curr_err = self.0.lock().unwrap();

        match &*curr_err {
            ErrorState::NotYetEncountered { err: polars_err } => {
                // Need to finish using `polars_err` here so that NLL considers `err` dropped
                let prev_err_msg = polars_err.to_string();
                // Place AlreadyEncountered in `self` for future users of `self`
                let prev_err = std::mem::replace(
                    &mut *curr_err,
                    ErrorState::AlreadyEncountered { prev_err_msg },
                );
                // Since we're in this branch, we know err was a NotYetEncountered
                match prev_err {
                    ErrorState::NotYetEncountered { err } => err,
                    ErrorState::AlreadyEncountered { .. } => unreachable!(),
                }
            },
            ErrorState::AlreadyEncountered { prev_err_msg } => {
                polars_err!(
                    ComputeError: "LogicalPlan already failed with error: '{}'", prev_err_msg,
                )
            },
        }
    }
}

impl From<PolarsError> for ErrorStateSync {
    fn from(err: PolarsError) -> Self {
        Self(Arc::new(Mutex::new(ErrorState::NotYetEncountered { err })))
    }
}

// https://stackoverflow.com/questions/1031076/what-are-projection-and-selection
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LogicalPlan {
    #[cfg_attr(feature = "serde", serde(skip))]
    AnonymousScan {
        function: Arc<dyn AnonymousScan>,
        file_info: FileInfo,
        predicate: Option<Expr>,
        options: Arc<AnonymousScanOptions>,
    },
    #[cfg(feature = "python")]
    PythonScan { options: PythonOptions },
    /// Filter on a boolean mask
    Selection {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    /// Cache the input at this point in the LP
    Cache {
        input: Box<LogicalPlan>,
        id: usize,
        count: usize,
    },
    Scan {
        path: PathBuf,
        file_info: FileInfo,
        predicate: Option<Expr>,
        file_options: FileScanOptions,
        scan_type: FileScan,
    },
    // we keep track of the projection and selection as it is cheaper to first project and then filter
    /// In memory DataFrame
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        projection: Option<Arc<Vec<String>>>,
        selection: Option<Expr>,
    },
    // a projection that doesn't have to be optimized
    // or may drop projected columns if they aren't in current schema (after optimization)
    LocalProjection {
        expr: Vec<Expr>,
        input: Box<LogicalPlan>,
        schema: SchemaRef,
    },
    /// Column selection
    Projection {
        expr: Vec<Expr>,
        input: Box<LogicalPlan>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    /// Groupby aggregation
    Aggregate {
        input: Box<LogicalPlan>,
        keys: Arc<Vec<Expr>>,
        aggs: Vec<Expr>,
        schema: SchemaRef,
        #[cfg_attr(feature = "serde", serde(skip))]
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
    },
    /// Join operation
    Join {
        input_left: Box<LogicalPlan>,
        input_right: Box<LogicalPlan>,
        schema: SchemaRef,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        options: Arc<JoinOptions>,
    },
    /// Adding columns to the table without a Join
    HStack {
        input: Box<LogicalPlan>,
        exprs: Vec<Expr>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    /// Remove duplicates from the table
    Distinct {
        input: Box<LogicalPlan>,
        options: DistinctOptions,
    },
    /// Sort the table
    Sort {
        input: Box<LogicalPlan>,
        by_column: Vec<Expr>,
        args: SortArguments,
    },
    /// Slice the table
    Slice {
        input: Box<LogicalPlan>,
        offset: i64,
        len: IdxSize,
    },
    /// A (User Defined) Function
    MapFunction {
        input: Box<LogicalPlan>,
        function: FunctionNode,
    },
    Union {
        inputs: Vec<LogicalPlan>,
        options: UnionOptions,
    },
    /// Catches errors and throws them later
    #[cfg_attr(feature = "serde", serde(skip))]
    Error {
        input: Box<LogicalPlan>,
        err: ErrorStateSync,
    },
    /// This allows expressions to access other tables
    ExtContext {
        input: Box<LogicalPlan>,
        contexts: Vec<LogicalPlan>,
        schema: SchemaRef,
    },
    FileSink {
        input: Box<LogicalPlan>,
        payload: FileSinkOptions,
    },
}

impl Default for LogicalPlan {
    fn default() -> Self {
        let df = DataFrame::new::<Series>(vec![]).unwrap();
        let schema = df.schema();
        DataFrameScan {
            df: Arc::new(df),
            schema: Arc::new(schema),
            output_schema: None,
            projection: None,
            selection: None,
        }
    }
}

impl LogicalPlan {
    pub fn describe(&self) -> String {
        format!("{self:#?}")
    }

    pub fn to_alp(self) -> PolarsResult<(Node, Arena<ALogicalPlan>, Arena<AExpr>)> {
        let mut lp_arena = Arena::with_capacity(16);
        let mut expr_arena = Arena::with_capacity(16);

        let node = to_alp(self, &mut expr_arena, &mut lp_arena)?;

        Ok((node, lp_arena, expr_arena))
    }
}

use rand::{distributions::Alphanumeric, Rng}; 

impl LogicalPlan {
    pub fn database_query(&self) -> SQLNode {
        fn random_string() -> String {
            rand::thread_rng()
                    .sample_iter(&Alphanumeric)
                    .take(7)
                    .map(char::from)
                    .collect()
        }
        

        fn sql_vec_expr(expr: &Vec<Expr>) -> (Vec<SQLExpr>, Option<SQLNode>) {
            let (vexpr, vnode) : (Vec<_>, Vec<_>)
                        = expr.iter().map(|val| val._database_query()).unzip();
            let node_fold = vnode.into_iter()
                                          .flatten()
                                          .fold(SQLNode::dummy_node().into(), |curr : SQLNode, new_node: SQLNode| curr.add_node(new_node));
            
            if node_fold.is_empty() {
                (vexpr, None)
            }
            else {
                (vexpr, Some(node_fold))
            }
        } 

        // fn sql_vec(expr: &Vec<LogicalPlan>, sep: &str) -> String {
        //     expr.iter().map(|val| val.database_query()).collect::<Vec<String>>().join(sep)
        // }

        match self {
            LogicalPlan::DataFrameScan {df, schema, output_schema, projection, selection} => {
                SQLNode::named_schema_node(schema.clone(), random_string())
            },
            LogicalPlan::Projection {expr, input, schema, options} => {
                let mut dummy_node = input.database_query();

                let (vexpr, node) = sql_vec_expr(expr);
                if !vexpr.is_empty() {
                    let options = SQLSelectOptions {distinct: false};
                    dummy_node = dummy_node.add_query(SQLQuery::Select{vexpr, options});
                }
                
                if let Some(val) = node {
                    val.add_node(dummy_node)
                } 
                else {
                    dummy_node
                }
            },
            LogicalPlan::Slice {input, offset, len} => {
                let mut dummy_node = input.database_query();
                dummy_node.add_query(SQLQuery::Fetch(*len)).add_query(SQLQuery::Offset(*offset))
            },
            LogicalPlan::Distinct {input, options} => {
                let mut dummy_node = input.database_query();
                let options = SQLSelectOptions {distinct: true};
                dummy_node.add_query(SQLQuery::Select{vexpr: vec![], options})
            },
            // LogicalPlan::Selection {input, predicate} => {
            //     format!("SELECT * FROM\n({})\nWHERE {}", 
            //             input.database_query(), 
            //             predicate._database_query()
            //         )
            // },
            // LogicalPlan::Aggregate {input, keys, aggs, schema, apply, maintain_order, options} => {
            //     format!("SELECT {} FROM\n({})\nGROUP BY {}",
            //             _sql_vec(aggs, ", "), 
            //             input.database_query(), 
            //             _sql_vec(keys, ", ")
            //         )
            // },
            // LogicalPlan::Sort {input, by_column, args} => {
            //     format!("SELECT * FROM\n({})\nORDER BY {}", 
            //             input.database_query(), 
            //             _sql_vec(by_column, ", ")
            //         )
            // },
            // LogicalPlan::Join {input_left, input_right, schema, left_on, right_on, options}  => {
            //     let mut on = String::from("");
            //     for i in 0..left_on.len() {
            //         on.push_str(&format!("{} = {}", 
            //                     left_on[i]._database_query(), 
            //                     right_on[i]._database_query()
            //                 ).to_string());
            //     }

            //     format!("SELECT * FROM\n({})\nJOIN ({})\nON {}", 
            //             input_left.database_query(), 
            //             input_right.database_query(), 
            //             on
            //         )
            // },
            // //LogicalPlan::HStack {input, exprs, schema, options} => ,
            // LogicalPlan::Union {inputs, options} => {
            //     format!("{}", sql_vec(inputs, "\nUNION ALL\n"))
            // },
            _ => SQLNode::dummy_node().into(),
            
            
        }
    //     #[cfg_attr(feature = "serde", serde(skip))]
    // AnonymousScan {
    //     function: Arc<dyn AnonymousScan>,
    //     file_info: FileInfo,
    //     predicate: Option<Expr>,
    //     options: Arc<AnonymousScanOptions>,
    // },
    // #[cfg(feature = "python")]
    // PythonScan { options: PythonOptions },

    // /// Cache the input at this point in the LP
    // Cache {
    //     input: Box<LogicalPlan>,
    //     id: usize,
    //     count: usize,
    // },
    // Scan {
    //     path: PathBuf,
    //     file_info: FileInfo,
    //     predicate: Option<Expr>,
    //     file_options: FileScanOptions,
    //     scan_type: FileScan,
    // },

    
    // // a projection that doesn't have to be optimized
    // // or may drop projected columns if they aren't in current schema (after optimization)
    // LocalProjection {
    //     expr: Vec<Expr>,
    //     input: Box<LogicalPlan>,
    //     schema: SchemaRef,
    // },


    // /// A (User Defined) Function
    // MapFunction {
    //     input: Box<LogicalPlan>,
    //     function: FunctionNode,
    // },

    // /// Catches errors and throws them later
    // #[cfg_attr(feature = "serde", serde(skip))]
    // Error {
    //     input: Box<LogicalPlan>,
    //     err: ErrorStateSync,
    // },
    // /// This allows expressions to access other tables
    // ExtContext {
    //     input: Box<LogicalPlan>,
    //     contexts: Vec<LogicalPlan>,
    //     schema: SchemaRef,
    // },
    // FileSink {
    //     input: Box<LogicalPlan>,
    //     payload: FileSinkOptions,
    // },
    }
}

// vec<expr> union function
// generics in rust
// map python function