use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use arrow::compute::filter::filter_chunk;
#[cfg(feature = "parquet")]
use polars_core::cloud::CloudOptions;
use polars_core::prelude::*;
use rayon::option;

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
    pub fn database_query(&self, table: &String) -> SQLNode {
        use LogicalPlan::*;

        // fn random_string() -> String {
        //     rand::thread_rng()
        //             .sample_iter(&Alphanumeric)
        //             .take(7)
        //             .map(char::from)
        //             .collect()
        // }

        fn vec_expr(expr: &Vec<Expr>) -> (Vec<SQLExpr>, Option<SQLNode>) {
            let (vexpr, vnode) : (Vec<_>, Vec<_>)
                        = expr.iter().map(|val| val._database_query()).unzip();
            let node_fold = vnode.into_iter()
                                          .flatten()
                                          .fold(SQLNode::empty_dummy_node().into(), |curr : SQLNode, new_node: SQLNode| curr.add_node(new_node));
            
            if node_fold.is_empty() {
                (vexpr, None)
            }
            else {
                (vexpr, Some(node_fold))
            }
        } 

        // fn i

        // fn sql_vec(expr: &Vec<LogicalPlan>, sep: &str) -> String {
        //     expr.iter().map(|val| val.database_query()).collect::<Vec<String>>().join(sep)
        // }

        match self {
            DataFrameScan {df, schema, output_schema, projection, selection} => {
                SQLNode::named_schema_node(schema.clone(), table.to_string())
            },
            Projection {expr, input, schema, options} => {
                let mut node = input.database_query(table);

                let (vexpr, expr_node) = vec_expr(expr);
                if !vexpr.is_empty() {
                    let options = SQLSelectOptions {distinct: false};
                    node = node.add_query(SQLQuery::Select{vexpr, options});
                }
                
                if let Some(expr_node) = expr_node {
                    node = expr_node.add_node(node);
                }
                node
            },

            Slice {input, offset, len} => {
                let node = input.database_query(table);
                node.add_query(SQLQuery::Fetch(*len)).add_query(SQLQuery::Offset(*offset))
            },

            Distinct {input, options} => {
                let node = input.database_query(table);
                node.add_query(SQLQuery::Select{
                    vexpr: vec![], 
                    options: SQLSelectOptions {distinct: true}
                })
            },

            Selection {input, predicate} => {
                let mut node = input.database_query(table);
                let (expr, expr_node) = predicate._database_query();
                
                if let Some(expr_node) = expr_node {
                    node = expr_node.add_node(node);
                }

                node.add_query(SQLQuery::Where(expr.remove_alias()))
            },

            HStack {input, exprs, schema, options} => {
                let mut node = input.database_query(table);

                let (mut vexpr, expr_node) = vec_expr(exprs);

                let schema = input.schema().unwrap();
                let names = schema.get_names().clone();
                let mut from_names = names.iter().map(
                    |&name| SQLExpr::Column(name.into())
                ).collect::<Vec<SQLExpr>>();
                
                from_names.append(&mut vexpr);
                vexpr = from_names;

                if !vexpr.is_empty() {
                    node = node.add_query(SQLQuery::Select{
                        vexpr, 
                        options: SQLSelectOptions {distinct: false}
                    });
                }
                
                if let Some(expr_node) = expr_node {
                    node = expr_node.add_node(node);
                }
                node
            },

            LogicalPlan::Aggregate {input, keys, aggs, schema, apply, maintain_order, options} => {
                let mut node = input.database_query(table);

                let (mut vexpr_aggs, expr_node_agg) = vec_expr(aggs);
                let (mut vexpr_keys, expr_node_keys) = vec_expr(keys);

                vexpr_aggs.append(&mut vexpr_keys.clone());

                node = node.add_query(SQLQuery::Select{
                    vexpr: vexpr_aggs, 
                    options: SQLSelectOptions {distinct: false}
                });

                vexpr_keys = vexpr_keys.iter().map(
                    |expr| expr.remove_alias()
                ).collect::<Vec<SQLExpr>>();

                node = node.add_query(SQLQuery::GroupBy(vexpr_keys.clone()));
                
                if let Some(expr_node_agg) = expr_node_agg {
                    node = expr_node_agg.add_node(node);
                } 

                if let Some(expr_node_keys) = expr_node_keys {
                    node = expr_node_keys.add_node(node);
                } 
                node
            },

            LogicalPlan::Sort {input, by_column, args} => {
                let mut node = input.database_query(table);

                let (mut vexpr, expr_node) = vec_expr(by_column);
                vexpr = vexpr.iter().map(
                    |expr| expr.remove_alias()
                ).collect::<Vec<SQLExpr>>();
                
                node = node.add_query(SQLQuery::OrderBy(vexpr));
                
                if let Some(expr_node) = expr_node {
                    node = expr_node.add_node(node)
                } 
                node
            },
            LogicalPlan::Join {input_left, input_right, schema, left_on, right_on, options}  => {
                let mut node_left = input_left.database_query(table);

                let (mut vexpr_left, expr_node_left) = vec_expr(left_on);
                vexpr_left = vexpr_left.iter().map(
                    |expr| expr.remove_alias()
                ).collect::<Vec<SQLExpr>>();
                
                if let Some(expr_node_left) = expr_node_left {
                    node_left = expr_node_left.add_node(node_left);
                } 

                let mut node_right = input_right.database_query(table);
                
                let (mut vexpr_right, expr_node_right) = vec_expr(right_on);
                vexpr_right = vexpr_right.iter().map(
                    |expr| expr.remove_alias()
                ).collect::<Vec<SQLExpr>>();
                
                if let Some(expr_node_right) = expr_node_right {
                    node_right = expr_node_right.add_node(node_right);
                } 

                let how = match options.args.how {
                    JoinType::Left => SQLJoinType::Left,
                    JoinType::Inner => SQLJoinType::Inner,
                    JoinType::Outer => SQLJoinType::Outer,
                    _ => panic!("Join type conversion to SQL not implemented yet!")
                    // AsOf(AsOfOptions),
                    // Cross,
                    // Semi,
                    // Anti,
                };

                let node_join = SQLNode::query_dummy_node(
                    SQLQuery::Join { 
                        node: node_right.into(), 
                        left_on: vexpr_left,
                        right_on: vexpr_right,
                        options: SQLJoinOptions {how: how}
                    });

                    node_join.add_node(node_left)
            },
            _ => panic!("LogicalPlan cannot be converted to database query.")
            
            
        }
    //#[cfg_attr(feature = "serde", serde(skip))]
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
    // LogicalPlan::Union {inputs, options} => {
        
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