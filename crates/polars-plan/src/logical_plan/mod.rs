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
use crate::dsl::database::*;

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
    pub fn database_query(&self) -> DBNode {
        use LogicalPlan::*;

        // fn random_string() -> String {
        //     rand::thread_rng()
        //             .sample_iter(&Alphanumeric)
        //             .take(7)
        //             .map(char::from)
        //             .collect()
        // }

        fn vector_expr(expr: &Vec<Expr>) -> (Vec<DBExpr>, Option<DBNode>) {
            let (vexpr, vnode) : (Vec<_>, Vec<_>)
                        = expr.iter().map(|val| val._database_query()).unzip();
            let node_fold = vnode.into_iter()
                                          .flatten()
                                          .fold(DBNode::empty_dummy_node().into(), |curr : DBNode, new_node: DBNode| curr.add_node(new_node));
            
            if node_fold.is_empty() {
                (vexpr, None)
            }
            else {
                (vexpr, Some(node_fold))
            }
        } 

        fn vector_remove_alias(vexpr: Vec<DBExpr>) -> Vec<DBExpr> {
            vexpr.iter().map(
                |expr| expr.remove_alias()
            ).collect::<Vec<DBExpr>>()
        }

        fn root_option_node(root_node: Option<DBNode>, tail_node: DBNode) -> DBNode {
            if let Some(root_node) = root_node {
                root_node.add_node(tail_node)
            }
            else {
                tail_node
            }
        }

        // fn 

        // fn sql_vec(expr: &Vec<LogicalPlan>, sep: &str) -> String {
        //     expr.iter().map(|val| val.database_query()).collect::<Vec<String>>().join(sep)
        // }

        match self {
            DataFrameScan {df, schema, output_schema, projection, selection} => {
                let name = match df.name.clone() {
                    Some(name) => name,
                    None => "".to_owned()
                };
                
                DBNode::named_schema_node(schema.clone(), name)
            },
            
            Projection {expr, input, schema, options} => {
                let mut node = input.database_query();

                let (vexpr, expr_node) = vector_expr(expr);
                if !vexpr.is_empty() {
                    let options = DBSelectOptions {distinct: false, top: None};
                    node = node.add_query(DBQuery::Select{vexpr, options});
                }
                
                root_option_node(expr_node, node)
            },

            Slice {input, offset, len} => {
                let node = input.database_query();
                node.add_query(DBQuery::Fetch(*len)).add_query(DBQuery::Offset(*offset))
            },

            Distinct {input, options} => {
                let node = input.database_query();
                node.add_query(DBQuery::Select{
                    vexpr: vec![], 
                    options: DBSelectOptions {distinct: true, top: None}
                })
            },

            Selection {input, predicate} => {
                let mut node = input.database_query();
                let (expr, expr_node) = predicate._database_query();
                
                root_option_node(expr_node, node).add_query(DBQuery::Where(expr.remove_alias()))
            },

            HStack {input, exprs, schema, options} => {
                let mut node = input.database_query();

                let (mut vexpr, expr_node) = vector_expr(exprs);

                let schema = input.schema().unwrap();
                let names = schema.get_names().clone();
                let mut from_names = names.iter().map(
                    |&name| DBExpr::Column(name.into())
                ).collect::<Vec<DBExpr>>();
                
                from_names.append(&mut vexpr);
                vexpr = from_names;

                if !vexpr.is_empty() {
                    node = node.add_query(DBQuery::Select{
                        vexpr, 
                        options: DBSelectOptions {distinct: false, top: None}
                    });
                }
                
                root_option_node(expr_node, node)
            },

            Aggregate {input, keys, aggs, schema, apply, maintain_order, options} => {
                let mut node = input.database_query();

                let (mut vexpr_aggs, expr_node_agg) = vector_expr(aggs);
                let (mut vexpr_keys, expr_node_keys) = vector_expr(keys);

                vexpr_aggs.append(&mut vexpr_keys.clone());

                node = node.add_query(DBQuery::Select {
                    vexpr: vexpr_aggs, 
                    options: DBSelectOptions {distinct: false, top: None}
                });

                vexpr_keys = vector_remove_alias(vexpr_keys);

                node = root_option_node(expr_node_agg, node);
                node = root_option_node(expr_node_keys, node);
                
                node.add_query(DBQuery::GroupBy(vexpr_keys.clone()))
            },

            Sort {input, by_column, args} => {
                let mut node = input.database_query();

                let (mut vexpr, expr_node) = vector_expr(by_column);
                vexpr = vector_remove_alias(vexpr);
                node = root_option_node(expr_node, node);

                let mut queries: Vec<DBSortQuery> = vec![];

                for i in 0..vexpr.len() {
                    let options = DBSortOptions {descending: args.descending[i], nulls_last: args.nulls_last};
                    queries.push(DBSortQuery {expr: vexpr[i].clone(), options});
                }

                node.add_query(DBQuery::OrderBy(queries))
            },
            Join {input_left, input_right, schema, left_on, right_on, options}  => {
                let mut node_left = input_left.database_query();

                let (mut vexpr_left, expr_node_left) = vector_expr(left_on);
                vexpr_left = vector_remove_alias(vexpr_left);
                node_left = root_option_node(expr_node_left, node_left);

                let mut node_right = input_right.database_query();
                
                let (mut vexpr_right, expr_node_right) = vector_expr(right_on);
                vexpr_right = vector_remove_alias(vexpr_right);                
                node_right = root_option_node(expr_node_right, node_right);

                let how = match options.args.how {
                    JoinType::Left => DBJoinType::Left,
                    JoinType::Inner => DBJoinType::Inner,
                    JoinType::Outer => DBJoinType::Outer,
                    _ => panic!("Join type conversion to SQL not implemented yet!")
                    // AsOf(AsOfOptions),
                    // Cross,
                    // Semi,
                    // Anti,
                };

                let node_join = DBNode::query_dummy_node(
                    DBQuery::Join { 
                        node: node_right.into(), 
                        left_on: vexpr_left,
                        right_on: vexpr_right,
                        options: DBJoinOptions {how: how}
                });

                node_join.add_node(node_left)
            },
            _ => {
                panic!("LogicalPlan cannot be converted to database query. {self:?}")
            }
            
            
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