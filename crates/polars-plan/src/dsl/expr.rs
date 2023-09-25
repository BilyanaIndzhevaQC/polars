use core::panic;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use polars_core::prelude::*;
use rayon::option;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use super::expr_dyn_fn::*;
use crate::dsl::function_expr::FunctionExpr;
use crate::prelude::*;
use crate::dsl::database::*;

#[derive(PartialEq, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AggExpr {
    Min {
        input: Box<Expr>,
        propagate_nans: bool,
    },
    Max {
        input: Box<Expr>,
        propagate_nans: bool,
    },
    Median(Box<Expr>),
    NUnique(Box<Expr>),
    First(Box<Expr>),
    Last(Box<Expr>),
    Mean(Box<Expr>),
    Implode(Box<Expr>),
    Count(Box<Expr>),
    Quantile {
        expr: Box<Expr>,
        quantile: Box<Expr>,
        interpol: QuantileInterpolOptions,
    },
    Sum(Box<Expr>),
    AggGroups(Box<Expr>),
    Std(Box<Expr>, u8),
    Var(Box<Expr>, u8),
}

impl AsRef<Expr> for AggExpr {
    fn as_ref(&self) -> &Expr {
        use AggExpr::*;
        match self {
            Min { input, .. } => input,
            Max { input, .. } => input,
            Median(e) => e,
            NUnique(e) => e,
            First(e) => e,
            Last(e) => e,
            Mean(e) => e,
            Implode(e) => e,
            Count(e) => e,
            Quantile { expr, .. } => expr,
            Sum(e) => e,
            AggGroups(e) => e,
            Std(e, _) => e,
            Var(e, _) => e,
        }
    }
}

/// Expressions that can be used in various contexts. Queries consist of multiple expressions. When using the polars
/// lazy API, don't construct an `Expr` directly; instead, create one using the functions in the `polars_lazy::dsl`
/// module. See that module's docs for more info.
#[derive(Clone, PartialEq)] 
#[must_use]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Expr {
    Alias(Box<Expr>, Arc<str>),
    Column(Arc<str>),
    Columns(Vec<String>),
    DtypeColumn(Vec<DataType>),
    Literal(LiteralValue),
    BinaryExpr {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    },
    Cast {
        expr: Box<Expr>,
        data_type: DataType,
        strict: bool,
    },
    Sort {
        expr: Box<Expr>,
        options: SortOptions,
    },
    Take {
        expr: Box<Expr>,
        idx: Box<Expr>,
    },
    SortBy {
        expr: Box<Expr>,
        by: Vec<Expr>,
        descending: Vec<bool>,
    },
    Agg(AggExpr),
    /// A ternary operation
    /// if true then "foo" else "bar"
    Ternary {
        predicate: Box<Expr>,
        truthy: Box<Expr>,
        falsy: Box<Expr>,
    },
    Function {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: FunctionExpr,
        options: FunctionOptions,
    },
    Explode(Box<Expr>),
    Filter {
        input: Box<Expr>,
        by: Box<Expr>,
    },
    /// See postgres window functions
    Window {
        /// Also has the input. i.e. avg("foo")
        function: Box<Expr>,
        partition_by: Vec<Expr>,
        order_by: Option<Box<Expr>>,
        options: WindowOptions,
    },
    Wildcard,
    Slice {
        input: Box<Expr>,
        /// length is not yet known so we accept negative offsets
        offset: Box<Expr>,
        length: Box<Expr>,
    },
    /// Can be used in a select statement to exclude a column from selection
    Exclude(Box<Expr>, Vec<Excluded>),
    /// Set root name as Alias
    KeepName(Box<Expr>),
    /// Special case that does not need columns
    Count,
    /// Take the nth column in the `DataFrame`
    Nth(i64),
    // skipped fields must be last otherwise serde fails in pickle
    #[cfg_attr(feature = "serde", serde(skip))]
    RenameAlias {
        function: SpecialEq<Arc<dyn RenameAliasFn>>,
        expr: Box<Expr>,
    },
    AnonymousFunction {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: SpecialEq<Arc<dyn SeriesUdf>>,
        /// output dtype of the function
        #[cfg_attr(feature = "serde", serde(skip))]
        output_type: GetOutput,
        options: FunctionOptions,
    },
    /// Expressions in this node should only be expanding
    /// e.g.
    /// `Expr::Columns`
    /// `Expr::Dtypes`
    /// `Expr::Wildcard`
    /// `Expr::Exclude`
    Selector(super::selector::Selector),
}

#[allow(clippy::derived_hash_with_manual_eq)]
impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let d = std::mem::discriminant(self);
        d.hash(state);
        match self {
            Expr::Column(name) => name.hash(state),
            Expr::Columns(names) => names.hash(state),
            Expr::DtypeColumn(dtypes) => dtypes.hash(state),
            Expr::Literal(lv) => std::mem::discriminant(lv).hash(state),
            Expr::Selector(s) => s.hash(state),
            Expr::Nth(v) => v.hash(state),
            Expr::Filter { input, by } => {
                input.hash(state);
                by.hash(state);
            },
            Expr::BinaryExpr { left, op, right } => {
                left.hash(state);
                right.hash(state);
                std::mem::discriminant(op).hash(state)
            },
            Expr::Cast {
                expr,
                data_type,
                strict,
            } => {
                expr.hash(state);
                data_type.hash(state);
                strict.hash(state)
            },
            Expr::Sort { expr, options } => {
                expr.hash(state);
                options.hash(state);
            },
            Expr::Alias(input, name) => {
                input.hash(state);
                name.hash(state)
            },
            Expr::KeepName(input) => input.hash(state),
            Expr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                predicate.hash(state);
                truthy.hash(state);
                falsy.hash(state);
            },
            Expr::Function {
                input,
                function,
                options,
            } => {
                input.hash(state);
                std::mem::discriminant(function).hash(state);
                options.hash(state);
            },
            // already hashed by discriminant
            Expr::Wildcard | Expr::Count => {},
            #[allow(unreachable_code)]
            _ => {
                // the panic checks if we hit this
                #[cfg(debug_assertions)]
                {
                    todo!("IMPLEMENT")
                }
                // TODO! derive. This is only a temporary fix
                // Because PartialEq will have a lot of `false`, e.g. on Function
                // Types, this may lead to many file reads, as we use predicate comparison
                // to check if we can cache a file
                let s = format!("{self:?}");
                s.hash(state)
            },
        }
    }
}

impl Eq for Expr {}

impl Default for Expr {
    fn default() -> Self {
        Expr::Literal(LiteralValue::Null)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]

pub enum Excluded {
    Name(Arc<str>),
    Dtype(DataType),
}

impl Expr {
    /// Get Field result of the expression. The schema is the input data.
    pub fn to_field(&self, schema: &Schema, ctxt: Context) -> PolarsResult<Field> {
        // this is not called much and th expression depth is typically shallow
        let mut arena = Arena::with_capacity(5);
        self.to_field_amortized(schema, ctxt, &mut arena)
    }
    pub(crate) fn to_field_amortized(
        &self,
        schema: &Schema,
        ctxt: Context,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Field> {
        let root = to_aexpr(self.clone(), expr_arena);
        expr_arena.get(root).to_field(schema, ctxt, expr_arena)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Operator {
    Eq,
    EqValidity,
    NotEq,
    NotEqValidity,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Multiply,
    Divide,
    TrueDivide,
    FloorDivide,
    Modulus,
    And,
    Or,
    Xor,
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Operator::*;
        let tkn = match self {
            Eq => "==",
            EqValidity => "==v",
            NotEq => "!=",
            NotEqValidity => "!=v",
            Lt => "<",
            LtEq => "<=",
            Gt => ">",
            GtEq => ">=",
            Plus => "+",
            Minus => "-",
            Multiply => "*",
            Divide => "//",
            TrueDivide => "/",
            FloorDivide => "floor_div",
            Modulus => "%",
            And => "&",
            Or => "|",
            Xor => "^",
        };
        write!(f, "{tkn}")
    }
}

impl Operator {
    pub(crate) fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Eq
                | Self::NotEq
                | Self::Lt
                | Self::LtEq
                | Self::Gt
                | Self::GtEq
                | Self::And
                | Self::Or
                | Self::Xor
                | Self::EqValidity
                | Self::NotEqValidity
        )
    }

    pub(crate) fn is_arithmetic(&self) -> bool {
        !(self.is_comparison())
    }
}

impl AggExpr {
    pub fn _database_query(&self) -> (DBAggExpr, Option<DBNode>) {
        match self {
            AggExpr::Min{input, propagate_nans} => {
                let (expr, node) = input._database_query();
                (DBAggExpr::Min{input: Box::new(expr), propagate_nans: *propagate_nans},
                    node)
            },
            AggExpr::Max{input, propagate_nans} => {
                let (expr, node) = input._database_query();
                (DBAggExpr::Max{input: Box::new(expr), propagate_nans: *propagate_nans},
                    node)
            },
            AggExpr::Mean(input) => {
                let (expr, node) = input._database_query();
                (DBAggExpr::Mean(Box::new(expr)), node)
            },
            AggExpr::Sum(input) => {
                let (expr, node) = input._database_query();
                (DBAggExpr::Sum(Box::new(expr)), node)
            },
            AggExpr::Count(input) => {
                let (expr, node) = input._database_query();
                (DBAggExpr::Count(Box::new(expr)), node)
            },
            AggExpr::Median(input) =>  {
                let (expr, node) = input._database_query();
                (DBAggExpr::Median(Box::new(expr)), node)
            },
            AggExpr::First(input) =>  {
                let (expr, node) = input._database_query();
                (DBAggExpr::First(Box::new(expr)), node)
            },
            
            // NUnique(Box<Expr>),
            // Last(Box<Expr>),
            // Implode(Box<Expr>),
            // Quantile {
            //     expr: Box<Expr>,
            //     quantile: Box<Expr>,
            //     interpol: QuantileInterpolOptions,
            // },
            // AggGroups(Box<Expr>),
            // Std(Box<Expr>, u8),
            // Var(Box<Expr>, u8),
            _ => panic!("Conversion from AggExpr to SQL is not implemented!")

        }
        
    }
}


impl Expr {
    pub fn _database_query(&self) -> (DBExpr, Option<DBNode>) {
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
 
        match self {
            Expr::Alias(expr, new_name) => {
                let (sql_expr, sql_node) = expr._database_query();
                (DBExpr::Alias(Box::new(sql_expr), new_name.clone()), sql_node)
            },
            Expr::Column(name) => {
                (DBExpr::Column(name.clone()), None)
            },
            Expr::Literal(literal) => {
                (DBExpr::Literal(literal.clone()), None)
            },
            Expr::BinaryExpr {left, op, right} => {
                let (left_sql_expr, left_sql_node) = left._database_query();
                let (right_sql_expr, right_sql_node) = right._database_query();
                
                let sql_node = 
                if let Some(left_val) = left_sql_node {
                    if let Some(right_val) = right_sql_node {
                        Some(left_val.add_node(right_val))
                    } else {
                        Some(left_val)
                    }
                } else {
                    right_sql_node
                }; 

                let expr = DBExpr::BinaryExpr {left:Box::new(left_sql_expr), op:*op, right:Box::new(right_sql_expr)};
                (expr, sql_node)
            }
            Expr::Agg(aggExpr) => {
                let (sql_aexpr, sql_node) = aggExpr._database_query();
                (DBExpr::Agg(sql_aexpr), sql_node)
            },
            Expr::Sort {
                expr,
                options,
            } => {
                let (mut expr, expr_node) = expr._database_query();
                
                let mut options: DBSortOptions = DBSortOptions {descending: options.descending, nulls_last: options.nulls_last};

                let query = vec![DBSortQuery {expr: expr.remove_alias(), options}];
                let node = DBNode::query_dummy_node(DBQuery::OrderBy(query)); 

                (expr, Some(root_option_node(expr_node, node)))
            },
            Expr::SortBy {
                expr,
                by,
                descending,       
            }  => {
                let (mut expr, option_node) = expr._database_query();

                let (mut vexpr, expr_node) = vector_expr(by);
                vexpr = vector_remove_alias(vexpr);


                let mut queries: Vec<DBSortQuery> = vec![];

                for i in 0..vexpr.len() {
                    let options = DBSortOptions {descending: descending[i], nulls_last: false};
                    queries.push(DBSortQuery {expr: vexpr[i].clone(), options});
                }

                let mut node = DBNode::query_dummy_node(DBQuery::OrderBy(queries));
                node = root_option_node(option_node, node);
                (expr, Some(root_option_node(expr_node, node)))

            },
            Expr::Filter {
                input,
                by,
            } => {
                let (mut expr, expr_node) = input._database_query();
                let (by_expr, by_expr_node) = by._database_query();
                
                let mut node = DBNode::query_dummy_node(DBQuery::Where(by_expr.remove_alias()));
                node = root_option_node(by_expr_node, node);
                (expr, Some(root_option_node(expr_node, node)))
            },
            _ => panic!("Expr cannot be converted to database query.") //panic

            // Slice {
            //     input: Box<Expr>,
            //     /// length is not yet known so we accept negative offsets
            //     offset: Box<Expr>,
            //     length: Box<Expr>,
            // },
            // Ternary {
            //     predicate: Box<Expr>,
            //     truthy: Box<Expr>,
            //     falsy: Box<Expr>,
            // },
            // Cast {
            //     expr: Box<Expr>,
            //     data_type: DataType,
            //     strict: bool,
            // },
            // Take {
            //     expr: Box<Expr>,
            //     idx: Box<Expr>,
            // },

            // Function {
            //     /// function arguments
            //     input: Vec<Expr>,
            //     /// function to apply
            //     function: FunctionExpr,
            //     options: FunctionOptions,
            // },

            // Explode(Box<Expr>),

            // /// See postgres window functions
            // Window {
            //     /// Also has the input. i.e. avg("foo")
            //     function: Box<Expr>,
            //     partition_by: Vec<Expr>,
            //     order_by: Option<Box<Expr>>,
            //     options: WindowOptions,
            // },

            // Wildcard,

            // /// Special case that does not need columns
            // Count,

            // /// Take the nth column in the `DataFrame`
            // Nth(i64),
        
            // AnonymousFunction {
            //     /// function arguments
            //     input: Vec<Expr>,
            //     /// function to apply
            //     function: SpecialEq<Arc<dyn SeriesUdf>>,
            //     /// output dtype of the function
            //     #[cfg_attr(feature = "serde", serde(skip))]
            //     output_type: GetOutput,
            //     options: FunctionOptions,
            // },

            


            
            //panic
            // // skipped fields must be last otherwise serde fails in pickle
            // #[cfg_attr(feature = "serde", serde(skip))]
            // RenameAlias {
            //     function: SpecialEq<Arc<dyn RenameAliasFn>>,
            //     expr: Box<Expr>,
            // },
            // /// Can be used in a select statement to exclude a column from selection
            // Exclude(Box<Expr>, Vec<Excluded>),
            // /// Set root name as Alias
            // KeepName(Box<Expr>),
            // Expr::Columns(names) => names.iter().map(|val| format!("{}", val)).collect::<Vec<String>>().join(", "),
            // Expr::DtypeColumn(dtypes: Vec<DataType>) => 
            // Selector(super::selector::Selector),
        }
    }
}
