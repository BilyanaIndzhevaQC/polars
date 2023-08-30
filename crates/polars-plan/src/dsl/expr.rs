use std::fmt::{Debug, Display, Result, Formatter};
use std::hash::{Hash, Hasher};

use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use super::expr_dyn_fn::*;
use crate::dsl::function_expr::FunctionExpr;
use crate::prelude::*;

#[derive(PartialEq, Clone)]
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

pub enum SQLExpr {
    Alias(Box<SQLExpr>, Arc<str>),
    Column(Arc<str>),
    Literal(LiteralValue),
    Agg(AggExpr),
    BinaryExpr {
        left: Box<SQLExpr>,
        op: Operator,
        right: Box<SQLExpr>,
    },

    // Take {
    //     expr: Box<Expr>,
    //     idx: Box<Expr>,
    // },

    // /// A ternary operation
    // /// if true then "foo" else "bar"
    // Ternary {
    //     predicate: Box<Expr>,
    //     truthy: Box<Expr>,
    //     falsy: Box<Expr>,
    // },
    // Function {
    //     /// function arguments
    //     input: Vec<Expr>,
    //     /// function to apply
    //     function: FunctionExpr,
    //     options: FunctionOptions,
    // },
    // Filter {
    //     input: Box<Expr>,
    //     by: Box<Expr>,
    // },
    // /// See postgres window functions
    // Window {
    //     /// Also has the input. i.e. avg("foo")
    //     function: Box<Expr>,
    //     partition_by: Vec<Expr>,
    //     order_by: Option<Box<Expr>>,
    //     options: WindowOptions,
    // },
    // Wildcard,
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
}

impl Display for SQLExpr {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use SQLExpr::*;
        let val = match self {
            Alias(expr, new_name) => format!("{expr} AS {new_name}"),
            Column(name) => format!("{name}"),
            Literal(literal) => format!("{literal}"),
            BinaryExpr {left, op, right} => format!("({left}) {op} ({right})"),
            Agg(AggExpr) => AggExpr._database_query(),
        };
        write!(f, "{}", val)
    }
}


pub enum InsideSQLQuery {
    Name(String),
    Node(Arc<SQLNode>)
}

impl Display for InsideSQLQuery {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use InsideSQLQuery::*;
        let val = match self {
            Name(string) => format!("{string}"),
            Name(node) => format!("{node}")
        };
        write!(f, "{}", val)
    }
}

pub enum SQLQuery {
    From(InsideSQLQuery),
    Join(Arc<SQLNode>),
    Select(Vec<SQLExpr>),
    Where(SQLExpr),
    GroupBy(SQLExpr),
    Having(SQLExpr),
    OrderBy(SQLExpr),
    Offset(i32),
    Fetch(i32),
}


impl SQLQuery {
    fn priority(&self) -> i32 {
        use SQLQuery::*;
        match self {
            From(_) => 1,
            Join(_) => 2,
            Select(_) => 3,
            Where(_) => 4,
            GroupBy(_) => 5,
            Having(_) => 6,
            OrderBy(_) => 7,
            Offset(_) => 8,
            Fetch(_) => 9
        }
    }
}


impl Display for SQLQuery {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use SQLQuery::*;
        let val = match self {
            From(val) => format!("FROM {val}"),
            Join(node) => format!("JOIN {node}"),
            Select(vexpr) => {
                let mut select = "SELECT ".to_owned();
                if vexpr.len() == 0 {
                    select.push_str("*");
                }
                else {
                    select.push_str(&vexpr.iter().map(|val| format!("{val}")).collect::<Vec<String>>().join(", "));
                }
                format!("SELECT {select}")
            },
            Where(expr) => format!("WHERE {expr}"),
            GroupBy(expr) => format!("GROUP BY {expr}"),
            Having(expr) => format!("HAVING {expr}"),
            OrderBy(expr) => format!("ORDER BY {expr}"),
            Offset(num) => format!("OFFSET {num}"),
            Fetch(num) => format!("LIMIT {num}")
        };
        write!(f, "{}", val)
    }
}


pub struct SQLNode {
    schema: SchemaRef,
    queries: Vec<SQLQuery>
}

impl Display for SQLNode {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.queries.iter().map(|val| format!("{val}")).collect::<Vec<String>>().join("\n"))
    }
}

impl SQLNode {
    fn add_node(&self, other: &SQLNode) -> &Self {

    //     bool has_from = false;
    //     for query in self.queries {
    //         match query {
    //             From(Name(string)) => {
    //                 *query = From(Node(other));
    //                 other.queries.push(From(Name(string)));
    //                 has_from = true;
    //             },
    //             From(Node(node)) => {
    //                 node.add_node(other);
    //                 has_from = true;
    //             },
    //             _ => {}
    //         };
    //     }
    //     if (!has_from) {
    //         self.queries.push(SQLQuery::From(Node(other)));
    //     }
    } 
}


impl AggExpr {
    pub fn _database_query(&self) -> String {
        match self {
            // AggExpr::Min{input, propagate_nans} => format!("MIN({})", input._database_query()),
            // AggExpr::Max{input, propagate_nans} => format!("MAX({})", input._database_query()),
            // AggExpr::Mean(input) => format!("AVG({})", input._database_query()),
            // AggExpr::Sum(input) => format!("SUM({})", input._database_query()),
            // AggExpr::Count(input) => format!("COUNT({})", input._database_query()),
            // AggExpr::Median(input) => format!("MEDIAN({})", input._database_query()),
            
            // NUnique(Box<Expr>),
            // First(Box<Expr>),
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
            _ => format!("\nPANIC!!!\n") //panic

        }
        
    }
}


impl Expr {
    pub fn _database_query(&self) -> (SQLExpr, Option<&mut SQLNode>) {
        match self {
            Expr::Alias(expr, new_name) => {
                let (sql_expr, sql_node) = expr._database_query();
                (SQLExpr::Alias(Box::new(sql_expr), new_name.clone()), sql_node)
            },
            Expr::Column(name) => (SQLExpr::Column(name.clone()), None),
            Expr::Literal(literal) => (SQLExpr::Literal(literal.clone()), None),


            Expr::BinaryExpr {left, op, right} => {
                let (left_sql_expr, left_sql_node) = left._database_query();
                let (right_sql_expr, right_sql_node) = right._database_query();
                
                let sql_node = match left_sql_node { //if let
                    Some(left_sql_node) => match right_sql_node {
                        Some(right_sql_node) => Some(left_sql_node.add_node(right_sql_node)),
                        None => Some(left_sql_expr)
                    },
                    None => right_sql_expr
                }; 

                (SQLExpr::BinaryExpr(Box::new(left_sql_expr), op, Box::new(right_sql_expr)), sql_node)
            }
            // Expr::Agg(AggExpr) => AggExpr._database_query(),
            _ => panic!("Expr cannot be converted to database query.") //panic
            
            // Sort { //maybe useless
            //     expr: Box<Expr>,
            //     options: SortOptions, //to impl
            // } => format!("ORDER BY {}", expr._database_query()),
            // SortBy {
            //     expr: Box<Expr>,
            //     by: Vec<Expr>,
            //     descending: Vec<bool>,       
            // }  => format!("SELECT {}\nORDER BY {}", expr.iter().map(_database_query()).collect::<Vec<String>>, by.),
            // Cast {
            //     expr: Box<Expr>,
            //     data_type: DataType,
            //     strict: bool,
            // },
            // Take {
            //     expr: Box<Expr>,
            //     idx: Box<Expr>,
            // },
            
            // /// A ternary operation
            // /// if true then "foo" else "bar"
            // Ternary {
            //     predicate: Box<Expr>,
            //     truthy: Box<Expr>,
            //     falsy: Box<Expr>,
            // },
            // Function {
            //     /// function arguments
            //     input: Vec<Expr>,
            //     /// function to apply
            //     function: FunctionExpr,
            //     options: FunctionOptions,
            // },

            // Explode(Box<Expr>),
            
            // Filter {
            //     input: Box<Expr>,
            //     by: Box<Expr>,
            // },

            // /// See postgres window functions
            // Window {
            //     /// Also has the input. i.e. avg("foo")
            //     function: Box<Expr>,
            //     partition_by: Vec<Expr>,
            //     order_by: Option<Box<Expr>>,
            //     options: WindowOptions,
            // },

            // Wildcard,

            // Slice {
            //     input: Box<Expr>,
            //     /// length is not yet known so we accept negative offsets
            //     offset: Box<Expr>,
            //     length: Box<Expr>,
            // },

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
            // /// Expressions in this node should only be expanding
            // /// e.g.
            // /// `Expr::Columns`
            // /// `Expr::Dtypes`
            // /// `Expr::Wildcard`
            // /// `Expr::Exclude`



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
            //Expr::Columns(names) => names.iter().map(|val| format!("{}", val)).collect::<Vec<String>>().join(", "),
            // Expr::DtypeColumn(dtypes: Vec<DataType>) => 
            // Selector(super::selector::Selector),
        }
    }
}