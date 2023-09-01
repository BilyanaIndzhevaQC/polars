use core::panic;
use std::fmt::{Debug, Display, Result, Formatter};
use std::hash::{Hash, Hasher};

use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use super::expr_dyn_fn::*;
use crate::dsl::function_expr::FunctionExpr;
use crate::prelude::*;

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
    pub fn _database_query(&self) -> (SQLAggExpr, Option<SQLNode>) {
        use AggExpr::*;

        match self {
            Min{input, propagate_nans} => {
                let (expr, node) = input._database_query();
                (SQLAggExpr::Min{input: Box::new(expr), propagate_nans: *propagate_nans},
                    node)
            },
            Max{input, propagate_nans} => {
                let (expr, node) = input._database_query();
                (SQLAggExpr::Max{input: Box::new(expr), propagate_nans: *propagate_nans},
                    node)
            },
            Mean(input) => {
                let (expr, node) = input._database_query();
                (SQLAggExpr::Mean(Box::new(expr)), node)
            },
            Sum(input) => {
                let (expr, node) = input._database_query();
                (SQLAggExpr::Sum(Box::new(expr)), node)
            },
            Count(input) => {
                let (expr, node) = input._database_query();
                (SQLAggExpr::Count(Box::new(expr)), node)
            },
            Median(input) =>  {
                let (expr, node) = input._database_query();
                (SQLAggExpr::Median(Box::new(expr)), node)
            },
            First(input) =>  {
                let (expr, node) = input._database_query();
                (SQLAggExpr::First(Box::new(expr)), node)
            },
            
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
            _ => panic!("Conversion from AggExpr to SQL is not implemented!")

        }
        
    }
}


impl Expr {
    pub fn _database_query(&self) -> (SQLExpr, Option<SQLNode>) {
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

                let expr = SQLExpr::BinaryExpr {left:Box::new(left_sql_expr), op:*op, right:Box::new(right_sql_expr)};
                (expr, sql_node)
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

#[derive(Clone, Debug)]
pub struct SQLSelectOptions {
    pub distinct: bool
}

impl Display for SQLSelectOptions {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let select = if self.distinct {"SELECT DISTINCT"} else {"SELECT"};
        write!(f, "{}", select)
    }
}

#[derive(Clone, Debug)]
pub enum SQLAggExpr {
    Min {
        input: Box<SQLExpr>,
        propagate_nans: bool,
    },
    Max {
        input: Box<SQLExpr>,
        propagate_nans: bool,
    },
    Median(Box<SQLExpr>),
    First(Box<SQLExpr>),
    Mean(Box<SQLExpr>),
    Count(Box<SQLExpr>),
    Sum(Box<SQLExpr>),

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
}

impl Display for SQLAggExpr {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use SQLAggExpr::*;
        let val = match self {
            Min{input, propagate_nans} => format!("MIN({})", input),
            Max{input, propagate_nans} => format!("MAX({})", input),
            Mean(input) => format!("AVG({})", input),
            Sum(input) => format!("SUM({})", input),
            Count(input) => format!("COUNT({})", input),
            Median(input) => format!("MEDIAN({})", input),
            First(input) => format!("{}", input),
        };
        write!(f, "{}", val)
    }
}

impl SQLAggExpr {
    fn get_expr(&self) -> &Box<SQLExpr> {
        use SQLAggExpr::*;
        match self {
            Min{input, propagate_nans} => input,
            Max{input, propagate_nans} => input,
            Mean(input) => input,
            Sum(input) => input,
            Count(input) => input,
            Median(input) => input,
            First(input) => input,
        }

    }
}


#[derive(Clone, Debug)]
pub enum SQLExpr {
    Alias(Box<SQLExpr>, Arc<str>),
    Column(Arc<str>),
    Literal(LiteralValue),
    Agg(SQLAggExpr),
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
            Agg(aggExpr) => format!("{aggExpr}"),
        };
        write!(f, "{}", val)
    }
}

impl SQLExpr {
    fn find_name(&self) -> Arc<str> {
        use SQLExpr::*;
        match self {
            Alias(expr, new_name) => new_name.clone(),
            Column(name) => name.clone(),
            Literal(literal) => "literal".into(),
            BinaryExpr {left, op, right} => left.find_name(),
            Agg(aggExpr) => aggExpr.get_expr().find_name(),
        }
    }

    fn remove_alias(&self) -> Self {
        use SQLExpr::*;
        match self {
            Alias(expr, new_name) => expr.remove_alias(),
            Column(name) => self.clone(),
            Literal(literal) => self.clone(),
            BinaryExpr {left, op, right} => {
                BinaryExpr {left: Box::new(left.remove_alias().clone()), op: *op, right: Box::new(right.remove_alias().clone())}
            },
            Agg(aggExpr) => aggExpr.get_expr().remove_alias(),
        }
    }

    pub fn fix_alias(&self) -> Self {
        SQLExpr::Alias(Box::new(self.remove_alias().clone()), self.find_name())
    }
}

#[derive(Clone, Debug)]
pub enum InsideSQLQuery {
    Name(String),
    Node(SQLNode)
}

impl Display for InsideSQLQuery {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use InsideSQLQuery::*;
        let val = match self {
            Name(string) => format!("{string}"),
            Node(node) => format!("FROM (\n{node}\n)\n")
        };
        write!(f, "{}", val)
    }
}

#[derive(Clone, Debug)]
pub enum SQLQuery {
    From(InsideSQLQuery),
    Join(Arc<SQLNode>),
    Select {
        vexpr: Vec<SQLExpr>, 
        options: SQLSelectOptions
    },
    Where(SQLExpr),
    GroupBy(SQLExpr),
    Having(SQLExpr),
    OrderBy(SQLExpr),
    Offset(i64),
    Fetch(u32),
}


impl SQLQuery {
    fn priority(&self) -> i32 {
        use SQLQuery::*;
        match self {
            From(_) => 2,
            Join(_) => 3,
            Select{vexpr, options} => 1,
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
            Join(node) => format!("JOIN ({node})"),
            Select {vexpr, options} => {
                let mut select = "".to_owned();
                if vexpr.len() == 0 {
                    select.push_str("*");
                }
                else {
                    select.push_str(&vexpr.iter().map(|val| format!("{}", val.fix_alias())).collect::<Vec<String>>().join(", "));
                }
                format!("{options} {select}")
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

#[derive(Clone, Debug)]
pub struct SQLNode {
    pub schema: Option<Arc<SchemaRef>>,
    pub queries: Vec<SQLQuery>
}

impl Display for SQLNode {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut node = self.clone();
        node.sort_by_priority();
        if !matches!(&node.queries[0], SQLQuery::Select{vexpr, options}) {
            write!(f, "SELECT *\n");
        }
        write!(f, "{}", node.queries.iter().map(|val| format!("{val}")).collect::<Vec<String>>().join("\n"))
    }
}

impl SQLNode {
    pub fn sort_by_priority(&mut self) {
        self.queries.sort_by_key(|a| a.priority())
    }

    pub fn is_empty(&self) -> bool {
        self.schema.is_none() && self.queries.is_empty()
    }

    pub fn dummy_node() -> Self {
        Self {
                schema: None,
                queries: vec![]
        }
    }

    pub fn named_dummy_node(name: String) -> Self {
        Self {
                schema: None,
                queries: vec![SQLQuery::From(InsideSQLQuery::Name(name))]
        }
    }

    pub fn named_schema_node(schema: SchemaRef, name: String) -> Self {
        Self {
                schema: Some(schema.into()),
                queries: vec![SQLQuery::From(InsideSQLQuery::Name(name))]
        }
    }

    pub fn add_query(&self, query: SQLQuery) -> SQLNode {
        let mut node = self.clone();
        node.queries.push(query.clone());//add_checks, if exists add new node
        node
    }
    
    pub fn add_node(&self, other: SQLNode) -> SQLNode {
        use SQLQuery::*;
        use InsideSQLQuery::*;

        let mut node = self.clone();

        let index = node.queries.iter().position(|val| matches!(val, SQLQuery::From(_)));
        let mut from = None;

        match index {
            Some(index) => {
                from = Some(node.queries.remove(index));
            },
            None => {
                node.queries.push(SQLQuery::From(Node(other.clone().into())));
            }
        };

        match from {
            Some(From(Name(from_name))) => {
                other.add_node(SQLNode::named_dummy_node(from_name.clone()).into());
            },
            Some(From(Node(from_node))) => {
                other.add_node(from_node);
            },
            _ => {}
        }

        node.into()
    } 
}


