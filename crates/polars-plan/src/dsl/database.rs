use std::fmt::{Debug, Display, Result, Formatter};

use polars_core::prelude::*;

pub use super::expr_dyn_fn::*;
use crate::prelude::*;


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
pub struct SQLSortOptions {
    pub descending: bool,
    pub nulls_last: bool,
}

impl Display for SQLSortOptions {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut output = "".to_owned();
        if self.descending {
            output += " DESC";
        }
        if self.nulls_last {
            output += " NULLS LAST";
        }
        write!(f, "{}", output)
    }
}


#[derive(Clone, Debug)]
pub struct SQLSortQuery {
    pub expr: SQLExpr,
    pub options: SQLSortOptions
}

impl Display for SQLSortQuery {
    fn fmt(&self, f: &mut Formatter) -> Result {
        
        write!(f, "{}{}", self.expr, self.options)
    }
}


#[derive(Clone, Debug)]
pub enum SQLJoinType {
    Left,
    Right,
    Inner,
    Outer,
    Union,
}


impl Display for SQLJoinType {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use SQLJoinType::*;
        let val = match self {
            Left => "LEFT JOIN",
            Right => "RIGHT JOIN",
            Inner => "INNER JOIN",
            Outer => "OUTER JOIN",
            Union => "UNION",
        };

        write!(f, "{}", val)
    }
}

#[derive(Clone, Debug)]
pub struct SQLJoinOptions {
    pub how: SQLJoinType
}


impl Display for SQLJoinOptions {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.how)
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

    pub fn remove_alias(&self) -> Self {
        use SQLAggExpr::*;
        match self {
            Min{input, propagate_nans} => Min{input: Box::new(input.remove_alias()), propagate_nans: *propagate_nans},
            Max{input, propagate_nans} => Max{input: Box::new(input.remove_alias()), propagate_nans: *propagate_nans},
            Mean(input) => Mean(Box::new(input.remove_alias())),
            Sum(input) => Sum(Box::new(input.remove_alias())),
            Count(input) => Count(Box::new(input.remove_alias())),
            Median(input) => Median(Box::new(input.remove_alias())),
            First(input) => First(Box::new(input.remove_alias())),
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

    pub fn remove_alias(&self) -> Self {
        use SQLExpr::*;
        match self {
            Alias(expr, new_name) => expr.remove_alias(),
            Column(name) => self.clone(),
            Literal(literal) => self.clone(),
            BinaryExpr {left, op, right} => {
                BinaryExpr {left: Box::new(left.remove_alias().clone()), op: *op, right: Box::new(right.remove_alias().clone())}
            },
            Agg(aggExpr) => Agg(aggExpr.remove_alias()),
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
            Name(string) => {
                if string.is_empty() {
                    "{}".to_owned()
                }
                else {
                    string.to_string()
                }
            },
            Node(node) => format!("(\n{node}\n)")
        };
        write!(f, "{}", val)
    }
}


#[derive(Clone, Debug)]
pub enum SQLQuery {
    From(InsideSQLQuery),
    Join {
        node: Arc<SQLNode>,
        left_on: Vec<SQLExpr>,
        right_on: Vec<SQLExpr>,
        options: SQLJoinOptions
    },
    Select {
        vexpr: Vec<SQLExpr>, 
        options: SQLSelectOptions
    },
    Where(SQLExpr),
    GroupBy(Vec<SQLExpr>),
    Having(SQLExpr),
    // OrderBy(Vec<SQLExpr>),
    OrderBy(Vec<SQLSortQuery>),
    Offset(i64),
    Fetch(u32),
}


impl SQLQuery {
    fn priority(&self) -> i32 {
        use SQLQuery::*;
        match self {
            From(_) => 2,
            Join{..} => 3,
            Select{..} => 1,
            Where(_) => 4,
            GroupBy(_) => 5,
            Having(_) => 6,
            OrderBy(_) => 7,
            Fetch(_) => 8,
            Offset(_) => 9,
        }
    }
}


impl Display for SQLQuery {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use SQLQuery::*;
        let val = match self {
            From(val) => format!("FROM {val}"),
            Join {
                node, 
                left_on, 
                right_on, 
                options
            } => { 
                let mut join_on = "".to_owned() ;
                for i in 0..left_on.len() {
                    join_on.push_str(&format!("{} = {}", left_on[i], right_on[i]).to_string());
                }
                format!("{options} (\n{node}\n)\nON {join_on}")
            },
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
            GroupBy(vexpr) => {
                let mut groupby = "".to_owned();
                groupby.push_str(&vexpr.iter().map(|val| format!("{val}")).collect::<Vec<String>>().join(", "));
                format!("GROUP BY {groupby}")
            }
            Having(expr) => format!("HAVING {expr}"),
            OrderBy(vquery) => {
                let mut orderby = "".to_owned();
                orderby += &vquery.iter().map(|val| format!("{val}")).collect::<Vec<String>>().join(", ");
                format!("ORDER BY {orderby}")
            },
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

        let mut output = "".to_owned();
        if !matches!(&node.queries[0], SQLQuery::Select{vexpr, options}) {
            output += "SELECT *\n";
        }
        output += &node.queries.iter().map(|val| format!("{val}")).collect::<Vec<String>>().join("\n");

        write!(f, "{}", output)
    }
}

impl SQLNode {
    pub fn sort_by_priority(&mut self) {
        self.queries.sort_by_key(|a| a.priority())
    }

    pub fn is_empty(&self) -> bool {
        self.schema.is_none() && self.queries.is_empty()
    }

    pub fn empty_dummy_node() -> Self {
        Self {
            schema: None,
            queries: vec![]
        }
    }

    pub fn query_dummy_node(query: SQLQuery) -> Self {
        Self {
            schema: None,
            queries: vec![query]
        }
    }

    pub fn named_dummy_node(name: String) -> Self {
        SQLNode::query_dummy_node(SQLQuery::From(InsideSQLQuery::Name(name)))
    }
    
    pub fn named_schema_node(schema: SchemaRef, name: String) -> Self {
        Self {
            schema: Some(schema.into()),
            queries: vec![SQLQuery::From(InsideSQLQuery::Name(name))]
        }
    }

    pub fn add_query(&self, query: SQLQuery) -> SQLNode {
        let mut node = self.clone();

        let index = node.queries.iter().position(|val| std::mem::discriminant(val) == std::mem::discriminant(&query));

        match index {
            Some(_) => {
                SQLNode::query_dummy_node(query).add_node(node)
            },
            None => {
                node.queries.push(query.clone());
                node
            }
        }
    }
    
    pub fn add_node(&self, other: SQLNode) -> SQLNode {
        use SQLQuery::*;
        use InsideSQLQuery::*;

        let mut node = self.clone();
        let mut other: SQLNode = other.clone();

        let index = node.queries.iter().position(|val| matches!(val, From(_)));
        let mut from = None;

        match index {
            Some(index) => {
                from = Some(node.queries.remove(index));
            },
            None => {}
        };

        match from {
            Some(From(Name(from_name))) => {
                other = other.add_query(From(Name(from_name.clone())));
            },
            Some(From(Node(from_node))) => {
                other = other.add_node(from_node);
            },
            _ => {}
        }

        node.queries.push(From(Node(other.clone().into())));
        node.into()
    } 
}


