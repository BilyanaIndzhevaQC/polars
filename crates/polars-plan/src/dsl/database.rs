use std::collections::HashSet;
use std::fmt::{Debug, Display, Result, Formatter};

use polars_core::export::chrono::format;
use polars_core::prelude::*;

pub use super::expr_dyn_fn::*;
use crate::prelude::*;


#[derive(Clone, Copy, Debug)]
pub enum DBVersion {
    MSSQL,
    SQLITE
}


impl DBVersion {
    pub fn get_string(&self) -> String {
        use DBVersion::*;

        match self {
            MSSQL => "MSSQL".to_owned(),
            SQLITE => "SQLITE".to_owned()
        }
    }

    pub fn from_string(db_version_str: String) -> DBVersion {
        match db_version_str.as_str() {
            "MSSQL" => DBVersion::MSSQL,
            "SQLITE" => DBVersion::SQLITE,
            _ => panic!("DB version is not supported yet.")
        }
    }
}


pub trait DBDisplay {
    fn db_display(&self, db_version: DBVersion) -> String;
}


#[derive(Clone, Debug)]
pub struct DBSelectOptions {
    pub distinct: bool,
    pub top: Option<u32>
}

impl DBDisplay for DBSelectOptions {
    fn db_display(&self, db_version: DBVersion) -> String {
        let mut select = (if self.distinct {"SELECT DISTINCT"} else {"SELECT"}).to_owned();
        if let Some(num) = self.top {
            select += format!(" TOP {num}").as_str();
        }
        format!("{select}")
    }
}

#[derive(Clone, Debug)]
pub struct DBSortOptions {
    pub descending: bool,
    pub nulls_last: bool,
}

impl DBDisplay for DBSortOptions {
    fn db_display(&self, db_version: DBVersion) -> String {
        let mut ordering = "".to_owned();
        if self.descending {
            ordering += "DESC";
        }
        else {
            ordering += "ASC";
        }

        let mut data_nulls = "".to_owned();
        match db_version {
            DBVersion::MSSQL => {
                if self.nulls_last {
                    data_nulls += " NULLS LAST";
                }
                else {
                    data_nulls += " NULLS FIRST";
                }
            }
            DBVersion::SQLITE => {
                if self.nulls_last {
                    data_nulls += "NULLS LAST";
                }
                else {
                    data_nulls += "NULLS FIRST";
                }
            }
        }
        format!("{}{}", data_nulls, ordering)
    }
}


#[derive(Clone, Debug)]
pub struct DBSortQuery {
    pub expr: DBExpr,
    pub options: DBSortOptions
}

impl DBDisplay for DBSortQuery {
    fn db_display(&self, db_version: DBVersion) -> String {
        let expr_display = self.expr.db_display(db_version);

        let mut ordering = "".to_owned();
        if self.options.descending {
            ordering += "DESC";
        }
        else {
            ordering += "ASC";
        }

        let mut data_nulls = "".to_owned();
        match db_version {
            DBVersion::MSSQL => {
                if self.options.nulls_last {
                    data_nulls += format!("(CASE WHEN [{expr_display}] IS NULL THEN 1 ELSE 0 END), [{expr_display}]").as_str();
                }
                else {
                    data_nulls += format!("(CASE WHEN [{expr_display}] IS NULL THEN 0 ELSE 1 END), [{expr_display}]").as_str();
                }
            }
            DBVersion::SQLITE => {
                data_nulls = expr_display;
                if self.options.nulls_last {
                    ordering += " NULLS LAST";
                }
                else {
                    ordering += " NULLS FIRST";
                }
            }
        }
        format!("{} {}", data_nulls, ordering)
    }
}


#[derive(Clone, Debug)]
pub enum DBJoinType {
    Left,
    Right,
    Inner,
    Outer,
    Union,
}


impl DBDisplay for DBJoinType {
    fn db_display(&self, db_version: DBVersion) -> String {
        use DBJoinType::*;
        let val = match self {
            Left => "LEFT JOIN",
            Right => "RIGHT JOIN",
            Inner => "INNER JOIN",
            Outer => "OUTER JOIN",
            Union => "UNION",
        };

        format!("{}", val)
    }
}

#[derive(Clone, Debug)]
pub struct DBJoinOptions {
    pub how: DBJoinType
}


impl DBDisplay for DBJoinOptions {
    fn db_display(&self, db_version: DBVersion) -> String {
        self.how.db_display(db_version)
    }
}


#[derive(Clone, Debug)]
pub enum DBAggExpr {
    Min {
        input: Box<DBExpr>,
        propagate_nans: bool,
    },
    Max {
        input: Box<DBExpr>,
        propagate_nans: bool,
    },
    Median(Box<DBExpr>),
    First(Box<DBExpr>),
    Mean(Box<DBExpr>),
    Count(Box<DBExpr>),
    Sum(Box<DBExpr>),

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

impl DBDisplay for DBAggExpr {
    fn db_display(&self, db_version: DBVersion) -> String {
        let agg_display = |agg: &str, input: &Box<DBExpr>| {
            format!("{agg}({})", input.db_display(db_version))
        };

        use DBAggExpr::*;
        match self {
            Min{input, propagate_nans} => agg_display("MIN", input),
            Max{input, propagate_nans} => agg_display("MAX", input),
            Mean(input) => format!("AVG(CAST({} as FLOAT))", input.db_display(db_version)),
            Sum(input) => agg_display("SUM", input),
            Count(input) => agg_display("COUNT", input),
            Median(input) => agg_display("MEDIAN", input),
            First(input) => format!("{}", input.db_display(db_version)),
        }
    }
}


impl DBAggExpr {
    fn get_expr(&self) -> &Box<DBExpr> {
        use DBAggExpr::*;
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
        use DBAggExpr::*;
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
pub enum DBExpr {
    Alias(Box<DBExpr>, Arc<str>),
    Column(Arc<str>),
    Literal(LiteralValue),
    Agg(DBAggExpr),
    BinaryExpr {
        left: Box<DBExpr>,
        op: Operator,
        right: Box<DBExpr>,
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

impl DBDisplay for DBExpr {
    fn db_display(&self, db_version: DBVersion) -> String {
        use DBExpr::*;
        let val = match self {
            Alias(expr, new_name) => {
                format!("{} AS {}", expr.db_display(db_version),
                new_name)
            },
            Column(name) => format!("{name}"),
            Literal(literal) => format!("{literal}"),
            BinaryExpr {left, op, right} => {
                format!(
                    "({}) {op} ({})", 
                    left.db_display(db_version), 
                    right.db_display(db_version)
                )
            },
            Agg(aggExpr) => aggExpr.db_display(db_version),
        };
        format!("{}", val)
    }
}


impl DBExpr {
    fn find_name(&self) -> Arc<str> {
        use DBExpr::*;
        match self {
            Alias(expr, new_name) => new_name.clone(),
            Column(name) => name.clone(),
            Literal(literal) => "literal".into(),
            BinaryExpr {left, op, right} => left.find_name(),
            Agg(aggExpr) => aggExpr.get_expr().find_name(),
        }
    }

    pub fn remove_alias(&self) -> Self {
        use DBExpr::*;
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
        DBExpr::Alias(Box::new(self.remove_alias().clone()), self.find_name())
    }
}


#[derive(Clone, Debug)]
pub enum InsideDBQuery {
    Name(String),
    Node(DBNode)
}


impl DBDisplay for InsideDBQuery {
    fn db_display(&self, db_version: DBVersion) -> String {
        use InsideDBQuery::*;
        let val = match self {
            Name(string) => {
                if string.is_empty() {
                    "{}".to_owned()
                }
                else {
                    string.to_string()
                }
            },
            Node(node) => format!("(\n{}\n) _", node.db_display(db_version))
        };
        format!("{}", val)
    }
}


#[derive(Clone, Debug)]
pub enum DBQuery {
    From(InsideDBQuery),
    Join {
        node: Arc<DBNode>,
        left_on: Vec<DBExpr>,
        right_on: Vec<DBExpr>,
        options: DBJoinOptions
    },
    Select {
        vexpr: Vec<DBExpr>, 
        options: DBSelectOptions
    },
    Where(DBExpr),
    GroupBy(Vec<DBExpr>),
    Having(DBExpr),
    // OrderBy(Vec<DBExpr>),
    OrderBy(Vec<DBSortQuery>),
    Fetch(u32),
    Offset(i64),
}


impl DBQuery {
    fn priority(&self) -> i32 {
        use DBQuery::*;
        match self {
            Select{..} => 1,
            From(_) => 2,
            Join{..} => 3,
            Where(_) => 4,
            GroupBy(_) => 5,
            Having(_) => 6,
            OrderBy(_) => 7,
            Fetch(_) => 8,
            Offset(_) => 9,
        }
    }

    fn name(&self) -> String {
        use DBQuery::*;
        let val = match self {
            Select{..} => "Select",
            From(_) => "From",
            Join{..} => "Join",
            Where(_) => "Where",
            GroupBy(_) => "GroupBy",
            Having(_) => "Having",
            OrderBy(_) => "OrderBy",
            Fetch(_) => "Fetch",
            Offset(_) => "Offset",
        };
        val.to_owned()
    }
}


impl DBDisplay for DBQuery {
    fn db_display(&self, db_version: DBVersion) -> String {
        use DBQuery::*;
        let val = match self {
            From(val) => format!("FROM {}", val.db_display(db_version)),
            Join {
                node, 
                left_on, 
                right_on, 
                options
            } => { 
                let mut join_on = "".to_owned() ;
                for i in 0..left_on.len() {
                    join_on.push_str(&format!(
                        "{} = {}", 
                        left_on[i].db_display(db_version), 
                        right_on[i].db_display(db_version)
                    ).to_string());
                }

                format!(
                    "{} (\n{}\n) _\nON {join_on}", 
                    options.db_display(db_version), 
                    node.db_display(db_version)
                )
            },
            Select {vexpr, options} => {
                let mut select = "".to_owned();
                if vexpr.len() == 0 {
                    select.push_str("*");
                }
                else {
                    select.push_str(&vexpr.iter().map(|val| val.fix_alias().db_display(db_version)).collect::<Vec<String>>().join(", "));
                }
                format!("{} {select}", options.db_display(db_version))
            },
            Where(expr) => format!("WHERE {}", expr.db_display(db_version)),
            GroupBy(vexpr) => {
                let mut groupby = "".to_owned();
                groupby.push_str(&vexpr.iter().map(|val| val.db_display(db_version)).collect::<Vec<String>>().join(", "));
                format!("GROUP BY {groupby}")
            }
            Having(expr) => format!("HAVING {}", expr.db_display(db_version)),
            OrderBy(vquery) => {
                let mut orderby = "".to_owned();
                orderby += &vquery.iter().map(|val| val.db_display(db_version)).collect::<Vec<String>>().join(", ");
                format!("ORDER BY {orderby}")
            },
            Offset(num) => format!("OFFSET {num}"),
            Fetch(num) => match db_version {
                DBVersion::MSSQL => format!("TOP {num}"),
                DBVersion::SQLITE => format!("LIMIT {num}"),
            }
        };
        format!("{}", val)
    }
}


#[derive(Clone, Debug)]
pub struct DBNode {
    pub schema: Option<Arc<SchemaRef>>,
    pub queries: Vec<DBQuery>
}

impl DBDisplay for DBNode {
    fn db_display(&self, db_version: DBVersion) -> String {
        let db_display_sqlite = || {
            let mut node = self.clone();
            node.sort_by_priority();
    
            let mut output = "".to_owned();
            if !matches!(&node.queries[0], DBQuery::Select{vexpr, options}) {
                output += "SELECT *\n";
            }
            output += &node.queries.iter().map(|val| val.db_display(db_version)).collect::<Vec<String>>().join("\n");
    
            format!("{}", output)
        };


        let db_display_mssql = || {
            let mut node = self.clone();
            node.sort_by_priority();

            let mut output = "".to_owned();
            let mut output_suffix = "".to_owned();
    
            let fetch_idx = node.queries.iter().position(|x| matches!(x, DBQuery::Fetch(_)));
            let sort_idx = node.queries.iter().position(|x| matches!(x, DBQuery::OrderBy(_)));

            if let Some(DBQuery::Offset(offset_value)) = node.queries.last() {
                if let None = sort_idx {
                    output_suffix += format!("\nORDER BY (SELECT NULL)").as_str();
                }

                output_suffix += format!("\nOFFSET {offset_value} ROWS").as_str();
                node.queries.pop();

                if let Some(fetch_idx) = fetch_idx {
                    if let DBQuery::Fetch(fetch_value) = node.queries[fetch_idx] {
                        output_suffix += format!("\nFETCH NEXT {fetch_value} ROWS ONLY").as_str();
                        node.queries.remove(fetch_idx);
                    }
                }

                if !matches!(&node.queries[0], DBQuery::Select{vexpr, options}) {
                    output += "SELECT *\n";
                }
            }
            else if let Some(fetch_idx) = fetch_idx {
                if !matches!(&node.queries[0], DBQuery::Select{vexpr, options}) {
                    output += format!(
                        "SELECT {} *\n", 
                        node.queries[fetch_idx].db_display(db_version)
                    ).as_str();
                }
                else if let DBQuery::Fetch(fetch_value) = node.queries[fetch_idx] {
                    if let DBQuery::Select { vexpr, options } = node.queries.remove(0) {
                        let options = DBSelectOptions { 
                            distinct: options.distinct, 
                            top: Some(fetch_value) 
                        };
                        node.queries[0] = DBQuery::Select { vexpr, options } 
                    }
                }
            }
            else if !matches!(&node.queries[0], DBQuery::Select{vexpr, options}) {
                output += "SELECT *\n";
            }

            output += &node.queries.iter().map(
                |val| val.db_display(db_version)
            ).collect::<Vec<String>>().join("\n");

            if let None = fetch_idx {
                if let Some(sort_idx) = sort_idx {
                    output_suffix += "\nOFFSET 0 ROWS";
                }
            }


            output += output_suffix.as_str();
    
            format!("{output}")
        };

        match db_version {
            DBVersion::MSSQL => db_display_mssql(),
            DBVersion::SQLITE => db_display_sqlite(),
        }
        
    }
}

impl DBNode {
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

    pub fn query_dummy_node(query: DBQuery) -> Self {
        Self {
            schema: None,
            queries: vec![query]
        }
    }

    pub fn named_dummy_node(name: String) -> Self {
        DBNode::query_dummy_node(DBQuery::From(InsideDBQuery::Name(name)))
    }
    
    pub fn named_schema_node(schema: SchemaRef, name: String) -> Self {
        Self {
            schema: Some(schema.into()),
            queries: vec![DBQuery::From(InsideDBQuery::Name(name))]
        }
    }

    pub fn add_query(&self, query: DBQuery) -> DBNode {
        let mut node = self.clone();

        let index = node.queries.iter().position(|val| std::mem::discriminant(val) == std::mem::discriminant(&query));

        match index {
            Some(_) => {
                DBNode::query_dummy_node(query).add_node(node)
            },
            None => {
                node.queries.push(query.clone());
                node
            }
        }
    }
    
    pub fn add_node(&self, other: DBNode) -> DBNode {
        use DBQuery::*;
        use InsideDBQuery::*;

        let mut node = self.clone();
        let mut other: DBNode = other.clone();

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


