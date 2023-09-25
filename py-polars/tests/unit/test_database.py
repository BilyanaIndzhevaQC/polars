from __future__ import annotations

import math
from pathlib import Path

import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal, assert_series_equal

# lf_sql = pl.SQLContext(lf = lf.collect())
# assert(lf_sql.execute(query_sql).collect() == query_exec.collect())

lf = pl.LazyFrame({"a":[1,5,3,5,1], "b":["h","c","c","h","h"], "c":[7,8,9,10, 7]})
lf2 = pl.LazyFrame({"a":[3,3,3], "b":["a","b","c"], "c":[2,3,4]})

# pl.scan_database(schema=...).select(pl.col(["a", "c"])).collect_graph()

def exec_query(query_str): 
    return eval(query_str)

def database_query(query_str):
    return exec_query(query_str).database_query()

def print_query(query_str):
    return f"{query_str}\n\t<=>\n{database_query(query_str)}\n\n{exec_query(query_str).collect()}\n\n"

queries = [
    "lf.slice(1, 2)", 
    "lf.unique()",
    "lf.with_columns(pl.col(\"a\").alias(\"sum\"))",
    "lf.group_by(pl.col(\"a\")).max()",
    
    
    "lf.sort(pl.col(\"a\"), nulls_last=True)",    
    "lf.sort(pl.col(\"a\"), descending=True)", 
    "lf.sort(pl.col(\"a\"), pl.col(\"b\"), descending=[True, False])", 
    
    
    "lf.filter(pl.col(\"a\") > 2)",
    "lf.filter((pl.col(\"a\") > 2).alias(\"b\"))",
    "lf.filter((pl.col(\"a\") > 2) & (pl.col(\"b\") == \"h\"))",
    "lf.select(pl.col(\"a\").filter(pl.col(\"a\") > 2))",
    
    "lf.select(pl.col(\"a\").cast(pl.Int8))",
    
       
    "lf.join(lf2, on=\"a\")",
        
    "lf.with_columns((pl.col(\"a\") + pl.lit(1)).alias(\"sum\")).slice(1,2).unique().filter((pl.col(\"sum\") > 3) & (pl.lit(2) > 1)).groupby(\"b\").min().sort(pl.col(\"sum\")).join(lf2, on=\"a\")",
    
    
]

def do_queries():
    for id, query in enumerate(queries):
        print(f"Query: {id}\n{print_query(query)}\n")
    


uri_sqlite="sqlite:///Users/bilyanaindzheva/Quantco/polars/polars/py-polars/sql_test"

str_mssql = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;PORT=1433;UID=sa;PWD=mssql!Password;TrustServerCertificate=yes;Encrypt=No"
uri_mssql = "mssql://sa:mssql!Password@localhost:1433/master?driver=SQL+Server&trusted_connection=true"


table = "Persons"
intColumn = "Age"
intColumn2 = "PersonID"
strColumn = "FirstName"
strColumn2 = "LastName"
bitColumn = "Disability"
bitColumn2 = "Drinks"


def query_test_sqlite(query, uri_sqlite) -> None:
    connection = pl.scan_database(connection_uri=uri_sqlite, table=table)
    scan_dataframe = query(connection).load_dataframe()
 
    read_dataframe = pl.read_database_uri(query=f"SELECT * FROM {table}", uri=uri_sqlite)
    # print("ORIGINAL\n", dataframe)
    read_dataframe = query(read_dataframe)
    
    # print("LEFT:\n", scan_dataframe, "\nRIGHT:\n", read_dataframe)
    assert_frame_equal(scan_dataframe, read_dataframe, check_dtype=False)


def query_test_mssql(query, str_mssql, uri_mssql) -> None:
    # import turbodbc as tdbc
    
    connection = pl.scan_database(connection_string=str_mssql, table=table)
    scan_dataframe = query(connection).load_dataframe()
 
    read_dataframe = pl.read_database_uri(query=f"SELECT * FROM {table}", uri=uri_mssql)
    # print("ORIGINAL\n", dataframe)
    read_dataframe = query(read_dataframe)
    
    # print("LEFT:\n", scan_dataframe, "\nRIGHT:\n", read_dataframe)
    assert_frame_equal(scan_dataframe, read_dataframe, check_dtype=False)


def query_test_all_databases(query) -> None:
    query_test_sqlite(query, uri_sqlite)
    query_test_mssql(query, str_mssql, uri_mssql)


def test_database_loading() -> None:
    query = lambda lf: lf
    query_test_all_databases(query)


def test_database_expr_column() -> None:
    query = lambda lf: lf.select(intColumn)
    query_test_all_databases(query)
    

def test_database_expr_columns() -> None:
    query = lambda lf: lf.select(intColumn, strColumn)
    query_test_all_databases(query)
    
    
def test_database_expr_alias() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn).alias("test")),
        lambda lf: lf.select(pl.col(intColumn).alias("test").alias("test2")),
        lambda lf: lf.select(pl.col(intColumn).alias("test").keep_name()),
    ]
        
    for query in queries:
        query_test_all_databases(query)  
    
    
def test_database_expr_lit() -> None:
    queries = [
        # lambda lf: lf.select(pl.lit(1)),
        lambda lf: lf.select(pl.lit(1), pl.col(intColumn)),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
    
    
def test_database_expr_agg() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn).max()),
        lambda lf: lf.select(pl.col(intColumn).min()),
        # lambda lf: lf.select(pl.col(intColumn).median()),
        lambda lf: lf.select(pl.col(intColumn).mean()),
        # lambda lf: lf.select(pl.col(intColumn).count()),#doesnt count nulls
        # lambda lf: lf.select(pl.col(intColumn).first()),#works in groupby
        # lambda lf: lf.select(pl.col(intColumn).last()),
        lambda lf: lf.select(pl.col(intColumn).sum()),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
    
    
def test_database_expr_binary() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn) + pl.col(intColumn2)),
        lambda lf: lf.select(pl.col(intColumn) + pl.lit(1)),
        # lambda lf: lf.select(pl.col(strColumn) + pl.col(strColumn2)),
        lambda lf: lf.select(pl.col(bitColumn) & pl.col(bitColumn2)),
        
    ]
    
    for query in queries:
        query_test_all_databases(query)  
        
        
def test_database_expr_sort() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn).sort()),
        lambda lf: lf.select(pl.col(intColumn).sort(descending=True)),
        lambda lf: lf.select(pl.col(intColumn).sort(nulls_last=True)),
        lambda lf: lf.select(pl.col(intColumn).sort(descending=True, nulls_last=True)),
        # lambda lf: lf.select(pl.col(intColumn).sort(), pl.col(intColumn2)),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
        
        
def test_database_expr_sort_by() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(intColumn))),
        lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(intColumn2))),
        lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(intColumn2), descending=True)),
        # lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(intColumn2), descending=True), pl.col(intColumn2)),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
        
        
def test_database_expr_filter() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn).filter(pl.col(intColumn) > 20)),
        # lambda lf: lf.select(pl.col(intColumn).filter(pl.col(intColumn) > 20), pl.col(intColumn2)),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
        
        
def test_database_projection() -> None:
    queries = [
        # lambda lf: lf.select(pl.col(intColumn).filter(pl.col(intColumn) > 20), pl.col(intColumn2)),
        # lambda lf: lf.select((pl.col(intColumn) + 3).alias(intColumn2).sort(descending=True)),
        # lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(bitColumn) & pl.col(bitColumn2), descending=True)),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
