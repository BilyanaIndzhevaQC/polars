from __future__ import annotations

import math
from pathlib import Path

import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal, assert_series_equal


uri_sqlite="sqlite:///Users/bilyanaindzheva/Quantco/polars/polars/py-polars/sql_test"

str_mssql = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;PORT=1433;UID=sa;PWD=mssql!Password;TrustServerCertificate=yes;Encrypt=No"
uri_mssql = "mssql://sa:mssql!Password@localhost:1433/master?driver=SQL+Server&trusted_connection=true"


table = "table10k"
intColumn = "int_null"
intColumn2 = "int_not_null"
strColumn = "string_notnull"
strColumn2 = "string_null"
bitColumn = "bit_notnull"
bitColumn2 = "bit_null"


test_functions_list = []
def test_function(f):
    global test_functions_list
    test_functions_list += [f]
    return f


def query_test_sqlite(query, uri_sqlite, **additional_args) -> None:
    connection = pl.scan_database(connection_uri=uri_sqlite, table=table)
    # print(query(connection).print_query())
    scan_dataframe = query(connection).load_dataframe()
 
    read_dataframe = pl.read_database_uri(query=f"SELECT * FROM {table}", uri=uri_sqlite)
    # print("ORIGINAL\n", read_dataframe)
    read_dataframe = query(read_dataframe)
    
    # print("LEFT:\n", scan_dataframe, "\nRIGHT:\n", read_dataframe)
    assert_frame_equal(scan_dataframe, read_dataframe, check_dtype=False, **additional_args)


def query_test_mssql(query, str_mssql, uri_mssql, **additional_args) -> None:
    connection = pl.scan_database(connection_string=str_mssql, table=table)
    # print(query(connection).print_query())
    
    scan_dataframe = query(connection).load_dataframe()
 
    read_dataframe = pl.read_database_uri(query=f"SELECT * FROM {table}", uri=uri_mssql)
    # print("ORIGINAL\n", read_dataframe)
    read_dataframe = query(read_dataframe)

    # print("LEFT:\n", scan_dataframe, "\nRIGHT:\n", read_dataframe)
    assert_frame_equal(scan_dataframe, read_dataframe, check_dtype=False, **additional_args)


def query_test_all_databases(query, **additional_args) -> None:
    query_test_sqlite(query, uri_sqlite, **additional_args)
    query_test_mssql(query, str_mssql, uri_mssql, **additional_args)


@test_function
def test_database_loading() -> None:
    query = lambda lf: lf
    query_test_all_databases(query)


@test_function
def test_database_expr_column() -> None:
    query = lambda lf: lf.select(intColumn)
    query_test_all_databases(query)
    

@test_function
def test_database_expr_columns() -> None:
    query = lambda lf: lf.select(intColumn, strColumn)
    query_test_all_databases(query)
    
    
@test_function
def test_database_expr_alias() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn).alias("test")),
        lambda lf: lf.select(pl.col(intColumn).alias("test").alias("test2")),
        lambda lf: lf.select(pl.col(intColumn).alias("test").keep_name()),
    ]
        
    for query in queries:
        query_test_all_databases(query)  
    
    
@test_function
def test_database_expr_lit() -> None:
    queries = [
        # lambda lf: lf.select(pl.lit(1)),
        lambda lf: lf.select(pl.lit(1), pl.col(intColumn)),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
    
    
@test_function
def test_database_expr_agg() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn).max()),
        lambda lf: lf.select(pl.col(intColumn).min()),
        # lambda lf: lf.select(pl.col(intColumn).median()),
        lambda lf: lf.select(pl.col(intColumn).mean()),
        # lambda lf: lf.select(pl.col(intColumn).count()), #doesnt count nulls
        # lambda lf: lf.select(pl.col(intColumn).first()), #works in groupby for sqlite
        # lambda lf: lf.select(pl.col(intColumn).last()),
        lambda lf: lf.select(pl.col(intColumn).sum()),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
    
    
@test_function
def test_database_expr_binary() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn) + pl.col(intColumn2)),
        lambda lf: lf.select(pl.col(intColumn) + pl.lit(1)),
        # lambda lf: lf.select(pl.col(strColumn) + pl.col(strColumn2)), #ok for mssql
        # lambda lf: lf.select(pl.col(bitColumn) & pl.col(bitColumn2)), #ok for sqlite
        
    ]
    
    for query in queries:
        query_test_all_databases(query)  
        
        
@test_function
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
        
        
@test_function
def test_database_expr_sort_by() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(intColumn))),
        lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(intColumn2))), #mssql random order for equal values
        lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(intColumn2), descending=True)),
        # lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(intColumn2), descending=True), pl.col(intColumn2)),
    ]
    
    for query in queries:
        query_test_all_databases(query, check_row_order=False)  
        
        
@test_function
def test_database_expr_filter() -> None:
    queries = [
        lambda lf: lf.select(pl.col(intColumn).filter(pl.col(intColumn) > 20)),
        # lambda lf: lf.select(pl.col(intColumn).filter(pl.col(intColumn) > 20), pl.col(intColumn2)),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
        
        
@test_function
def test_database_projection() -> None:
    queries = [
        #should fail in polars
        # lambda lf: lf.select(pl.col(intColumn).filter(pl.col(intColumn) > 20), pl.col(intColumn2)),
        # lambda lf: lf.select((pl.col(intColumn) + 3).alias(intColumn2).sort(descending=True)),
        # lambda lf: lf.select(pl.col(intColumn).sort_by(pl.col(bitColumn) & pl.col(bitColumn2), descending=True)),
    ]
    
    for query in queries:
        query_test_all_databases(query)  
        
        
@test_function
def test_database_slice() -> None:
    queries = [
        lambda lf: lf.slice(0, 5),
        lambda lf: lf.slice(5, 5),
    ]
    
    for query in queries:
        query_test_all_databases(query)  


@test_function
def test_database_distinct() -> None:
    queries = [
        lambda lf: lf.unique()
    ]
    
    for query in queries:
        query_test_all_databases(query, check_row_order=False)  
    

@test_function
def test_database_selection() -> None:
    queries = [
        lambda lf: lf.filter(pl.col(intColumn) > 2),
        lambda lf: lf.filter((pl.col(intColumn) > 2).alias("test")),
        # lambda lf: lf.filter((pl.col(intColumn) > 2) & (pl.col(bitColumn) == 1)), # redo &
    ]
    
    for query in queries:
        query_test_all_databases(query)
        
        
@test_function
def test_database_hstack() -> None:
    queries = [
        lambda lf: lf.with_columns(pl.col(intColumn).alias("test"))
    ]
    
    for query in queries:
        query_test_all_databases(query)
        
        
@test_function
def test_database_groupby() -> None:
    queries = [
        # lambda lf: lf.group_by(pl.col(intColumn)).max(), #mssql bit column cant max
        # lambda lf: lf.group_by(pl.col(intColumn)).min(),
        # lambda lf: lf.group_by(pl.col(intColumn)).first(), #mssql first not working at all
        lambda lf: lf.group_by(pl.col(intColumn)).agg(pl.col(intColumn2).max()),
        lambda lf: lf.group_by(pl.col(intColumn)).agg(pl.col(intColumn2).min()),
    ]
    
    for query in queries:
        query_test_all_databases(query, check_column_order=False, check_row_order=False)
        
        
@test_function
def test_database_sort() -> None:
    queries = [
        lambda lf: lf.sort(pl.col(intColumn)),
        lambda lf: lf.sort(pl.col(intColumn), nulls_last=True),
        lambda lf: lf.sort(pl.col(intColumn), descending=True),
        lambda lf: lf.sort(pl.col(intColumn), nulls_last=True, descending=True),
        lambda lf: lf.sort(pl.col(intColumn), pl.col(intColumn2), descending=[True, False]),
    ]
    
    for query in queries:
        query_test_all_databases(query, check_row_order=False)
        
           
@test_function
def test_database_join() -> None:
    queries = [
        # to do        
    ]
    
    for query in queries:
        query_test_all_databases(query)
    
        
@test_function
def test_database_random() -> None:
    queries = [
        # to do        
    ]
    
    for query in queries:
        query_test_all_databases(query)
        

def test_one(f) -> None: 
    try:
        f()
        print(f.__name__, " OK")
    except Exception as e:
        print(f.__name__, " NOT OK")
        print("ERROR MESSAGE:", e)
            

def test_all() -> None: 
    for test_function in test_functions_list:
        test_one(test_function)
            
            
# test_one(test_database_random)
test_all()