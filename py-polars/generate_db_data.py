import polars as pl
from polars.testing import assert_frame_equal
import random
import string
import re
import time
import logging


import turbodbc as tdbc

connection_string = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;PORT=1433;UID=sa;PWD=mssql!Password;TrustServerCertificate=yes;Encrypt=No"
connection_string_uri = "mssql://sa:mssql!Password@localhost:1433/master?driver=SQL+Server&trusted_connection=true"


options = tdbc.make_options(autocommit=True)
connection = tdbc.connect(connection_string=connection_string, turbodbc_options=options)
cur = connection.cursor()

name_length = 10
row_number = 100

null_percentage = 10

max_int = pow(2, 16) - 1
max_smallint = pow(2, 8) - 1
max_float = pow(2, 13) - 1
min_date = "1/1/2000"
max_date = "1/1/2023"

# query = lambda lf: lf
# query = lambda lf: lf.select(pl.col(pl.INTEGER_DTYPES))
# query = lambda lf: lf.select(pl.all().exclude(["bit_null", "bit_null2", "bit_notnull"])).group_by("int_null").max()
# query = lambda lf: lf.sort(by="int_null")
# query = lambda lf: lf.filter(pl.col("int_null2") > 3)
# query = lambda lf: lf.select(pl.all().exclude(["bit_null", "bit_null2", "bit_notnull"]), (pl.col("int_null") + pl.lit(1)).alias("sum")).unique().filter((pl.col("int_null") > 3)).group_by("int_null2").min().sort(pl.col("int_null2"))

table1k = "table1k"
table10k = "table10k"
table100k = "first_table"
table1m = "table1m"

global_schema = {
    "int_null": "INT NULL",
    "int_null2": "INT NULL",
    "int_not_null": "INT NOT NULL",
    "int_not_null2": "INT NOT NULL",
    "sint_null": "SMALLINT NULL",
    "sint_notnull": "SMALLINT NOT NULL",
    "int5": "SMALLINT NOT NULL",
    "float_null": "FLOAT NULL",
    "float_notnull": "FLOAT NOT NULL",
    "datetime_null": "DATETIME NULL",
    "datetime_null2": "DATETIME NULL",
    "datetime_notnull": "DATETIME NOT NULL",
    "bit_null": "BIT NULL",
    "bit_null2": "BIT NULL",
    "bit_notnull": "BIT NOT NULL",
    "string_null": "VARCHAR(20) NULL",
    "string_null2": "VARCHAR(20) NULL",
    "string_notnull": "VARCHAR(20) NOT NULL",
    "string_notnull2": "VARCHAR(20) NOT NULL",
}
    
def str_time_prop(start, end, time_format, prop):
    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


def random_date(start, end, prop):
    return str_time_prop(start, end, '%m/%d/%Y', prop)

def random_string(size) -> str:
    size = random.randint(1, size-1)
    letters = string.ascii_lowercase
    
    return (''.join(random.choice(letters) for i in range(size)))

def split_nonalpha(s):
   return map(s, lambda x: x if x.isalpha() else ' ').split()


def create_table(table = None, schema = None) -> (str, dict):
    if table is None:
        table = random_string(name_length)
        
    # if schema is None:
    #     schema = generate_random_schema(20)
        
    schema_str = ','.join([f"{key} {value}" for (key, value) in schema.items()])

    cur.execute(f"""
        CREATE TABLE {table}(
            {schema_str}
        );
    """)
    
    return (table, schema)


def delete_table(table) -> None:
    cur.execute(f"""
        DROP TABLE {table};
    """)
      
def row_count(table, schema):
    res = cur.execute(f"""SELECT COUNT(int_null) FROM {table}""")
    print(res.fetchmany(1))
      
def generate_row(schema) -> str: 
    row_str = ""
    for value in schema.values():
        value = re.findall(r'\w+', value) 
        
        if value[-2] != "NOT" and random.randint(1,100) <= null_percentage:
            row_str += "NULL, "
            continue
        
        new_val = ""
        match value[0]:
            case "INT":       new_val = random.randint(-max_int, max_int)
            case "SMALLINT":  new_val = random.randint(-max_smallint, max_smallint)
            case "SMALLINT":  new_val = random.randint(-max_smallint, max_smallint)
            case "FLOAT":     new_val = round(random.uniform(-max_float, max_float), 2)
            case "DATETIME":  new_val = random_date(min_date, max_date, random.random())
            case "BIT":       new_val = random.randint(0, 1)
            case "VARCHAR":   new_val = f"\'{random_string(int(value[1]))}\'"
        
        row_str += f"{new_val}, "
    
    return row_str[:-2]
        
    
def generate_rows(table, schema, row_number) -> None: 
    data = ""
    batch = f"INSERT INTO {table} VALUES"
    for i in range(row_number):
        row = generate_row(schema)
        # print("ROW - ", row)
        
        batch += f"({row}),\n"
       
        if i % 1000 == 0:
            data += batch[:-2] + ';'
            batch = f"INSERT INTO {table} VALUES"
            
    
    data += batch[:-2] + ';'
    
    cur.execute(data)
    
    
def time_scan_query(query, table, connection_string) -> (time, pl.DataFrame):
    begin_time = time.time()
    
    dbconnection = pl.scan_database(connection_string=connection_string, table=table)
    dataframe = query(dbconnection).load_dataframe()
    
    end_time = time.time()
    dtime = end_time - begin_time
    
    return (dtime, dataframe)
    
    
def time_read_query(query, table, connection_string) -> (time, pl.DataFrame):
    begin_time = time.time()
    
    query_temp = f"SELECT * FROM {table}"
    dataframe = pl.read_database_uri(query_temp, connection_string)
    dataframe = query(dataframe)
    
    end_time = time.time()
    dtime = end_time - begin_time
    
    return (dtime, dataframe)


def check_table_equality(scan_dataframe, read_dataframe) -> bool:
    try:
        assert_frame_equal(scan_dataframe, read_dataframe, check_dtype=False, check_column_order=False, check_row_order=False)
        return True
    except:
        return False 


def pipeline(table, schema, connection_string, connection_string_uri, query) -> None:
    # (table, schema) = create_table(table=table, schema=schema)
    # print(f"Table {table} created.")
    
    # generate_rows(table, schema, row_number)
    # print(f"Generated {row_number} rows for {table}.")

    (scan_query_time, scan_lazyframe) = time_scan_query(query, table, connection_string)
    print(f"Database query's time is {scan_query_time}.")
    
    (read_query_time, read_lazyframe) = time_read_query(query, table, connection_string_uri)
    print(f"Current frame query's time is {read_query_time}.")    
    
    frame_equality = check_table_equality(scan_lazyframe, read_lazyframe)
    print(f"Assert frame equality {frame_equality}.")    
    
    # row_count(table, schema)
    
    # delete_table(table)
    # print(f"Table {table} deleted.")

    

pipeline(table1k, global_schema, connection_string, connection_string_uri, query)

