import os
from typing import List
import sqlalchemy
from sqlalchemy.engine.base import Engine
from sqlalchemy import text, create_engine
import pandas as pd
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from utils.ask_cli import ask  # assume available


@tool
def list_tables_tool(config: RunnableConfig) -> List[str]:
    """List all tables in database"""
    db_engine: Engine = config.get("configurable", {}).get("db_engine")
    inspector = sqlalchemy.inspect(db_engine)
    return inspector.get_table_names()


@tool
def get_table_schema_tool(table_name: str, config: RunnableConfig) -> List[str]:
    """
    Get schema information about a table. Returns a list of dictionaries.
    - name is the column name
    - type is the column type
    - nullable is whether the column is nullable or not
    - default is the default value of the column
    - primary_key is whether the column is a primary key or not
    """
    db_engine: Engine = config.get("configurable", {}).get("db_engine")
    inspector = sqlalchemy.inspect(db_engine)
    return inspector.get_columns(table_name)


@tool
def execute_sql_tool(query: str, config: RunnableConfig):
    """
    Execute SQL query and return result (list of rows).
    """
    db_engine: Engine = config.get("configurable", {}).get("db_engine")
    with db_engine.begin() as connection:
        return connection.execute(text(query)).fetchall()


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(SCRIPT_DIR, "../../artifacts/sales.db")
    db_engine = create_engine(f"sqlite:///{db_path}")
    config = {"configurable": {"db_engine": db_engine}}

    while True:
        print("\n=== SQL Tools Demo (Interactive) ===")
        print("1) List tables")
        print("2) Show table schema")
        print("3) Run SQL query")
        print("4) Quick preview (SELECT * LIMIT 10)")
        print("q) Quit")

        choice = ask("Choose an option").lower()

        if choice == "q":
            break

        if choice == "1":
            tables = list_tables_tool.invoke({}, config)
            print("\nTables:", tables)

        elif choice == "2":
            tables = list_tables_tool.invoke({}, config)
            print("\nTables:", tables)
            table = ask("Table name", tables[0] if tables else "")
            schema = get_table_schema_tool.invoke({"table_name": table}, config)
            print("\nSchema:")
            for col in schema:
                print(col)

        elif choice == "3":
            query = ask("SQL query", "SELECT * FROM sales LIMIT 10")
            try:
                result = execute_sql_tool.invoke({"query": query}, config)
                print("\nResult (first 10 rows):", result[:10])
                if result:
                    # try to display as DataFrame if columns can be inferred
                    try:
                        table_guess = ""
                        ql = query.lower()
                        if "from" in ql:
                            table_guess = ql.split("from", 1)[1].strip().split()[0].strip(";")
                        if table_guess:
                            cols = [c["name"] for c in sqlalchemy.inspect(db_engine).get_columns(table_guess)]
                            print("\nAs DataFrame:")
                            print(pd.DataFrame(result, columns=cols).head(10))
                    except Exception:
                        pass
            except Exception as e:
                print("SQL error:", e)

        elif choice == "4":
            tables = list_tables_tool.invoke({}, config)
            if not tables:
                print("No tables found.")
                continue
            print("\nTables:", tables)
            table = ask("Table name", tables[0])
            limit = ask("LIMIT", "10")

            sql = f"SELECT * FROM {table} LIMIT {limit}"
            try:
                result = execute_sql_tool.invoke({"query": sql}, config)
                schema = get_table_schema_tool.invoke({"table_name": table}, config)
                cols = [c["name"] for c in schema]
                print("\nAs DataFrame:")
                print(pd.DataFrame(result, columns=cols))
            except Exception as e:
                print("SQL error:", e)

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
