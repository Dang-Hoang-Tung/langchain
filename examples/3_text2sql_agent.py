import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
import sqlalchemy
from sqlalchemy.engine.base import Engine
from sqlalchemy import text, create_engine
from utils.llm_openai import instantiate_llm
from dotenv import load_dotenv
from langchain_core.tools import tool
from typing import List
from langchain_core.runnables.config import RunnableConfig
from utils.ask_cli import ask
from utils.display_image import open_png_bytes


load_dotenv()

llm = instantiate_llm()


### ----- Initialize the State Graph ----- ###
# You can either chose from creating it with TypedDict, Pydantic or using the MessagesState.
class State(MessagesState):
    user_query: str

workflow = StateGraph(State)


### ----- Define Database Tools ----- ###

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


dba_tools = [list_tables_tool, get_table_schema_tool, execute_sql_tool]

workflow.add_node("dba_tools", ToolNode(dba_tools))

dba_llm = llm.bind_tools(dba_tools, tool_choice="auto")


### ----- Define Agent Nodes ----- ###

def messages_builder(state: State):
    dba_sys_msg = (
        "You are a Sr. SQL developer tasked with generating SQL queries. Perform the following steps:\n"
        "First, find out the appropriate table name based on all tables. "
        "Then get the table's schema to understand the columns. "
        "With the table name and the schema, generate the ANSI SQL query you think is applicable to the user question. "
        "Finally, use a tool to execute the above SQL query and output the result based on the user question."
    )
    messages = [
        SystemMessage(dba_sys_msg),
        HumanMessage(state["user_query"])
    ]
    return {"messages": messages}


def dba_agent(state: State):
   ai_message = dba_llm.invoke(state["messages"])
   ai_message.name = "dba_agent"
   return {"messages": ai_message}


workflow.add_node("messages_builder", messages_builder)
workflow.add_node("dba_agent", dba_agent)


### ----- Define Workflow Edges ----- ###

def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "dba_tools"
    return END


workflow.add_edge(START, "messages_builder")
workflow.add_edge("messages_builder", "dba_agent")
workflow.add_conditional_edges(
    source="dba_agent", 
    path=should_continue, 
    path_map=["dba_tools", END]
)
workflow.add_edge("dba_tools", "dba_agent")


react_graph = workflow.compile()

open_png_bytes(react_graph.get_graph().draw_mermaid_png())


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(SCRIPT_DIR, "../artifacts/sales.db")
db_engine = create_engine(f"sqlite:///{db_path}")

config = {
    "configurable": {
        "db_engine": db_engine
    }
}

inputs = {
    "user_query": "How many Dell XPS 15 were sold?"
}

messages = react_graph.invoke(
    input=inputs, 
    config=config,
)

for m in messages['messages']:
    m.pretty_print()
