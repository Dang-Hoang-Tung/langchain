import random
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display
from dotenv import load_dotenv
from utils.llm_hf import instantiate_llm
from utils.display_image import open_png_bytes

load_dotenv()

class State(TypedDict):
    input: int
    output: int


def node_a(state: State)->State:
    input_value = state['input']
    offset = random.randint(1,10)
    output =  input_value + offset
    print(
        f"NODE A:\n "
        f"->input:{input_value}\n "
        f"->offset:{offset}\n "
        f"->output:{output}\n "
    )
    return State(output=output)


def node_b(state: State):
    input_value = state['output']
    offset = random.randint(1,10)
    output =  input_value + offset
    print(
        f"NODE B:\n "
        f"->input:{input_value}\n "
        f"->offset:{offset}\n "
        f"->output:{output}\n "
    )
    return {"output": output}


workflow = StateGraph(state_schema=State)


workflow.add_node(node_a)
workflow.add_node(node_b)


workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

graph = workflow.compile()


open_png_bytes(graph.get_graph().draw_mermaid_png())


graph.invoke(
    input = {
        "input": 1,
    },
)


llm = instantiate_llm()


class State(TypedDict):
    question:str
    response:str


def model(state: State):
    question = state["question"]
    response = llm.invoke([
        SystemMessage("You're a Pokémon specialist"),
        HumanMessage(question)
    ])

    return {"response": response.content}


workflow = StateGraph(State)

workflow.add_node("model", model)

workflow.add_edge(START, "model")
workflow.add_edge("model", END)

graph = workflow.compile()


open_png_bytes(graph.get_graph().draw_mermaid_png())


result = graph.invoke(
    input={
        "question": "What's the name of Ash's first pokémon?"
    },
)

print(result)

