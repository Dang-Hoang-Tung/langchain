from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from utils.display_image import open_png_bytes


class State(TypedDict):
    input: str
    action: Literal['reverse', 'upper']
    output: str


workflow = StateGraph(State)

def node_a(state: State):
    print("Node A")

    # Action: Reverse
    output = state['input'][::-1]

    print(f"output: {output}")
    return {"output": output}


def node_b(state: State):
    print("Node B")

    # Action: Uppercase
    output = state['input'].upper()

    print(f"output: {output}")
    return {"output": output}


workflow.add_node(node_a)
workflow.add_node(node_b)


# TODO - The routing function
def routing_function(state: State):
    action = state["action"]
    if action == "reverse":
        return "node_a"
    if action == "upper":
        return "node_b"


# TODO - Add your condital edges
workflow.add_conditional_edges(
    source=START,
    path=routing_function,
    path_map=["node_a", "node_b"]
)


workflow.add_edge("node_a", END)
workflow.add_edge("node_b", END)


graph = workflow.compile()


open_png_bytes(graph.get_graph().draw_mermaid_png())


graph.invoke(
    input = {
        "input": "Hello World",
        "action": "upper",
    },
)

graph.invoke(
    input = {
        "input": "Hello World",
        "action": "reverse",
    },
)
