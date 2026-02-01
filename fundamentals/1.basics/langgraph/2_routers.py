from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from utils.display_image import open_png_bytes
from utils.ask_cli import ask  # assume available

class State(TypedDict):
    input: str
    action: Literal["reverse", "upper"]
    output: str


workflow = StateGraph(State)

def node_a(state: State):
    print("Node A")
    output = state["input"][::-1]
    print(f"output: {output}")
    return {"output": output}

def node_b(state: State):
    print("Node B")
    output = state["input"].upper()
    print(f"output: {output}")
    return {"output": output}

workflow.add_node(node_a)
workflow.add_node(node_b)

def routing_function(state: State):
    action = state["action"]
    if action == "reverse":
        return "node_a"
    if action == "upper":
        return "node_b"

workflow.add_conditional_edges(
    source=START,
    path=routing_function,
    path_map=["node_a", "node_b"],
)

workflow.add_edge("node_a", END)
workflow.add_edge("node_b", END)

graph = workflow.compile()

open_png_bytes(graph.get_graph().draw_mermaid_png())

while True:
    text = ask("Input text (or 'q' to quit)", "Hello World")
    if text.lower() == "q":
        break

    action = ask("Action (upper/reverse)", "upper").lower()
    if action not in ("upper", "reverse"):
        print("Invalid action. Use 'upper' or 'reverse'.")
        continue

    graph.invoke(
        input={
            "input": text,
            "action": action,
        },
    )
