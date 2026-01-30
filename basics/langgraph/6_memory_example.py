from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from utils.ask_cli import ask  # assume available


class MemoryState(TypedDict):
    messages: List[str]
    user_id: str
    user_memory: Dict[str, Any]  # long-term memory for preferences and counts


def remember_preferences(state: MemoryState) -> MemoryState:
    last_msg = state["messages"][-1].lower() if state["messages"] else ""

    diet = state["user_memory"].get("diet", [])

    if "vegan" in last_msg and "vegan" not in diet:
        diet.append("vegan")
    if "vegetarian" in last_msg and "vegetarian" not in diet:
        diet.append("vegetarian")
    if "gluten-free" in last_msg or "gluten free" in last_msg:
        if "gluten-free" not in diet:
            diet.append("gluten-free")

    if diet:
        state["user_memory"]["diet"] = diet

    count = state["user_memory"].get("visits", 0) + 1

    if count > 1:
        state["messages"].append(f"Welcome back! This is visit #{count}.")

    state["user_memory"]["visits"] = count

    if diet:
        if "vegan" in diet:
            state["messages"].append("May I recommend our vegan Buddha bowl?")
        elif "vegetarian" in diet:
            state["messages"].append("How about a fresh vegetarian salad?")
        elif "gluten-free" in diet:
            state["messages"].append("Try our gluten-free pasta?")

    return state


workflow = StateGraph(MemoryState)
workflow.add_node("remember", remember_preferences)
workflow.add_edge(START, "remember")
workflow.add_edge("remember", END)

app = workflow.compile(checkpointer=InMemorySaver())


# Interactive loop
user_id = ask("user_id", "user123")
config = {"configurable": {"thread_id": user_id}}

# initialize once
state = {
    "messages": [],
    "user_id": user_id,
    "user_memory": {},
}

print("\n=== Dietary Preference Memory ===")
print("Type messages like: 'I'm vegan', 'I'm also gluten-free'. Type 'show' or 'q'.\n")

while True:
    msg = ask("You", "I'm vegan and looking for options")
    msg_l = msg.lower()

    if msg_l == "q":
        break

    if msg_l == "show":
        saved = app.get_state(config).values
        if saved:
            print("\nUser Memory:", saved["user_memory"])
            print("Last messages:", saved["messages"][-5:])
        else:
            print("\nNo saved state yet.")
        print()
        continue

    saved_state = app.get_state(config).values
    if not saved_state:
        saved_state = state

    saved_state["messages"].append(msg)
    result = app.invoke(saved_state, config=config)

    print("\nUser Memory:", result["user_memory"])
    print("Latest messages:", result["messages"][-2:])
    print()
