from typing import TypedDict, List, Optional, Literal, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

from utils.ask_cli import ask, ask_int, ask_list  # assume available


# -------------------------
# Example 1: Thread isolation
# -------------------------
class SupportState(TypedDict):
    messages: List[str]
    user_name: str
    visit: int

def handle_support(state: SupportState) -> SupportState:
    state["visit"] += 1
    state["messages"].append(f"Support visit {state['visit']} for {state['user_name']}")
    return state

def run_example_1():
    memory = InMemorySaver()
    app = (
        StateGraph(SupportState)
        .add_node("support", handle_support)
        .add_edge(START, "support")
        .add_edge("support", END)
        .compile(checkpointer=memory)
    )

    # track multiple users/threads
    threads: dict[str, dict] = {}

    user_name = ask("User name")
    thread_id = f"{user_name.lower()}_support"
    config = {"configurable": {"thread_id": thread_id}}

    # init current thread if needed
    if thread_id not in threads:
        app.invoke({"messages": [], "user_name": user_name, "visit": 0}, config=config)
        threads[thread_id] = {"user_name": user_name}

    print("\n=== Thread isolation demo ===")
    print(f"Using thread_id={thread_id}\n")

    while True:
        cmd = ask("'add_visit' to add a visit, 'show' to view ALL threads, 'change_user' to switch, or 'q' to quit", "")
        cmd_l = cmd.lower()

        if cmd_l == "q":
            break

        if cmd_l == "change_user":
            user_name = ask("New user name")
            thread_id = f"{user_name.lower()}_support"
            config = {"configurable": {"thread_id": thread_id}}

            # init new thread if not present
            if thread_id not in threads:
                app.invoke({"messages": [], "user_name": user_name, "visit": 0}, config=config)
                threads[thread_id] = {"user_name": user_name}

            print(f"\nSwitched to thread_id={thread_id}\n")
            continue

        if cmd_l == "show":
            print("\n--- All threads ---")
            for tid, meta in threads.items():
                cfg = {"configurable": {"thread_id": tid}}
                saved = app.get_state(cfg).values
                print(f"\nthread_id: {tid} (user_name: {meta['user_name']})")
                print(saved)
            print()
            continue

        if cmd_l == "add_visit":
            saved = app.get_state(config).values
            result = app.invoke(saved, config=config)
            print(result["messages"][-1])
            continue

        



# -------------------------
# Example 2: Config vs State
# -------------------------
class DetailedState(TypedDict):
    messages: List[str]
    current_issue: str
    resolved: bool
    timestamp: str

def process_with_config(state: DetailedState, config: RunnableConfig) -> DetailedState:
    settings = config.get("configurable", {})

    if settings.get("priority") == "high":
        state["messages"].append("HIGH PRIORITY!!!: Escalating immediately!")
    else:
        state["messages"].append("Standard processing...")

    if settings.get("language") == "es":
        state["messages"].append("Hola! ¿Cómo puedo ayudarte?")
    else:
        state["messages"].append("Hello! How can I help you?")

    state["timestamp"] = datetime.utcnow().isoformat()
    return state

def run_example_2():
    app = (
        StateGraph(DetailedState)
        .add_node("process", process_with_config)
        .add_edge(START, "process")
        .add_edge("process", END)
        .compile(checkpointer=InMemorySaver())
    )

    print("\n=== Config vs State demo ===")

    user_name = ask("User name")
    thread_id = f"{user_name.lower()}_detailed"

    priority = ask("priority (normal/high)").lower()
    language = ask("language (en/es)").lower()
    issue = ask("current_issue", "Login problem")

    state = {
        "messages": [f"{user_name} needs help"],
        "current_issue": issue,
        "resolved": False,
        "timestamp": "",
    }

    config = {"configurable": {"thread_id": thread_id, "priority": priority, "language": language}}
    result = app.invoke(state, config=config)

    print("\nResult messages:")
    for m in result["messages"]:
        print("-", m)
    print("timestamp:", result["timestamp"])


# -------------------------
# Example 3: Time travel with checkpoint history
# -------------------------
class ConversationState(TypedDict):
    messages: List[str]
    conversation_topic: str
    mood: str

def conversation_node(state: ConversationState) -> ConversationState:
    topic = state["messages"][-1].split(" ")[-1].lower()
    mood = state["mood"]

    if topic == "weather" and mood == "happy":
        response = "What a beautiful sunny day!"
    elif topic == "weather":
        response = "Looks like it might rain today."
    else:
        response = f"Let's talk about {topic}. I'm feeling {mood} today."

    state["messages"].append(f"AI: {response}")
    state["conversation_topic"] = topic
    return state

def run_example_3():
    memory = InMemorySaver()
    app = (
        StateGraph(ConversationState)
        .add_node("chat", conversation_node)
        .add_edge(START, "chat")
        .add_edge("chat", END)
        .compile(checkpointer=memory)
    )

    print("\n=== Time travel demo ===")

    user_name = ask("User name")
    thread_id = f"{user_name.lower()}_time_travel"
    mood = ask("initial mood", "happy")

    config = {"configurable": {"thread_id": thread_id}}

    first_msg = ask("Type your first message", "Let's talk about the weather")
    initial = {"messages": [first_msg], "mood": mood}

    res1 = app.invoke(initial, config=config)
    print("\nAfter first invoke:")
    print(res1["messages"][-1])

    while True:
        cmd = ask("Type 'travel' to change mood at an earlier checkpoint, 'history' to list, or 'q' to quit", "travel").lower()
        if cmd == "q":
            break

        if cmd == "history":
            history = list(app.get_state_history(config))
            print(f"checkpoints: {len(history)}")
            for i, h in enumerate(history):
                vals = h.values
                last = vals["messages"][-1] if vals.get("messages") else ""
                print(f"{i}: mood={vals.get('mood')} last='{last}'")
            continue

        # travel
        history = list(app.get_state_history(config))
        if len(history) < 2:
            print("Not enough history to time travel yet.")
            continue

        # Use the earlier checkpoint like your example (index 1)
        checkpoint_index = ask_int("Checkpoint index to update (e.g., 1)", 1)
        if checkpoint_index < 0 or checkpoint_index >= len(history):
            print("Invalid checkpoint index.")
            continue

        new_mood = ask("New mood", "sad")
        chosen = history[checkpoint_index]
        new_config = app.update_state(chosen.config, {"mood": new_mood})
        res2 = app.invoke(None, config=new_config)

        print("\nAfter time travel invoke:")
        print(res2["messages"][-1])


# -------------------------
# Example 4: Practical memory features
# -------------------------
class MemoryState(TypedDict):
    messages: List[str]
    user_id: str
    user_memory: Dict[str, Any]

def remember_user_preferences(state: MemoryState) -> MemoryState:
    new_preference = state["messages"][-1].split(" ")[-1].lower()
    preferences = state["user_memory"].get("food_preferences", [])
    preferences.append(new_preference)
    state["user_memory"]["food_preferences"] = preferences
    return state

def suggest_a_dessert(state: MemoryState) -> MemoryState:
    preferences = state["user_memory"].get("food_preferences", [])

    if "chocolate" in preferences:
        if "cookies" in preferences:
            state["messages"].append("How about a chocolate cookies for dessert?")
            return state
        elif "cake" in preferences:
            state["messages"].append("How about a chocolate cake for dessert?")
            return state

    state["messages"].append("Can you tell me more about your food preferences?")
    return state

def run_example_4():
    workflow = StateGraph(MemoryState)
    workflow.add_node("remember", remember_user_preferences)
    workflow.add_node("suggest", suggest_a_dessert)
    workflow.add_edge(START, "remember")
    workflow.add_edge("remember", "suggest")
    workflow.add_edge("suggest", END)

    app = workflow.compile(checkpointer=InMemorySaver())

    print("\n=== Practical memory demo ===")

    user_name = ask("User name")
    user_id = ask("user_id", "user_123")
    thread_id = f"{user_name}_memory"
    config = {"configurable": {"thread_id": thread_id}}

    # init state
    state = {
        "messages": [],
        "user_id": user_id,
        "user_memory": {},
    }

    while True:
        msg = ask("Tell me a food preference (or 'q' to quit)", "I like chocolate")
        if msg.lower() == "q":
            break

        saved = app.get_state(config).values
        if not saved:
            saved = state

        # append new message then invoke
        saved = {
            **saved,
            "messages": saved.get("messages", []) + [msg],
        }

        result = app.invoke(saved, config=config)

        print("\nMemory:", result["user_memory"])
        print("Response:", result["messages"][-1])
        print()


# -------------------------
# Menu
# -------------------------
if __name__ == "__main__":
    while True:
        print("\n=== LangGraph Checkpointing Demos (Interactive) ===")
        print("1) Thread isolation")
        print("2) Config vs State")
        print("3) Time travel with checkpoint history")
        print("4) Practical memory features")
        print("q) Quit")

        choice = ask("Choose an option").lower()
        if choice == "q":
            break
        elif choice == "1":
            run_example_1()
        elif choice == "2":
            run_example_2()
        elif choice == "3":
            run_example_3()
        elif choice == "4":
            run_example_4()
        else:
            print("Invalid option.")
