from langgraph.checkpoint.memory import InMemorySaver
from typing import List, Optional, Literal, Dict, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from utils.display_image import open_png_bytes
from utils.ask_cli import ask  # assume available


class LibraryState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    section: Optional[Literal["borrow", "return", "overdue", "unknown"]]
    books_borrowed: List[str]
    resolved: bool
    last_user_message: HumanMessage


# Router: return a partial update dict
def route_library(state: LibraryState) -> Dict[str, Any]:
    last_msg = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            last_msg = msg.content.lower()
            break

    if "borrow" in last_msg:
        intent = "borrow"
    elif "return" in last_msg:
        intent = "return"
    elif "overdue" in last_msg or "fine" in last_msg:
        intent = "overdue"
    else:
        intent = "unknown"

    return {
        "last_user_message": last_msg,
        "section": intent,
        "resolved": False,
    }


def handle_borrow(state: LibraryState) -> Dict[str, Any]:
    book_title = None
    content = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            content = msg.content
            break

    if content.lower().startswith("borrow "):
        book_title = content[6:].strip().title()

    current_books = state.get("books_borrowed", [])
    updates = {}

    if book_title:
        if book_title not in current_books:
            updates["books_borrowed"] = current_books + [book_title]
            ai_text = f"Sure! I've added '{book_title}' to your borrowed books."
        else:
            ai_text = f"It looks like '{book_title}' is already on your list."
    else:
        ai_text = "I'm sorry, I couldn't identify the book you'd like to borrow."

    return {"messages": [AIMessage(content=ai_text)], **updates, "resolved": True}


def handle_return(state: LibraryState) -> Dict[str, Any]:
    book_title = None
    content = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            content = msg.content
            break

    if content.lower().startswith("return "):
        book_title = content[6:].strip().title()

    current_books = state.get("books_borrowed", [])
    updates = {}

    if book_title:
        if book_title in current_books:
            updates["books_borrowed"] = [x for x in current_books if x != book_title]
            ai_text = f"Sure! I've removed '{book_title}' from your borrowed books."
        else:
            ai_text = f"It looks like '{book_title}' is not on your list."
    else:
        ai_text = "I'm sorry, I couldn't identify the book you'd like to return."

    return {"messages": [AIMessage(content=ai_text)], **updates, "resolved": True}


def handle_overdue(state: LibraryState) -> Dict[str, Any]:
    current_books = state.get("books_borrowed", [])
    if current_books:
        books_list = ", ".join([f"'{book}'" for book in current_books])
        ai_text = (
            f"You currently have {len(current_books)} book(s) borrowed: {books_list}. "
            "Please visit the library to check if any fines apply."
        )
    else:
        ai_text = "You have no books borrowed at the moment. There are no overdue fines."
    return {"messages": [AIMessage(content=ai_text)], "resolved": True}


def handle_unknown(state: LibraryState) -> Dict[str, Any]:
    return {
        "messages": [
            AIMessage(
                content="Sorry, I didn't understand your request. Please clarify whether you would like to borrow, return, or check for overdue items."
            )
        ],
        "resolved": True,
    }


def next_step(state: LibraryState) -> str:
    if state.get("resolved", False):
        return END
    section = state.get("section", None)
    return section if section else END


# Build workflow
workflow = StateGraph(LibraryState)
workflow.add_node("router", route_library)
workflow.add_node("borrow", handle_borrow)
workflow.add_node("return", handle_return)
workflow.add_node("overdue", handle_overdue)
workflow.add_node("unknown", handle_unknown)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    next_step,
    {
        "borrow": "borrow",
        "return": "return",
        "overdue": "overdue",
        "unknown": "unknown",
        END: END,
    },
)

workflow.add_edge("borrow", END)
workflow.add_edge("return", END)
workflow.add_edge("overdue", END)
workflow.add_edge("unknown", END)

app = workflow.compile(checkpointer=InMemorySaver())

open_png_bytes(app.get_graph().draw_mermaid_png())


def print_state(state) -> None:
    print(f"  Books borrowed: {state.get('books_borrowed', [])}")
    if state.get("messages"):
        print(f"  Last message: {state['messages'][-1].content}")
    print()


if __name__ == "__main__":
    thread_id = 'demo_user'
    config = {"configurable": {"thread_id": thread_id}}

    print("=== Library Assistant (Interactive) ===")
    print("Try: 'borrow <book name>', 'return <book name>', 'any overdue?', or 'q' to quit.\n")

    # initialize state once
    state = {
        "messages": [],
        "books_borrowed": [],
        "resolved": False,
    }

    while True:
        user_text = ask("You")
        if user_text.lower() == "q":
            break

        # pull current saved state (if any), append new message
        current_state = app.get_state(config).values or state
        new_state = {
            **current_state,
            "messages": current_state.get("messages", []) + [HumanMessage(content=user_text)],
            "resolved": False,
        }

        result = app.invoke(new_state, config=config)
        print_state(result)
