import os
from typing import Dict
import requests
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from utils.ask_cli import ask
from utils.display_image import open_png_bytes

load_dotenv()


@tool
def random_got_quote_tool() -> Dict:
    """Return a random Game of Thrones quote and the character who said it"""
    response = requests.get("https://api.gameofthronesquotes.xyz/v1/random")
    return response.json()


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(question: str) -> Dict:
    """Return top search results for a given search query"""
    return tavily_client.search(question)


tools = [random_got_quote_tool, web_search]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

llm_with_tools = llm.bind_tools(tools)


def agent(state: MessagesState):
    ai_message = llm_with_tools.invoke(state["messages"])
    return {"messages": ai_message}


def router(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(source="agent", path=router, path_map=["tools", END])
workflow.add_edge("tools", "agent")

graph = workflow.compile()


open_png_bytes(graph.get_graph().draw_mermaid_png())

system_prompt = ask(
    "System prompt",
    (
        "You are a Web Researcher focused on Game of Thrones. "
        "If user asks you a random quote about GoT. You will not only "
        "provide it, but also search the web to find the actor or actress "
        "who perform the character who said that. "
        "So, your output should be: Quote, Character and Performer."
    ),
)

messages = [SystemMessage(system_prompt)]

print("\n=== GoT Web Research Agent (Interactive) ===")
print("Type a request like: 'Give me a random GoT quote' or ask about GoT. Type 'q' to quit.\n")

while True:
    user_text = ask("You", "Give me a random GoT quote")
    if user_text.lower() == "q":
        break

    messages.append(HumanMessage(user_text))

    result = graph.invoke(input={"messages": messages})

    # keep conversation going by updating the running messages list
    messages = result["messages"]

    print("\n--- Assistant ---")
    for m in result["messages"][-3:]:
        # print the latest few messages (often includes tool + final)
        try:
            m.pretty_print()
        except Exception:
            print(getattr(m, "content", m))
    print()
