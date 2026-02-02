import os 
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from tavily import TavilyClient
from utils.display_image import open_png_bytes
from utils.llm_openai import instantiate_llm

from dotenv import load_dotenv
load_dotenv()

embeddings_fn = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_store = Chroma(
    collection_name="udacity",
    embedding_function=embeddings_fn
)

file_path = "artifacts/compact-guide-to-large-language-models.pdf"
loader = PyPDFLoader(file_path)
# Load pages synchronously to avoid using `async for` at top-level
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(pages)

_ = vector_store.add_documents(documents=all_splits)


llm = instantiate_llm()


class State(MessagesState):
    question: str
    documents: List[Document]
    search_required: str = "NO"
    answer: str


def retrieve(state: State):
    question = state["question"]
    retrieved_docs = vector_store.similarity_search(question)
    return {"documents": retrieved_docs}


def evaluator(state: State):
    question = state["question"]
    documents = state["documents"]
    docs_content = "\n\n".join(doc.page_content for doc in documents)
    
    template = ChatPromptTemplate([
        ("system", "You are an assistant for evaluating context."),
        ("human", "Use the following pieces of retrieved context to determine if an " 
                  "additional search is required. If the context is relevant to the " 
                  "question, your response should be 'NO', i.e. no need for search. "
                  "If the context is not relevant, say 'YES', i.e. search the web for an answer. " 
                  "Don't explain anything, if it's to search the web, say YES, otherwise, NO. "
                  "\n# Question: \n-> {question} "
                  "\n# Context: \n-> {context} "
                  "\n# Answer: "),
    ])

    chain = template | llm | StrOutputParser()

    search_required = chain.invoke(
        {"context": docs_content, "question": question}
    )

    return {"search_required": search_required}
    

@tool
def web_search(question:str)->Dict:
    """
    Return top search results for a given search query
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(question)
    return response


llm_with_tools = llm.bind_tools([web_search])

def researcher(state: State):
    question = state["question"]
    messages = state["messages"]
    if not messages:
        human_message = HumanMessage(
            "Conduct a web research to findout the answer to the following question: "
            f"```{question}```"
        )
        messages = [human_message]
    
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)
    return {"messages": messages, "answer": ai_message.content}


def augment(state: State):
    question = state["question"]
    documents = state["documents"]
    docs_content = "\n\n".join(doc.page_content for doc in documents)
    
    template = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks."),
        ("human", "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. " 
                "Use three sentences maximum and keep the answer concise. "
                "\n# Question: \n-> {question} "
                "\n# Context: \n-> {context} "
                "\n# Answer: "),
    ])

    messages = template.invoke(
        {"context": docs_content, "question": question}
    ).to_messages()

    return {"messages": messages}


def generate(state: State):
    ai_message = llm.invoke(state["messages"])
    return {
        "answer": ai_message.content,
        "messages": state["messages"] + [ai_message],
    }


def rag_router(state: State):
    search_required = (state.get("search_required") or "NO").strip().lower()
    return "researcher" if search_required == "yes" else "augment"


def tool_router(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "web_search"

    return END


workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("augment", augment)
workflow.add_node("generate", generate)
workflow.add_node("evaluator", evaluator)
workflow.add_node("researcher", researcher)
workflow.add_node("web_search", ToolNode([web_search]))

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "evaluator")

workflow.add_conditional_edges(
    source="evaluator", 
    path=rag_router, 
    path_map=["augment", "researcher"]
)

workflow.add_conditional_edges(
    source="researcher", 
    path=tool_router, 
    path_map=["web_search", END]
)
workflow.add_edge("web_search", "researcher")

workflow.add_edge("augment", "generate")
workflow.add_edge("generate", END)


graph = workflow.compile()

open_png_bytes(graph.get_graph().draw_mermaid_png())


output = graph.invoke(
    input={"question": "What is Pokemon?"},
)

print(output["answer"])


for message in output["messages"]:
    message.pretty_print()

print(output["search_required"])

output = graph.invoke(
    input={"question": "What is Open Source model?"},
)

print(output["search_required"])

for message in output["messages"]:
    message.pretty_print()