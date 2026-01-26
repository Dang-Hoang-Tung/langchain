from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_core.output_parsers.openai_tools import parse_tool_calls
from dotenv import load_dotenv
from utils.llm_hf import instantiate_llm
from utils.ask_cli import ask

load_dotenv()

llm = instantiate_llm()

# Tool creation
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [multiply]
tool_map = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)

# Interactive loop
system_prompt = ask("Enter a system prompt", default="You're a helpful assistant")

while True:
    question = ask("Question (or 'q' to quit)", default="3 multiplied by 2")
    if question.lower() == "q":
        break

    messages = [
        SystemMessage(system_prompt),
        HumanMessage(question),
    ]

    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    try:
        parsed_tool_calls = parse_tool_calls(ai_message.additional_kwargs.get("tool_calls"))
        print(parsed_tool_calls)
    except Exception as e:
        print(f"Error parsing tool calls: {e}")
        print("Required tool most likely does not exist.")
        continue

    for tool_call in parsed_tool_calls:
        tool_call_id = tool_call["id"]
        function_name = tool_call["name"]
        arguments = tool_call["args"]
        func = tool_map[function_name]
        result = func.invoke(arguments)
        tool_message = ToolMessage(
            content=str(result),
            name=function_name,
            tool_call_id=tool_call_id,
        )
        messages.append(tool_message)

    ai_message = llm_with_tools.invoke(messages)
    print(ai_message)
