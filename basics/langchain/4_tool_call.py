from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain.tools import tool
from langchain_core.output_parsers.openai_tools import parse_tool_calls
from dotenv import load_dotenv
from utils.llm_hf import instantiate_llm

load_dotenv()

llm = instantiate_llm()

# Tool creation
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tools = [multiply]

tool_map = {tool.name:tool for tool in tools}


llm_with_tools = llm.bind_tools(tools)


question = "3 multiplied by 2"

messages = [
    SystemMessage("You're a helpful assistant"),
    HumanMessage(question)
]

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

parsed_tool_calls = parse_tool_calls(
    ai_message.additional_kwargs.get("tool_calls")
)

print(parsed_tool_calls)

for tool_call in parsed_tool_calls:
    tool_call_id = tool_call['id']
    function_name = tool_call['name']
    arguments = tool_call['args']
    func = tool_map[function_name]
    result = func.invoke(arguments)
    tool_message = ToolMessage(
        content=result,
        name=function_name,
        tool_call_id=tool_call_id,
    )
    messages.append(tool_message)


ai_message = llm_with_tools.invoke(messages)

print(ai_message)
