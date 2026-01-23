import os
from typing import List
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


idea_prompt = PromptTemplate(
    template=(
        "Generate a business idea for the {industry} industry."
    )
)

logs = []
parser = StrOutputParser()

parse_and_log_output_chain = RunnableParallel(
    output=parser,
    log=RunnableLambda(lambda x: logs.append(x))
)


idea_chain = idea_prompt | llm | parse_and_log_output_chain

idea_result = idea_chain.invoke("agriculture")
print(idea_result["output"])

print(logs[0].content)


analysis_prompt = PromptTemplate(
    template=(
        "Analyze the following business idea: {idea}"
        "Identify 3 key strengths and 3 potential weaknesses of the idea."
    )
)

analysis_chain = (
    analysis_prompt | llm | parse_and_log_output_chain
)

analysis_result = analysis_chain.invoke(idea_result["output"])

print(analysis_result["output"])

print(logs[1].content)


report_prompt = PromptTemplate(
    template=(
        "Generate a structured business report from this analysis of strengths and weaknesses: "
        "{output}"
    )
)


class AnalysisReport(TypedDict):
    """Strengths and Weaknesses about a business idea"""
    strengths: Annotated[List[str], "Idea's strength list"]
    weaknesses: Annotated[List[str], "Idea's weakness list"]


report_chain = (
    report_prompt | llm.with_structured_output(AnalysisReport, method="function_calling")
)

# report_result = report_chain.invoke(analysis_result["output"])
# print(report_result)

e2e_chain = (
    RunnablePassthrough()
    | idea_chain
    | RunnableParallel(idea=RunnablePassthrough())
    | analysis_chain
    | report_chain
)

e2e_chain.get_graph().print_ascii()

# Change the industry if you want
e2e_result = e2e_chain.invoke("agriculture")

for message in logs:
    print(message.content)

print("------------------------")
print("Final structured report:")
print(e2e_result)
