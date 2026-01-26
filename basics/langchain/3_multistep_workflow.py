from typing import List
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from utils.llm_openai import instantiate_llm
from utils.ask_cli import ask, ask_list

load_dotenv()
llm = instantiate_llm()


def demo_pipeline(
    industry: str,
    print_chain_graph: bool = True,
    print_logs: bool = True,
):
    # Prompts
    idea_prompt = PromptTemplate(
        template="Generate a business idea for the {industry} industry."
    )

    analysis_prompt = PromptTemplate(
        template=(
            "Analyze the following business idea: {idea}\n"
            "Identify 3 key strengths and 3 potential weaknesses of the idea."
        )
    )

    report_prompt = PromptTemplate(
        template=(
            "Generate a structured business report from this analysis of strengths and weaknesses:\n"
            "{output}"
        )
    )

    # Output + logging
    logs = []
    parser = StrOutputParser()

    parse_and_log_output_chain = RunnableParallel(
        output=parser,
        log=RunnableLambda(lambda x: logs.append(x)),
    )

    # Chains
    idea_chain = idea_prompt | llm | parse_and_log_output_chain
    analysis_chain = analysis_prompt | llm | parse_and_log_output_chain

    class AnalysisReport(TypedDict):
        """Strengths and Weaknesses about a business idea"""
        strengths: Annotated[List[str], "Idea's strength list"]
        weaknesses: Annotated[List[str], "Idea's weakness list"]

    report_chain = report_prompt | llm.with_structured_output(
        AnalysisReport, method="function_calling"
    )

    e2e_chain = (
        RunnablePassthrough()
        | idea_chain
        | RunnableParallel(idea=RunnablePassthrough())
        | analysis_chain
        | report_chain
    )

    if print_chain_graph:
        print("\n[Chain graph]")
        e2e_chain.get_graph().print_ascii()

    # Run end-to-end
    e2e_result = e2e_chain.invoke(industry)

    if print_logs:
        print("\n[Logs: raw LLM messages]")
        for i, message in enumerate(logs, start=1):
            print(f"\n--- Message {i} ---")
            print(message.content)

    print("\n------------------------")
    print("Final structured report:")
    print(e2e_result)

    return e2e_result, logs


def main():
    while True:
        print("\n=== Business Idea -> Analysis -> Structured Report (Interactive) ===")
        print("1) Run pipeline")
        print("2) Run pipeline multiple times (industries list)")
        print("q) Quit")

        choice = ask("Choose an option", default="1").lower()

        if choice == "q":
            break

        if choice == "1":
            industry = ask("Choose an industry to generate business idea", default="agriculture")
            show_graph = ask("Print ASCII graph? (y/n)", default="y").lower() in ("y", "yes")
            show_logs = ask("Print raw logs? (y/n)", default="y").lower() in ("y", "yes")
            demo_pipeline(industry=industry, print_chain_graph=show_graph, print_logs=show_logs)

        elif choice == "2":
            industries = ask_list(
                "Industries (comma-separated)",
                default_csv="agriculture,fintech,education",
            )
            show_graph = ask("Print ASCII graph? (y/n)", default="n").lower() in ("y", "yes")
            show_logs = ask("Print raw logs? (y/n)", default="n").lower() in ("y", "yes")

            for ind in industries:
                print(f"\n\n===== RUN: {ind} =====")
                demo_pipeline(industry=ind, print_chain_graph=show_graph, print_logs=show_logs)

        else:
            print("Unknown choice. Try again.")


if __name__ == "__main__":
    main()
