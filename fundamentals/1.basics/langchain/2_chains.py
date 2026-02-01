from dotenv import load_dotenv
from utils.llm_hf import instantiate_llm
from utils.ask_cli import ask, ask_int, ask_list

load_dotenv()
llm = instantiate_llm()

def demo_basic_usage(llm):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    topic = ask("Enter topic to generate a joke about", default="Python")

    prompt = PromptTemplate(template="Tell me a joke about {topic}")
    parser = StrOutputParser()

    output = parser.invoke(
        llm.invoke(
            prompt.invoke({"topic": topic})
        )
    )
    print("\n[Basic usage output]")
    print(output)


def demo_runnables(llm):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.tracers.context import collect_runs

    topic = ask("Joke topic (for runnable introspection)", default="Python")

    prompt = PromptTemplate(template="Tell me a joke about {topic}")
    parser = StrOutputParser()
    runnables = [prompt, llm, parser]

    print("\n[Runnable capabilities + schemas]\n")
    for runnable in runnables:
        print(f"{repr(runnable).split('(')[0]}")
        print(f"\tINVOKE: {repr(runnable.invoke)}")
        print(f"\tBATCH: {repr(runnable.batch)}")
        print(f"\tSTREAM: {repr(runnable.stream)}\n")
        print(f"\tINPUT: {repr(runnable.get_input_schema())}")
        print(f"\tOUTPUT: {repr(runnable.get_output_schema())}")
        print(f"\tCONFIG: {repr(runnable.config_schema())}\n")

    # Demonstrate tracing with collect_runs
    text = ask("Text to send to llm.invoke for tracing", default="Hello")

    run_name = ask("run_name", default="demo_run")
    tags_csv = ask("tags (comma-separated)", default="demo,lcel")
    tags = [t.strip() for t in tags_csv.split(",") if t.strip()]

    with collect_runs() as run_collection:
        _ = llm.invoke(
            text,
            config={
                "run_name": run_name,
                "tags": tags,
                "metadata": {"lesson": 2},
            },
        )
        print("\n[Traced run dict]")
        print(run_collection.traced_runs[0].dict())

    # Also show the prompt runnable quickly
    print("\n[Prompt example output]")
    print(llm.invoke(prompt.invoke({"topic": topic})))


def demo_runnable_chains(llm):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableSequence

    topic = ask("Topic for chain invoke/stream", default="Python")
    batch_topics = ask_list("Batch topics (comma-separated)", "Python,Data,Machine Learning")

    prompt = PromptTemplate(template="Tell me a joke about {topic}")
    parser = StrOutputParser()

    chain = RunnableSequence(prompt, llm, parser)

    # Chain invoke
    output = chain.invoke({"topic": topic})
    print("\n[Chain invoke output]")
    print(output)

    # Chain stream
    print("\n[Chain stream output]")
    for chunk in chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)
    print("\n")

    # Chain batch
    output = chain.batch([{"topic": t} for t in batch_topics])
    print("\n[Chain batch output]")
    for t, out in zip(batch_topics, output):
        print(f"- {t}: {out}")

    # Visualize chain
    show_graph = ask("Print ASCII graph? (y/n)", default="y").lower() in ("y", "yes")
    if show_graph:
        print("\n[Chain graph]")
        chain.get_graph().print_ascii()


def demo_advanced_runnable():
    from langchain_core.runnables import RunnableLambda, RunnableParallel

    n = ask_int("Number to double/triple", default=3)

    def double(x: int) -> int:
        return 2 * x

    runnable = RunnableLambda(double)
    print("\n[RunnableLambda double]")
    print(runnable.invoke(n))

    parallel_chain = RunnableParallel(
        double=RunnableLambda(lambda x: x * 2),
        triple=RunnableLambda(lambda x: x * 3),
    )

    print("\n[RunnableParallel output]")
    print(parallel_chain.invoke(n))

    show_graph = ask("Print ASCII graph? (y/n)", default="y").lower() in ("y", "yes")
    if show_graph:
        print("\n[Parallel graph]")
        parallel_chain.get_graph().print_ascii()


def demo_lcel(llm):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    topic = ask("Topic for LCEL pipe", default="computers")

    prompt = PromptTemplate(template="Tell me a joke about {topic}")
    parser = StrOutputParser()

    chain = prompt | llm | parser
    output = chain.invoke({"topic": topic})
    print("\n[LCEL output]")
    print(output)


def main():
    actions = {
        "1": ("Basic usage (PromptTemplate + parser)", lambda: demo_basic_usage(llm)),
        "2": ("Runnables + tracing (collect_runs)", lambda: demo_runnables(llm)),
        "3": ("RunnableSequence chains (invoke/stream/batch + graph)", lambda: demo_runnable_chains(llm)),
        "4": ("Advanced runnable (Lambda + Parallel + graph)", demo_advanced_runnable),
        "5": ("LCEL (prompt | llm | parser)", lambda: demo_lcel(llm)),
    }

    while True:
        print("\n=== HF LLM + LangChain Core Demos (Interactive) ===")
        for k, (label, _) in actions.items():
            print(f"{k}) {label}")
        print("a) Run ALL")
        print("q) Quit")

        choice = ask("Choose an option").lower()

        if choice == "q":
            break
        elif choice == "a":
            for _, fn in actions.values():
                fn()
        elif choice in actions:
            actions[choice][1]()
        else:
            print("Unknown choice. Try again.")


if __name__ == "__main__":
    main()
