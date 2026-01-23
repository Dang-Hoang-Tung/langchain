from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


model = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    # repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

llm = ChatHuggingFace(llm=model)


if False:
    ### ----- Basic usage ----- ###

    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate(
        template="Tell me a joke about {topic}"
    )
    parser = StrOutputParser()
    output = parser.invoke(
        llm.invoke(
            prompt.invoke(
                {"topic": "Python"}
            )
        )
    )
    print(output)


if False:
    ### ----- Runnables ----- ###
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.tracers.context import collect_runs

    prompt = PromptTemplate(
        template="Tell me a joke about {topic}"
    )
    parser = StrOutputParser()
    
    runnables = [prompt, llm, parser]

    for runnable in runnables:
        print(f"{repr(runnable).split('(')[0]}")
        print(f"\tINVOKE: {repr(runnable.invoke)}")
        print(f"\tBATCH: {repr(runnable.batch)}")
        print(f"\tSTREAM: {repr(runnable.stream)}\n")

        print(f"\tINPUT: {repr(runnable.get_input_schema())}")
        print(f"\tOUTPUT: {repr(runnable.get_output_schema())}")
        print(f"\tCONFIG: {repr(runnable.config_schema())}\n")


    with collect_runs() as run_collection:
        result = llm.invoke(
            "Hello",
            config={
                'run_name': 'demo_run',
                'tags': ['demo', 'lcel'],
                'metadata': {'lesson': 2}
            }
        )
        print(run_collection.traced_runs[0].dict())


if True:
    ### ----- Runnable Chains, Parallels, and Lambdas ----- ###
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel

    prompt = PromptTemplate(
        template="Tell me a joke about {topic}"
    )
    parser = StrOutputParser()

    chain = RunnableSequence(prompt, llm, parser)

    # Chain invoke
    output = chain.invoke({"topic": "Python"})
    print(output)

    # Chain stream
    for chunk in chain.stream({"topic": "Python"}):
        print(chunk, end="", flush=True)

    # Chain batch
    output = chain.batch([
        {"topic": "Python"},
        {"topic": "Data"},
        {"topic": "Machine Learning"},
    ])
    print(output)

    # Visualize chain
    chain.get_graph().print_ascii()


if False:
    ### ----- Advanced runnable ----- ###
    from langchain_core.runnables import RunnableLambda, RunnableParallel

    def double(x:int)->int:
        return 2*x
    
    runnable = RunnableLambda(double)
    print(runnable.invoke(5))  # Should print 10

    parallel_chain = RunnableParallel(
        double=RunnableLambda(lambda x: x * 2),
        triple=RunnableLambda(lambda x: x * 3),
    )

    parallel_chain.invoke(3)  # Should print {'double': 6, 'triple': 9}

    parallel_chain.get_graph().print_ascii()


if True:
    ### ----- Using LCEL ----- ###
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate(
        template="Tell me a joke about {topic}"
    )
    parser = StrOutputParser()

    chain = prompt | llm | parser
    output = chain.invoke({"topic": "computers"})
    print(output)
