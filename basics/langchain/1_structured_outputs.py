from dotenv import load_dotenv
from utils.llm_openai import instantiate_llm
from utils.ask_cli import ask, ask_int

load_dotenv()
llm = instantiate_llm()

def demo_basic_parsers(llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_classic.output_parsers.boolean import BooleanOutputParser
    from langchain_classic.output_parsers.datetime import DatetimeOutputParser

    user_prompt = ask("Enter a prompt for StrOutputParser", default="hello")
    output = StrOutputParser().invoke(llm.invoke(user_prompt))
    print("\n[StrOutputParser output]")
    print(output)

    dt_prompt = ask(
        "Enter a prompt to request a datetime",
        default="Output a random datetime in %Y-%m-%dT%H:%M:%S.%fZ. Don't say anything else",
    )
    output = DatetimeOutputParser().invoke(llm.invoke(dt_prompt))
    print("\n[DatetimeOutputParser output]")
    print(output)

    bool_question = ask(
        "Enter a YES/NO question for BooleanOutputParser",
        default="Are you an AI? YES or NO only",
    )
    output = BooleanOutputParser().invoke(input=llm.invoke(bool_question))
    print("\n[BooleanOutputParser output]")
    print(output)


def demo_typeddict_structured(llm):
    from typing_extensions import Annotated, TypedDict

    class UserInfo(TypedDict):
        """User's info."""
        name: Annotated[str, "", "User's name. Defaults to ''"]
        country: Annotated[str, "", "Where the user lives. Defaults to ''"]

    llm_with_structure = llm.with_structured_output(UserInfo)

    typed_prompt = ask(
        "Enter a sentence containing name + country (TypedDict)",
        default="My name is Henrique, and I am from Brazil",
    )
    output = llm_with_structure.invoke(typed_prompt)
    print("\n[TypedDict structured output]")
    print(output)


def demo_pydantic_structured(llm):
    from pydantic import BaseModel, Field
    from typing_extensions import Annotated

    class PydanticUserInfo(BaseModel):
        """User's info."""
        name: Annotated[str, Field(description="User's name. Defaults to ''", default=None)]
        country: Annotated[str, Field(description="Where the user lives. Defaults to ''", default=None)]

    llm_with_structure = llm.with_structured_output(PydanticUserInfo)

    pyd_prompt = ask(
        "Enter a sentence containing a person's name + their country (Pydantic)",
        default="Hello, my name is the same as the capital of the U.S. But I'm from a country where we usually associate with kangaroos",
    )
    output = llm_with_structure.invoke(pyd_prompt)
    print("\n[Pydantic structured output]")
    print(output)


def demo_error_handling(llm):
    from typing import List
    from typing_extensions import Annotated
    from pydantic import BaseModel, Field
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.exceptions import OutputParserException
    from langchain_classic.output_parsers import OutputFixingParser

    class Performer(BaseModel):
        """Filmography info about an actor/actress"""
        name: Annotated[str, Field(description="name of an actor/actress")]
        film_names: Annotated[List[str], Field(description="list of names of films they starred in")]

    llm_with_structure = llm.with_structured_output(Performer)

    actor = ask("Actor/actress name for filmography", default="Scarlett Johansson")
    top_n = ask_int("How many films?", default=5)

    response = llm_with_structure.invoke(f"Generate the filmography for {actor}. Top {top_n} only")
    print("\n[Structured Performer response]")
    print(response)

    parser = PydanticOutputParser(pydantic_object=Performer)

    # If response is already a Pydantic object from LC, parse its JSON dump:
    parser.parse(response.model_dump_json())

    misformatted_result = ask(
        "Paste a misformatted result to test OutputFixingParser",
        default="{'name': 'Scarlett Johansson', 'film_names': ['The Avengers']}",
    )

    try:
        parser.parse(misformatted_result)
    except OutputParserException as e:
        print("\n[Parse error caught]")
        print(e)

    fixed = OutputFixingParser.from_llm(parser=parser, llm=llm).parse(misformatted_result)
    print("\n[Fixed parsed output]")
    print(fixed)


def main():
    actions = {
        "1": ("Basic output parsers", demo_basic_parsers),
        "2": ("TypedDict structured output", demo_typeddict_structured),
        "3": ("Pydantic structured output", demo_pydantic_structured),
        "4": ("Error handling / OutputFixingParser", demo_error_handling),
    }

    while True:
        print("\n=== LangChain Output Parser Demos ===")
        for k, (label, _) in actions.items():
            print(f"{k}) {label}")
        print("a) Run ALL")
        print("q) Quit")

        choice = ask("Choose an option").lower()

        if choice == "q":
            break
        elif choice == "a":
            for _, fn in actions.values():
                fn(llm)
        elif choice in actions:
            actions[choice][1](llm)
        else:
            print("Unknown choice. Try again.")


if __name__ == "__main__":
    main()
