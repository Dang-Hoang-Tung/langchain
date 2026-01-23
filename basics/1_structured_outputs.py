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
    ### ----- Using basic output parsers ----- ###
    from langchain_core.output_parsers import StrOutputParser
    from langchain_classic.output_parsers.boolean import BooleanOutputParser
    from langchain_classic.output_parsers.datetime import DatetimeOutputParser

    parser = StrOutputParser()
    output = parser.invoke(
        llm.invoke("hello")
    )
    print(output)

    parser = DatetimeOutputParser()
    output = parser.invoke(
        llm.invoke(
            "Output a random datetime in %Y-%m-%dT%H:%M:%S.%fZ. "
            "Don't say anything else"
        )
    )
    print(output)


    parser = BooleanOutputParser()
    output = parser.invoke(
        input=llm.invoke(
            "Are you an AI? YES or NO only"
        )
    )
    print(output)


if True:
    ### ----- Using TypedDict structured output ----- ###
    from typing_extensions import Annotated, TypedDict

    class UserInfo(TypedDict):
        """User's info."""
        name: Annotated[str, "", "User's name. Defaults to ''"]
        country: Annotated[str, "", "Where the user lives. Defaults to ''"]

    llm_with_structure = llm.with_structured_output(UserInfo)

    output = llm_with_structure.invoke(
        "My name is Henrique, and I am from Brazil"
    )
    print(output)


if False:
    ### ----- Using Pydantic structured output (depends on model support) ----- ###
    from pydantic import BaseModel, Field
    from typing_extensions import Annotated

    class PydanticUserInfo(BaseModel):
        """User's info."""
        name: Annotated[str, Field(description="User's name. Defaults to ''", default=None)]
        country: Annotated[str, Field(description="Where the user lives. Defaults to ''", default=None, )]

    llm_with_structure = llm.with_structured_output(PydanticUserInfo)

    output = llm_with_structure.invoke(
        "Hello, my name is the same as the capital of the U.S.  "
        "But I'm from a country where we usually associate with kangaroos"
    )
    print(output)


if False:
    ### ----- Dealing with Errors in Structured Output Parsing ----- ###

    from typing_extensions import Annotated, TypedDict
    from typing import List
    from pydantic import BaseModel, Field
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.exceptions import OutputParserException
    from langchain_classic.output_parsers import OutputFixingParser

    class Performer(BaseModel):
        """Filmography info about an actor/actress"""
        name: Annotated[str, Field(description="name of an actor/actress")]
        film_names: Annotated[List[str], Field(description="list of names of films they starred in")]

    llm_with_structure = llm.with_structured_output(Performer)
    
    response = llm_with_structure.invoke(
        "Generate the filmography for Scarlett Johansson. Top 5 only"
    )
    print(response)

    parser = PydanticOutputParser(pydantic_object=Performer)
    parser.parse(response.json())

    misformatted_result = "{'name': 'Scarlett Johansson', 'film_names': ['The Avengers']}"

    try:
        parser.parse(misformatted_result)
    except OutputParserException as e:
        print(e)
    
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    output = new_parser.parse(misformatted_result)

    print(output)
