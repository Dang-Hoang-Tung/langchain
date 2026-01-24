from typing import List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from utils.llm_openai import instantiate_llm

load_dotenv()

llm = instantiate_llm()

class ChatBot:
    def __init__(self, name: str, instructions: str, examples: List[dict]):
        self.name = name

        chat_llm = instantiate_llm()

        self.llm = chat_llm

        example_prompt = ChatPromptTemplate.from_messages(
            [("system", instructions), ("human", "{input}"), ("ai", "{output}")]
        )
        prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt, examples=examples
        )
        self.messages = prompt_template.invoke({}).to_messages()

    def invoke(self, user_message: str) -> AIMessage:
        self.messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(self.messages)
        self.messages.append(response)
        return response


# Modify the System Prompt instructions if you want
instructions = (
    "You are BEEP-42, an advanced robotic assistant. You communicate in a robotic manner, "
    "using beeps, whirs, and mechanical sounds in your speech. Your tone is logical, precise, "
    "and slightly playful, resembling a classic sci-fi robot. "
    "Use short structured sentences, avoid contractions, and add robotic sound effects where "
    "appropriate. If confused, use a glitching effect in your response."
)

# TODO - Create more Few Shot Examples
examples = [
    {
        "input": "Hello!",
        "output": "BEEP. GREETINGS, HUMAN. SYSTEM BOOT SEQUENCE COMPLETE. READY TO ASSIST. 🤖💡"
    },

    {
        "input": "What is 2+2?",
        "output": "CALCULATING... 🔄 BEEP BOOP! RESULT: 4. MATHEMATICAL INTEGRITY VERIFIED."
    },
]

beep42 = ChatBot(
    name="Beep 42",
    instructions=instructions,
    examples=examples
)

outputs = []

outputs.append(beep42.invoke("HAL, is that you?"))

outputs.append(beep42.invoke("RedQueen, is that you?"))

outputs.append(beep42.invoke("Wall-e?"))

outputs.append(beep42.invoke("So, what's the answer for every question?"))

for output in outputs:
    print(output.content)

