"""
This is a setup file to initialize the LLM for the examples in the basics folder.
"""

# --- TO USE OPENAI INFERENCE ---
import os
from langchain_openai import ChatOpenAI

def instantiate_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
