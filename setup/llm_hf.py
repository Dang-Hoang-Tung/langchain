"""
This is a setup file to initialize the LLM for the examples in the basics folder.
"""

# --- TO USE HUGGINGFACE INFERENCE ---
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

def instantiate_llm():
    model = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        # repo_id="deepseek-ai/DeepSeek-R1",
        # repo_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        provider="auto",  # let Hugging Face choose the best provider for you
    )
    return ChatHuggingFace(llm=model)
