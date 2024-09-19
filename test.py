import os
import asyncio
import logging
logging.basicConfig(level=logging.DEBUG)
from gpt_researcher.llm_provider.generic import GenericLLMProvider
from gpt_researcher.utils.llm import get_llm

# Set up environment variables. Change them as per your environment. Currently this code assumes you are running it from a docker container
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_BASE_URL"] = "http://host.docker.internal:11434"
os.environ["FAST_LLM_MODEL"] = "llama3.1:latest"

# Create the GenericLLMProvider instance
llm_provider = get_llm(
    "ollama",
    base_url=os.environ["OLLAMA_BASE_URL"],
    model=os.environ["FAST_LLM_MODEL"],
    temperature=0.7,
    max_tokens=2000,
    verify_ssl=False  # Add this line
)

# Test the connection with a simple query
messages = [{"role": "user", "content": "sup?"}]

async def test_ollama():
    try:
        response = await llm_provider.get_chat_response(messages, stream=False)
        print("Ollama response:", response)
    except Exception as e:
        print(f"Error: {e}")

# Run the async function
asyncio.run(test_ollama())