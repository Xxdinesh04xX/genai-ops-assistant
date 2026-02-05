from typing import Any, Dict

from llm.client import LLMClient


def generate_text(instruction: str) -> Dict[str, Any]:
    client = LLMClient()
    system_prompt = (
        "You are a helpful assistant that writes clear, concise content. "
        "Follow the user's instruction and keep the response direct."
    )
    text = client.chat_text(system_prompt, instruction)
    return {"text": text.strip()}
