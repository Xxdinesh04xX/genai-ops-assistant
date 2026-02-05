import json
import os
import re
from typing import Any, Dict

from openai import OpenAI


def _extract_json(text: str) -> Dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _default_base_url() -> str:
    return "https://api.groq.com/openai/v1"


def _default_model(base_url: str | None) -> str:
    if base_url and "api.groq.com" in base_url:
        return "llama-3.1-8b-instant"
    return "gpt-4o-mini"


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or _default_base_url()
        )
        model_env = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")
        self.model = model or model_env or _default_model(base_url)
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(max_retries + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                extracted = _extract_json(content)
                if extracted is not None:
                    return extracted
        raise ValueError("LLM did not return valid JSON") from last_error

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 1,
    ) -> str:
        last_error: Exception | None = None
        for _ in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise ValueError("LLM did not return text") from last_error
