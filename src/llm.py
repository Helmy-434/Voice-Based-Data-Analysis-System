import os
from typing import Generator
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        _client = Groq(api_key=api_key)
    return _client


def _prepare_messages(system: str, messages: list[dict]) -> list[dict]:
    formatted = []
    if system:
        formatted.append({"role": "system", "content": system})
    for m in messages:
        formatted.append({"role": m["role"], "content": m["content"]})
    return formatted


def complete(system: str, messages: list[dict], max_tokens: int = 1000) -> str:
    client = get_client()
    formatted_messages = _prepare_messages(system, messages)
    response = client.chat.completions.create(
        model=MODEL,
        messages=formatted_messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def stream(system: str, messages: list[dict]) -> Generator[str, None, None]:
    client = get_client()
    formatted_messages = _prepare_messages(system, messages)
    response = client.chat.completions.create(
        model=MODEL,
        messages=formatted_messages,
        stream=True,
        temperature=0.2,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

