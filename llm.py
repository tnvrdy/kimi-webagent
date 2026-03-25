"""
Thin wrapper around the OpenAI chat completion api for action generation

Swap model/base_url/api_key to point at e.g. Kimi, or any other OpenAI-compatible endpoint

Later will try two-role split using oai for action gen and Kimi for reasoning about trajectories
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


def chat(
    messages: list[dict],
    model: str = "gpt-5.4-mini", # should be good rn, altho 3.5/7 sonnet are best at web bench :>
    temperature: float = 0.0, # 0 for deterministic action output; dont want same observation to produce diff actions
    max_completion_tokens: int = 256,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Send a list of chat messages and return the model's reply as a plain string.

    Args:
        messages:    OpenAI-format message list, e.g. [{"role": "system", "content": "..."},
                     {"role": "user", "content": "..."}]
        model:       Model name. Override w/ MODEL env var or pass directly
        temperature: 0.0 for deterministic action output (will we want to raise this for variety?)
        max_completion_tokens:  Budget for the reply. 256 is enough for one action line
        base_url:    Override API endpoint (e.g. Kimi: "https://api.moonshot.cn/v1")
                     Falls back to OPENAI_BASE_URL env var, then the default OpenAI endpoint
        api_key:     Falls back to OPENAI_API_KEY env var
    """
    client = OpenAI(
        api_key=api_key or os.environ["OPENAI_API_KEY"],
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
    )

    response = client.chat.completions.create(
        model=model or os.getenv("MODEL", "gpt-5.4-mini"),
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )

    return response.choices[0].message.content or ""

# local test llm call, requires api key set in env
if __name__ == "__main__":
    reply = chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say: hi tanvi"},
        ],
        max_completion_tokens=16,
    )
    print(repr(reply))
