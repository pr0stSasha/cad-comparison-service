"""Клиент к локальной LLM через OpenAI-совместимый API (Ollama / vLLM / LM Studio)."""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings


class LLMError(RuntimeError):
    """Ошибка обращения к локальной модели."""


class LLMClient:
    """Тонкий асинхронный клиент /chat/completions."""

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers: Dict[str, str] = {}
        if settings.llm_api_key and settings.llm_api_key.lower() not in {"none", ""}:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

        url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Не удалось обратиться к LLM ({url}): {exc}") from exc

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Неожиданный ответ LLM: {data}") from exc
        if not content:
            raise LLMError("LLM вернула пустой ответ")
        return content


def encode_image_png(image_bytes: bytes) -> str:
    """Кодирует PNG в data-URL для content-части 'image_url'."""
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
