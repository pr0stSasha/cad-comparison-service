"""Конфигурация приложения через переменные окружения / .env-файл."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_dotenv(path: str = ".env") -> None:
    """Мини-загрузчик .env, чтобы не тянуть лишнюю зависимость."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # OpenAI-совместимый API локальной LLM (Ollama / vLLM / LM Studio)
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5vl:27b")
    llm_api_key: str = os.getenv("LLM_API_KEY", "none")
    llm_supports_vision: bool = field(
        default_factory=lambda: _bool("LLM_SUPPORTS_VISION", True)
    )
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # Обработка изображения
    max_image_px: int = int(os.getenv("MAX_IMAGE_PX", "2048"))


settings = Settings()
