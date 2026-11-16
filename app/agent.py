"""Агент: растровый чертёж -> векторизация -> сравнение с описанием -> словесный отчёт."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.config import settings
from app.llm.client import LLMClient, encode_image_png
from app.pipeline.features import compute_features, format_features
from app.pipeline.preprocess import load_image, to_binary
from app.pipeline.projections import segment_views
from app.pipeline.vectorize import detect_straight_segments, extract_entities, to_svg

SYSTEM_PROMPT = (
    "Ты — опытный инженер-конструктор и эксперт по машиностроительным чертежам. "
    "Тебе дают: (1) изображение чертежа детали с проекциями/разрезами, "
    "(2) автоматически извлечённое векторное резюме этого изображения, "
    "(3) текстовое описание детали. "
    "Твоя задача — сравнить изображение с описанием и перечислить конкретные расхождения: "
    "чего не хватает, что лишнее, где несовпадение геометрии, размеров или проекций. "
    "Отвечай по-русски, конкретно и технически грамотно."
)

USER_TEMPLATE = """## Текстовое описание детали
{description}

## Автоматически извлечённое векторное резюме чертежа
{vector_summary}

## Задача
Сравни чертёж на изображении с текстовым описанием (и с векторным резюме).
Найди расхождения. Ответь СТРОГО валидным JSON без markdown-ограждений по схеме:
{{
  "summary": "краткое резюме сравнения (1-3 предложения)",
  "differences": [
    {{
      "type": "missing | extra | wrong_dimension | geometry | other",
      "what": "что именно расходится",
      "where": "где на чертеже (какая проекция/область)",
      "severity": "high | medium | low"
    }}
  ],
  "report_md": "подробный человеческий отчёт в Markdown"
}}"""


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Пытается вытащить JSON-объект из ответа модели (устойчиво к лишнему тексту)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class DrawingAgent:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()

    async def analyze(self, image_bytes: bytes, description: str) -> Dict[str, Any]:
        # 1. Предобработка растра
        img = load_image(image_bytes)
        binary = to_binary(img, max_px=settings.max_image_px)

        # 2. Векторизация
        entities = extract_entities(binary)
        segments = detect_straight_segments(binary)

        # 3. 2D-разбиение на проекции/виды
        views = segment_views(binary, entities)

        # 4. Метрики + текстовое резюме для модели
        features = compute_features(binary, entities, segments, views)
        vector_summary = format_features(features)
        svg = to_svg(binary.shape, entities)

        # 5. Запрос к локальной LLM
        user_text = USER_TEMPLATE.format(description=description, vector_summary=vector_summary)
        if settings.llm_supports_vision:
            content: list = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": encode_image_png(image_bytes)}},
            ]
        else:
            # Модель без зрения: анализируем только по векторному резюме
            content = [{"type": "text", "text": user_text}]

        raw_report = await self.llm.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=3000,
        )

        return {
            "model": settings.llm_model,
            "vector_summary": vector_summary,
            "features": features,
            "views": [v.__dict__ for v in views],
            "entities_count": len(entities),
            "svg": svg,
            "report": parse_json_block(raw_report) or {"report_md": raw_report},
            "raw_model_output": raw_report,
        }
