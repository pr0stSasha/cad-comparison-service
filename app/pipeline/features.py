"""Агрегированные метрики векторизованного чертежа (для промпта в LLM)."""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from app.pipeline.projections import View
from app.pipeline.vectorize import Entity


def compute_features(
    binary: np.ndarray,
    entities: List[Entity],
    segments: List[Entity],
    views: List[View],
) -> Dict:
    h, w = binary.shape[:2]

    kind_counts: Dict[str, int] = {}
    for e in entities:
        kind_counts[e.kind] = kind_counts.get(e.kind, 0) + 1

    # Проверка симметрии: сравниваем левую и зеркальную правую половину
    left = binary[:, : w // 2]
    right = cv2.flip(binary[:, w - w // 2 :], 1)
    min_w = min(left.shape[1], right.shape[1])
    diff = cv2.bitwise_xor(left[:, :min_w], right[:, :min_w])
    symmetry = round(1.0 - diff.mean() / 255.0, 3)

    # Мелкие компоненты — вероятные размерные числа/надписи
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    small_components = int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < 100))

    return {
        "image_size": {"width": int(w), "height": int(h)},
        "entity_counts": kind_counts,
        "long_straight_segments": len(segments),
        "detected_views": [
            {
                "name": v.name,
                "bbox": list(v.bbox),
                "aspect_ratio": round(v.bbox[2] / max(v.bbox[3], 1), 2),
                "entity_count": v.entity_count,
            }
            for v in views
        ],
        "symmetry_left_right": symmetry,
        "small_components_probably_text": small_components,
    }


def format_features(features: Dict) -> str:
    """Человекочитаемое резюме метрик для подстановки в промпт."""
    lines: List[str] = []
    size = features["image_size"]
    lines.append(f"Размер изображения: {size['width']}x{size['height']} px")

    counts = features["entity_counts"]
    counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "нет"
    lines.append(f"Извлечённые примитивы — {counts_str}")
    lines.append(f"Длинных прямых отрезков (Hough): {features['long_straight_segments']}")
    lines.append(f"Симметрия лево/право: {features['symmetry_left_right']:.1%}")
    lines.append(
        f"Мелких компонент (вероятно, размерные числа/надписи): "
        f"{features['small_components_probably_text']}"
    )

    lines.append(f"Обнаружено областей (видов/проекций): {len(features['detected_views'])}")
    for v in features["detected_views"]:
        lines.append(
            f"  - {v['name']}: bbox={v['bbox']}, "
            f"соотношение сторон={v['aspect_ratio']}, примитивов={v['entity_count']}"
        )
    return "\n".join(lines)
