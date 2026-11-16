"""Сегментация чертежа на проекции/виды (связные области)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from app.pipeline.vectorize import Entity


@dataclass
class View:
    """Одна область чертежа — потенциальная проекция/разрез."""

    name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    entity_count: int = 0


def segment_views(
    binary: np.ndarray, entities: List[Entity], min_area: int = 2000
) -> List[View]:
    """Разбивает чертёж на связные области (виды/проекции), отсортированные по площади.

    Области ищутся на дилатированной маске, чтобы близкие линии одного вида
    слились в одну компоненту.
    """
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(binary, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)

    candidates: List[Tuple[int, int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        candidates.append((area, x, y, w, h))
    candidates.sort(reverse=True)

    views: List[View] = []
    for idx, (_, x, y, w, h) in enumerate(candidates, start=1):
        view = View(name=f"Область {idx}", bbox=(int(x), int(y), int(w), int(h)))
        x0, y0, w0, h0 = view.bbox
        view.entity_count = sum(
            1
            for e in entities
            if e.points
            and x0 <= e.points[0][0] < x0 + w0
            and y0 <= e.points[0][1] < y0 + h0
        )
        views.append(view)
    return views
