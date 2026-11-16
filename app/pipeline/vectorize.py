"""Векторизация растрового чертежа: извлечение примитивов и генерация SVG."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


@dataclass
class Entity:
    """Векторный примитив, извлечённый из растра."""

    kind: str  # line | polyline | circle
    points: List[Point]
    extra: Dict = field(default_factory=dict)

    def describe(self) -> str:
        if self.kind == "line" and len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            return f"отрезок ({x1},{y1})-({x2},{y2}), длина={length:.0f}px, угол={angle:.0f}°"
        if self.kind == "circle":
            (cx, cy), r = self.points[0], self.extra.get("radius", 0)
            return f"окружность центр=({cx},{cy}), радиус={r:.0f}px"
        return f"{self.kind} с {len(self.points)} вершинами"


def extract_entities(binary: np.ndarray, min_area: int = 30) -> List[Entity]:
    """Извлекает контуры и классифицирует их как линии/полилинии/окружности.

    `binary` — маска, где линии чертежа белые (255) на чёрном фоне.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    entities: List[Entity] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        circularity = 4 * math.pi * area / (peri * peri) if peri > 0 else 0.0
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True).reshape(-1, 2)

        if circularity > 0.85 and len(approx) > 6:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            entities.append(Entity("circle", [(int(cx), int(cy))], {"radius": radius}))
            continue

        pts: List[Point] = [(int(x), int(y)) for x, y in approx]
        if len(pts) == 2:
            entities.append(Entity("line", pts))
        else:
            entities.append(Entity("polyline", pts))
    return entities


def detect_straight_segments(
    binary: np.ndarray, min_length: int = 40, max_gap: int = 6
) -> List[Entity]:
    """Ищет длинные прямые отрезки (размерные/осевые/контурные линии) через HoughLinesP."""
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    segments: List[Entity] = []
    if lines is None:
        return segments
    # OpenCV может вернуть форму (N,1,4) или (N,4) — нормализуем
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        segments.append(Entity("line", [(int(x1), int(y1)), (int(x2), int(y2))]))
    return segments


def to_svg(shape: Tuple[int, int], entities: List[Entity]) -> str:
    """Генерирует SVG-представление векторизованного чертежа (для предпросмотра в UI)."""
    h, w = shape[:2]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0b0f14"/>',
    ]
    for e in entities:
        if e.kind == "circle":
            (cx, cy), r = e.points[0], e.extra.get("radius", 1)
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" '
                'stroke="#4ade80" stroke-width="1.5"/>'
            )
        else:
            d = "M " + " L ".join(f"{x} {y}" for x, y in e.points)
            if e.kind == "polyline":
                d += " Z"
            color = "#38bdf8" if e.kind == "line" else "#f59e0b"
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "\n".join(parts)
