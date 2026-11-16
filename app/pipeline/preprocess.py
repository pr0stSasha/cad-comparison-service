"""Предобработка растра: декодирование, ресайз, бинаризация."""
from __future__ import annotations

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    pass


def load_image(data: bytes) -> np.ndarray:
    """Декодирует PNG (или другой растр) в BGR-массив OpenCV."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError(
            "Не удалось декодировать изображение — ожидается PNG или другой растровый формат."
        )
    return img


def to_binary(img: np.ndarray, max_px: int = 2048) -> np.ndarray:
    """Приводит изображение к бинарной маске: линии чертежа = 255, фон = 0."""
    h, w = img.shape[:2]
    scale = max(h, w) / max_px
    if scale > 1:
        img = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.medianBlur(gray, 3)

    # Otsu + инверсия: на чертеже тёмные линии на светлом фоне
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Склеиваем разорванные штрихи
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return binary
