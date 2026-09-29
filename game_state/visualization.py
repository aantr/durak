"""Разметка кропов детекторов и вставка обратно в исходный BGR-кадр."""

import cv2
import numpy as np


def draw_label(image, text, x, y, color=(0, 255, 0)):
    """Подпись с подложкой, помещающаяся даже в небольшой OCR-кроп."""
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    scale *= min(1.0, max(1, width - 4) / max(1, tw))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x = max(0, min(int(x), width - tw - 4))
    y = max(th + 2, min(int(y), height - baseline - 2))
    cv2.rectangle(image, (x, y - th - 2), (x + tw + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 2, y), font, scale, color, 1, cv2.LINE_AA)


class DetectionVisualization:
    def __init__(self, image, *, layers=None):
        self.image = image
        self.crops = []
        # Общий кеш разметки детекторов; None — обычный однокадровый режим.
        self.layers = layers

    def add_crop(self, crop):
        """Координаты и fallback совпадают с detect_*.crop_by_size()."""
        height, width = self.image.shape[:2]
        x1, y1, x2, y2 = crop.get((width, height), (0, 0, width, height))
        x1, x2 = np.clip((x1, x2), 0, width)
        y1, y2 = np.clip((y1, y2), 0, height)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, width, height
        annotated = self.image[y1:y2, x1:x2].copy()
        self.crops.append(((x1, y1, x2, y2), annotated))
        return annotated

    def extract_layers(self, start=0):
        """Сохраняет только координаты/цвета разметки, без старого фона.

        Вызывается один раз после запуска детектора, не на пропущенных кадрах.
        """
        layers = []
        for (x1, y1, x2, y2), annotated in self.crops[start:]:
            original = self.image[y1:y2, x1:x2]
            # Кроп кнопки лежит внутри кропа руки. Переносим изменённые
            # пиксели, чтобы вставка одного кропа не стёрла чужую разметку.
            changed = np.any(annotated != original, axis=2)
            ys, xs = np.nonzero(changed)
            if ys.size:
                layers.append((ys + y1, xs + x1, annotated[ys, xs]))
        return layers

    def render(self):
        result = self.image.copy()
        groups = self.layers.values() if self.layers is not None else (self.extract_layers(),)
        for layers in groups:
            for ys, xs, colors in layers:
                result[ys, xs] = colors
        return result
