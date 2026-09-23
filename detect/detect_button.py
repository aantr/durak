"""
Обрезка изображения по CROP_BUTTON и распознавание текста
с помощью PaddleOCR (GPU).
"""

import cv2
from paddleocr import PaddleOCR
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants.constants import CROP_BUTTON

# --- путь к изображению ---
SOURCE = "screenshots/IMG_1416.PNG"


def crop_by_size(img, crop):
    """Обрезает изображение по crop, если для его размера есть правило."""
    h, w = img.shape[:2]
    key = (w, h)
    if key not in crop:
        return img
    x1, y1, x2, y2 = crop[key]
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def main():
    # --- 1. Загружаем и обрезаем изображение ---
    img = cv2.imread(SOURCE)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть: {SOURCE}")

    img = crop_by_size(img, CROP_BUTTON)

    # --- 2. Инициализируем PaddleOCR на GPU (PaddleOCR 3.x) ---
    ocr = PaddleOCR(
        use_textline_orientation=True,  # определять поворот текста
        lang="en",                      # язык: 'en', 'ru', 'ch' и т.д.
        device="gpu",                   # GPU (например, "gpu:0")
    )

    # --- 3. Распознаём текст ---
    result = ocr.predict(img)

    # --- 4. Печатаем результат ---
    print("=== Результат распознавания ===")
    if not result:
        print("Текст не найден.")
        

    # Собираем все строки из всех страниц результата
    lines = []  # список кортежей (box, text, conf)
    for res in result:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        polys = res.get("rec_polys", res.get("dt_polys", []))
        for text, conf, box in zip(texts, scores, polys):
            lines.append((box, text, conf))

    if not lines:
        print("Текст не найден.")


    for box, text, conf in lines:
        print(f"{text!r}  conf={conf:.3f}  box={box}")

    # --- 5. Опционально: рисуем рамки и текст на картинке ---
    for box, text, conf in lines:
        pts = [(int(x), int(y)) for x, y in box]
        for i in range(4):
            cv2.line(img, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{text} {conf:.2f}",
            (pts[0][0], pts[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("PaddleOCR", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return lines


if __name__ == "__main__":
    main()