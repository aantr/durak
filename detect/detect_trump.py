"""
Обрезка изображения по CROP_TRUMP и распознавание масти
с помощью обученной модели YOLO.
"""

import cv2
from ultralytics import YOLO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import constants
from constants.constants import CLASSIFY_SUIT_WEIGHTS_PATH, IMAGE_SIZE_SUIT

# --- путь к изображению ---
SOURCE = "screenshots/IMG_1413.PNG"


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

    crop = getattr(constants, "CROP_TRUMP", {})
    if (img.shape[1], img.shape[0]) in crop:
        img = crop_by_size(img, crop)
    else:
        window = "Select trump suit (Enter to confirm)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 600, 800)
        try:
            x, y, w, h = cv2.selectROI(window, img, fromCenter=False)
        finally:
            cv2.destroyWindow(window)
        if not w or not h:
            return None, 0.0
        img = img[y:y + h, x:x + w]

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    model = YOLO(str(CLASSIFY_SUIT_WEIGHTS_PATH))
    result = model.predict(source=img, imgsz=IMAGE_SIZE_SUIT, verbose=False)[0]
    suit, confidence = None, 0.0
    label = "Suit not recognized"
    if result.probs is not None:
        name = result.names[int(result.probs.top1)]
        suit = {"clubs": "Clubs", "spades": "Spades",
                "diamonds": "Diamonds", "hearts": "Hearts"}.get(name, name)
        confidence = float(result.probs.top1conf)
        label = f"{suit}: {confidence:.1%}"

    print(label)
    width = max(360, img.shape[1])
    preview = cv2.copyMakeBorder(
        img, 55, 0, 0, width - img.shape[1],
        cv2.BORDER_CONSTANT, value=(30, 30, 30),
    )
    cv2.putText(preview, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    try:
        cv2.imshow("Trump suit", preview)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()
    return suit, confidence


if __name__ == "__main__":
    main()
