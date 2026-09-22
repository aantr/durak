from pathlib import Path
import cv2
from ultralytics import YOLO

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants.constants import (
    DETECTION_ENGINE_PATH, CROP_CARDS, IMAGE_SIZE, CROP_FIELD,
    DETECTION_FIELD_ENGINE_PATH, IMAGE_SIZE_FIELD,
    CLASSIFY_SUIT_WEIGHTS_PATH, IMAGE_SIZE_SUIT, CROP_CARDS_ALLOWED
)

# Источник: путь к картинке, папке, видео или 0 для вебки
source = "screenshots/IMG_1417.PNG"

SHOW_CLS_AREA = True
SHOW_CLS_CROP_WINDOW = False

# Во сколько раз увеличить область, подаваемую в классификатор
SCALE_CLASSIFY_AREA = 1.2

SCALE = 1


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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def intersect(box, point):
    """Точка внутри бокса (включая границы)."""
    (x1, y1), (x2, y2) = box
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_suit(crop_bgr, suit_model):
    """Классифицирует масть по вырезанному изображению карты."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    res = suit_model.predict(
        source=(crop_bgr,),
        imgsz=IMAGE_SIZE_SUIT,
        conf=0.25,
        save=False,
        show=False,
        verbose=False,
    )[0]
    probs = res.probs
    if probs is None:
        return None, 0.0
    cls_idx = int(probs.top1)
    conf = float(probs.top1conf)
    name = res.names[cls_idx]
    return name, conf

def main():
    # Загружаем engine
    model = YOLO(str(DETECTION_ENGINE_PATH))
    # model = YOLO(str(DETECTION_FIELD_ENGINE_PATH))

    suit_model = YOLO(str(CLASSIFY_SUIT_WEIGHTS_PATH))


    img = cv2.imread(source)
    # img = cv2.resize(img, (1280, 1280))

    height, width = img.shape[:2]

    # Обрезаем ДО подачи в модель
    img = crop_by_size(img, CROP_CARDS)
    # img = crop_by_size(img, CROP_FIELD)

    results = model.predict(
        source=img,
        imgsz=IMAGE_SIZE,
        # imgsz=IMAGE_SIZE_FIELD,
        conf=0.5,
        iou=0.45,
        save=False,
        show=False,
        line_width=2,
        show_labels=True,
        show_conf=True,
    )

    # Если нужно обработать результаты вручную
    for r in results:
        boxes = r.boxes
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            name = r.names[cls]
            print(f"{name} {conf:.2f} {xyxy.tolist()}")



    # Размер области классификации с учётом масштаба
    cls_w = int(IMAGE_SIZE_SUIT * SCALE_CLASSIFY_AREA)
    cls_h = int(IMAGE_SIZE_SUIT * SCALE_CLASSIFY_AREA)

    r = results[0]

    img_vis = img.copy()
    h_img, w_img = img_vis.shape[:2]

    # --- Переводим CROP_CARDS_ALLOWED в координаты обрезанного img ---
    allowed_box_full = None
    if (width, height) in CROP_CARDS_ALLOWED:
        ax1, ay1, ax2, ay2 = CROP_CARDS_ALLOWED[(width, height)]

        # Смещение кропа CROP_CARDS в координатах исходного изображения
        if (width, height) in CROP_CARDS:
            ox1, oy1, _, _ = CROP_CARDS[(width, height)]
        else:
            ox1, oy1 = 0, 0

        # Координаты ALLOWED в системе обрезанного img
        ax1 -= ox1
        ay1 -= oy1
        ax2 -= ox1
        ay2 -= oy1

        # Клипуем по границам обрезанного изображения
        ax1 = clamp(ax1, 0, w_img)
        ay1 = clamp(ay1, 0, h_img)
        ax2 = clamp(ax2, 0, w_img)
        ay2 = clamp(ay2, 0, h_img)

        if ax2 > ax1 and ay2 > ay1:
            allowed_box = ((ax1, ay1), (ax2, ay2))
            allowed_box_full = (ax1, ay1, ax2, ay2)
        else:
            allowed_box = None
    else:
        allowed_box = None


    # --- Ручная отрисовка боксов детектора (только попавших в ALLOWED) ---
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        name = r.names[cls]

        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Фильтр по ALLOWED
        if allowed_box is not None and not intersect(allowed_box, center):
            continue

        color = (0, 255, 0)

        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        ty = max(y1, th + 5)

        cv2.rectangle(img_vis, (x1, ty - th - 5), (x1 + tw + 4, ty + base), color, -1)
        cv2.putText(
            img_vis,
            label,
            (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # --- Дорисовываем распознавание масти (только для боксов в ALLOWED) ---
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
        center_x = (x1 + x2) // 2
        center = (center_x, (y1 + y2) // 2)

        if allowed_box is not None and not intersect(allowed_box, center):
            continue

        cx1 = clamp(center_x - cls_w // 2, 0, w_img)
        cy1 = clamp(y2, 0, h_img)
        cx2 = clamp(center_x + cls_w // 2, 0, w_img)
        cy2 = clamp(y2 + cls_h, 0, h_img)

        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = img[cy1:cy2, cx1:cx2]

        if SHOW_CLS_AREA:
            cv2.rectangle(img_vis, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
            L = 8
            for (px, py, dx, dy) in [
                (cx1, cy1, 1, 1),
                (cx2, cy1, -1, 1),
                (cx1, cy2, 1, -1),
                (cx2, cy2, -1, -1),
            ]:
                cv2.line(img_vis, (px, py), (px + dx * L, py), (0, 255, 255), 2)
                cv2.line(img_vis, (px, py), (px, py + dy * L), (0, 255, 255), 2)

        if SHOW_CLS_CROP_WINDOW:
            cv2.imshow("cls crop", crop)

        suit_name, suit_conf = classify_suit(crop, suit_model)
        suit_text = "suit: n/a" if suit_name is None else f"{suit_name} {suit_conf:.2f}"

        (stw, sth), sbase = cv2.getTextSize(suit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        pad = 3
        sx1 = x1
        sy1 = y2 + 2
        sx2 = min(w_img, sx1 + stw + 2 * pad)
        sy2 = min(h_img, sy1 + sth + sbase + 2 * pad)

        cv2.rectangle(img_vis, (sx1, sy1), (sx2, sy2), (255, 200, 0), -1)
        cv2.putText(img_vis, suit_text, (sx1 + pad, sy1 + sth + pad - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # (опционально) Показать границу ALLOWED для отладки
    if allowed_box_full is not None:
        ax1, ay1, ax2, ay2 = allowed_box_full
        cv2.rectangle(img_vis, (ax1, ay1), (ax2, ay2), (255, 0, 255), 1)

    small = cv2.resize(
        img_vis, (int(w_img * SCALE), int(h_img * SCALE)), interpolation=cv2.INTER_AREA
    )

    cv2.imshow("YOLO", small)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()