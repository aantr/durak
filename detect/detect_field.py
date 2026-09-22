from pathlib import Path
import cv2
from ultralytics import YOLO

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants.constants import (
    DETECTION_ENGINE_PATH, CROP_CARDS, IMAGE_SIZE, CROP_FIELD,
    DETECTION_FIELD_ENGINE_PATH, IMAGE_SIZE_FIELD, ALLOW_CARD, DISCARD_CARD,
    AREA_SHIFT, CLASSIFY_SUIT_WEIGHTS_PATH, IMAGE_SIZE_SUIT,
)

source = "screenshots/IMG_1417.PNG"

# Показывать область, которая подаётся в классификатор масти
SHOW_CLS_AREA = True

# Показывать сам вырезанный кроп в отдельном окне
SHOW_CLS_CROP_WINDOW = False


def crop_by_size(img, crop):
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


def intersect(box, point):
    (x1, y1), (x2, y2) = box
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# --- Детектор поля ---
model = YOLO(str(DETECTION_FIELD_ENGINE_PATH))

# --- Классификатор масти ---
suit_model = YOLO(str(CLASSIFY_SUIT_WEIGHTS_PATH))

img = cv2.imread(source)
height, width = img.shape[:2]

img = crop_by_size(img, CROP_FIELD)

results = model.predict(
    source=(img,),
    imgsz=IMAGE_SIZE_FIELD,
    conf=0.5,
    iou=0.45,
    save=False,
    show=False,
    line_width=2,
    show_labels=True,
    show_conf=True,
)

for r in results:
    for b in r.boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        name = r.names[cls]
        print(f"{name} {conf:.2f} {xyxy.tolist()}")

SCALE = 1

r = results[0]
img_vis = img.copy()

boxes = []
avail = []
for b in r.boxes:
    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
    cls = int(b.cls[0])
    conf = float(b.conf[0])
    boxes.append((x1, y1, x2, y2, cls, conf))

boxes = sorted(boxes, key=lambda x: x[:4], reverse=True)
process = []
while boxes:
    b = boxes.pop(-1)
    process.append(("simple", b))

    while process:
        t, (x1, y1, x2, y2, cls, conf) = process.pop(-1)
        avail.append((x1, y1, x2, y2, cls, conf, t))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        box_allow = [
            (cx + AREA_SHIFT[(width, height)][0], cy + AREA_SHIFT[(width, height)][1]),
            (cx + ALLOW_CARD[(width, height)][0] + AREA_SHIFT[(width, height)][0],
             cy + ALLOW_CARD[(width, height)][1] + AREA_SHIFT[(width, height)][1]),
        ]
        box_discard = [
            (cx + AREA_SHIFT[(width, height)][0], cy + AREA_SHIFT[(width, height)][1]),
            (cx + DISCARD_CARD[(width, height)][0] + AREA_SHIFT[(width, height)][0],
             cy + DISCARD_CARD[(width, height)][1] + AREA_SHIFT[(width, height)][1]),
        ]

        for b in list(boxes):
            x1_, y1_, x2_, y2_, cls_, conf_ = b
            c = ((x1_ + x2_) // 2, (y1_ + y2_) // 2)
            if intersect(box_allow, c):
                process.append((cls, b))
                boxes.remove(b)

        for b in list(boxes):
            x1_, y1_, x2_, y2_, cls_, conf_ = b
            c = ((x1_ + x2_) // 2, (y1_ + y2_) // 2)
            if intersect(box_discard, c):
                boxes.remove(b)


def classify_suit(crop_bgr):
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


# --- Отрисовка ---
h_img, w_img = img_vis.shape[:2]

for (x1, y1, x2, y2, cls, conf, tpe) in avail:
    color = (0, 255, 0) if tpe == "simple" else (0, 0, 255)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Основной бокс детектора
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

    name = r.names[cls]
    label = f"{name} {conf:.2f}"

    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    ty = max(y1, th + 5)
    cv2.rectangle(img_vis, (x1, ty - th - 5), (x1 + tw + 4, ty + base), color, -1)
    cv2.putText(img_vis, label, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    
    cx1 = clamp(center_x - IMAGE_SIZE_SUIT // 2, 0, w_img)
    cy1 = clamp(y2, 0, h_img)
    cx2 = clamp(center_x + IMAGE_SIZE_SUIT // 2, 0, w_img)
    cy2 = clamp(y2 + IMAGE_SIZE_SUIT, 0, h_img)

    if cx2 <= cx1 or cy2 <= cy1:
        continue

    crop = img[cy1:cy2, cx1:cx2]

    # Рисуем именно эту область (жёлтая рамка)
    if SHOW_CLS_AREA:
        cv2.rectangle(img_vis, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
        # уголки-«маркеры», чтобы было видно даже если совпадает с основным боксом
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

    suit_name, suit_conf = classify_suit(crop)
    if suit_name is None:
        suit_text = "suit: n/a"
    else:
        suit_text = f"{suit_name} {suit_conf:.2f}"

    (stw, sth), sbase = cv2.getTextSize(suit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    pad = 3
    sx1 = x1
    sy1 = y2 + 2
    sx2 = min(w_img, sx1 + stw + 2 * pad)
    sy2 = min(h_img, sy1 + sth + sbase + 2 * pad)

    cv2.rectangle(img_vis, (sx1, sy1), (sx2, sy2), (255, 200, 0), -1)
    cv2.putText(img_vis, suit_text, (sx1 + pad, sy1 + sth + pad - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


h, w = img_vis.shape[:2]
small = cv2.resize(img_vis, (int(w * SCALE), int(h * SCALE)),
                   interpolation=cv2.INTER_AREA)

cv2.imshow("YOLO", small)
cv2.waitKey(0)
cv2.destroyAllWindows()