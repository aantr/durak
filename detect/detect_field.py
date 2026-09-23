from dataclasses import dataclass
import cv2
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants.constants import (
    CROP_FIELD,
    DETECTION_FIELD_ENGINE_PATH, IMAGE_SIZE_FIELD, ALLOW_CARD, DISCARD_CARD,
    AREA_SHIFT, CLASSIFY_SUIT_WEIGHTS_PATH, IMAGE_SIZE_SUIT,
)

source = "screenshots/IMG_1417.PNG"

# Показывать область, которая подаётся в классификатор масти
SHOW_CLS_AREA = True

# Показывать сам вырезанный кроп в отдельном окне
SHOW_CLS_CROP_WINDOW = False

SCALE = 1


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


@dataclass(frozen=True)
class FieldBox:
    bbox: tuple[int, int, int, int]
    class_id: int
    confidence: float
    # Индекс нижней карты в результате, а не её класс: ранги могут повторяться.
    covers: int | None = None


@dataclass(frozen=True)
class FieldCard:
    box: FieldBox
    rank: str
    suit: str | None
    suit_confidence: float


@dataclass
class FieldDetection:
    cards: list[FieldCard]
    annotated_crop: np.ndarray | None = None

    def legacy_avail(self):
        """Сохраняет прежний формат результата main(): bbox, cls, conf, tpe."""
        return [
            (*card.box.bbox, card.box.class_id, card.box.confidence,
             "simple" if card.box.covers is None else self.cards[card.box.covers].box.class_id)
            for card in self.cards
        ]


def field_regions(bbox, image_size):
    """Зоны относительно центра бокса в координатах кропа поля."""
    x1, y1, x2, y2 = bbox
    shift_x, shift_y = AREA_SHIFT[image_size]
    origin = ((x1 + x2) // 2 + shift_x, (y1 + y2) // 2 + shift_y)
    allow_w, allow_h = ALLOW_CARD[image_size]
    discard_w, discard_h = DISCARD_CARD[image_size]
    return (
        (origin, (origin[0] + allow_w, origin[1] + allow_h)),
        (origin, (origin[0] + discard_w, origin[1] + discard_h)),
    )


def draw_field_regions(image, boxes, image_size):
    """Рисует те же зоны, которые используются при фильтрации боксов."""
    height, width = image.shape[:2]
    for box in boxes:
        allow, discard = field_regions(box.bbox, image_size)
        for label, region, color in (
            ("DISCARD_CARD", discard, (0, 165, 255)),
            ("ALLOW_CARD", allow, (255, 255, 0)),
        ):
            start, end = region
            cv2.rectangle(image, start, end, color, 1)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            x = max(0, min(start[0] + 4, width - text_w - 2))
            y = max(text_h + 2, min(end[1] - 5, height - baseline - 2))
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def postprocess_field_boxes(boxes, image_size):
    """Исходная сортировка и фильтрация ALLOW_CARD/DISCARD_CARD.

    boxes: (x1, y1, x2, y2, class_id, confidence) в координатах кропа.
    image_size: (width, height) полного кадра для выбора правил геометрии.
    covers ссылается на индекс в возвращённом списке.
    """
    avail = []
    boxes = sorted(boxes, key=lambda x: x[:4], reverse=True)
    if not boxes:
        return avail
    if any(image_size not in rules for rules in (AREA_SHIFT, ALLOW_CARD, DISCARD_CARD)):
        raise ValueError(f"Нет правил геометрии поля для размера {image_size}")
    process = []
    while boxes:
        b = boxes.pop(-1)
        process.append((None, b))

        while process:
            t, (x1, y1, x2, y2, cls, conf) = process.pop(-1)
            parent_index = len(avail)
            avail.append(FieldBox((x1, y1, x2, y2), cls, conf, t))
            box_allow, box_discard = field_regions((x1, y1, x2, y2), image_size)

            for b in list(boxes):
                x1_, y1_, x2_, y2_, cls_, conf_ = b
                c = ((x1_ + x2_) // 2, (y1_ + y2_) // 2)
                if intersect(box_allow, c):
                    process.append((parent_index, b))
                    boxes.remove(b)

            for b in list(boxes):
                x1_, y1_, x2_, y2_, cls_, conf_ = b
                c = ((x1_ + x2_) // 2, (y1_ + y2_) // 2)
                if intersect(box_discard, c):
                    boxes.remove(b)
    return avail


def detect_field(img, model, suit_model, *, draw=False,
                 show_cls_area=SHOW_CLS_AREA, show_cls_crop_window=False):
    """Общий проход для демо и бота: детекция, фильтрация, масти и разметка.

    Принимает полный BGR-кадр и уже загруженные модели.
    annotated_crop содержит только размеченную область CROP_FIELD.
    """
    height, width = img.shape[:2]
    img = crop_by_size(img, CROP_FIELD)
    r = model.predict(
        source=(img,), imgsz=IMAGE_SIZE_FIELD, conf=0.5, iou=0.45,
        save=False, show=False, verbose=False,
    )[0]
    boxes = [
        (*map(int, b.xyxy[0].cpu().numpy()), int(b.cls[0]), float(b.conf[0]))
        for b in r.boxes
    ]
    avail = postprocess_field_boxes(boxes, (width, height))
    img_vis = img.copy() if draw else None
    if draw:
        # Зоны под рамками карт и подписями, чтобы не перекрывать распознавания.
        draw_field_regions(img_vis, avail, (width, height))
    h_img, w_img = img.shape[:2]
    cards = []
    for box in avail:
        x1, y1, x2, y2 = box.bbox
        cls, conf = box.class_id, box.confidence
        center_x = (x1 + x2) // 2
        cx1 = clamp(center_x - IMAGE_SIZE_SUIT // 2, 0, w_img)
        cy1 = clamp(y2, 0, h_img)
        cx2 = clamp(center_x + IMAGE_SIZE_SUIT // 2, 0, w_img)
        cy2 = clamp(y2 + IMAGE_SIZE_SUIT, 0, h_img)
        crop = img[cy1:cy2, cx1:cx2]
        suit_name, suit_conf = classify_suit(crop, suit_model)
        cards.append(FieldCard(box, str(r.names[cls]), suit_name, suit_conf))
        if not draw:
            continue

        color = (0, 255, 0) if box.covers is None else (0, 0, 255)

        # Основной бокс детектора
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

        name = r.names[cls]
        label = f"{name} {conf:.2f}"

        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        ty = max(y1, th + 5)
        cv2.rectangle(img_vis, (x1, ty - th - 5), (x1 + tw + 4, ty + base), color, -1)
        cv2.putText(img_vis, label, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

        if cx2 <= cx1 or cy2 <= cy1:
            continue

        # Рисуем именно эту область (жёлтая рамка)
        if show_cls_area:
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

        if show_cls_crop_window:
            cv2.imshow("cls crop", crop)

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


    return FieldDetection(cards, img_vis)


def main():
    from ultralytics import YOLO

    model = YOLO(str(DETECTION_FIELD_ENGINE_PATH))
    suit_model = YOLO(str(CLASSIFY_SUIT_WEIGHTS_PATH))
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть: {source}")
    result = detect_field(
        img, model, suit_model, draw=True,
        show_cls_area=SHOW_CLS_AREA, show_cls_crop_window=SHOW_CLS_CROP_WINDOW,
    )
    for card in result.cards:
        print(f"{card.rank} {card.box.confidence:.2f} {list(card.box.bbox)}")
    img_vis = result.annotated_crop
    h, w = img_vis.shape[:2]
    small = cv2.resize(img_vis, (int(w * SCALE), int(h * SCALE)),
                    interpolation=cv2.INTER_AREA)

    cv2.imshow("YOLO", small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result.legacy_avail()

if __name__ == "__main__":
    main()
