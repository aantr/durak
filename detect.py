from pathlib import Path
import cv2
from ultralytics import YOLO

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import DETECTION_ENGINE_PATH, CROP_CARDS, IMAGE_SIZE

# Источник: путь к картинке, папке, видео или 0 для вебки
source = "screenshots/IMG_1408.PNG"

def crop_by_size(img):
    """Обрезает изображение по CROP_CARDS, если для его размера есть правило."""
    h, w = img.shape[:2]
    key = (w, h)
    if key not in CROP_CARDS:
        return img
    x1, y1, x2, y2 = CROP_CARDS[key]
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


# Загружаем engine
model = YOLO(str(DETECTION_ENGINE_PATH))


img = cv2.imread(source)
# img = cv2.resize(img, (1280, 1280))

# Обрезаем ДО подачи в модель
img = crop_by_size(img)

results = model.predict(
    source=img,
    imgsz=IMAGE_SIZE,        # ВАЖНО: должно совпадать с тем, на чём экспортировали engine
    conf=0.5,
    iou=0.45,
    save=False,        # сохранить в runs/detect/predict*/
    show=False,        # показать окно (в headless не сработает)
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

SCALE = 0.5   # в 2 раза меньше

for r in results:
    img_vis = r.plot(line_width=2)
    h, w = img_vis.shape[:2]
    small = cv2.resize(
        img_vis, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_AREA
    )

    cv2.imshow("YOLO", small)
    cv2.waitKey(0)

cv2.destroyAllWindows()