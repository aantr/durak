from pathlib import Path
import cv2
from ultralytics import YOLO

from constants import DETECTION_ENGINE_PATH

# Загружаем engine
model = YOLO(str(DETECTION_ENGINE_PATH))

# Источник: путь к картинке, папке, видео или 0 для вебки
source = "screenshots/IMG_1408.PNG"

results = model.predict(
    source=source,
    imgsz=1280,        # ВАЖНО: должно совпадать с тем, на чём экспортировали engine
    conf=0.15,
    iou=0.45,
    save=False,         # сохранить в runs/detect/predict*/
    show=False,         # показать окно (в headless не сработает)
    line_width=2,
    show_labels=True,
    show_conf=True,
)

# Если нужно обработать результаты вручную
for r in results:
    boxes = r.boxes
    for b in boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        cls  = int(b.cls[0])
        conf = float(b.conf[0])
        name = r.names[cls]
        print(f"{name} {conf:.2f} {xyxy.tolist()}")

SCALE = 0.5   # в 2 раза меньше

for r in results:
    img = r.plot(line_width=2)
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_AREA)

    cv2.imshow("YOLO", small)
    cv2.waitKey(0)

cv2.destroyAllWindows()