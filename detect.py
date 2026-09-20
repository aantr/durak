from pathlib import Path
from ultralytics import YOLO

from constants import DETECTION_ENGINE_PATH

# Загружаем engine
model = YOLO(str(DETECTION_ENGINE_PATH))

# Источник: путь к картинке, папке, видео или 0 для вебки
source = "screenshots/IMG_1407.PNG"

results = model.predict(
    source=source,
    imgsz=1280,        # ВАЖНО: должно совпадать с тем, на чём экспортировали engine
    conf=0.25,
    iou=0.45,
    save=True,         # сохранить в runs/detect/predict*/
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