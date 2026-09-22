from pathlib import Path

from ultralytics import YOLO

from constants import IMAGE_SIZE_SUIT

DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = DIR / "runs" / "cls" / "runs_suit"

model = YOLO("yolo11n-cls.pt")
model.train(
    data="generation_suit",
    imgsz=IMAGE_SIZE_SUIT,
    epochs=100,
    batch=32,
    patience=100,
    project=str(PROJECT_DIR),
    name="cards_suit",
)