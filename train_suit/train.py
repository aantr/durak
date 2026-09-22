from pathlib import Path

from ultralytics import YOLO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants.constants import IMAGE_SIZE_SUIT

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