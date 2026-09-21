import sys
from pathlib import Path

from ultralytics import YOLO

DIR = Path(__file__).resolve().parents[1]
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

from constants import DETECTION_FIELD_WEIGHTS_PATH, IMAGE_SIZE_FIELD


def main() -> None:
    model = YOLO(str(DETECTION_FIELD_WEIGHTS_PATH))

    model.export(
        format="engine",
        imgsz=IMAGE_SIZE_FIELD,
        quantize=16,
        device=0,
    )


if __name__ == "__main__":
    main()
