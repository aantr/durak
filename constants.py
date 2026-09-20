from pathlib import Path


DIR = Path(__file__).resolve().parent

DETECTION_WEIGHTS_PATH = (
    DIR / "runs/detect/runs_cards/yolo26s_p2_1280-5/weights/best.pt"
)

DETECTION_ENGINE_PATH = (
    DIR / "runs/detect/runs_cards/yolo26s_p2_1280-5/weights/best.engine"
)