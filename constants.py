from pathlib import Path


DIR = Path(__file__).resolve().parent

IMAGE_SIZE = 640

DETECTION_WEIGHTS_PATH = (
    DIR / "runs/detect/runs_cards/yolo26s_p2_1280-8/weights/best.pt"
)

DETECTION_ENGINE_PATH = (
    DIR / "runs/detect/runs_cards/yolo26s_p2-5/weights/last.pt"
)

CROP_CARDS = {
    (1206, 2622): (0, 3 * 2622 // 4, 1206, 2622)
}
