from pathlib import Path


DIR = Path(__file__).resolve().parent.parent

IMAGE_SIZE = 640
IMAGE_SIZE_FIELD = 640
IMAGE_SIZE_SUIT = 64

DETECTION_WEIGHTS_PATH = (
    DIR / "runs/detect/runs_cards/yolo26s_p2-6/weights/best.pt"
)

DETECTION_ENGINE_PATH = (
    DIR / "runs/detect/runs_cards/yolo26s_p2-6/weights/best.engine"
)

DETECTION_FIELD_WEIGHTS_PATH = (
    DIR / "runs/detect/runs_field/yolo26s_p2/weights/best.pt"
)

DETECTION_FIELD_ENGINE_PATH = (
    DIR / "runs/detect/runs_field/yolo26s_p2/weights/best.pt"
)

CLASSIFY_SUIT_WEIGHTS_PATH = (
    DIR / "runs/cls/runs_suit/cards_suit/weights/best.pt"
)

CROP_CARDS = {
    (1206, 2622): (0, 3 * 2622 // 4, 1206, 2622)
}

CROP_CARDS_ALLOWED = {
    (1206, 2622): (0, 3 * 2622 // 4, 1206, 2322)
}

CROP_FIELD = {
    (1206, 2622): (0, 1 * 2622 // 4, 1206, 3 * 2622 // 4)
}

CROP_DEQUE_LOST_CARDS = {
    (1206, 2622): (0, 575, 100, 650)
}

CROP_OPPONENT = {
    (1206, 2622): (500, 200, 700, 300)
}

CROP_MINE = {
    (1206, 2622): (500, 2622 - 350, 700, 2622 - 250)
}

CROP_BUTTON = {
    (1206, 2622): (50, 2622 - 275, 450, 2622 - 125)
}

ALLOW_CARD = {
    (1206, 2622): (150, 100)
}

DISCARD_CARD = {
    (1206, 2622): (210, 300)
}

AREA_SHIFT = {
    (1206, 2622): (-20, -20)
}