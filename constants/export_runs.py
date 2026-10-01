import shutil
from pathlib import Path

from constants import *


DIR = Path(__file__).resolve().parent.parent

dst = DIR / "export_runs"
dst.mkdir(parents=True, exist_ok=True)

for k, v in [
    (DETECTION_WEIGHTS_PATH, "detect_cards.pt"), 
    (DETECTION_FIELD_WEIGHTS_PATH, "detect_field.pt"),
    (CLASSIFY_SUIT_WEIGHTS_PATH, "classify_suit.pt")
    ]:
    shutil.copy(k, dst / v)
