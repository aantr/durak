"""Построчный журнал только изменившихся результатов распознавания."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from constants.constants import CROP_FIELD


DEFAULT_RECOGNITION_LOG = Path(__file__).resolve().parents[1] / "logs" / "recognition.jsonl"


class RecognitionChangeLog:
    """Сохраняет полное состояние при изменении, включая детали детекции."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._previous: dict[str, Any] | None = None
        self._session = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._image_number = 0

    def record(self, state: Any, snapshot: Mapping[str, Any],
               frame: np.ndarray | None = None) -> bool:
        current = dict(snapshot)
        if current == self._previous:
            return False
        if self._previous is None:
            changed = list(current)
        else:
            changed = [key for key in current if current[key] != self._previous.get(key)]
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session": self._session,
            "frame": state.frame_number,
            "frame_size": state.frame_size,
            "changed": changed,
            "state": current,
            "detection": {
                "hand_layout": state.hand_layout,
                "field_layout": state.field_layout,
                "last_mine_action": state._last_mine_action,
                "last_opponent_action": state._last_opponent_action,
                "round_destination": state._round_destination,
                "confirmed_table": sorted(state._confirmed_table),
            },
        }
        if frame is not None and self._needs_field_image(changed, current):
            relative, region = self._save_field_image(frame, state.frame_number)
            record["field_image"] = relative
            record["field_image_region"] = region
        self._file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        self._file.flush()
        self._previous = current
        return True

    @staticmethod
    def _needs_field_image(changed: list[str], current: Mapping[str, Any]) -> bool:
        if {"field_cards", "field_layout", "out_cards", "last_out_cards"}.intersection(changed):
            return True
        return any(key in changed and current.get(key) in {"bat", "itake"}
                   for key in ("mine", "opponent"))

    def _save_field_image(self, frame: np.ndarray, frame_number: int) -> tuple[str, list[int]]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = CROP_FIELD.get((width, height), (0, 0, width, height))
        crop = frame[y1:y2, x1:x2]
        self._image_number += 1
        session = self._session.replace(":", "-").replace("+", "_")
        directory = self.path.parent / f"{self.path.stem}_frames" / session
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"{self._image_number:06}_f{frame_number:06}.jpg"
        if not cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise OSError(f"Не удалось сохранить кадр поля: {image_path}")
        return str(image_path.relative_to(self.path.parent)), [x1, y1, x2, y2]

    def close(self) -> None:
        self._file.close()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Не удалось сериализовать значение {type(value).__name__}")
