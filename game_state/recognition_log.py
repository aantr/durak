"""Построчный журнал только изменившихся результатов распознавания."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_RECOGNITION_LOG = Path(__file__).resolve().parents[1] / "logs" / "recognition.jsonl"


class RecognitionChangeLog:
    """Сохраняет полное состояние при изменении, включая детали детекции."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._previous: dict[str, Any] | None = None
        self._session = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def record(self, state: Any, snapshot: Mapping[str, Any]) -> bool:
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
        self._file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        self._file.flush()
        self._previous = current
        return True

    def close(self) -> None:
        self._file.close()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Не удалось сериализовать значение {type(value).__name__}")
