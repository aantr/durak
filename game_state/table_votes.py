"""Подтверждение карт стола по месту и связи атаки с защитой."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


RANK_VALUE = {rank: value for value, rank in enumerate(
    ("6", "7", "8", "9", "10", "J", "Q", "K", "A"), start=6
)}


def _bbox(item: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    box = item.get("bbox")
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    fx = (first[0] + first[2]) / 2
    fy = (first[1] + first[3]) / 2
    sx = (second[0] + second[2]) / 2
    sy = (second[1] + second[3]) / 2
    width = max(first[2] - first[0], second[2] - second[0], 1)
    height = max(first[3] - first[1], second[3] - second[1], 1)
    return max(abs(fx - sx) / width, abs(fy - sy) / height)


def _beats(cover: str, attack: str, trump: str | None) -> bool:
    cover_rank = RANK_VALUE.get(cover[:-1])
    attack_rank = RANK_VALUE.get(attack[:-1])
    if cover_rank is None or attack_rank is None:
        return False
    if cover[-1] == attack[-1]:
        return cover_rank > attack_rank
    if trump is None:
        # До распознавания козыря опираемся на геометрию накрытия.
        return True
    return trump is not None and cover[-1] == trump and attack[-1] != trump


@dataclass
class _Vote:
    count: int = 0
    confidence: float = 0.0
    latest: int = -1
    streak: int = 0


@dataclass
class _Slot:
    bbox: tuple[float, float, float, float]
    cards: dict[str, _Vote] = field(default_factory=dict)
    covers: dict[int, int] = field(default_factory=dict)


class TableVotes:
    """Одна физическая позиция получает один итоговый код карты."""

    def __init__(self):
        self.slots: list[_Slot] = []

    def _match(self, box: tuple[float, float, float, float], used: set[int]) -> int | None:
        candidates = (( _distance(box, slot.bbox), index)
                      for index, slot in enumerate(self.slots) if index not in used)
        distance, index = min(candidates, default=(float("inf"), -1))
        return index if distance <= 0.7 else None

    def add(self, layout: list[Mapping[str, Any]], *, weight: int = 1,
            frame: int = 0, excluded: set[str] | None = None) -> None:
        """Добавляет по одному голосу за видимую карту в каждой позиции кадра."""
        used: set[int] = set()
        indices: dict[int, int] = {}
        for item_index, item in enumerate(layout):
            box = _bbox(item)
            if box is None:
                continue
            slot_index = self._match(box, used)
            if slot_index is None:
                slot_index = len(self.slots)
                self.slots.append(_Slot(box))
            used.add(slot_index)
            indices[item_index] = slot_index
            code = item.get("card")
            if not isinstance(code, str) or not code or code in (excluded or ()):
                continue
            vote = self.slots[slot_index].cards.setdefault(code, _Vote())
            vote.count += weight
            vote.confidence += weight * float(item.get("confidence", 1.0)) * float(item.get("suit_confidence", 1.0))
            vote.streak = vote.streak + 1 if frame == vote.latest + 1 else weight
            vote.latest = max(vote.latest, frame)
        for item_index, item in enumerate(layout):
            parent = item.get("covers")
            child_slot = indices.get(item_index)
            parent_slot = indices.get(parent) if type(parent) is int else None
            if child_slot is not None and parent_slot is not None and child_slot != parent_slot:
                covers = self.slots[child_slot].covers
                covers[parent_slot] = covers.get(parent_slot, 0) + weight

    def confirmed(self, threshold: int, *, bat: bool = False,
                  trump: str | None = None) -> set[str]:
        chosen: dict[int, str] = {}
        for index, slot in enumerate(self.slots):
            eligible = [(code, vote) for code, vote in slot.cards.items()
                        if vote.count >= threshold]
            stable = [(code, vote) for code, vote in eligible if vote.streak >= threshold]
            if stable:
                winner = max(stable, key=lambda pair: (
                    pair[1].latest, pair[1].streak, pair[1].confidence))
            else:
                winner = max(eligible, key=lambda pair: (
                    pair[1].count, pair[1].latest, pair[1].confidence), default=None)
            if winner is not None:
                chosen[index] = winner[0]
        if not bat:
            return set(chosen.values())
        # В биту попадают только распознанные пары с допустимой защитой.
        pairs: dict[int, tuple[int, int]] = {}
        for child, slot in enumerate(self.slots):
            if child not in chosen or not slot.covers:
                continue
            parent, votes = max(slot.covers.items(), key=lambda item: item[1])
            if parent not in chosen or not _beats(chosen[child], chosen[parent], trump):
                continue
            if parent not in pairs or votes > pairs[parent][1]:
                pairs[parent] = (child, votes)
        return {chosen[index] for parent, (child, _) in pairs.items()
                for index in (parent, child)}
