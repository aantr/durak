"""Состояние партии в «Дурака», обновляемое по кадрам OpenCV."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Any, Callable, Mapping

import numpy as np

from game_state.visualization import DetectionVisualization, draw_label

RANKS = ("6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("C", "D", "H", "S")
FULL_DECK = frozenset(f"{rank}{suit}" for rank in RANKS for suit in SUITS)


def normalize_text(value: Any) -> str:
    """Удаляет все пробелы из OCR-текста и приводит его к нижнему регистру."""
    if value is None:
        return ""
    if isinstance(value, str):
        return "".join(value.split()).lower()
    if isinstance(value, Mapping):
        value = value.get("text", value.get("rec_texts", ""))
    if isinstance(value, (tuple, list)):
        # PaddleOCR: (box, text, confidence)
        if len(value) == 3 and not isinstance(value[0], str) and isinstance(value[1], str):
            return normalize_text(value[1])
        return "".join(normalize_text(item) for item in value)
    return normalize_text(str(value))


def _card_code(card: Any) -> str | None:
    """Нормализует карту к формату ``8H``/``10S``."""
    if isinstance(card, Mapping):
        if card.get("rank") is not None and card.get("suit") is not None:
            card = (card["rank"], card["suit"])
        else:
            card = card.get("card", card.get("name"))
    if isinstance(card, (tuple, list)) and len(card) >= 2:
        # Детектор ранга обучен на именах наподобие ``8R`` и ``8B``;
        # цвет там избыточен, поскольку точную масть даёт классификатор.
        rank = re.match(r"10|[6-9JQKA]", str(card[0]).upper())
        suit = _normalise_suit(card[1])
        card = f"{rank.group() if rank else card[0]}{suit}"
    if not isinstance(card, str):
        return None
    token = re.sub(r"[\s_-]", "", card).upper()
    match = re.fullmatch(r"(10|[6-9JQKA])([CDHS♣♦♥♠])", token)
    if not match:
        return None
    rank, suit = match.groups()
    return rank + {"♣": "C", "♦": "D", "♥": "H", "♠": "S"}.get(suit, suit)


def _cards(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, Mapping):
        value = value.get("cards", ())
    if isinstance(value, str):
        value = (value,)
    result = set()
    for item in value:
        code = _card_code(item)
        if code:
            result.add(code)
    return result


@lru_cache(maxsize=1)
def _ocr() -> Any:
    from paddleocr import PaddleOCR
    # Экземпляр создаётся один раз: без этого OCR не выдержит поток 5–60 FPS.
    return PaddleOCR(use_textline_orientation=True, lang="en", device="gpu")


def _ocr_lines(image: np.ndarray, crop: Mapping[tuple[int, int], tuple[int, int, int, int]], cropper: Callable[..., np.ndarray], *, visualization=None) -> list[str]:
    """OCR зоны, подготовленной соответствующим ``detect_*.py``."""
    result = _ocr().predict(cropper(image, crop))
    annotated = visualization.add_crop(crop) if visualization is not None else None
    lines = []
    for page in result or []:
        texts = page.get("rec_texts", [])
        lines.extend(texts)
        if annotated is not None:
            import cv2

            scores = page.get("rec_scores", [])
            polygons = page.get("rec_polys", page.get("dt_polys", []))
            for index, text in enumerate(texts):
                x, y = 0, 20 + index * 22
                if index < len(polygons):
                    points = np.asarray(polygons[index], dtype=np.int32).reshape(-1, 2)
                    if len(points):
                        cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
                        x, y = points.min(axis=0)
                        y -= 5
                confidence = f" {float(scores[index]):.2f}" if index < len(scores) else ""
                draw_label(annotated, normalize_text(text) + confidence, x, y)
    return lines


def _default_button(image: np.ndarray, *, visualization=None) -> str:
    from constants.constants import CROP_BUTTON
    from detect.detect_button import crop_by_size
    return normalize_text(_ocr_lines(image, CROP_BUTTON, crop_by_size, visualization=visualization))


def _default_opponent(image: np.ndarray, *, visualization=None) -> str:
    from constants.constants import CROP_OPPONENT
    from detect.detect_opponent import crop_by_size
    return normalize_text(_ocr_lines(image, CROP_OPPONENT, crop_by_size, visualization=visualization))


def _default_mine(image: np.ndarray, *, visualization=None) -> str:
    from constants.constants import CROP_MINE
    from detect.detect_mine import crop_by_size
    return normalize_text(_ocr_lines(image, CROP_MINE, crop_by_size, visualization=visualization))


def _default_deque(image: np.ndarray, *, visualization=None) -> int | None:
    from constants.constants import CROP_DEQUE_LOST_CARDS
    from detect.detect_deque import crop_by_size
    match = re.search(r"\d+", normalize_text(_ocr_lines(image, CROP_DEQUE_LOST_CARDS, crop_by_size, visualization=visualization)))
    return int(match.group()) if match else None


def _normalise_suit(suit: Any) -> str:
    # В текущем обучающем датасете clubs содержит пики, spades — крести.
    # Исправляем имена классов модели; канонические C/♣ и S/♠ не меняем.
    names = {
        "clubs": "S", "club": "S", "piki": "S",
        "spades": "C", "spade": "C", "kresti": "C",
        "diamonds": "D", "diamond": "D", "bubi": "D",
        "hearts": "H", "heart": "H",
    }
    return names.get(normalize_text(suit), str(suit or ""))


@lru_cache(maxsize=2)
def _models(on_field: bool) -> tuple[Any, Any]:
    from ultralytics import YOLO
    from constants.constants import DETECTION_ENGINE_PATH, DETECTION_FIELD_ENGINE_PATH, CLASSIFY_SUIT_WEIGHTS_PATH
    weights = DETECTION_FIELD_ENGINE_PATH if on_field else DETECTION_ENGINE_PATH
    return YOLO(str(weights)), YOLO(str(CLASSIFY_SUIT_WEIGHTS_PATH))


def _detect_hand_cards(image: np.ndarray, *, visualization=None) -> list[str]:
    """Распознавание руки с фильтром CROP_CARDS_ALLOWED."""
    from constants.constants import CROP_CARDS, CROP_CARDS_ALLOWED, IMAGE_SIZE, IMAGE_SIZE_SUIT
    from detect.detect import classify_suit, crop_by_size as crop_hand, intersect

    original_h, original_w = image.shape[:2]
    cropped = crop_hand(image, CROP_CARDS)
    model, suit_model = _models(False)
    result = model.predict(source=(cropped,), imgsz=IMAGE_SIZE, conf=.5, iou=.45, save=False, show=False, verbose=False)[0]
    allowed = None
    if (original_w, original_h) in CROP_CARDS_ALLOWED:
        ax1, ay1, ax2, ay2 = CROP_CARDS_ALLOWED[(original_w, original_h)]
        ox1, oy1, _, _ = CROP_CARDS.get((original_w, original_h), (0, 0, 0, 0))
        allowed = ((ax1 - ox1, ay1 - oy1), (ax2 - ox1, ay2 - oy1))
    height, width = cropped.shape[:2]
    annotated = visualization.add_crop(CROP_CARDS) if visualization is not None else None
    if annotated is not None:
        import cv2

        if allowed is not None:
            cv2.rectangle(annotated, allowed[0], allowed[1], (255, 0, 255), 1)
    found = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        if allowed and not intersect(allowed, center):
            continue
        rank = str(result.names[int(box.cls[0])])
        if annotated is not None:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_label(annotated, f"{rank} {float(box.conf[0]):.2f}", x1, y1 - 5)
        cx1, cx2 = max(0, center[0] - IMAGE_SIZE_SUIT // 2), min(width, center[0] + IMAGE_SIZE_SUIT // 2)
        cy1, cy2 = max(0, y2), min(height, y2 + IMAGE_SIZE_SUIT)
        if cx1 >= cx2 or cy1 >= cy2:
            continue
        suit, suit_conf = classify_suit(cropped[cy1:cy2, cx1:cx2], suit_model)
        code = _card_code((rank, _normalise_suit(suit)))
        if annotated is not None:
            cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
            label = f"{code or rank}: {suit} {suit_conf:.2f}" if suit is not None else f"{rank}: suit n/a"
            draw_label(annotated, label, x1, cy2 + 20, (255, 200, 0))
        if code:
            found.append(code)
    return found


def _default_hand(image: np.ndarray, *, visualization=None) -> list[str]:
    return _detect_hand_cards(image, visualization=visualization)


def _default_field(image: np.ndarray, *, visualization=None) -> dict:
    from constants.constants import CROP_FIELD
    from detect.detect_field import detect_field

    model, suit_model = _models(True)
    result = detect_field(image, model, suit_model, draw=visualization is not None)
    if visualization is not None:
        visualization.add_crop(CROP_FIELD)[:] = result.annotated_crop
    layout = [
        {
            "card": _card_code((card.rank, _normalise_suit(card.suit))),
            "rank": card.rank,
            "suit": card.suit,
            "bbox": card.box.bbox,
            "confidence": card.box.confidence,
            "suit_confidence": card.suit_confidence,
            "covers": card.box.covers,
        }
        for card in result.cards
    ]
    return {"cards": [card["card"] for card in layout if card["card"]], "layout": layout}


_DEFAULT_DETECTORS = {
    "button": _default_button, "deque": _default_deque, "field": _default_field,
    "opponent": _default_opponent, "mine": _default_mine, "hand": _default_hand,
}


@dataclass
class DurakGameState:
    """Наблюдаемое состояние одной 36-карточной игры.

    ``detectors`` позволяет подменить любой детектор функцией ``image ->
    result``. Допустимые ключи: ``button``, ``deque``, ``field``, ``opponent``,
    ``mine`` и ``hand``. Без него вызываются написанные детекторы и их модели лениво,
    только при первом ``update``.
    """
    detectors: Mapping[str, Callable[[np.ndarray], Any]] | None = None
    deck_remaining: int | None = None
    hand_cards: set[str] = field(default_factory=set)
    field_cards: set[str] = field(default_factory=set)
    out_cards: set[str] = field(default_factory=set)
    known_opponent_cards: set[str] = field(default_factory=set)
    unknown_opponent_cards: set[str] = field(default_factory=lambda: set(FULL_DECK))
    opponent_card_count: int | None = None
    button_text: str = ""
    opponent_text: str = ""
    mine_text: str = ""
    phase: str = "opponent_turn"
    frame_number: int = 0
    # Для необратимого переноса в биту нужен устойчивый снимок стола.
    field_confirmation_frames: int = 2
    # covers=None — нижняя карта; иначе индекс нижней карты в field_layout.
    # card=None сохраняет геометрию даже при нераспознанной масти.
    field_layout: list[dict[str, Any]] = field(default_factory=list)
    annotated_frame: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _last_opponent_action: str = field(default="", init=False, repr=False)
    _last_mine_action: str = field(default="", init=False, repr=False)
    _round_cards: set[str] = field(default_factory=set, init=False, repr=False)
    _round_destination: str | None = field(default=None, init=False, repr=False)
    _transferred_cards: set[str] = field(default_factory=set, init=False, repr=False)
    _table_was_empty: bool = field(default=True, init=False, repr=False)
    _confirmed_table: set[str] = field(default_factory=set, init=False, repr=False)
    _table_candidate: set[str] = field(default_factory=set, init=False, repr=False)
    _table_candidate_frames: int = field(default=0, init=False, repr=False)
    # Взятые карты остаются в руке до подтверждения её детектором либо розыгрыша.
    _pending_hand_cards: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.field_confirmation_frames, int) or self.field_confirmation_frames < 1:
            raise ValueError("field_confirmation_frames должен быть положительным целым числом")
        defaults = _DEFAULT_DETECTORS.copy()
        if self.detectors:
            defaults.update(self.detectors)
        self.detectors = defaults
        self._round_cards.update(self.field_cards)
        self._confirmed_table.update(self.field_cards)
        self._refresh_opponent()

    def update(self, image: np.ndarray, *, draw_detections: bool = False) -> "DurakGameState":
        """Обновляет состояние; опционально сохраняет разметку в annotated_frame.

        Разметку предоставляют встроенные детекторы. Переданные через
        detectors функции по-прежнему получают только исходное изображение.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image должен быть непустым изображением cv2/numpy.ndarray")
        self.frame_number += 1
        visualization = DetectionVisualization(image) if draw_detections else None

        def detect(key):
            detector = self.detectors[key]
            if visualization is not None and detector is _DEFAULT_DETECTORS[key]:
                return detector(image, visualization=visualization)
            return detector(image)

        raw_button, raw_opponent, raw_mine, raw_deque = (detect(key) for key in ("button", "opponent", "mine", "deque"))
        self.button_text, self.opponent_text = normalize_text(raw_button), normalize_text(raw_opponent)
        self.mine_text = normalize_text(raw_mine)
        self._set_phase()
        try:
            number = int(raw_deque)
            if 0 <= number <= len(FULL_DECK):
                self.deck_remaining = number
        except (TypeError, ValueError):
            pass
        self.hand_cards = _cards(detect("hand"))
        self._pending_hand_cards.difference_update(self.hand_cards)
        field_result = detect("field")
        self.field_cards = _cards(field_result)
        self.field_layout = list(field_result.get("layout", [])) if isinstance(field_result, Mapping) else []
        self._apply_table_events()
        self.hand_cards.update(self._pending_hand_cards)
        self.hand_cards.difference_update(self.out_cards)
        self.known_opponent_cards.difference_update(self.hand_cards | self.out_cards)
        self._refresh_opponent()
        self.annotated_frame = visualization.render() if visualization is not None else None
        return self

    def _set_phase(self) -> None:
        self.phase = {"pass": "throw_in", "yourturn": "your_turn", "ready": "ready", "itake": "defend_or_take"}.get(self.button_text, "opponent_turn")

    def _apply_table_events(self) -> None:
        """Переносит накопленный стол по надписям игроков, не по кнопке.

        Состав стола сохраняется при исчезновении карт до прихода OCR-сигнала.
        Для Bat берётся последний подтверждённый снимок, а не объединение
        всех распознаваний. После Bat состав биты заморожен до нового стола
        и исчезновения терминальных надписей. При I take можно ещё подкидывать.
        """
        actions = {"pass", "bat", "itake"}
        mine = self.mine_text if self.mine_text in actions else ""
        opponent = self.opponent_text if self.opponent_text in actions else ""
        destinations = set()
        for actor, action, previous in (
            ("mine", mine, self._last_mine_action),
            ("opponent", opponent, self._last_opponent_action),
        ):
            if action != previous:
                if action == "bat":
                    destinations.add("out")
                elif action == "itake":
                    destinations.add(actor)
        self._last_mine_action, self._last_opponent_action = mine, opponent

        terminal_visible = mine in {"bat", "itake"} or opponent in {"bat", "itake"}
        if self._round_destination == "out":
            # Bat часто остаётся на экране во время анимации/следующей раздачи.
            # Повторные сигналы от обоих игроков не должны дополнять старую биту.
            self.field_cards.difference_update(self.out_cards)
            if terminal_visible or not self.field_cards:
                self._table_was_empty = not self.field_cards
                self.field_cards.clear()
                self.field_layout.clear()
                return
            self._reset_table_tracking()

        if self._round_destination is not None and self.field_cards and self._round_cards:
            # Новый стол без старых карт либо повторный розыгрыш взятых карт
            # после пустого стола и завершения прежних надписей.
            if (self.field_cards.isdisjoint(self._round_cards)
                    or (self._table_was_empty and not terminal_visible)):
                self._reset_table_tracking()

        if (self._round_destination is None and self._table_was_empty
                and self.field_cards and self._round_cards and not destinations
                and self.field_cards.isdisjoint(self._round_cards)):
            # Между столами мог быть пропущен I take/Bat. Старый снимок не
            # наследуется новым розыгрышем даже до подтверждения его карт.
            self._reset_table_tracking()

        if self._round_destination is None:
            visible = self.field_cards - self.out_cards - self.hand_cards
            if visible and visible == self._table_candidate:
                self._table_candidate_frames += 1
            else:
                self._table_candidate = set(visible)
                self._table_candidate_frames = 1 if visible else 0
            if self._table_candidate_frames >= self.field_confirmation_frames:
                # Заменяем снимок: исправленная масть/ранг не оставляет
                # в памяти розыгрыша старую ошибочную карту.
                self._confirmed_table = set(visible)

        self._table_was_empty = not self.field_cards
        self._round_cards.update(self.field_cards)
        if self._round_destination is None and len(destinations) == 1:
            self._round_destination = destinations.pop()

        if self._round_destination is None:
            self._pending_hand_cards.difference_update(self.field_cards)
            self.hand_cards.difference_update(self.field_cards)
            self.known_opponent_cards.difference_update(self.field_cards)
            return

        moved = self._round_cards - self._transferred_cards
        if self._round_destination == "out":
            moved = self._confirmed_table - self.hand_cards - self.out_cards
            self._round_cards = set(moved)
            self.out_cards.update(moved)
            self._pending_hand_cards.difference_update(moved)
        elif self._round_destination == "mine":
            self._pending_hand_cards.update(moved - self.hand_cards)
        else:
            self.known_opponent_cards.update(moved)
            self._pending_hand_cards.difference_update(moved)
            self.hand_cards.difference_update(moved)
        self._transferred_cards.update(moved)
        self.field_cards.clear()
        self.field_layout.clear()

    def _reset_table_tracking(self) -> None:
        self._round_cards.clear()
        self._transferred_cards.clear()
        self._confirmed_table.clear()
        self._table_candidate.clear()
        self._table_candidate_frames = 0
        self._round_destination = None

    def _refresh_opponent(self) -> None:
        known = self.hand_cards | self.field_cards | self.out_cards | self.known_opponent_cards
        self.unknown_opponent_cards = set(FULL_DECK - known)
        if self.deck_remaining is None:
            self.opponent_card_count = None
        else:
            self.opponent_card_count = max(0, len(FULL_DECK) - self.deck_remaining - len(self.hand_cards) - len(self.field_cards) - len(self.out_cards))
