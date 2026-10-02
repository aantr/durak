"""Состояние партии в «Дурака», обновляемое по кадрам OpenCV."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Any, Callable, Mapping

import numpy as np

from game_state.visualization import DetectionVisualization, draw_label
from game_state.table_votes import TableVotes

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
    return PaddleOCR(use_textline_orientation=False, use_doc_orientation_classify=False,
                     lang="en", device="gpu")


def _ocr_lines(image: np.ndarray, crop: Mapping[tuple[int, int], tuple[int, int, int, int]], cropper: Callable[..., np.ndarray], *, visualization=None, fixed_orientation: bool = False) -> list[str]:
    """OCR зоны, подготовленной соответствующим ``detect_*.py``."""
    # Одиночные 6/9 нельзя автоматически переворачивать как строки документа.
    options = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                   use_textline_orientation=False) if fixed_orientation else {}
    result = _ocr().predict(cropper(image, crop), **options)
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
    match = re.search(r"\d+", normalize_text(_ocr_lines(
        image, CROP_DEQUE_LOST_CARDS, crop_by_size,
        visualization=visualization, fixed_orientation=True)))
    return int(match.group()) if match else None


def _normalise_suit(suit: Any) -> str:
    # Имена классов классификатора соответствуют стандартным названиям мастей.
    names = {
        "clubs": "C", "club": "C", "kresti": "C",
        "spades": "S", "spade": "S", "piki": "S",
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


@lru_cache(maxsize=1)
def preload_models() -> None:
    """Загружает модели и инициализирует YOLO/TensorRT до получения видео."""
    from constants.constants import IMAGE_SIZE, IMAGE_SIZE_FIELD, IMAGE_SIZE_SUIT

    _ocr()
    suit_image = np.zeros((IMAGE_SIZE_SUIT, IMAGE_SIZE_SUIT, 3), dtype=np.uint8)
    for on_field, size in ((False, IMAGE_SIZE), (True, IMAGE_SIZE_FIELD)):
        model, suit_model = _models(on_field)
        # Одного YOLO(path) недостаточно: backend создаётся в первом predict().
        model.predict(source=(np.zeros((size, size, 3), dtype=np.uint8),),
                      imgsz=size, conf=.5, iou=.45, save=False, show=False, verbose=False)
        suit_model.predict(source=suit_image, imgsz=IMAGE_SIZE_SUIT, verbose=False)


def _detect_hand_cards(image: np.ndarray, *, visualization=None) -> dict:
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
    layout = []
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
            layout.append({"card": code, "bbox": (int(x1), int(y1), int(x2), int(y2))})
    return {"cards": found, "layout": layout}


def _default_trump(image: np.ndarray, *, visualization=None) -> str | None:
    import cv2
    from constants.constants import CROP_TRUMP, IMAGE_SIZE_SUIT

    height, width = image.shape[:2]
    bounds = CROP_TRUMP.get((width, height))
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    cropped = cv2.rotate(image[y1:y2, x1:x2], cv2.ROTATE_90_COUNTERCLOCKWISE)
    _, model = _models(False)
    result = model.predict(source=cropped, imgsz=IMAGE_SIZE_SUIT, verbose=False)[0]
    if result.probs is None:
        return None
    confidence = float(result.probs.top1conf)
    suit = _normalise_suit(result.names[int(result.probs.top1)])
    if visualization is not None:
        draw_label(visualization.add_crop(CROP_TRUMP), f"Trump {suit} {confidence:.2f}", 0, 20)
    return suit if suit in SUITS and confidence >= 0.8 else None


def _default_hand(image: np.ndarray, *, visualization=None) -> dict:
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
    "trump": _default_trump,
}


TERMINAL_FIELD_FRAMES = 5


@dataclass
class DurakGameState:
    """Наблюдаемое состояние одной 36-карточной игры.

    ``detectors`` позволяет подменить любой детектор функцией ``image ->
    result``. Допустимые ключи: ``button``, ``deque``, ``field``, ``opponent``,
    ``mine``, ``hand`` и ``trump``. Без него вызываются написанные детекторы и их модели лениво,
    только при первом ``update``.
    """
    detectors: Mapping[str, Callable[[np.ndarray], Any]] | None = None
    deck_remaining: int | None = None
    hand_cards: set[str] = field(default_factory=set)
    field_cards: set[str] = field(default_factory=set)
    out_cards: set[str] = field(default_factory=set)
    # Только фактически добавленные карты последнего отбоя; не отдельная зона.
    last_out_cards: set[str] = field(default_factory=set, init=False)
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
    # После Bat или I take соперника поле читается на каждом из этих кадров.
    terminal_field_frames: int = TERMINAL_FIELD_FRAMES
    # covers=None — нижняя карта; иначе индекс нижней карты в field_layout.
    # card=None сохраняет геометрию даже при нераспознанной масти.
    field_layout: list[dict[str, Any]] = field(default_factory=list)
    hand_layout: list[dict[str, Any]] = field(default_factory=list)
    frame_size: tuple[int, int] | None = field(default=None, init=False)
    # При пустой колоде интерфейс скрывает число; защищаемся от пропусков OCR.
    deck_empty_confirmation_frames: int = 3
    trump: str | None = None
    # Новое ненулевое число принимается после одинаковых последовательных кадров.
    deck_confirmation_frames: int = 3
    _deck_candidate: int | None = field(default=None, init=False, repr=False)
    _deck_candidate_frames: int = field(default=0, init=False, repr=False)
    _trump_candidate: str | None = field(default=None, init=False, repr=False)
    _trump_candidate_frames: int = field(default=0, init=False, repr=False)
    annotated_frame: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _detection_overlays: dict[str, list] = field(default_factory=dict, init=False, repr=False, compare=False)
    _deck_missing_frames: int = field(default=0, init=False, repr=False)
    _last_opponent_action: str = field(default="", init=False, repr=False)
    _last_mine_action: str = field(default="", init=False, repr=False)
    _round_cards: set[str] = field(default_factory=set, init=False, repr=False)
    _round_destination: str | None = field(default=None, init=False, repr=False)
    _transferred_cards: set[str] = field(default_factory=set, init=False, repr=False)
    _table_was_empty: bool = field(default=True, init=False, repr=False)
    _confirmed_table: set[str] = field(default_factory=set, init=False, repr=False)
    _confirmed_layout: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _table_candidate: set[str] = field(default_factory=set, init=False, repr=False)
    _table_candidate_layout: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _table_candidate_frames: int = field(default=0, init=False, repr=False)
    # Последовательные реальные наблюдения известной карты в новой зоне.
    _opponent_move_candidates: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)
    # Взятые карты остаются в руке до подтверждения её детектором либо розыгрыша.
    _pending_hand_cards: set[str] = field(default_factory=set, init=False, repr=False)
    _last_field_cards: set[str] = field(default_factory=set, init=False, repr=False)
    _last_field_layout: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _terminal_frames_left: int = field(default=0, init=False, repr=False)
    _terminal_mine_action: str = field(default="", init=False, repr=False)
    _terminal_opponent_action: str = field(default="", init=False, repr=False)
    _terminal_card_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _terminal_votes: TableVotes | None = field(default=None, init=False, repr=False)
    _terminal_confirmed_before: set[str] = field(default_factory=set, init=False, repr=False)
    _terminal_snapshot_candidate: set[str] = field(default_factory=set, init=False, repr=False)
    _terminal_snapshot_frames: int = field(default=0, init=False, repr=False)
    _terminal_latest_stable: set[str] | None = field(default=None, init=False, repr=False)
    _terminal_seen_cards: set[str] = field(default_factory=set, init=False, repr=False)
    _terminal_had_gap: bool = field(default=False, init=False, repr=False)
    _terminal_new_round: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.field_confirmation_frames, int) or self.field_confirmation_frames < 1:
            raise ValueError("field_confirmation_frames должен быть положительным целым числом")
        if isinstance(self.terminal_field_frames, bool) or not isinstance(self.terminal_field_frames, int) or self.terminal_field_frames < 1:
            raise ValueError("terminal_field_frames должен быть положительным целым числом")
        if not isinstance(self.deck_empty_confirmation_frames, int) or self.deck_empty_confirmation_frames < 1:
            raise ValueError("deck_empty_confirmation_frames должен быть положительным целым числом")
        if isinstance(self.deck_confirmation_frames, bool) or not isinstance(self.deck_confirmation_frames, int) or self.deck_confirmation_frames < 1:
            raise ValueError("deck_confirmation_frames должен быть положительным целым числом")
        defaults = _DEFAULT_DETECTORS.copy()
        if self.detectors:
            defaults.update(self.detectors)
        self.detectors = defaults
        self._round_cards.update(self.field_cards)
        self._confirmed_table.update(self.field_cards)
        self._confirmed_layout = [dict(item) for item in self.field_layout]
        self._refresh_opponent()

    def reset(self, *, trump: str | None = None) -> None:
        """Новая партия с прежними детекторами и настройками подтверждения."""
        fresh = type(self)(
            detectors=self.detectors, trump=trump,
            field_confirmation_frames=self.field_confirmation_frames,
            terminal_field_frames=self.terminal_field_frames,
            deck_empty_confirmation_frames=self.deck_empty_confirmation_frames,
            deck_confirmation_frames=self.deck_confirmation_frames,
        )
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)

    def update(self, image: np.ndarray, *, draw_detections: bool = False,
               slow_every: int = 1) -> "DurakGameState":
        """Обновляет состояние; опционально сохраняет разметку в annotated_frame.

        Разметку предоставляют встроенные детекторы. Переданные через
        detectors функции по-прежнему получают только исходное изображение.
        При slow_every > 1 mine/opponent читаются каждый кадр, остальные
        детекторы — каждый N-й вызов, начиная с первого. Изменение надписей
        либо размера кадра запускает полный проход вне очереди. После Bat
        или I take соперника поле читается terminal_field_frames кадров подряд.
        Разметка пропущенных детекторов сохраняется до следующего запуска;
        пустой результат удаляет их прежние рамки и подписи.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image должен быть непустым изображением cv2/numpy.ndarray")
        if isinstance(slow_every, bool) or not isinstance(slow_every, int) or slow_every < 1:
            raise ValueError("slow_every должен быть положительным целым числом")
        previous_size = self.frame_size
        if previous_size != (image.shape[1], image.shape[0]):
            self._opponent_move_candidates.clear()
        self.frame_number += 1
        self.frame_size = (image.shape[1], image.shape[0])
        starting_draw = draw_detections and self.annotated_frame is None
        if not draw_detections or previous_size != self.frame_size:
            self._detection_overlays.clear()
        visualization = (DetectionVisualization(image, layers=self._detection_overlays)
                         if draw_detections else None)

        def detect(key):
            detector = self.detectors[key]
            start = len(visualization.crops) if visualization is not None else 0
            if visualization is not None and detector is _DEFAULT_DETECTORS[key]:
                result = detector(image, visualization=visualization)
            else:
                result = detector(image)
            if visualization is not None:
                # Заменяем даже пустым слоем: пропуск запуска != ничего не найдено.
                self._detection_overlays[key] = visualization.extract_layers(start)
            return result

        # Надписи действий приоритетны: читаем их до любых тяжёлых детекторов.
        mine_text = normalize_text(detect("mine"))
        opponent_text = normalize_text(detect("opponent"))
        previous_mine, previous_opponent = self.mine_text, self.opponent_text
        actions_changed = (mine_text, opponent_text) != (previous_mine, previous_opponent)
        self.mine_text, self.opponent_text = mine_text, opponent_text
        terminal_started = ((mine_text == "bat" and previous_mine != "bat")
                            or (opponent_text in {"bat", "itake"} and opponent_text != previous_opponent))
        if terminal_started and self._terminal_frames_left == 0:
            self._terminal_frames_left = self.terminal_field_frames
            self._terminal_card_counts = {
                card: self._table_candidate_frames for card in self._table_candidate
            }
            self._terminal_confirmed_before = set(self._confirmed_table)
            self._terminal_votes = TableVotes()
            if self._confirmed_layout:
                excluded = {item.get("card") for item in self._confirmed_layout
                            if item.get("card") not in self._confirmed_table}
                self._terminal_votes.add(self._confirmed_layout, weight=self.field_confirmation_frames,
                                         frame=-2, excluded=excluded)
            if self._table_candidate_layout and self._table_candidate != self._confirmed_table:
                excluded = {item.get("card") for item in self._table_candidate_layout
                            if item.get("card") not in self._table_candidate}
                self._terminal_votes.add(self._table_candidate_layout,
                                         weight=self._table_candidate_frames, frame=-1, excluded=excluded)
            self._terminal_snapshot_candidate = set()
            self._terminal_snapshot_frames = 0
            self._terminal_latest_stable = None
            self._terminal_seen_cards = self._confirmed_table | self._table_candidate
            self._terminal_had_gap = False
            self._terminal_new_round = False
            self._terminal_mine_action = ""
            self._terminal_opponent_action = ""
        if self._terminal_frames_left:
            if mine_text in {"bat", "itake"}:
                self._terminal_mine_action = mine_text
            if opponent_text in {"bat", "itake"}:
                self._terminal_opponent_action = opponent_text
        full_update = ((self.frame_number - 1) % slow_every == 0
                       or actions_changed or previous_size != self.frame_size or starting_draw
                       or self._terminal_frames_left > 0)
        if not full_update:
            # Не выдаём кеш карт/колоды за новое наблюдение: не подтверждаем
            # повторно стол, пустую колоду и не выполняем переносы карт.
            self.annotated_frame = visualization.render() if visualization is not None else None
            return self

        self.button_text = normalize_text(detect("button"))
        self._set_phase()
        self._update_deck_count(detect("deque"))
        if self.deck_remaining == 0 or self.phase == "ready":
            self._detection_overlays.pop("trump", None)
        hand_result = detect("hand")
        self.hand_cards = _cards(hand_result)
        self.hand_layout = list(hand_result.get("layout", [])) if isinstance(hand_result, Mapping) else []
        if self.trump is None and self.hand_cards and self.phase != "ready" and self.deck_remaining != 0:
            trump = detect("trump")
            if trump in SUITS:
                self._trump_candidate_frames = self._trump_candidate_frames + 1 if trump == self._trump_candidate else 1
                self._trump_candidate = trump
                if self._trump_candidate_frames >= 2:
                    self.trump = trump
            else:
                self._trump_candidate = None
                self._trump_candidate_frames = 0
        self._pending_hand_cards.difference_update(self.hand_cards)
        field_result = detect("field")
        self.field_cards = _cards(field_result)
        self.field_layout = list(field_result.get("layout", [])) if isinstance(field_result, Mapping) else []
        observed_field = self.field_cards - self.hand_cards
        self._confirm_opponent_card_transitions()
        if self._terminal_frames_left:
            can_throw_more = (self._terminal_opponent_action == "itake"
                              and self.button_text == "pass" and mine_text != "pass")
            if observed_field and self._terminal_had_gap and not can_throw_more:
                if observed_field.isdisjoint(self._terminal_seen_cards):
                    # После исчезновения прежнего стола начался новый розыгрыш.
                    self._terminal_new_round = True
            if not observed_field and self._terminal_seen_cards:
                self._terminal_had_gap = True
            if not self._terminal_new_round:
                for card in observed_field:
                    self._terminal_card_counts[card] = self._terminal_card_counts.get(card, 0) + 1
                self._terminal_votes.add(self.field_layout, frame=self.frame_number,
                                         excluded=self.hand_cards | self.out_cards)
                if observed_field:
                    if observed_field == self._terminal_snapshot_candidate:
                        self._terminal_snapshot_frames += 1
                    else:
                        self._terminal_snapshot_candidate = set(observed_field)
                        self._terminal_snapshot_frames = 1
                    if self._terminal_snapshot_frames >= self.field_confirmation_frames:
                        self._terminal_latest_stable = set(observed_field)
                else:
                    self._terminal_snapshot_candidate.clear()
                    self._terminal_snapshot_frames = 0
                self._terminal_seen_cards.update(observed_field)
                if self.field_cards:
                    self._last_field_cards = set(self.field_cards)
                    self._last_field_layout = [dict(item) for item in self.field_layout]
            self._terminal_frames_left -= 1
            if self._terminal_frames_left == 0:
                additional = {
                    card for card, count in self._terminal_card_counts.items()
                    if count >= self.field_confirmation_frames
                }
                latest = self._terminal_latest_stable
                if latest is not None and len(latest) >= len(self._terminal_confirmed_before):
                    confirmed = latest | (additional - self._terminal_confirmed_before)
                else:
                    confirmed = self._terminal_confirmed_before | additional
                bat = self._terminal_mine_action == "bat" or self._terminal_opponent_action == "bat"
                if self._terminal_votes.slots:
                    confirmed = self._terminal_votes.confirmed(
                        self.field_confirmation_frames, bat=bat, trump=self.trump)
                new_field = set(self.field_cards) if self._terminal_new_round else set()
                new_layout = [dict(item) for item in self.field_layout] if self._terminal_new_round else []
                if self._terminal_new_round:
                    self.field_cards.clear()
                    self.field_layout.clear()
                if self._terminal_opponent_action == "itake" and self.button_text == "pass":
                    self._last_field_cards = set(confirmed)
                    self._last_field_layout = [
                        item for item in self._last_field_layout if item.get("card") in confirmed
                    ]
                self._apply_table_events(
                    observed_field=set() if self._terminal_new_round else observed_field,
                    mine_action=self._terminal_mine_action or mine_text,
                    opponent_action=self._terminal_opponent_action or opponent_text,
                    confirmed_override=confirmed,
                )
                if new_field:
                    self.field_cards = new_field
                    self.field_layout = new_layout
                self._terminal_card_counts.clear()
                self._terminal_votes = None
                self._terminal_confirmed_before.clear()
                self._terminal_latest_stable = None
                self._terminal_snapshot_candidate.clear()
                self._terminal_snapshot_frames = 0
                self._terminal_seen_cards.clear()
                self._terminal_had_gap = self._terminal_new_round = False
                self._terminal_mine_action = self._terminal_opponent_action = ""
        else:
            self._apply_table_events(observed_field=observed_field)
        self.hand_cards.update(self._pending_hand_cards)
        self.hand_cards.difference_update(self.out_cards)
        self.known_opponent_cards.difference_update(self.hand_cards | self.out_cards)
        self._refresh_opponent()
        self.annotated_frame = visualization.render() if visualization is not None else None
        return self

    @property
    def terminal_pending(self) -> bool:
        """Истина, пока собираются кадры после конечной надписи розыгрыша."""

        return self._terminal_frames_left > 0

    def _confirm_opponent_card_transitions(self) -> None:
        """Не забывает руку соперника из-за одиночной ошибки детектора.

        До field_confirmation_frames наблюдений в одной зоне приоритет у
        известной руки соперника. Проверка покарточная: изменение других карт
        не сбрасывает подтверждение. Вызывается только при свежей детекции.
        """
        previous = self._opponent_move_candidates
        current = {}
        ambiguous = self.hand_cards & self.field_cards
        for zone, cards, layout in (
            ("hand", self.hand_cards, self.hand_layout),
            ("field", self.field_cards, self.field_layout),
        ):
            pending = set()
            for card in cards & self.known_opponent_cards:
                key = (card, zone)
                count = 0 if card in ambiguous else previous.get(key, 0) + 1
                if count:
                    current[key] = min(count, self.field_confirmation_frames)
                if count < self.field_confirmation_frames:
                    pending.add(card)
            cards.difference_update(pending)
            # Не удаляем геометрию: covers содержит индексы элементов layout.
            # Копируем записи, чтобы не менять возвращённый детектором объект.
            layout[:] = [dict(item, card=None) if item.get("card") in pending else item
                         for item in layout]
        self._opponent_move_candidates = current

    def _update_deck_count(self, raw_count: Any) -> None:
        if raw_count is None or (isinstance(raw_count, str) and not raw_count.strip()):
            self._deck_candidate = None
            self._deck_candidate_frames = 0
            self._deck_missing_frames = min(
                self._deck_missing_frames + 1, self.deck_empty_confirmation_frames)
            if self._deck_missing_frames >= self.deck_empty_confirmation_frames:
                self.deck_remaining = 0
            return

        self._deck_missing_frames = 0
        try:
            number = int(raw_count)
        except (TypeError, ValueError, OverflowError):
            self._deck_candidate = None
            self._deck_candidate_frames = 0
            return
        if not 0 <= number <= len(FULL_DECK):
            self._deck_candidate = None
            self._deck_candidate_frames = 0
            return
        if number == 0 or number == self.deck_remaining:
            self._deck_candidate = None
            self._deck_candidate_frames = 0
            self.deck_remaining = number
            return
        self._deck_candidate_frames = self._deck_candidate_frames + 1 if number == self._deck_candidate else 1
        self._deck_candidate = number
        if self._deck_candidate_frames >= self.deck_confirmation_frames:
            self.deck_remaining = number
            self._deck_candidate = None
            self._deck_candidate_frames = 0

    def _set_phase(self) -> None:
        self.phase = {"pass": "throw_in", "bat": "throw_in", "yourturn": "your_turn", "ready": "ready", "itake": "defend_or_take"}.get(self.button_text, "opponent_turn")

    def _apply_table_events(self, *, observed_field: set[str] | None = None,
                            mine_action: str | None = None, opponent_action: str | None = None,
                            confirmed_override: set[str] | None = None) -> None:
        """Переносит накопленный стол по надписям игроков, не по кнопке.

        Состав стола сохраняется при исчезновении карт до прихода OCR-сигнала.
        Для Bat берутся подтверждённые позиции карт. После Bat состав биты
        заморожен до нового стола и исчезновения терминальных надписей.
        При I take можно ещё подкидывать.
        """
        # Даже ещё не подтверждённая карта означает непустое наблюдение стола.
        # Иначе фильтр переходов задерживал бы начало следующего розыгрыша.
        observed = self.field_cards if observed_field is None else observed_field
        observed = observed - self.out_cards
        actions = {"pass", "bat", "itake"}
        mine_text = self.mine_text if mine_action is None else mine_action
        opponent_text = self.opponent_text if opponent_action is None else opponent_action
        mine = mine_text if mine_text in actions else ""
        opponent = opponent_text if opponent_text in actions else ""
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
        can_throw_to_taking_opponent = (
            self.button_text == "pass" and mine != "pass"
            and (opponent == "itake" or self._round_destination == "opponent")
        )
        if self._round_destination == "out":
            # Bat часто остаётся на экране во время анимации/следующей раздачи.
            # Повторные сигналы от обоих игроков не должны дополнять старую биту.
            self.field_cards.difference_update(self.out_cards)
            if terminal_visible or not observed:
                self._table_was_empty = not observed
                self.field_cards.clear()
                self.field_layout.clear()
                return
            self._reset_table_tracking()

        if (self._round_destination is not None and observed and self._round_cards
                and not can_throw_to_taking_opponent):
            # Новый стол без старых карт либо повторный розыгрыш взятых карт
            # после пустого стола и завершения прежних надписей.
            if (observed.isdisjoint(self._round_cards)
                    or (self._table_was_empty and not terminal_visible)):
                self._reset_table_tracking()

        if (self._round_destination is None and self._table_was_empty
                and observed and self._round_cards and not destinations
                and observed.isdisjoint(self._round_cards)):
            # Между столами мог быть пропущен I take/Bat. Старый снимок не
            # наследуется новым розыгрышем даже до подтверждения его карт.
            self._reset_table_tracking()

        if self._round_destination in (None, "mine") or (
            self._round_destination == "opponent" and can_throw_to_taking_opponent
        ):
            # Подтверждаем исходные наблюдения параллельно переходу карты
            # соперника, а не начинаем второй цикл ожидания после него.
            visible = observed - self.hand_cards
            if visible and visible == self._table_candidate:
                self._table_candidate_frames += 1
            else:
                self._table_candidate = set(visible)
                self._table_candidate_frames = 1 if visible else 0
            self._table_candidate_layout = [dict(item) for item in self.field_layout]
            if self._table_candidate_frames >= self.field_confirmation_frames:
                # Заменяем снимок: исправленная масть/ранг не оставляет
                # в памяти розыгрыша старую ошибочную карту.
                self._confirmed_table = set(visible)
                self._confirmed_layout = [dict(item) for item in self.field_layout]
        if confirmed_override is not None:
            self._confirmed_table = set(confirmed_override)

        if self.field_cards:
            self._last_field_cards = set(self.field_cards)
            self._last_field_layout = [dict(item) for item in self.field_layout]
        self._table_was_empty = not observed
        self._round_cards.update(self.field_cards)
        if self._round_destination is None and len(destinations) == 1:
            self._round_destination = destinations.pop()

        if self._round_destination == "opponent" and can_throw_to_taking_opponent:
            # I take — объявление взятия, но кнопка Pass ещё разрешает подкинуть.
            # До окончания атаки карты остаются на столе и не входят в руку
            # соперника. Короткий пропуск детекции не должен завершать взятие.
            if not self.field_cards:
                self.field_cards = set(self._last_field_cards)
                self.field_layout = [dict(item) for item in self._last_field_layout]
            self.hand_cards.difference_update(self.field_cards)
            self.known_opponent_cards.difference_update(self.field_cards)
            return

        if self._round_destination is None:
            self._pending_hand_cards.difference_update(self.field_cards)
            self.hand_cards.difference_update(self.field_cards)
            self.known_opponent_cards.difference_update(self.field_cards)
            return

        moved = self._round_cards - self._transferred_cards
        if self._round_destination == "out":
            moved = self._confirmed_table - self.hand_cards - self.out_cards
            self.last_out_cards = set(moved)
            self._round_cards = set(moved)
            self.out_cards.update(moved)
            self._pending_hand_cards.difference_update(moved)
        elif self._round_destination == "mine":
            # Не переносим в руку накопленные ошибки детектора стола.
            moved = self._confirmed_table - self._transferred_cards - self.out_cards
            self._pending_hand_cards.update(moved - self.hand_cards)
        else:
            # Ошибочные варианты распознавания не становятся известной рукой.
            moved = self._confirmed_table - self._transferred_cards - self.hand_cards - self.out_cards
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
        self._confirmed_layout.clear()
        self._table_candidate.clear()
        self._table_candidate_layout.clear()
        self._table_candidate_frames = 0
        self._round_destination = None
        self._last_field_cards.clear()
        self._last_field_layout.clear()

    def _refresh_opponent(self) -> None:
        known = self.hand_cards | self.field_cards | self.out_cards | self.known_opponent_cards
        self.unknown_opponent_cards = set(FULL_DECK - known)
        if self.deck_remaining is None:
            self.opponent_card_count = None
        else:
            self.opponent_card_count = max(0, len(FULL_DECK) - self.deck_remaining - len(self.hand_cards) - len(self.field_cards) - len(self.out_cards))
