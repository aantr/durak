"""Графическая таблица состояния игры из изображений обучающих датасетов."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from game_state.game import RANKS, SUITS


WIDTH = 1160
CARD_WIDTH = 60
CARD_HEIGHT = 83
CARD_GAP = 6
PANEL_WIDTH = 556
SUIT_FILES = {
    "C": ("clubs", "kresti.png"),
    "D": ("diamonds", "bubi.png"),
    "H": ("hearts", "heart.png"),
    "S": ("spades", "piki.png"),
}
SUIT_SYMBOLS = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
PHASES = {
    "throw_in": "Подкинуть карты сопернику",
    "your_turn": "Ваш ход",
    "ready": "Можно начать игру",
    "defend_or_take": "Отбиваться или взять",
    "opponent_turn": "Ход соперника",
}
ACTIONS = {
    "pass": "Pass",
    "bat": "Bat",
    "itake": "I take",
}


def _card_sort_key(code: str) -> tuple[int, int, str]:
    if not isinstance(code, str) or not code:
        return (len(SUITS), len(RANKS), str(code))
    rank, suit = code[:-1], code[-1]
    return (SUITS.index(suit) if suit in SUITS else len(SUITS),
            RANKS.index(rank) if rank in RANKS else len(RANKS), code)


def _card_text(code: str | None) -> str:
    if not code:
        return "?"
    return code[:-1] + SUIT_SYMBOLS.get(code[-1], code[-1])


class StateWindowRenderer:
    """Рисует BGR-кадр для отдельного окна OpenCV; PNG и шрифты кешируются."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        self.font = self._font(18)
        self.small_font = self._font(14)
        self.title_font = self._font(25)
        self.card_font = self._font(12)
        self._ranks: dict[str, Image.Image | None] = {}
        self._suits: dict[str, Image.Image | None] = {}
        self._cards: dict[str, Image.Image] = {}

    @staticmethod
    def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _read(path: Path) -> Image.Image | None:
        try:
            with Image.open(path) as source:
                return source.convert("RGBA")
        except (OSError, ValueError):
            return None

    def _rank(self, rank: str, suit: str) -> Image.Image | None:
        name = rank + ("R" if suit in "DH" else "B")
        if name not in self._ranks:
            self._ranks[name] = self._read(self.root / "dataset_field" / name / f"{name}.png")
        return self._ranks[name]

    def _suit(self, suit: str) -> Image.Image | None:
        if suit not in self._suits:
            parts = SUIT_FILES.get(suit)
            self._suits[suit] = (self._read(self.root / "dataset_suit" / parts[0] / parts[1])
                                 if parts else None)
        return self._suits[suit]

    @staticmethod
    def _paste_contained(target: Image.Image, source: Image.Image | None,
                         box: tuple[int, int, int, int]) -> None:
        if source is None:
            return
        x, y, width, height = box
        resized = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
        target.alpha_composite(resized, (x + (width - resized.width) // 2,
                                         y + (height - resized.height) // 2))

    def card(self, code: str | None) -> Image.Image:
        key = code or "?"
        if key in self._cards:
            return self._cards[key]
        tile = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)
        draw.rounded_rectangle((0, 0, CARD_WIDTH - 1, CARD_HEIGHT - 1), radius=6,
                               fill=(250, 250, 245, 255), outline=(176, 184, 184, 255), width=1)
        if code and code[-1] in SUITS and code[:-1] in RANKS:
            self._paste_contained(tile, self._rank(code[:-1], code[-1]), (5, 4, 23, 52))
            self._paste_contained(tile, self._suit(code[-1]), (31, 9, 24, 29))
        label = _card_text(code)
        color = (180, 39, 57) if code and code[-1] in "DH" else (25, 36, 45)
        ImageDraw.Draw(tile).text((CARD_WIDTH // 2, 65), label, font=self.card_font,
                                  fill=color, anchor="mm")
        self._cards[key] = tile
        return tile

    @staticmethod
    def _fit(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont,
             width: int) -> str:
        if draw.textlength(text, font=font) <= width:
            return text
        while text and draw.textlength(text + "…", font=font) > width:
            text = text[:-1]
        return text + "…"

    def _text(self, draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
              *, width: int, font: ImageFont.ImageFont | None = None,
              fill: str = "#E9F2F3") -> None:
        font = font or self.font
        draw.text(xy, self._fit(draw, text, font, width), font=font, fill=fill)

    def _zone(self, image: Image.Image, x: int, y: int, title: str,
              cards: list[str], *, note: str = "") -> int:
        cards = sorted(cards, key=_card_sort_key)
        columns = 8
        rows = (len(cards) + columns - 1) // columns
        height = 37 + max(1, rows) * (CARD_HEIGHT + CARD_GAP) + (23 if note else 0) + 8
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((x, y, x + PANEL_WIDTH, y + height), radius=11,
                               fill="#1B2E3B", outline="#38515E", width=2)
        self._text(draw, (x + 14, y + 8), f"{title}  [{len(cards)}]",
                   width=PANEL_WIDTH - 28, font=self.font)
        if cards:
            for index, code in enumerate(cards):
                cx = x + 14 + (index % columns) * (CARD_WIDTH + CARD_GAP)
                cy = y + 37 + (index // columns) * (CARD_HEIGHT + CARD_GAP)
                tile = self.card(code)
                image.paste(tile, (cx, cy), tile)
        else:
            self._text(draw, (x + 16, y + 53), "Нет карт", width=PANEL_WIDTH - 32,
                       font=self.small_font, fill="#A8BDC5")
        if note:
            self._text(draw, (x + 14, y + height - 28), note,
                       width=PANEL_WIDTH - 28, font=self.small_font, fill="#C4D5D8")
        return y + height + 10

    @staticmethod
    def _field_note(layout: list[Mapping[str, Any]]) -> str:
        pairs = []
        for item in layout:
            parent = item.get("covers")
            if isinstance(parent, int) and 0 <= parent < len(layout):
                pairs.append(f"{_card_text(layout[parent].get('card'))} → {_card_text(item.get('card'))}")
        return "Накрытия: " + "  ·  ".join(pairs) if pairs else ""

    @staticmethod
    def _move_text(recommendation: Mapping[str, Any] | None) -> str:
        if not recommendation:
            return "Ход: расчёт не запускался"
        move = recommendation if recommendation.get("status") == "ok" else recommendation.get("last_move")
        if not move:
            return "Ход: " + str(recommendation.get("reason") or recommendation.get("status") or "—")
        action = move.get("action", {})
        kind = action.get("type")
        if kind == "defend":
            description = f"Отбить {_card_text(action.get('target_card'))} картой {_card_text(action.get('card'))}"
        elif kind == "attack":
            description = f"Сыграть {_card_text(action.get('card'))}"
        elif kind == "pass":
            description = "Нажать " + str(action.get("button", "Pass"))
        else:
            description = "Взять карты"
        prefix = "Рекомендация" if recommendation.get("status") == "ok" else "Предыдущий ход"
        details = []
        if move.get("iterations") is not None:
            details.append(f"{move['iterations']} симуляций")
        if move.get("elapsed_ms") is not None:
            details.append(f"{move['elapsed_ms']:.0f} мс")
        suffix = " · " + " · ".join(details) if details else ""
        return f"{prefix}: {description}{suffix}"

    def render(self, snapshot: Mapping[str, Any] | None, *,
               auto_enabled: bool = False, auto_delay: float = 1.0,
               auto_status: str = "") -> np.ndarray:
        """Возвращает готовую таблицу в формате BGR для cv2.imshow."""
        snapshot = snapshot or {}
        left = [
            ("Ваша рука", list(snapshot.get("hand_cards", [])), ""),
            ("Стол", list(snapshot.get("field_cards", [])),
             self._field_note(snapshot.get("field_layout", []))),
            ("Известные карты соперника", list(snapshot.get("known_opponent_cards", [])), ""),
            ("Последняя бита", list(snapshot.get("last_out_cards", [])), ""),
        ]
        right = [
            ("Бита", list(snapshot.get("out_cards", [])), ""),
            ("Неустановленные карты · колода / соперник",
             list(snapshot.get("unknown_opponent_cards", [])), ""),
        ]
        def height(zones):
            return 170 + sum(37 + max(1, (len(cards) + 7) // 8) * (CARD_HEIGHT + CARD_GAP)
                             + (23 if note else 0) + 18 for _, cards, note in zones)
        canvas_height = max(770, height(left), height(right))
        image = Image.new("RGB", (WIDTH, canvas_height), "#10202C")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, WIDTH, 160), fill="#172C39")
        self._text(draw, (18, 11), "ДУРАК · СОСТОЯНИЕ ИГРЫ", width=530,
                   font=self.title_font, fill="#F0F7F6")
        mode = "АВТО ВКЛ" if auto_enabled else "АВТО ВЫКЛ"
        self._text(draw, (687, 19), f"{mode} · Enter · задержка {auto_delay:g} с",
                   width=456, font=self.small_font, fill="#74DFCA")
        self._text(draw, (18, 47),
                   "Ход: " + PHASES.get(snapshot.get("phase"), snapshot.get("phase", "Ожидание первого кадра")),
                   width=660)
        self._text(draw, (687, 48), f"Колода: {snapshot.get('deck_remaining') if snapshot.get('deck_remaining') is not None else '?'}"
                   f"   ·   Соперник: {snapshot.get('opponent_card_count') if snapshot.get('opponent_card_count') is not None else '?'}",
                   width=456)
        self._text(draw, (18, 78),
                   "Кнопка: " + ACTIONS.get(snapshot.get("button"), snapshot.get("button") or "—")
                   + "   ·   Вы: " + ACTIONS.get(snapshot.get("mine"), snapshot.get("mine") or "—")
                   + "   ·   Соперник: " + ACTIONS.get(snapshot.get("opponent"), snapshot.get("opponent") or "—"),
                   width=880, font=self.small_font)
        trump = snapshot.get("trump")
        self._text(draw, (978, 78), "Козырь:", width=97, font=self.small_font)
        if trump in SUITS:
            suit = self._suit(trump)
            if suit is not None:
                # На RGB-холст вставляем прозрачный PNG с его альфа-каналом.
                icon = ImageOps.contain(suit, (30, 30), Image.Resampling.LANCZOS)
                image.paste(icon, (1076, 73), icon)
            self._text(ImageDraw.Draw(image), (1111, 78), SUIT_SYMBOLS[trump], width=35,
                       font=self.small_font)
        else:
            self._text(draw, (1076, 78), "?", width=35, font=self.small_font)
        evaluation = snapshot.get("last_evaluation")
        value = f"{evaluation:.3f}" if isinstance(evaluation, (int, float)) else "—"
        self._text(draw, (18, 108), "Последняя оценка игры: " + value,
                   width=420, font=self.small_font, fill="#F7DA83")
        self._text(draw, (440, 108), self._move_text(snapshot.get("recommendation")),
                   width=702, font=self.small_font, fill="#F7DA83")
        status = auto_status or "Space: расчёт · ↑: сыграть · R: сброс · Q/Esc: выход"
        self._text(draw, (18, 133), status, width=1120, font=self.small_font,
                   fill="#A9C5CC")
        y = 170
        for title, cards, note in left:
            y = self._zone(image, 14, y, title, cards, note=note)
        y = 170
        for title, cards, note in right:
            y = self._zone(image, 590, y, title, cards, note=note)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
