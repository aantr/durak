"""Получение экрана iPhone и обновление состояния игры.

Запуск из корня проекта::

    python -m game_state.bot --mac-ip 10.10.10.1 --fps 10
    python -m game_state.bot --draw-detections
    python -m game_state.bot --no-window

Esc/Q или Ctrl+C — выход. Клик по окну передаётся на iPhone.
Автоматический выбор и выполнение ходов здесь пока не реализованы.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap
import time
from typing import TYPE_CHECKING

import cv2

# Поддерживаем также запуск python game_state/bot.py из IDE.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game_state.game import DurakGameState, RANKS, SUITS

if TYPE_CHECKING:
    from iphone_screen.iphone_client_v2 import IPhoneRemote


MAC_IP = "10.10.10.1"
WINDOW_NAME = "Durak - iPhone"
logger = logging.getLogger(__name__)


def format_state(snapshot: dict, *, width: int = 80, color: bool = False) -> str:
    """Читаемая панель из снимка, без обращения к изменяемому состоянию.

    Перенос строк выполняется до добавления ANSI-цветов. Без color результат
    подходит для файла или обычной консоли и не содержит управляющих кодов.
    """
    width = max(40, width)
    content_width = width - 4
    symbols = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
    phases = {
        "throw_in": "Подкинуть карты сопернику",
        "your_turn": "Ваш ход",
        "ready": "Можно начать игру",
        "defend_or_take": "Отбиваться или взять",
        "opponent_turn": "Ход соперника",
    }
    opponent_actions = {
        "pass": "Больше не подкидывает",
        "bat": "Карты уходят в биту",
        "itake": "Берёт карты со стола",
    }
    mine_actions = {
        "pass": "Больше не подкидываю",
        "bat": "Карты уходят в биту",
        "itake": "Беру карты со стола",
    }

    def clean(value):
        return "".join(char if char.isprintable() else " " for char in str(value))

    def card(code):
        if not code:
            return "?"
        code = clean(code)
        return code[:-1] + symbols.get(code[-1], code[-1])

    def card_key(code):
        rank, suit = code[:-1], code[-1:]
        return (SUITS.index(suit) if suit in SUITS else len(SUITS),
                RANKS.index(rank) if rank in RANKS else len(RANKS), code)

    def cards(codes):
        return "  ".join(card(code) for code in sorted(codes, key=card_key)) or "—"

    def count(value):
        return "?" if value is None else str(value)

    lines = ["╭" + "─" * (width - 2) + "╮"]

    def row(text=""):
        wrapped = textwrap.wrap(clean(text), width=content_width, subsequent_indent="  ") or [""]
        lines.extend("│ " + line.ljust(content_width) + " │" for line in wrapped)

    def separator():
        lines.append("├" + "─" * (width - 2) + "┤")

    row("ДУРАК · СОСТОЯНИЕ ИГРЫ")
    separator()
    row("Ход:      " + phases.get(snapshot["phase"], snapshot["phase"]))
    row(f"Колода:   {count(snapshot['deck_remaining'])} карт    ·    Соперник: {count(snapshot['opponent_card_count'])} карт")
    row("Кнопка:   " + (snapshot["button"] or "—"))
    opponent = snapshot["opponent"]
    row("Соперник: " + opponent_actions.get(opponent, opponent or "—"))
    mine = snapshot.get("mine", "")
    row("Вы:       " + mine_actions.get(mine, mine or "—"))
    separator()
    row(f"Ваша рука [{len(snapshot['hand_cards'])}]:  " + cards(snapshot["hand_cards"]))
    separator()
    row(f"Стол [{len(snapshot['field_cards'])} распознано] · нижняя → накрывающая")
    layout = snapshot.get("field_layout", [])
    if layout:
        parents = {item["covers"] for item in layout if isinstance(item["covers"], int) and 0 <= item["covers"] < len(layout)}
        for index, item in enumerate(layout):
            parent = item["covers"]
            if isinstance(parent, int) and 0 <= parent < len(layout):
                row(f"  {card(layout[parent]['card'])} → {card(item['card'])}")
            elif index not in parents:
                row(f"  {card(item['card'])} → …")
    else:
        row("  " + cards(snapshot["field_cards"]))
    separator()
    row(f"Известные карты соперника [{len(snapshot['known_opponent_cards'])}]:  " + cards(snapshot["known_opponent_cards"]))
    row(f"Бита [{len(snapshot['out_cards'])}]:  " + cards(snapshot["out_cards"]))
    separator()
    unknown = snapshot["unknown_opponent_cards"]
    row(f"Неустановленные карты [{len(unknown)}] · колода / соперник")
    if unknown:
        for suit in SUITS:
            group = [code for code in unknown if code.endswith(suit)]
            if group:
                row("  " + cards(group))
    else:
        row("  —")
    lines.append("╰" + "─" * (width - 2) + "╯")
    panel = "\n".join(lines)
    if color:
        panel = re.sub(r"(?:10|[6-9JQKA])[♥♦]", lambda m: f"\033[91m{m.group()}\033[0m", panel)
        panel = re.sub(r"(?:10|[6-9JQKA])[♣♠]", lambda m: f"\033[97m{m.group()}\033[0m", panel)
        panel = panel.replace("ДУРАК · СОСТОЯНИЕ ИГРЫ", "\033[1;36mДУРАК · СОСТОЯНИЕ ИГРЫ\033[0m")
    return panel


def state_snapshot(state: DurakGameState) -> dict:
    """Снимок для вывода; номер кадра не считается изменением игры."""
    return {
        "phase": state.phase,
        "button": state.button_text,
        "opponent": state.opponent_text,
        "mine": state.mine_text,
        "deck_remaining": state.deck_remaining,
        "hand_cards": sorted(state.hand_cards),
        "field_cards": sorted(state.field_cards),
        "field_layout": [{"card": card["card"], "covers": card["covers"]} for card in state.field_layout],
        "out_cards": sorted(state.out_cards),
        "opponent_card_count": state.opponent_card_count,
        "known_opponent_cards": sorted(state.known_opponent_cards),
        # Это кандидаты: часть этих карт может ещё находиться в колоде.
        "unknown_opponent_cards": sorted(state.unknown_opponent_cards),
    }


def run_bot(
    iphone: IPhoneRemote,
    state: DurakGameState | None = None,
    *,
    fps: float = 10.0,
    show_window: bool = True,
    draw_detections: bool = False,
    state_format: str = "pretty",
) -> DurakGameState:
    """Читает новые кадры до Esc/Q, закрытия окна или Ctrl+C.

    ``fps`` — верхняя граница частоты распознавания, а не гарантия скорости
    моделей. Пока выполняется update(), входящие кадры только отображаются.
    Очередь распознавания не накапливается. Переданный клиент закрывает
    вызывающий код; run_bot освобождает своё окно и поток распознавания.
    ``draw_detections`` показывает последний распознанный кадр с разметкой
    всех областей; он обновляется с фактической частотой распознавания.
    ``state_format``: pretty — панель на русском, json — исходный JSON.
    """
    if not math.isfinite(fps) or not 5 <= fps <= 60:
        raise ValueError("fps должен быть в диапазоне от 5 до 60")
    if state_format not in ("pretty", "json"):
        raise ValueError("state_format должен быть pretty или json")
    if state is None:
        state = DurakGameState()

    def tap_done(future: Future) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("Не удалось отправить нажатие на iPhone")

    def mouse(event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                iphone.send_tap_async(x, y).add_done_callback(tap_done)
            except Exception:
                logger.exception("Не удалось поставить нажатие в очередь")

    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="durak-state")
    pending: Future | None = None
    previous_snapshot = None
    annotated_frame = None
    next_update = 0.0
    last_timeout_log = float("-inf")
    window_created = False
    try:
        if show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(WINDOW_NAME, 460, 900)
            cv2.setMouseCallback(WINDOW_NAME, mouse)

        logger.info("Ожидание видео iPhone; частота распознавания до %g FPS", fps)
        first_frame = True
        while True:
            # Только этот worker изменяет state. Читаем его после завершения
            # Future и до постановки следующего кадра, чтобы избежать гонок.
            if pending is not None and pending.done():
                pending.result()  # Ошибка модели должна быть видна вызывающему коду.
                pending = None
                if draw_detections:
                    annotated_frame = state.annotated_frame
                    if show_window and annotated_frame is not None:
                        cv2.imshow(WINDOW_NAME, annotated_frame)
                snapshot = state_snapshot(state)
                if snapshot != previous_snapshot:
                    if state_format == "json":
                        logger.info("Состояние: %s", json.dumps(snapshot, ensure_ascii=False))
                    else:
                        use_color = sys.stderr.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
                        width = min(90, shutil.get_terminal_size(fallback=(80, 24)).columns)
                        logger.info("\n%s", format_state(snapshot, width=width, color=use_color))
                    previous_snapshot = snapshot

            try:
                frame = iphone.get_screen(
                    wait_new=not first_frame,
                    timeout=0.1,
                    copy=False,
                )
            except TimeoutError as exc:
                now = time.monotonic()
                if now - last_timeout_log >= 5.0:
                    logger.warning("Ожидание кадра: %s", exc)
                    last_timeout_log = now
            else:
                first_frame = False
                now = time.monotonic()
                if pending is None and now >= next_update:
                    # Передаём отдельный BGR-кадр; отображение и декодер
                    # не могут изменить изображение во время распознавания.
                    if draw_detections:
                        pending = worker.submit(state.update, frame.copy(), draw_detections=True)
                    else:
                        pending = worker.submit(state.update, frame.copy())
                    next_update = now + 1.0 / fps
                if show_window and (not draw_detections or annotated_frame is None):
                    cv2.imshow(WINDOW_NAME, frame)

            # Обрабатываем клавиатуру и при таймаутах видео.
            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
    finally:
        try:
            if window_created:
                cv2.destroyWindow(WINDOW_NAME)
        finally:
            # Активный update завершается до возврата объекта state.
            worker.shutdown(wait=True, cancel_futures=True)

    # Не теряем ошибку распознавания, завершившегося одновременно с выходом.
    if pending is not None and not pending.cancelled():
        pending.result()
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-ip", default=MAC_IP, help="IP Mac с DeviceKit bridge")
    parser.add_argument("--control-port", type=int, default=22004)
    parser.add_argument("--video-port", type=int, default=22005)
    parser.add_argument("--fps", type=float, default=30.0, help="Максимальная частота распознавания (5–60)")
    parser.add_argument("--no-window", action="store_true", help="Только вывод состояния в консоль")
    parser.add_argument("--draw-detections", action="store_true", help="Вставлять размеченные кропы всех детекторов обратно в кадр")
    parser.add_argument("--state-format", choices=("pretty", "json"), default="pretty", help="Формат состояния в консоли (по умолчанию pretty)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or not 5 <= args.fps <= 60:
        parser.error("--fps должен быть в диапазоне от 5 до 60")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from iphone_screen.iphone_client_v2 import IPhoneRemote

        with IPhoneRemote(
            mac_ip=args.mac_ip,
            control_port=args.control_port,
            video_port=args.video_port,
        ) as iphone:
            run_bot(iphone, fps=args.fps, show_window=not args.no_window, draw_detections=args.draw_detections, state_format=args.state_format)
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
    except Exception:
        logger.exception("Работа бота прервана")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
