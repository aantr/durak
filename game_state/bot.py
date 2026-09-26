"""Получение экрана iPhone и обновление состояния игры.

Запуск из корня проекта::

    python -m game_state.bot --mac-ip 10.10.10.1 --fps 10
    python -m game_state.bot --draw-detections
    python -m game_state.bot --no-window
    python -m game_state.bot --suggest-moves --trump S

Esc/Q или Ctrl+C — выход. R — сброс состояния партии. Клик по окну передаётся на iPhone.
Подсказки включаются через --suggest-moves; Space рассчитывает ход, ↑ выполняет последний рассчитанный ход.
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
from game_state.engine_process import EngineProcess
from game_state.move_input import execute_move

if TYPE_CHECKING:
    from iphone_screen.iphone_client_v2 import IPhoneRemote
    from game_engine import DurakEngine


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
    phase_label = phases.get(snapshot["phase"], snapshot["phase"])
    if snapshot["phase"] == "throw_in" and snapshot["button"] in ("pass", "bat"):
        phase_label = "Подкинуть карты или нажать " + ("Bat" if snapshot["button"] == "bat" else "Pass")
    row("Ход:      " + phase_label)
    row("Козырь:   " + symbols.get(snapshot.get("trump"), "?"))
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
    recommendation = snapshot.get("recommendation")
    if recommendation:
        separator()
        move = recommendation if recommendation.get("status") == "ok" else recommendation.get("last_move")
        if move:
            action = move["action"]
            if action["type"] == "defend":
                description = f"Отбить {card(action['target_card'])} картой {card(action['card'])}"
            elif action["type"] == "attack":
                verb = "Подкинуть" if recommendation.get("status") == "ok" and snapshot["phase"] == "throw_in" else "Походить"
                description = f"{verb} {card(action['card'])}"
            elif action["type"] == "pass":
                button = action.get("button", "Bat" if snapshot["button"] == "bat" else "Pass")
                description = f"Нажать {button} — закончить подкидывание"
            else:
                description = "Взять карты"
            status = "Готово" if recommendation.get("status") == "ok" else recommendation["reason"]
            prefix = "MCTS: " if recommendation.get("status") == "ok" else "Предыдущий ход: "
            row(prefix + description + " | " + status)
            value = move["moves"][0]["value"]
            row(f"Оценка {value:.3f} · {move['iterations']} симуляций · {move['elapsed_ms']:.0f} мс")
            if "threads" in move:
                details = f"Потоки: {move['threads']} · exploration: {move['exploration']:.3g}"
                if move.get("search_mode") == "determinized":
                    details += (f" · расклады: {move['deals_completed']}/{move['deals_requested']}"
                                f" · rollouts: {move['rollouts']}")
                row(details)
        else:
            row("MCTS: " + recommendation["reason"])
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


def recommendation_overlay(recommendation: dict) -> str:
    status = recommendation.get("status")
    move = recommendation if status == "ok" else recommendation.get("last_move")
    label = {"ok": "Ready", "calculating": "Calculating...", "waiting": "Waiting",
             "idle": "Space: calculate", "invalid_state": "Waiting for valid state"}.get(status, str(status))
    if not move:
        return label
    action = move["action"]
    kind = action["type"]
    if kind == "defend":
        description = f"Defend {action['target_card']} with {action['card']}"
    elif kind == "attack":
        description = f"Play {action['card']}"
    elif kind == "pass":
        description = action.get("button", "Pass")
    else:
        description = "Take"
    prefix = "" if status == "ok" else "Previous: "
    return f"{prefix}{description} | {label}"


def state_snapshot(state: DurakGameState) -> dict:
    """Снимок для вывода; номер кадра не считается изменением игры."""
    return {
        "phase": state.phase,
        "trump": state.trump,
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
    engine: DurakEngine | None = None,
    engine_options: dict | None = None,
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

    if engine is not None and engine_options is not None:
        raise ValueError("Передайте engine или engine_options, не оба параметра")
    if engine is not None:
        engine_options = {name: getattr(engine, name) for name in (
            "trump", "bottom_trump", "iterations", "time_limit_ms", "seed",
            "rollout_depth", "simultaneous_winner", "rollouts", "deals", "exploration", "threads",
        )}
    engine_process = EngineProcess(engine_options) if engine_options is not None else None
    initial_trump = state.trump

    def update_state(frame):
        if draw_detections:
            state.update(frame, draw_detections=True)
        else:
            state.update(frame)

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
    current_snapshot = None
    recommendation = None
    latest_frame = None
    annotated_frame = None
    next_update = 0.0
    last_timeout_log = float("-inf")
    window_created = False
    reset_requested = False
    geometry = None
    move_pending: Future | None = None
    executed_snapshot = None
    executed_move = None

    try:
        if show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(WINDOW_NAME, 460, 900)
            cv2.setMouseCallback(WINDOW_NAME, mouse)

        logger.info("Ожидание видео iPhone; частота распознавания до %g FPS", fps)
        first_frame = True
        while True:
            if move_pending is not None and move_pending.done():
                try:
                    move_pending.result()
                except Exception:
                    logger.exception("Не удалось выполнить ход на iPhone")
                    executed_snapshot = None
                    executed_move = None
                move_pending = None
            # Только этот worker изменяет state. Читаем его после завершения
            # Future и до постановки следующего кадра, чтобы избежать гонок.
            if pending is not None and pending.done():
                pending.result()  # Ошибки вычислений не скрываем.
                pending = None
                if not reset_requested:
                    if draw_detections:
                        annotated_frame = state.annotated_frame
                    current_snapshot = state_snapshot(state)
                    geometry = {
                        "frame_size": state.frame_size,
                        "hand_layout": [dict(item) for item in state.hand_layout],
                        "field_layout": [dict(item) for item in state.field_layout],
                    }
            if reset_requested and pending is None:
                state.reset(trump=initial_trump)
                reset_requested = False
                next_update = 0.0
                logger.info("Состояние партии сброшено")
            if current_snapshot is not None:
                if engine_process is not None:
                    recommendation = engine_process.poll()
                snapshot = dict(current_snapshot)
                if recommendation is not None:
                    snapshot["recommendation"] = recommendation
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
                latest_frame = frame
                now = time.monotonic()
                if pending is None and now >= next_update:
                    # Передаём отдельный BGR-кадр; отображение и декодер
                    # не могут изменить изображение во время распознавания.
                    pending = worker.submit(update_state, frame.copy())
                    next_update = now + 1.0 / fps

            if show_window:
                display = annotated_frame if draw_detections and annotated_frame is not None else latest_frame
                if display is not None:
                    if recommendation is not None:
                        display = display.copy()
                        from game_state.visualization import draw_label
                        label = recommendation_overlay(recommendation)
                        if recommendation.get("status") == "ok" or recommendation.get("last_move"):
                            label += " | Up: play | Space: recalculate"
                        draw_label(display, label, 20, 60, (0, 255, 255))
                    cv2.imshow(WINDOW_NAME, display)

            # Обрабатываем клавиатуру и при таймаутах видео.
            if show_window:
                key = cv2.waitKeyEx(1)
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    reset_requested = True
                    current_snapshot = previous_snapshot = None
                    recommendation = annotated_frame = None
                    geometry = executed_snapshot = None
                    executed_move = None
                    if engine_process is not None:
                        engine_process.reset()
                if key == ord(" "):
                    if engine_process is None:
                        logger.info("Для расчёта ходов включите --suggest-moves")
                    elif current_snapshot is None or reset_requested:
                        logger.info("Дождитесь распознавания состояния")
                    elif engine_process.request(current_snapshot):
                        recommendation = engine_process.poll()
                        logger.info("Расчёт хода запущен по Space")
                    else:
                        logger.info("Расчёт ещё выполняется")
                # Полные коды Up: Qt, GTK/X11, Windows и Cocoa.
                if key in (16777235, 65362, 2490368, 63232):
                    move = None if recommendation is None else (
                        recommendation if recommendation.get("status") == "ok" else recommendation.get("last_move"))
                    if move_pending is not None:
                        logger.info("Предыдущий ввод ещё выполняется")
                    elif move is None:
                        logger.info("Сначала рассчитайте ход пробелом")
                    elif move is executed_move or (current_snapshot is not None and current_snapshot == executed_snapshot):
                        logger.info("Ход уже отправлен; ожидается изменение состояния")
                    elif geometry is None or current_snapshot is None or reset_requested:
                        logger.info("Дождитесь распознавания состояния")
                    else:
                        try:
                            if latest_frame is None or geometry["frame_size"] != (latest_frame.shape[1], latest_frame.shape[0]):
                                raise ValueError("Размер кадра изменился; дождитесь распознавания")
                            move_pending = execute_move(iphone, move, current_snapshot, geometry)
                            executed_snapshot = current_snapshot
                            executed_move = move
                            logger.info("Ход отправлен: %s", recommendation_overlay(move))
                        except ValueError as exc:
                            logger.info("Ход не отправлен: %s", exc)
                        except Exception:
                            logger.exception("Не удалось поставить ход в очередь")
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
            try:
                worker.shutdown(wait=True, cancel_futures=True)
            finally:
                if engine_process is not None:
                    engine_process.close()

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
    parser.add_argument("--suggest-moves", action="store_true", help="Подсказки C++ MCTS без автоматических ходов")
    parser.add_argument("--trump", choices=("C", "D", "H", "S"), help="Задать козырь вручную вместо распознавания: C=крести, D=бубны, H=червы, S=пики")
    parser.add_argument("--trump-card", help="Известная нижняя карта колоды, например 6S")
    parser.add_argument("--mcts-ms", type=float, default=250, help="Бюджет поиска в мс (0 — только лимит итераций)")
    parser.add_argument("--mcts-iterations", type=int, default=3000, help="Максимум симуляций MCTS на состояние")
    parser.add_argument("--mcts-rollouts", type=int, help="Включить поиск по раскладам: итераций на каждый расклад (заменяет --mcts-iterations)")
    parser.add_argument("--mcts-deals", type=int, default=1, help="Число раскладов скрытых карт; требует --mcts-rollouts")
    parser.add_argument("--mcts-exploration", type=float, default=1.41421356237, help="Коэффициент исследования UCT (>= 0)")
    parser.add_argument("--mcts-threads", type=int, default=1, help="Число потоков C++ поиска (1–256)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or not 5 <= args.fps <= 60:
        parser.error("--fps должен быть в диапазоне от 5 до 60")
    from game_engine.game_engine import validate_search_options
    try:
        validate_search_options(iterations=args.mcts_iterations, time_limit_ms=args.mcts_ms,
                                rollouts=args.mcts_rollouts, deals=args.mcts_deals,
                                exploration=args.mcts_exploration, threads=args.mcts_threads)
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        engine_options = None
        if args.suggest_moves:
            engine_options = dict(trump=args.trump, bottom_trump=args.trump_card,
                                  iterations=args.mcts_iterations, time_limit_ms=args.mcts_ms,
                                  rollouts=args.mcts_rollouts, deals=args.mcts_deals,
                                  exploration=args.mcts_exploration, threads=args.mcts_threads)
        from iphone_screen.iphone_client_v2 import IPhoneRemote

        with IPhoneRemote(
            mac_ip=args.mac_ip,
            control_port=args.control_port,
            video_port=args.video_port,
        ) as iphone:
            run_bot(iphone, state=DurakGameState(trump=args.trump), fps=args.fps, show_window=not args.no_window, draw_detections=args.draw_detections,
                    state_format=args.state_format, engine_options=engine_options)
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
    except Exception:
        logger.exception("Работа бота прервана")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
