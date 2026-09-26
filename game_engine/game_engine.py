"""Адаптер DurakGameState/снимка бота к C++ ISMCTS. Игрок 0 — мы."""

from __future__ import annotations

from collections.abc import Mapping
import secrets
import math


def validate_search_options(*, iterations=3000, time_limit_ms=250, rollout_depth=256,
                            rollouts=None, deals=1, exploration=1.41421356237, threads=1):
    """Проверка до запуска фонового процесса или загрузки расширения."""
    for name, value, maximum in (("iterations", iterations, 10000000),
                                 ("rollout_depth", rollout_depth, 10000),
                                 ("deals", deals, 100000), ("threads", threads, 256)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{name} должен быть целым числом от 1 до {maximum}")
    if rollouts is not None:
        if isinstance(rollouts, bool) or not isinstance(rollouts, int) or not 1 <= rollouts <= 10000000:
            raise ValueError("rollouts должен быть целым числом от 1 до 10000000")
        if rollouts * deals > 10000000:
            raise ValueError("rollouts * deals не должно превышать 10000000")
    elif deals != 1:
        raise ValueError("deals требует задания rollouts")
    for name, value in (("time_limit_ms", time_limit_ms), ("exploration", exploration)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} должен быть конечным числом >= 0")


def _native_module():
    try:
        from . import _native
    except ImportError as exc:
        raise ImportError(
            "Сначала соберите движок тем же Python: "
            "python -m pip install --no-build-isolation ./game_engine"
        ) from exc
    return _native


def _get(state, key, default=None):
    return state.get(key, default) if isinstance(state, Mapping) else getattr(state, key, default)


def _text(value):
    return "".join(str(value or "").split()).lower()


def _phase(state):
    # Pass и Bat на кнопке — доступное действие, не уже произошедшее событие.
    button = _text(_get(state, "button_text", _get(state, "button", "")))
    return "throw_in" if button in ("pass", "bat") else _get(state, "phase")


def observation_from_state(
    state, *, trump: str, bottom_trump: str | None = None,
    attacker: int | None = None, turn: int | None = None,
    defender_start_count: int | None = None,
    taking: bool | None = None, attacker_passed: bool | None = None,
    simultaneous_winner: str = "attacker",
) -> dict:
    """Снимок только наблюдаемой информации; скрытые карты не угадываются здесь.

    Для your_turn/throw_in атакующий — мы, для defend_or_take — соперник.
    opponent_turn неоднозначен и требует явных attacker/turn. attack_limit
    восстанавливается из оставшейся руки защитника и числа уже отбитых карт;
    defender_start_count позволяет передать точное число в начале розыгрыша.
    """
    phase = _phase(state)
    if phase == "ready":
        raise ValueError("Игра ещё не началась")
    hand = sorted(_get(state, "hand_cards", ()))
    field = set(_get(state, "field_cards", ()))
    layout = _get(state, "field_layout", ())
    if field and not layout:
        raise ValueError("Для непустого стола необходим field_layout с парами карт")
    if any(item.get("card") is None for item in layout):
        raise ValueError("На столе есть карта с нераспознанной мастью/рангом")
    layout_cards = [item["card"] for item in layout]
    if set(layout_cards) != field or len(layout_cards) != len(field):
        raise ValueError("field_cards и field_layout не совпадают или содержат дубликаты")
    roots = [i for i, item in enumerate(layout) if item.get("covers") is None]
    root_to_pair = {index: n for n, index in enumerate(roots)}
    pairs = [{"attack": layout[i]["card"], "defense": None} for i in roots]
    for item in layout:
        parent = item.get("covers")
        if parent is None:
            continue
        if type(parent) is not int or parent not in root_to_pair:
            raise ValueError("Некорректная связь covers: требуется нижняя карта, не цепочка")
        pair = pairs[root_to_pair[parent]]
        if pair["defense"] is not None:
            raise ValueError("На одну карту распознано несколько защитных карт")
        pair["defense"] = item["card"]
    if attacker is None:
        attacker = {"your_turn": 0, "throw_in": 0, "defend_or_take": 1}.get(phase)
    if attacker not in (0, 1):
        raise ValueError("Неизвестен атакующий: передайте attacker=0 или 1")
    if turn is None:
        turn = 0 if phase in ("your_turn", "throw_in", "defend_or_take") else 1
    if phase == "your_turn" and pairs:
        raise ValueError("your_turn с непустым столом: дождитесь завершения анимации")
    if phase in ("throw_in", "defend_or_take") and not pairs:
        raise ValueError("Нет распознанного стола для текущей фазы")
    deck_count = _get(state, "deck_remaining")
    opponent_count = _get(state, "opponent_card_count")
    if type(deck_count) is not int or type(opponent_count) is not int:
        raise ValueError("Неизвестно количество карт колоды или соперника")
    known_opponent = sorted(_get(state, "known_opponent_cards", ()))
    if len(known_opponent) > opponent_count:
        raise ValueError(
            f"Противоречие распознавания: известных карт соперника {len(known_opponent)}, "
            f"всего по подсчёту {opponent_count}. Если не проходит — R для сброса партии"
        )
    mine = _text(_get(state, "mine_text", _get(state, "mine", "")))
    opponent = _text(_get(state, "opponent_text", _get(state, "opponent", "")))
    if mine == "bat" or opponent == "bat":
        raise ValueError("Идёт бита: дождитесь следующего розыгрыша")
    defender_text = opponent if attacker == 0 else mine
    attacker_text = mine if attacker == 0 else opponent
    if taking is None:
        taking = defender_text == "itake"
    if attacker_passed is None:
        attacker_passed = attacker_text == "pass"
    if defender_start_count is None:
        remaining = opponent_count if attacker == 0 else len(hand)
        defender_start_count = remaining + sum(p["defense"] is not None for p in pairs)
    if type(defender_start_count) is not int or defender_start_count < 0:
        raise ValueError("Некорректное число карт защитника в начале розыгрыша")
    return {
        "hand": hand,
        "known_opponent": known_opponent,
        "discard": sorted(_get(state, "out_cards", ())),
        "table": pairs,
        "deck_count": deck_count, "opponent_count": opponent_count,
        "trump": trump.upper(), "bottom_trump": bottom_trump.upper() if bottom_trump and deck_count else None,
        "attacker": attacker, "turn": turn,
        "attack_limit": min(6, defender_start_count),
        "taking": taking, "attacker_passed": attacker_passed,
        "simultaneous_winner": simultaneous_winner,
    }


class DurakEngine:
    """ISMCTS или MCTS по раскладам; не гарантирует оптимальный ход.

    rollouts=None сохраняет ISMCTS с общим бюджетом iterations.
    rollouts=N включает deals независимых деревьев, до N итераций на дерево.
    threads задаёт число C++ потоков, time_limit_ms — общий бюджет времени.
    """

    def __init__(self, trump: str | None = None, *, bottom_trump: str | None = None,
                 iterations: int = 3000, time_limit_ms: float = 250,
                 seed: int | None = None, rollout_depth: int = 256,
                 rollouts: int | None = None, deals: int = 1,
                 exploration: float = 1.41421356237, threads: int = 1,
                 simultaneous_winner: str = "attacker"):
        validate_search_options(iterations=iterations, time_limit_ms=time_limit_ms,
                                rollout_depth=rollout_depth, rollouts=rollouts, deals=deals,
                                exploration=exploration, threads=threads)
        if trump is not None and trump.upper() not in ("C", "D", "H", "S"):
            raise ValueError("Козырь: C (крести), D (бубны), H (червы), S (пики)")
        if simultaneous_winner not in ("attacker", "defender"):
            raise ValueError("simultaneous_winner: attacker или defender")
        self.trump, self.bottom_trump = trump.upper() if trump is not None else None, bottom_trump
        self.iterations, self.time_limit_ms = iterations, time_limit_ms
        self.seed, self.rollout_depth = seed, rollout_depth
        self.rollouts, self.deals = rollouts, deals
        self.exploration, self.threads = exploration, threads
        self.simultaneous_winner = simultaneous_winner
        self._native = _native_module()

    def suggest(self, state, **overrides) -> dict:
        phase = _phase(state)
        if phase in ("ready", "opponent_turn") and overrides.get("turn") != 0:
            return {"status": "waiting", "reason": "Игра ещё не началась" if phase == "ready" else "Ход соперника", "action": None}
        trump = self.trump or _get(state, "trump")
        if trump not in ("C", "D", "H", "S"):
            return {"status": "waiting", "reason": "Козырь ещё не распознан", "action": None}
        observation = observation_from_state(
            state, trump=trump, bottom_trump=self.bottom_trump,
            simultaneous_winner=self.simultaneous_winner, **overrides,
        )
        if observation["turn"] != 0:
            return {"status": "waiting", "reason": "Ход соперника", "action": None}
        result = self._native.analyze(
            observation, iterations=self.iterations, time_limit_ms=self.time_limit_ms,
            seed=self.seed if self.seed is not None else secrets.randbits(64),
            rollout_depth=self.rollout_depth,
            rollouts=self.rollouts if self.rollouts is not None else 0,
            deals=self.deals, exploration=self.exploration, threads=self.threads,
        )
        result["status"] = "ok"
        result["value_kind"] = "rollout_score"  # not a calibrated probability
        for action in [result["action"], *result["moves"]]:
            if action["type"] == "defend":
                action["target_card"] = observation["table"][action["target"]]["attack"]
            elif action["type"] == "pass":
                button = _text(_get(state, "button_text", _get(state, "button", "")))
                action["button"] = "Bat" if button == "bat" else "Pass"
        return result
