"""Генерация JSON-партий существующим C++ движком и ансамблем стратегий.

python generate_games_dataset.py --games 100 --strategies random:1 heuristic:2 greedy:1 mcts:4
python generate_games_dataset.py \
  --games 1000 \
  --seed 10000 \
  --workers 4 \
  --iterations 3000 \
  --rollout-depth 256 \
  --strategies random:1 heuristic:2 greedy:1 mcts:4
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import random

from game_engine.game_engine import _native_module, validate_search_options

SCHEMA_VERSION = 1
STRATEGIES = ("random", "heuristic", "greedy", "mcts")
RANKS = ("6", "7", "8", "9", "10", "J", "Q", "K", "A")


def action_key(action):
    return action["type"], action.get("card"), action.get("target")


def softmax(values, temperature=1.0):
    peak = max(values)
    weights = [math.exp((v - peak) / temperature) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def parse_strategies(items):
    weights = {}
    for item in items:
        name, separator, raw = item.partition(":")
        weight = float(raw) if separator else 1.0
        if name not in STRATEGIES or name in weights:
            raise ValueError(f"Стратегии без повторений: {', '.join(STRATEGIES)}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Вес стратегии должен быть конечным и > 0")
        weights[name] = weight
    if not weights or not math.isfinite(sum(weights.values())):
        raise ValueError("Нужна хотя бы одна стратегия с конечным суммарным весом")
    return weights


class PublicMemory:
    """Только публично раскрытые карты: сыгранные/взятые и нижний козырь."""

    def __init__(self, history_size=32):
        self.known = [set(), set()]
        self.history = []
        self.history_size = history_size

    def update(self, before, action, after):
        player = before["turn"]
        if action.get("card"):
            self.known[player].discard(action["card"])
        round_ended = bool(before["table"]) and not after["table"]
        if round_ended:
            if before["taking"] or action["type"] == "take":
                for pair in before["table"]:
                    self.known[1 - before["attacker"]].update(
                        card for card in pair.values() if card is not None
                    )
            # Получатель открытой последней карты выводится из публичного порядка добора.
            if before["deck"] and not after["deck"]:
                attacker = before["attacker"]
                attacker_draws = min(len(before["deck"]), max(0, 6 - len(before["hands"][attacker])))
                owner = attacker if attacker_draws == len(before["deck"]) else 1 - attacker
                self.known[owner].add(before["deck"][-1])
        target = action.get("target")
        self.history.append({"player_id": player, **action,
                             "target_card": before["table"][target]["attack"]
                             if target is not None else None})
        self.history = self.history[-self.history_size:] if self.history_size else []

    def observation(self, full, player):
        """Тот же формат, что observation_from_state(); игрок-наблюдатель всегда 0."""
        opponent = 1 - player
        history = [{**event, "player_id": int(event["player_id"] != player)}
                   for event in self.history]
        return {
            "hand": sorted(full["hands"][player]),
            "known_opponent": sorted(self.known[opponent]),
            "opponent_count": len(full["hands"][opponent]),
            "deck_count": len(full["deck"]),
            "discard": sorted(full["discard"]),
            "table": [dict(pair) for pair in full["table"]],
            "trump": full["trump"],
            "bottom_trump": full["deck"][-1] if full["deck"] else None,
            "attacker": int(full["attacker"] != player),
            "turn": int(full["turn"] != player),
            "attack_limit": full["attack_limit"],
            "taking": full["taking"], "attacker_passed": full["attacker_passed"],
            "simultaneous_winner": full["simultaneous_winner"],
            "history": history,
        }


def strategy_scores(name, state, legal, *, seed, iterations, rollout_depth):
    """Сырые оценки имеют разный масштаб; ансамбль объединяет распределения."""
    if name == "random":
        scores = [0.0] * len(legal)
    elif name in ("heuristic", "greedy"):
        rank_counts = Counter(card[:-1] for card in state["hand"])
        scores = []
        for action in legal:
            card = action.get("card")
            if card is None:
                scores.append(-5.0 if action["type"] == "take" else -4.0)
                continue
            rank = RANKS.index(card[:-1])
            trump = card[-1] == state["trump"]
            if name == "greedy":
                # Выбрасывает старшие карты, оставляя дешёвые для последующих ходов.
                score = 1.0 + rank / 8.0 - float(trump)
            else:
                # Бережёт козыри, использует дешёвые карты, начинает с пар рангов.
                score = -(rank + 12 * trump) / 4.0
                if action["type"] == "attack":
                    score += 0.3 * (rank_counts[card[:-1]] - 1)
            scores.append(score)
    elif name == "mcts":
        result = _native_module().analyze(
            state, iterations=max(iterations, len(legal)), time_limit_ms=0,
            seed=seed, rollout_depth=rollout_depth,
        )
        by_action = {action_key(move): move for move in result["moves"]}
        moves = [by_action[action_key(action)] for action in legal]
        # При достаточном числе итераций все корневые действия посещены.
        if any(move["value"] is None for move in moves):
            raise RuntimeError("MCTS не оценил все допустимые действия")
        scores = [move["value"] for move in moves]
        return {"scores": scores, "probabilities": softmax(scores, 0.15),
                "visits": [move["visits"] for move in moves],
                "value_kind": "rollout_score", "iterations": result["iterations"]}
    else:
        raise ValueError(f"Неизвестная стратегия: {name}")
    return {"scores": scores, "probabilities": softmax(scores), "value_kind": "heuristic"}


def ensemble(state, legal, weights, rng, *, iterations=128, rollout_depth=128):
    evaluations = {
        name: strategy_scores(name, state, legal, seed=rng.getrandbits(64),
                              iterations=iterations, rollout_depth=rollout_depth)
        for name in weights
    }
    total = sum(weights.values())
    probabilities = [sum(weights[name] * evaluations[name]["probabilities"][i]
                         for name in weights) / total for i in range(len(legal))]
    return probabilities, evaluations


def generate_game(seed, *, weights=None, max_steps=2000, iterations=128,
                  rollout_depth=128, history_size=32, selection="sample"):
    weights = parse_strategies([f"{name}:{weight}" for name, weight in
                                (weights or {"random": 1, "heuristic": 2,
                                             "greedy": 1, "mcts": 4}).items()])
    validate_search_options(iterations=iterations, rollout_depth=rollout_depth)
    if max_steps < 1 or history_size < 0 or selection not in ("sample", "argmax"):
        raise ValueError("max_steps >= 1, history_size >= 0, selection: sample/argmax")
    game = _native_module().Game.new_game(seed)
    memory = PublicMemory(history_size)
    rng = random.Random(seed)
    transitions, pending = [], {}

    def finish_pending(player, state, legal, step, *, done=False, truncated=False):
        if player not in pending:
            return
        row = pending.pop(player)
        row.update(next_state=state, next_legal_actions=legal, done=done,
                   truncated=truncated, steps_to_next=step - row["turn_id"],
                   reward=(1.0 if game.winner == player else -1.0) if done else 0.0)

    for step in range(max_steps):
        full = game.snapshot()
        player = full["turn"]
        state, legal = memory.observation(full, player), game.legal_moves()
        finish_pending(player, state, legal, step)
        probabilities, evaluations = ensemble(
            state, legal, weights, rng, iterations=iterations, rollout_depth=rollout_depth
        )
        index = (rng.choices(range(len(legal)), weights=probabilities)[0]
                 if selection == "sample" else max(range(len(legal)), key=probabilities.__getitem__))
        action = legal[index]
        row = {"player_id": player, "turn_id": step, "state": state,
               "legal_actions": legal, "action": action, "action_index": index,
               "action_probabilities": probabilities, "strategy_evaluations": evaluations,
               "policy_id": "ensemble"}
        transitions.append(row)
        pending[player] = row
        game.apply(action)
        after = game.snapshot()
        memory.update(full, action, after)
        if game.winner != -1:
            break
    done = game.winner != -1
    full = game.snapshot()
    for player in list(pending):
        next_legal = game.legal_moves() if not done and full["turn"] == player else []
        finish_pending(player, memory.observation(full, player), next_legal,
                       len(transitions), done=done, truncated=not done)
    return {"schema_version": SCHEMA_VERSION, "game_id": f"game-{seed}", "seed": seed,
            "winner": game.winner if done else None, "truncated": not done,
            "config": {"strategies": weights, "selection": selection,
                       "iterations": iterations, "rollout_depth": rollout_depth,
                       "history_size": history_size, "max_steps": max_steps},
            "transitions": transitions}


def _write_game(task):
    path, seed, options = task
    game = generate_game(seed, **options)
    # Exclusive create: повторный seed не перезаписывает уже собранные партии.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(game, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return path, len(game["transitions"]), game["truncated"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("games_dataset"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--strategies", nargs="+", default=["random:1", "heuristic:2", "greedy:1", "mcts:4"])
    parser.add_argument("--selection", choices=("sample", "argmax"), default="sample")
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--rollout-depth", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--history-size", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        weights = parse_strategies(args.strategies)
        validate_search_options(iterations=args.iterations, rollout_depth=args.rollout_depth)
        if min(args.games, args.workers, args.max_steps) < 1 or args.history_size < 0:
            raise ValueError("games/workers/max_steps >= 1, history_size >= 0")
        if not 0 <= args.seed <= 2**64 - args.games:
            raise ValueError("seed и seed + games - 1 должны помещаться в uint64")
        args.output.mkdir(parents=True, exist_ok=True)
        paths = [args.output / f"game_{args.seed + i:020d}.json" for i in range(args.games)]
        if any(path.exists() for path in paths):
            raise ValueError("Файлы для этих seed уже существуют; задайте другой --seed или --output")
        options = dict(weights=weights, max_steps=args.max_steps, iterations=args.iterations,
                       rollout_depth=args.rollout_depth, history_size=args.history_size,
                       selection=args.selection)
        tasks = [(str(path), args.seed + i, options) for i, path in enumerate(paths)]
        if args.workers == 1:
            for result in map(_write_game, tasks):
                print(f"{result[0]}: {result[1]} ходов, truncated={result[2]}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(_write_game, tasks):
                    print(f"{result[0]}: {result[1]} ходов, truncated={result[2]}", flush=True)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
