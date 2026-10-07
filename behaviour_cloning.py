"""Behaviour cloning наблюдений DurakEngine из games_dataset/*.json.

python behaviour_cloning.py train --dataset games_dataset --epochs 20
python behaviour_cloning.py predict --checkpoint runs/bc/best.pt --state observation.json
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from game_engine import observation_from_state
from game_engine.game_engine import _native_module
from generate_games_dataset import RANKS, SCHEMA_VERSION, action_key

CARDS = tuple(rank + suit for rank in RANKS for suit in "CDHS")
CARD_INDEX = {card: i for i, card in enumerate(CARDS)}
PAD_CARD = 36
ACTION_TYPES = ("attack", "defend", "take", "pass")
HISTORY_SIZE = 32
CONTEXT_SIZE = 17


def card_id(card):
    if card is None:
        return PAD_CARD
    try:
        return CARD_INDEX[card]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Неизвестная карта: {card!r}") from exc


def legal_actions(state):
    """Список движка. Расклад скрытых карт не влияет на наши допустимые ходы."""
    if state["turn"] != 0:
        raise ValueError("Нужен ход игрока-наблюдателя: turn=0")
    native = _native_module()
    return native.Game(native.sample_world(state, seed=0)).legal_moves()


def encode_action(action, state):
    kind = action["type"]
    if kind not in ACTION_TYPES:
        raise ValueError(f"Неизвестный тип действия: {kind}")
    card, target = action.get("card"), action.get("target")
    if kind in ("attack", "defend") and card not in state["hand"]:
        raise ValueError("Карта действия отсутствует в руке")
    target_card = None
    if kind == "defend":
        if type(target) is not int or not 0 <= target < len(state["table"]):
            raise ValueError("Некорректная цель защиты")
        pair = state["table"][target]
        if pair.get("defense") is not None:
            raise ValueError("Цель защиты уже отбита")
        target_card = pair["attack"]
        same_suit = card[-1] == target_card[-1]
        if not ((same_suit and card_id(card) // 4 > card_id(target_card) // 4)
                or (card[-1] == state["trump"] and not same_suit)):
            raise ValueError("Карта не бьёт цель защиты")
    elif target is not None:
        raise ValueError("target разрешён только для defend")
    if kind in ("take", "pass") and card is not None:
        raise ValueError("take/pass не содержат карту")
    return [ACTION_TYPES.index(kind), card_id(card), card_id(target_card)]


def encode_state(state, actions):
    if not actions:
        raise ValueError("Список legal_actions пуст")
    if len({action_key(a) for a in actions}) != len(actions):
        raise ValueError("Повторяющиеся legal_actions")
    if state["turn"] != 0:
        raise ValueError("Состояние решения должно иметь turn=0")
    sets = [state["hand"], state.get("known_opponent", []), state["discard"],
            [state["bottom_trump"]] if state.get("bottom_trump") else []]
    if any(len(cards) > 36 or len(set(cards)) != len(cards) for cards in sets):
        raise ValueError("Некорректный набор карт")
    table = state["table"]
    if len(table) > 6:
        raise ValueError("На столе не может быть больше шести пар")
    visible = [card for cards in sets for card in cards]
    visible.extend(card for pair in table for card in (pair["attack"], pair.get("defense"))
                   if card is not None)
    if len(visible) != len(set(visible)):
        raise ValueError("Карта присутствует в нескольких зонах")
    opponent, deck = state["opponent_count"], state["deck_count"]
    if (type(opponent) is not int or type(deck) is not int or
            not len(sets[1]) <= opponent <= 36 or not 0 <= deck <= 36):
        raise ValueError("Некорректное число карт соперника/колоды")
    table_count = sum(1 + (pair.get("defense") is not None) for pair in table)
    if len(sets[0]) + len(sets[2]) + table_count + opponent + deck != 36:
        raise ValueError("Число карт по зонам должно равняться 36")
    trump = "CDHS".index(state["trump"])
    if sets[3] and (not deck or sets[3][0][-1] != state["trump"]):
        raise ValueError("Некорректный нижний козырь")
    if state["attacker"] not in (0, 1) or not 0 <= state["attack_limit"] <= 6:
        raise ValueError("Некорректный контекст розыгрыша")
    if {action_key(a) for a in actions} != {action_key(a) for a in legal_actions(state)}:
        raise ValueError("legal_actions не совпадают с допустимыми ходами движка")
    context = [float(i == trump) for i in range(4)] + [
        len(sets[0]) / 36, len(sets[1]) / 36, len(sets[2]) / 36,
        opponent / 36, deck / 36, state["attack_limit"] / 6,
        float(state["attacker"]), float(state["turn"]), float(state["taking"]),
        float(state["attacker_passed"]),
        float(state.get("simultaneous_winner", "attacker") == "attacker"),
        len(table) / 6, sum(pair.get("defense") is None for pair in table) / 6,
    ]
    history = state.get("history", [])[-HISTORY_SIZE:]
    history_rows = []
    for event in history:
        if event["player_id"] not in (0, 1):
            raise ValueError("История использует относительные player_id 0/1")
        history_rows.append([ACTION_TYPES.index(event["type"]), card_id(event.get("card")),
                             card_id(event.get("target_card")), event["player_id"]])
    return {
        "cards": [[card_id(c) for c in cards] + [PAD_CARD] * (36 - len(cards)) for cards in sets],
        "table": [[card_id(p["attack"]), card_id(p.get("defense"))] for p in table]
                 + [[PAD_CARD, PAD_CARD]] * (6 - len(table)),
        "context": context, "trump": trump,
        "history": history_rows + [[4, PAD_CARD, PAD_CARD, 0]] * (HISTORY_SIZE - len(history)),
        "history_length": len(history),
        "actions": [encode_action(a, state) for a in actions],
    }


def encode_transition(row, target="action"):
    actions = row["legal_actions"]
    keys = [action_key(action) for action in actions]
    if len(keys) != len(set(keys)):
        raise ValueError("Повторяющиеся legal_actions")
    try:
        index = keys.index(action_key(row["action"]))
    except ValueError as exc:
        raise ValueError("Действие датасета отсутствует в legal_actions") from exc
    if "action_index" in row and row["action_index"] != index:
        raise ValueError("action_index не соответствует action")
    probabilities = [float(i == index) for i in range(len(actions))]
    if target == "scores":
        probabilities = row.get("action_probabilities")
        if (not isinstance(probabilities, list) or len(probabilities) != len(actions)
                or any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0
                       for p in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)):
            raise ValueError("Нужны action_probabilities: распределение по legal_actions")
    elif target != "action":
        raise ValueError("target: action или scores")
    return {**encode_state(row["state"], actions), "label": index, "probabilities": probabilities}


def collate_examples(rows):
    count = max(len(row["actions"]) for row in rows)
    batch = {}
    for key in ("cards", "table", "context", "trump", "history", "history_length"):
        batch[key] = torch.tensor([row[key] for row in rows],
                                 dtype=torch.float32 if key == "context" else torch.long)
    batch["actions"] = torch.tensor([
        row["actions"] + [[4, PAD_CARD, PAD_CARD]] * (count - len(row["actions"])) for row in rows
    ], dtype=torch.long)
    batch["action_mask"] = torch.tensor([
        [True] * len(row["actions"]) + [False] * (count - len(row["actions"])) for row in rows
    ])
    if "label" in rows[0]:
        batch["label"] = torch.tensor([row["label"] for row in rows])
        batch["probabilities"] = torch.tensor([
            row["probabilities"] + [0.0] * (count - len(row["actions"])) for row in rows
        ], dtype=torch.float32)
    return batch


def read_game(path):
    with Path(path).open(encoding="utf-8") as stream:
        game = json.load(stream)
    if (not isinstance(game, dict) or game.get("schema_version") != SCHEMA_VERSION
            or not isinstance(game.get("game_id"), str)
            or not isinstance(game.get("transitions"), list) or not game["transitions"]):
        raise ValueError(f"{path}: ожидается JSON партии schema_version={SCHEMA_VERSION}")
    return game


def split_games(directory, validation_fraction=0.15, seed=42):
    """Одна партия (включая копии с тем же game_id) целиком в одном split."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction должна быть в [0, 1)")
    paths = sorted(Path(directory).rglob("*.json"))
    if not paths:
        raise ValueError(f"В {directory} нет JSON-партий; сначала запустите generate_games_dataset.py")
    groups = {}
    for path in paths:
        game = read_game(path)
        groups.setdefault(game["game_id"], []).append(path)
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    if validation_fraction and len(ids) < 2:
        raise ValueError("Для train/validation нужны минимум две партии или --validation-fraction 0")
    n_val = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction))) if validation_fraction else 0
    return ([path for gid in ids[n_val:] for path in groups[gid]],
            [path for gid in ids[:n_val] for path in groups[gid]])


class GamesDataset(IterableDataset):
    """Загружает по одной партии на worker, не весь датасет в память."""

    def __init__(self, paths, *, target="action", shuffle=False, seed=42):
        super().__init__()
        self.paths, self.target, self.shuffle, self.seed = list(paths), target, shuffle, seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        paths = list(self.paths)
        if self.shuffle:
            rng.shuffle(paths)
        worker = get_worker_info()
        if worker:
            paths = paths[worker.id::worker.num_workers]
        for path in paths:
            rows = read_game(path)["transitions"]
            if self.shuffle:
                rng.shuffle(rows)
            for row in rows:
                try:
                    yield encode_transition(row, self.target)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}, turn_id={row.get('turn_id')}: {exc}") from exc


class BehaviourCloningPolicy(nn.Module):
    """Shared card DeepSets + пары стола + GRU истории + encoder легальных действий."""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank_embedding = nn.Embedding(10, 16, padding_idx=9)
        self.suit_embedding = nn.Embedding(5, 8, padding_idx=4)
        self.trump_embedding = nn.Embedding(3, 8, padding_idx=2)
        self.card_encoder = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        self.type_embedding = nn.Embedding(5, 16, padding_idx=4)
        self.actor_embedding = nn.Embedding(2, 8)
        self.action_encoder = nn.Sequential(nn.Linear(80, 64), nn.ReLU(), nn.Linear(64, 64))
        self.table_encoder = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.history_encoder = nn.GRU(72, 64, batch_first=True)
        self.state_encoder = nn.Sequential(
            nn.Linear(4 * 32 + 64 + 64 + CONTEXT_SIZE, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(nn.Linear(hidden_size + 64, 128), nn.ReLU(), nn.Linear(128, 1))

    def embed_cards(self, cards, trump):
        missing = cards == PAD_CARD
        suits = (cards % 4).masked_fill(missing, 4)
        trump = trump.reshape((-1,) + (1,) * (cards.ndim - 1))
        flags = (suits == trump).long().masked_fill(missing, 2)
        encoded = self.card_encoder(torch.cat((self.rank_embedding(cards // 4),
                                              self.suit_embedding(suits),
                                              self.trump_embedding(flags)), dim=-1))
        return encoded * (~missing).unsqueeze(-1)

    def embed_actions(self, actions, trump):
        cards = self.embed_cards(actions[..., 1:], trump).flatten(-2)
        return self.action_encoder(torch.cat((self.type_embedding(actions[..., 0]), cards), dim=-1))

    @staticmethod
    def pool(encoded, mask):
        return (encoded * mask.unsqueeze(-1)).sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)

    def state_features(self, batch):
        trump = batch["trump"]
        sets = self.pool(self.embed_cards(batch["cards"], trump), batch["cards"] != PAD_CARD).flatten(1)
        pairs = self.table_encoder(self.embed_cards(batch["table"], trump).flatten(-2))
        table = self.pool(pairs, batch["table"][..., 0] != PAD_CARD)
        history = batch["history"]
        history_inputs = torch.cat((self.embed_actions(history[..., :3], trump),
                                    self.actor_embedding(history[..., 3])), dim=-1)
        outputs, _ = self.history_encoder(history_inputs)
        lengths = batch["history_length"]
        history_vector = outputs[torch.arange(len(lengths), device=lengths.device), (lengths - 1).clamp_min(0)]
        history_vector = history_vector * (lengths > 0).unsqueeze(-1)
        return self.state_encoder(torch.cat((sets, table, history_vector, batch["context"]), dim=-1))

    def forward(self, batch):
        state = self.state_features(batch)
        actions = self.embed_actions(batch["actions"], batch["trump"])
        scores = self.policy_head(torch.cat((state.unsqueeze(1).expand(-1, actions.size(1), -1), actions), dim=-1)).squeeze(-1)
        return scores.masked_fill(~batch["action_mask"], -torch.inf)

    @torch.inference_mode()
    def suggest(self, state, *, actions=None, **state_options):
        if not isinstance(state, Mapping) or "hand" not in state:
            state_options.setdefault("trump", state.get("trump") if isinstance(state, Mapping)
                                     else getattr(state, "trump", None))
            if state_options["trump"] is None:
                raise ValueError("Нужна козырная масть trump")
            state = observation_from_state(state, **state_options)
        self.eval()
        actual = legal_actions(state)
        if actions is not None and {action_key(a) for a in actions} != {action_key(a) for a in actual}:
            raise ValueError("Переданный список действий не совпадает с легальными ходами движка")
        actions = actual if actions is None else actions
        batch = collate_examples([encode_state(state, actions)])
        device = next(self.parameters()).device
        logits = self({key: value.to(device) for key, value in batch.items()})[0]
        probabilities = logits.softmax(-1).cpu().tolist()
        index = max(range(len(actions)), key=probabilities.__getitem__)
        return {"status": "ok", "action": actions[index],
                "moves": [{**action, "probability": p} for action, p in zip(actions, probabilities)],
                "value_kind": "policy_probability"}


def batch_loss(logits, batch):
    # Обнуляем log p у padding до умножения: 0 * -inf иначе даёт NaN.
    log_probabilities = F.log_softmax(logits, dim=-1).masked_fill(~batch["action_mask"], 0)
    return -(batch["probabilities"] * log_probabilities).sum(-1).mean()


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    count, loss_sum, correct, teacher_correct = 0, 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch)
            loss = batch_loss(logits, batch)
            if not torch.isfinite(loss):
                raise ValueError("Loss не является конечным числом")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            size = len(batch["label"])
            count += size
            loss_sum += loss.item() * size
            correct += (logits.argmax(-1) == batch["label"]).sum().item()
            teacher_correct += (logits.argmax(-1) == batch["probabilities"].argmax(-1)).sum().item()
    if not count:
        raise ValueError("Выборка не содержит примеров")
    return {"samples": count, "loss": loss_sum / count, "accuracy": correct / count,
            "target_accuracy": teacher_correct / count}


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Неподдерживаемая версия checkpoint")
    model = BehaviourCloningPolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class BehaviourCloningEngine:
    """Адаптер BC для фонового процесса live-бота, с тем же протоколом подсказок."""

    engine_name = "bc"

    def __init__(self, checkpoint, *, device="cpu", trump=None, bottom_trump=None):
        torch.set_num_threads(2)
        self.policy = load_policy(checkpoint, device)
        self.trump = trump
        self.bottom_trump = bottom_trump

    def suggest(self, state):
        from game_engine.game_engine import _get, _phase, _text

        phase = _phase(state)
        if phase in ("ready", "opponent_turn"):
            return {"status": "waiting", "action": None, "engine": self.engine_name,
                    "reason": "Игра ещё не началась" if phase == "ready" else "Ход соперника"}
        trump = self.trump or _get(state, "trump")
        if trump not in ("C", "D", "H", "S"):
            return {"status": "waiting", "action": None, "engine": self.engine_name,
                    "reason": "Козырь ещё не распознан"}
        start = perf_counter()
        observation = observation_from_state(state, trump=trump, bottom_trump=self.bottom_trump)
        # Live-снимки пока не содержат надёжного журнала публичных действий.
        result = self.policy.suggest(observation)
        for action in [result["action"], *result["moves"]]:
            if action["type"] == "defend":
                action["target_card"] = observation["table"][action["target"]]["attack"]
            elif action["type"] == "pass":
                button = _text(_get(state, "button_text", _get(state, "button", "")))
                action["button"] = "Bat" if button == "bat" else "Pass"
        result["engine"] = self.engine_name
        result["elapsed_ms"] = (perf_counter() - start) * 1000
        result["policy_probability"] = next(
            move["probability"] for move in result["moves"]
            if action_key(move) == action_key(result["action"])
        )
        return result


def train(args):
    if (min(args.epochs, args.batch_size, args.hidden_size, args.threads) < 1 or args.workers < 0
            or not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("epochs/batch-size/hidden-size/threads/lr > 0, workers >= 0")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    train_paths, val_paths = split_games(args.dataset, args.validation_fraction, args.seed)
    training = GamesDataset(train_paths, target=args.target, shuffle=True, seed=args.seed)
    validation = GamesDataset(val_paths, target=args.target)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_examples)
    train_loader = DataLoader(training, **kwargs)
    val_loader = DataLoader(validation, **kwargs) if val_paths else None
    model = BehaviourCloningPolicy(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {"config": config, "train_files": [str(p) for p in train_paths],
                "validation_files": [str(p) for p in val_paths]}
    (args.output / "split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={device}, train_games={len(train_paths)}, validation_games={len(val_paths)}", flush=True)
    if not val_paths:
        print("Validation отключена: best.pt выбирается по train loss.", flush=True)
    best_loss = math.inf
    with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        for epoch in range(args.epochs):
            training.epoch = epoch
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            val_metrics = run_epoch(model, val_loader, device) if val_loader is not None else None
            metrics = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
            print(json.dumps(metrics), flush=True)
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            loss = (val_metrics or train_metrics)["loss"]
            checkpoint = {"schema_version": SCHEMA_VERSION, "model_config": {"hidden_size": args.hidden_size},
                          "model_state_dict": model.state_dict(), "metrics": metrics, "training_config": config}
            torch.save(checkpoint, args.output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(checkpoint, args.output / "best.pt")
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", type=Path, default=Path("games_dataset"))
    training.add_argument("--output", type=Path, default=Path("runs/bc"))
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--hidden-size", type=int, default=256)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--validation-fraction", type=float, default=0.15)
    training.add_argument("--target", choices=("action", "scores"), default="action")
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="cpu", help="cpu или cuda / cuda:0")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True, type=Path)
    prediction.add_argument("--state", required=True, type=Path, help="JSON наблюдения DurakEngine или transition")
    prediction.add_argument("--device", default="cpu")
    try:
        args = parser.parse_args(argv)
        if args.command == "train":
            train(args)
        else:
            payload = json.loads(args.state.read_text(encoding="utf-8"))
            state = payload.get("state", payload)
            model = load_policy(args.checkpoint, args.device)
            print(json.dumps(model.suggest(state), ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
