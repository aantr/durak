"""Offline IQL на JSON-партиях Durak; обучение, predict и матчи без iPhone.

python offline_iql.py train --dataset games_dataset --init-bc runs/bc_scores/best.pt
python -m game_state.bot --iql-checkpoint runs/iql/last.pt

Algorithm: https://arxiv.org/abs/2110.06169
Reference: https://github.com/ikostrikov/implicit_q_learning
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import islice
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from behaviour_cloning import (
    BehaviourCloningEngine, BehaviourCloningPolicy, collate_examples,
    encode_state, encode_transition, load_policy as load_bc_policy, read_game, split_games,
)
from generate_games_dataset import PublicMemory, SCHEMA_VERSION, strategy_scores
from game_engine.game_engine import _native_module


def encode_iql_transition(row):
    """gamma применяется один раз между решениями одного игрока, без смены знака."""
    for key in ("done", "truncated"):
        if type(row[key]) is not bool:
            raise ValueError(f"{key} должен быть bool")
    if row["done"] and row["truncated"]:
        raise ValueError("Переход не может быть одновременно terminal и truncated")
    reward = row["reward"]
    if type(reward) not in (float, int) or not math.isfinite(reward) or reward not in (-1, 0, 1):
        raise ValueError("Ожидается награда -1/0/+1")
    if (row["done"] and reward == 0) or (not row["done"] and reward != 0):
        raise ValueError("Награда ±1 только при завершении партии")
    if type(row["steps_to_next"]) is not int or row["steps_to_next"] < 1:
        raise ValueError("steps_to_next должен быть положительным целым")
    if row["truncated"]:
        # На границе обрезания next_state может ещё принадлежать ходу противника.
        return None
    current = encode_transition(row, target="action")
    if row["done"]:
        if row["next_legal_actions"]:
            raise ValueError("У terminal-перехода next_legal_actions должен быть пустым")
        following = None
    else:
        # encode_state проверяет turn=0 и полный набор допустимых действий.
        following = encode_state(row["next_state"], row["next_legal_actions"])
    return {"current": current, "next": following, "reward": float(reward)}


class IQLDataset(IterableDataset):
    def __init__(self, paths, *, shuffle=False, seed=42):
        super().__init__()
        self.paths, self.shuffle, self.seed, self.epoch = list(paths), shuffle, seed, 0

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
                    example = encode_iql_transition(row)
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"{path}, turn_id={row.get('turn_id')}: {exc}") from exc
                if example is not None:
                    yield example


def collate_iql(rows):
    indices = [i for i, row in enumerate(rows) if row["next"] is not None]
    return {"current": collate_examples([row["current"] for row in rows]),
            "next": collate_examples([rows[i]["next"] for i in indices]) if indices else None,
            "next_indices": torch.tensor(indices, dtype=torch.long),
            "reward": torch.tensor([row["reward"] for row in rows], dtype=torch.float32)}


class IQLPolicy(BehaviourCloningPolicy):
    def __init__(self, hidden_size=256, history_mode="none"):
        super().__init__(hidden_size)
        if history_mode not in ("none", "dataset"):
            raise ValueError("history_mode: none или dataset")
        self.history_mode = history_mode

    def state_features(self, batch):
        if self.history_mode == "none":
            batch = {**batch, "history_length": torch.zeros_like(batch["history_length"])}
        return super().state_features(batch)


class ValueNetwork(IQLPolicy):
    def __init__(self, **config):
        super().__init__(**config)
        del self.policy_head
        self.value_head = nn.Sequential(nn.Linear(self.hidden_size, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, batch):
        return self.value_head(self.state_features(batch)).squeeze(-1)


class DoubleQ(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.q1, self.q2 = IQLPolicy(**config), IQLPolicy(**config)

    def forward(self, batch):
        # Q обучается и используется для targets только на действии из датасета.
        index = batch["label"].view(-1, 1, 1).expand(-1, 1, 3)
        chosen = {**batch, "actions": batch["actions"].gather(1, index),
                  "action_mask": torch.ones_like(batch["label"], dtype=torch.bool).unsqueeze(1)}
        return self.q1(chosen).squeeze(1), self.q2(chosen).squeeze(1)


def expectile_loss(difference, expectile):
    return (torch.where(difference > 0, expectile, 1 - expectile) * difference.square()).mean()


def advantage_weights(advantage, beta, max_weight):
    return (beta * advantage.detach()).clamp(max=math.log(max_weight)).exp().clamp_max(max_weight)


def bellman_target(reward, next_value, gamma):
    # next_value уже содержит нули для terminal; не кодируем terminal-состояния.
    return reward + gamma * next_value


@torch.no_grad()
def soft_update(source, target, tau):
    for parameter, target_parameter in zip(source.parameters(), target.parameters()):
        target_parameter.lerp_(parameter, tau)


def move_batch(batch, device):
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    return batch.to(device) if isinstance(batch, torch.Tensor) else batch


class IQLLearner:
    def __init__(self, *, hidden_size=None, history_mode="none", device="cpu", init_bc=None,
                 gamma=0.99, expectile=0.7, beta=3.0, max_weight=100.0, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, value_lr=3e-4):
        for name, value in dict(gamma=gamma, expectile=expectile, beta=beta, max_weight=max_weight,
                                tau=tau, actor_lr=actor_lr, critic_lr=critic_lr, value_lr=value_lr).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} должен быть конечным")
        if not (0 < gamma <= 1 and 0.5 < expectile < 1 and beta >= 0 and max_weight >= 1
                and 0 < tau <= 1 and min(actor_lr, critic_lr, value_lr) > 0):
            raise ValueError("Нужны 0<gamma<=1, .5<expectile<1, beta>=0, max_weight>=1, 0<tau<=1, lr>0")
        pretrained = load_bc_policy(init_bc) if init_bc else None
        if pretrained is not None:
            if hidden_size is not None and hidden_size != pretrained.hidden_size:
                raise ValueError("hidden_size не совпадает с BC-checkpoint")
            hidden_size = pretrained.hidden_size
        self.config = {"hidden_size": hidden_size or 256, "history_mode": history_mode}
        self.device = torch.device(device)
        self.actor = IQLPolicy(**self.config).to(self.device)
        self.critic = DoubleQ(**self.config).to(self.device)
        self.value = ValueNetwork(**self.config).to(self.device)
        if pretrained is not None:
            self.actor.load_state_dict(pretrained.state_dict())
            features = {key: value for key, value in pretrained.state_dict().items()
                        if not key.startswith("policy_head.")}
            for network in (self.critic.q1, self.critic.q2, self.value):
                network.load_state_dict(features, strict=False)
        self.target_critic = deepcopy(self.critic).requires_grad_(False).eval()
        self.optimizers = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=actor_lr),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=critic_lr),
            "value": torch.optim.Adam(self.value.parameters(), lr=value_lr),
        }
        self.gamma, self.expectile, self.beta = gamma, expectile, beta
        self.max_weight, self.tau, self.updates = max_weight, tau, 0

    def optimize(self, name, loss):
        if not torch.isfinite(loss):
            raise ValueError(f"Неконечный {name} loss")
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(getattr(self, name).parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()

    def step(self, batch, *, training=True):
        batch = move_batch(batch, self.device)
        current = batch["current"]
        for network in (self.actor, self.critic, self.value):
            network.train(training)
        with torch.no_grad():
            q_target = torch.minimum(*self.target_critic(current))
        with torch.set_grad_enabled(training):
            value = self.value(current)
            value_loss = expectile_loss(q_target - value, self.expectile)
            if training:
                self.optimize("value", value_loss)
        with torch.no_grad():
            advantage = q_target - self.value(current)
            weights = advantage_weights(advantage, self.beta, self.max_weight)
            next_value = torch.zeros_like(batch["reward"])
            if batch["next"] is not None:
                next_value[batch["next_indices"]] = self.value(batch["next"])
            target = bellman_target(batch["reward"], next_value, self.gamma)
        with torch.set_grad_enabled(training):
            logits = self.actor(current)
            nll = F.cross_entropy(logits, current["label"], reduction="none")
            actor_loss = (weights * nll).mean()
            if training:
                self.optimize("actor", actor_loss)
            q1, q2 = self.critic(current)
            critic_loss = ((q1 - target).square() + (q2 - target).square()).mean()
            if training:
                self.optimize("critic", critic_loss)
                soft_update(self.critic, self.target_critic, self.tau)
                self.updates += 1
        result = {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                  "value_loss": value_loss.item(), "nll": nll.mean().item(),
                  "accuracy": (logits.argmax(-1) == current["label"]).float().mean().item(),
                  "q": torch.minimum(q1, q2).mean().item(), "v": value.mean().item(),
                  "advantage": advantage.mean().item(), "weight": weights.mean().item(),
                  "clipped_fraction": (weights >= self.max_weight).float().mean().item()}
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Неконечные метрики IQL")
        return result

    def checkpoint(self, metrics, config):
        return {"algorithm": "iql", "checkpoint_version": 1, "schema_version": SCHEMA_VERSION,
                "model_config": self.config, "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(), "value_state_dict": self.value.state_dict(),
                "target_critic_state_dict": self.target_critic.state_dict(),
                "optimizer_states": {key: optimizer.state_dict() for key, optimizer in self.optimizers.items()},
                "updates": self.updates, "metrics": metrics, "training_config": config}


def load_iql_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("algorithm") != "iql" or checkpoint.get("checkpoint_version") != 1
            or checkpoint.get("schema_version") != SCHEMA_VERSION):
        raise ValueError("Нужен IQL-checkpoint версии 1, а не BC-checkpoint")
    model = IQLPolicy(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["actor_state_dict"])
    return model.eval()


class IQLEngine(BehaviourCloningEngine):
    engine_name = "iql"

    def __init__(self, checkpoint, *, device="cpu", trump=None, bottom_trump=None):
        torch.set_num_threads(2)
        self.policy = load_iql_policy(checkpoint, device)
        self.trump, self.bottom_trump = trump, bottom_trump


def epoch_metrics(learner, loader, *, training=False, max_batches=None):
    totals, count = Counter(), 0
    for batch in islice(loader, max_batches):
        size = len(batch["reward"])
        metrics = learner.step(batch, training=training)
        for key, value in metrics.items():
            totals[key] += value * size
        count += size
    if not count:
        raise ValueError("Нет пригодных IQL-переходов: проверьте датасет и truncated")
    return {"samples": count, **{key: value / count for key, value in totals.items()}}


def dataset_split(args):
    split_path = args.split
    if split_path is None and args.init_bc is not None:
        candidate = args.init_bc.parent / "split.json"
        if candidate.is_file():
            split_path = candidate
    if split_path is None:
        return split_games(args.dataset, args.validation_fraction, args.seed)
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    train_paths = [Path(p).resolve() for p in manifest["train_files"]]
    val_paths = [Path(p).resolve() for p in manifest["validation_files"]]
    actual = {p.resolve() for p in args.dataset.rglob("*.json")}
    if not train_paths or set(train_paths + val_paths) != actual:
        raise ValueError(f"Файлы --dataset не совпадают с {split_path}; используйте тот же датасет и split, что BC")
    ids = [{read_game(path)["game_id"] for path in paths} for paths in (train_paths, val_paths)]
    if ids[0] & ids[1]:
        raise ValueError("Один game_id присутствует и в train, и в validation")
    print(f"Используется разделение партий из {split_path}", flush=True)
    return train_paths, val_paths


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(args):
    if (min(args.epochs, args.batch_size, args.threads) < 1 or args.workers < 0
            or (args.hidden_size is not None and args.hidden_size < 1)
            or (args.max_batches is not None and args.max_batches < 1)):
        raise ValueError("epochs/batch_size/threads/hidden_size/max_batches > 0, workers >= 0")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_paths, val_paths = dataset_split(args)
    learner = IQLLearner(hidden_size=args.hidden_size, history_mode=args.history_mode, device=args.device,
                         init_bc=args.init_bc, gamma=args.gamma, expectile=args.expectile, beta=args.beta,
                         max_weight=args.max_weight, tau=args.tau, actor_lr=args.actor_lr,
                         critic_lr=args.critic_lr, value_lr=args.value_lr)
    training = IQLDataset(train_paths, shuffle=True, seed=args.seed)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_iql)
    train_loader = DataLoader(training, **kwargs)
    val_loader = DataLoader(IQLDataset(val_paths), **kwargs) if val_paths else None
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.init_bc and args.output.resolve() == args.init_bc.parent.resolve():
        raise ValueError("Выходная директория IQL должна отличаться от директории исходной BC-модели")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "train_files": [str(p.resolve()) for p in train_paths],
                "validation_files": [str(p.resolve()) for p in val_paths]}
    (args.output / "split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={args.device}, train_games={len(train_paths)}, validation_games={len(val_paths)}, "
          f"history={args.history_mode}; truncated-переходы пропускаются", flush=True)
    best = math.inf
    with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(args.epochs):
            training.epoch = epoch
            metrics = {"epoch": epoch + 1,
                       "train": epoch_metrics(learner, train_loader, training=True, max_batches=args.max_batches),
                       "validation": epoch_metrics(learner, val_loader, max_batches=args.max_batches)
                       if val_loader is not None else None}
            print(json.dumps(metrics), flush=True)
            stream.write(json.dumps(metrics) + "\n")
            stream.flush()
            checkpoint = learner.checkpoint(metrics, config)
            atomic_save(checkpoint, args.output / "last.pt")
            # Диагностика Bellman residual; НЕ выбор по силе игры или winrate.
            score = (metrics["validation"] or metrics["train"])["critic_loss"]
            if score < best:
                best = score
                atomic_save(checkpoint, args.output / "best.pt")
    return learner


@torch.inference_mode()
def evaluate(policy, *, games=20, seed=100000, opponent="heuristic", max_steps=2000):
    if games < 1 or max_steps < 1 or not 0 <= seed <= 2**64 - games:
        raise ValueError("games/max_steps > 0, допустимый uint64 seed")
    counts = Counter(wins=0, losses=0, truncated=0)
    for i in range(games):
        game_seed = seed + i
        game = _native_module().Game.new_game(game_seed)
        memory, rng = PublicMemory(), random.Random(game_seed)
        player = i % 2
        for _ in range(max_steps):
            before = game.snapshot()
            state, actions = memory.observation(before, before["turn"]), game.legal_moves()
            if before["turn"] == player:
                action = policy.suggest(state)["action"]
            else:
                scores = strategy_scores(opponent, state, actions, seed=rng.getrandbits(64),
                                         iterations=128, rollout_depth=128)
                action = rng.choices(actions, weights=scores["probabilities"])[0]
            game.apply(action)
            memory.update(before, action, game.snapshot())
            if game.winner != -1:
                break
        counts["truncated" if game.winner == -1 else "wins" if game.winner == player else "losses"] += 1
    finished = counts["wins"] + counts["losses"]
    return {**counts, "games": games, "opponent": opponent, "seed": seed,
            "winrate_finished": counts["wins"] / finished if finished else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", type=Path, default=Path("games_dataset"))
    training.add_argument("--output", type=Path, default=Path("runs/iql"))
    training.add_argument("--init-bc", type=Path, help="Инициализация actor и encoders из BC")
    training.add_argument("--split", type=Path, help="JSON со списками train_files/validation_files; иначе берётся split BC")
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--hidden-size", type=int, help="По умолчанию размер BC либо 256")
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--expectile", type=float, default=0.7)
    training.add_argument("--beta", type=float, default=3.0, help="Множитель advantage в exp(beta * advantage)")
    training.add_argument("--max-weight", type=float, default=100.0)
    training.add_argument("--tau", type=float, default=0.005, help="Коэффициент EMA целевых Q-сетей")
    for network in ("actor", "critic", "value"):
        training.add_argument(f"--{network}-lr", type=float, default=3e-4)
    training.add_argument("--history-mode", choices=("none", "dataset"), default="none")
    training.add_argument("--validation-fraction", type=float, default=0.15)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--max-batches", type=int, help="Для smoke-test: ограничить train и validation на каждой эпохе")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True, type=Path)
    prediction.add_argument("--state", required=True, type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--games", type=int, default=20)
    evaluation.add_argument("--opponent", choices=("random", "heuristic", "greedy", "mcts"), default="heuristic")
    evaluation.add_argument("--max-steps", type=int, default=2000)
    for command in (training, prediction, evaluation):
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=2)
    training.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args(argv)
    try:
        if args.threads < 1:
            raise ValueError("threads > 0")
        torch.set_num_threads(args.threads)
        if args.command == "train":
            train(args)
        else:
            policy = load_iql_policy(args.checkpoint, args.device)
            if args.command == "predict":
                payload = json.loads(args.state.read_text(encoding="utf-8"))
                result = policy.suggest(payload.get("state", payload))
            else:
                result = evaluate(policy, games=args.games, seed=args.seed,
                                  opponent=args.opponent, max_steps=args.max_steps)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
