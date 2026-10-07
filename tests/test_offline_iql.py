from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from behaviour_cloning import BehaviourCloningPolicy, collate_examples, encode_transition
from generate_games_dataset import generate_game
from offline_iql import (
    IQLDataset, IQLEngine, IQLLearner, IQLPolicy, advantage_weights, bellman_target,
    collate_iql, encode_iql_transition, evaluate, expectile_loss, load_iql_policy, main, soft_update,
)
from game_state.bot import main as bot_main, format_state, recommendation_overlay, state_snapshot
from game_state.engine_process import _initialize_engine, _suggest
from game_state.game import DurakGameState


class IQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.game = generate_game(901, iterations=24, rollout_depth=48, max_steps=1000)
        cls.rows = cls.game["transitions"]

    def setUp(self):
        torch.manual_seed(7)

    def test_expectile_and_detached_clipped_weights(self):
        difference = torch.tensor([-2., 2.], requires_grad=True)
        loss = expectile_loss(difference, 0.7)
        self.assertAlmostEqual(loss.item(), 2.0)
        loss.backward()
        torch.testing.assert_close(difference.grad, torch.tensor([-0.6, 1.4]))
        weights = advantage_weights(torch.tensor([-1., 0., 1., 1e6], requires_grad=True), 3, 100)
        torch.testing.assert_close(weights, torch.tensor([math.exp(-3), 1, math.exp(3), 100]))
        self.assertFalse(weights.requires_grad)
        self.assertTrue(torch.isfinite(weights).all())

    def test_bootstrap_keeps_player_perspective_and_terminal_rewards(self):
        rewards = torch.tensor([0., 1., -1.])
        target = bellman_target(rewards, torch.tensor([0.8, 0., 0.]), 0.9)
        torch.testing.assert_close(target, torch.tensor([0.72, 1., -1.]))
        terminal = [row for row in self.rows if row["done"]]
        self.assertEqual(sorted(row["reward"] for row in terminal), [-1, 1])
        rows = [encode_iql_transition(row) for row in terminal]
        batch = collate_iql(rows)
        self.assertIsNone(batch["next"])
        self.assertEqual(batch["next_indices"].numel(), 0)
        learner = IQLLearner(hidden_size=32)
        with patch.object(learner.value, "forward", wraps=learner.value.forward) as forward:
            learner.step(batch)
        # Два обращения V к текущим состояниям; terminal-next не кодируется и не оценивается.
        self.assertEqual(forward.call_count, 2)

    def test_invalid_next_player_and_cutoff(self):
        row = deepcopy(next(row for row in self.rows if not row["done"]))
        row["next_state"]["turn"] = 1
        with self.assertRaisesRegex(ValueError, "turn=0"):
            encode_iql_transition(row)
        row["truncated"] = True
        self.assertIsNone(encode_iql_transition(row))
        row["done"] = True
        with self.assertRaises(ValueError):
            encode_iql_transition(row)
        row = deepcopy(self.rows[0])
        row["reward"] = float("nan")
        with self.assertRaises(ValueError):
            encode_iql_transition(row)

    def test_both_terminal_and_nonterminal_batch(self):
        terminal = next(row for row in self.rows if row["done"])
        ordinary = next(row for row in self.rows if not row["done"])
        batch = collate_iql([encode_iql_transition(terminal), encode_iql_transition(ordinary)])
        self.assertEqual(batch["next_indices"].tolist(), [1])
        learner = IQLLearner(hidden_size=32, tau=0.1)
        old_target = {key: value.clone() for key, value in learner.target_critic.state_dict().items()}
        old = {name: [p.detach().clone() for p in getattr(learner, name).parameters()]
               for name in ("actor", "critic", "value")}
        metrics = learner.step(batch)
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        for name, parameters in old.items():
            self.assertTrue(any(not torch.equal(before, after) for before, after in
                                zip(parameters, getattr(learner, name).parameters())), name)
        for key, value in learner.target_critic.state_dict().items():
            torch.testing.assert_close(value, old_target[key] * 0.9 + learner.critic.state_dict()[key] * 0.1)
        self.assertTrue(all(p.grad is None for p in learner.target_critic.parameters()))
        self.assertTrue(all(not p.requires_grad for p in learner.target_critic.parameters()))

    def test_validation_does_not_update_any_network(self):
        learner = IQLLearner(hidden_size=32)
        batch = collate_iql([encode_iql_transition(self.rows[0])])
        networks = (learner.actor, learner.critic, learner.value, learner.target_critic)
        before = [[p.clone() for p in network.parameters()] for network in networks]
        learner.step(batch, training=False)
        for parameters, network in zip(before, networks):
            for original, actual in zip(parameters, network.parameters()):
                torch.testing.assert_close(original, actual)
                self.assertIsNone(actual.grad)
        self.assertEqual(learner.updates, 0)

    def test_q_uses_only_dataset_action_and_value_ignores_actions(self):
        learner = IQLLearner(hidden_size=32)
        batch = collate_examples([encode_transition(self.rows[0])])
        with patch.object(learner.critic.q1, "forward", wraps=learner.critic.q1.forward) as forward:
            learner.critic(batch)
            chosen = forward.call_args.args[0]
            self.assertEqual(chosen["actions"].shape[1], 1)
            torch.testing.assert_close(chosen["actions"][0, 0], batch["actions"][0, batch["label"][0]])
        value = learner.value(batch)
        reversed_actions = {**batch, "actions": batch["actions"].flip(1)}
        torch.testing.assert_close(value, learner.value(reversed_actions))

    def test_bc_initialization_and_history_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bc.pt"
            bc = BehaviourCloningPolicy(32)
            torch.save({"schema_version": 1, "model_config": {"hidden_size": 32},
                        "model_state_dict": bc.state_dict()}, path)
            learner = IQLLearner(init_bc=path)
            self.assertEqual(learner.config["hidden_size"], 32)
            for key, value in bc.state_dict().items():
                torch.testing.assert_close(value, learner.actor.state_dict()[key])
            for key, value in bc.state_dict().items():
                if not key.startswith("policy_head."):
                    torch.testing.assert_close(value, learner.critic.q1.state_dict()[key])
            self.assertNotEqual(learner.actor.rank_embedding.weight.data_ptr(),
                                learner.critic.q1.rank_embedding.weight.data_ptr())
            with self.assertRaisesRegex(ValueError, "hidden_size"):
                IQLLearner(init_bc=path, hidden_size=64)
            with self.assertRaisesRegex(ValueError, "IQL-checkpoint"):
                load_iql_policy(path)
        row = next(row for row in self.rows if row["state"].get("history"))
        batch = collate_examples([encode_transition(row)])
        no_history = {**batch, "history_length": torch.zeros_like(batch["history_length"])}
        torch.testing.assert_close(learner.actor(batch), learner.actor(no_history))

    def test_training_cli_checkpoints_and_streaming(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            for i in range(2):
                (data / f"game{i}.json").write_text(json.dumps({**self.game, "game_id": f"game{i}"}))
            self.assertEqual(len(list(IQLDataset(list(data.glob("*.json"))))), len(self.rows) * 2)
            output = root / "iql"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["train", "--dataset", str(data), "--output", str(output),
                                       "--epochs", "2", "--hidden-size", "32", "--threads", "1",
                                       "--max-batches", "2", "--batch-size", "8"]), 0)
            checkpoint = torch.load(output / "last.pt", weights_only=True)
            self.assertEqual(checkpoint["updates"], 4)
            self.assertEqual(set(checkpoint["optimizer_states"]), {"actor", "critic", "value"})
            policy = load_iql_policy(output / "last.pt")
            result = policy.suggest(self.rows[0]["state"])
            self.assertIn(result["action"], self.rows[0]["legal_actions"])
            self.assertTrue((output / "best.pt").is_file())
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 2)

    def test_reuse_bc_split_and_reject_mixed_game_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            paths = [data / "a.json", data / "b.json"]
            for i, path in enumerate(paths):
                path.write_text(json.dumps({**self.game, "game_id": f"g{i}"}))
            bc = BehaviourCloningPolicy(32)
            checkpoint = root / "bc.pt"
            torch.save({"schema_version": 1, "model_config": {"hidden_size": 32},
                        "model_state_dict": bc.state_dict()}, checkpoint)
            manifest = {"train_files": [str(paths[1])], "validation_files": [str(paths[0])]}
            (root / "split.json").write_text(json.dumps(manifest))
            args = ["train", "--dataset", str(data), "--init-bc", str(checkpoint),
                    "--output", str(root / "iql"), "--epochs", "1", "--max-batches", "1", "--threads", "1"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            saved = json.loads((root / "iql" / "split.json").read_text())
            self.assertEqual(saved["train_files"], manifest["train_files"])
            self.assertEqual(saved["validation_files"], manifest["validation_files"])
            paths[1].write_text(paths[0].read_text())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)

    def test_evaluation_cutoff_is_not_loss(self):
        counts = evaluate(IQLPolicy(hidden_size=32), games=2, max_steps=1)
        self.assertEqual(counts["truncated"], 2)
        self.assertEqual(counts["losses"], 0)
        self.assertIsNone(counts["winrate_finished"])

    def test_live_adapter_cli_and_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "iql.pt"
            learner = IQLLearner(hidden_size=32)
            with torch.no_grad():
                for p in learner.actor.parameters():
                    p.zero_()
            torch.save(learner.checkpoint({}, {}), path)
            engine = IQLEngine(path)
            state = state_snapshot(DurakGameState())
            from behaviour_cloning import CARDS
            state.update(phase="defend_or_take", button="itake", trump="S", hand_cards=["7C"],
                         field_cards=["6C"], field_layout=[{"card": "6C", "covers": None}],
                         opponent_card_count=1, known_opponent_cards=["8D"], deck_remaining=0,
                         out_cards=sorted(set(CARDS) - {"7C", "6C", "8D"}))
            result = engine.suggest(state)
            self.assertEqual(result["engine"], "iql")
            self.assertEqual(result["action"]["target_card"], "6C")
            self.assertIn("Defend 6C", recommendation_overlay(result))
            self.assertIn("IQL:", format_state({**state, "recommendation": result}))
            self.assertEqual(engine.suggest({"phase": "ready"})["status"], "waiting")
            with patch("game_state.engine_process._engine"), patch("game_engine.DurakEngine") as mcts:
                _initialize_engine({"iql_checkpoint": str(path)})
                self.assertEqual(_suggest(state)["engine"], "iql")
                mcts.assert_not_called()
            with patch("game_state.bot.preload_models"), patch("iphone_screen.iphone_client_v2.IPhoneRemote"), patch(
                "game_state.bot.run_bot"
            ) as run:
                self.assertEqual(bot_main(["--iql-checkpoint", str(path)]), 0)
            self.assertEqual(run.call_args.kwargs["engine_options"],
                             {"iql_checkpoint": str(path), "device": "cpu", "trump": None, "bottom_trump": None})
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bot_main(["--iql-checkpoint", str(path), "--bc-checkpoint", str(path)])


if __name__ == "__main__":
    unittest.main()
