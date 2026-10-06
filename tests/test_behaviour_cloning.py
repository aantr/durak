import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from behaviour_cloning import (
    BehaviourCloningPolicy, GamesDataset, batch_loss, collate_examples,
    encode_state, encode_transition, legal_actions, load_policy, main, split_games,
)
from generate_games_dataset import PublicMemory, action_key, generate_game, parse_strategies
from game_engine import _native


def position(hands, *, table=None, deck=None, **options):
    table, deck = table or [], deck or []
    cards = {rank + suit for rank in ("6", "7", "8", "9", "10", "J", "Q", "K", "A") for suit in "CDHS"}
    used = set(hands[0] + hands[1] + deck)
    used.update(c for pair in table for c in pair.values() if c is not None)
    full = dict(hands=hands, deck=deck, table=table, discard=sorted(cards - used),
                trump="S", attacker=0, turn=0, taking=False, attacker_passed=False, winner=-1,
                attack_limit=min(6, len(hands[1])), simultaneous_winner="attacker")
    full.update(options)
    return _native.Game(full)


class DatasetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = generate_game(901, iterations=24, rollout_depth=48, max_steps=1000)

    def test_reproducible_ensemble_and_legal_states(self):
        game = self.game
        self.assertEqual(game, generate_game(901, iterations=24, rollout_depth=48, max_steps=1000))
        self.assertFalse(game["truncated"])
        players = {row["player_id"] for row in game["transitions"]}
        self.assertEqual(players, {0, 1})
        for row in game["transitions"]:
            state, actions = row["state"], row["legal_actions"]
            self.assertNotIn("hands", state)
            self.assertNotIn("deck", state)
            self.assertNotIn("winner", state)
            self.assertEqual(state["turn"], 0)
            self.assertEqual({action_key(a) for a in actions}, {action_key(a) for a in legal_actions(state)})
            self.assertIn(row["action"], actions)
            self.assertAlmostEqual(sum(row["action_probabilities"]), 1)
            weights = game["config"]["strategies"]
            for i, probability in enumerate(row["action_probabilities"]):
                expected = sum(weights[name] * evaluation["probabilities"][i]
                               for name, evaluation in row["strategy_evaluations"].items()) / sum(weights.values())
                self.assertAlmostEqual(probability, expected)
            self.assertTrue(all(v is not None for v in row["strategy_evaluations"]["mcts"]["scores"]))
            encode_transition(row, "scores")

    def test_transitions_return_to_same_player_and_rewards(self):
        rows = self.game["transitions"]
        for player in (0, 1):
            decisions = [row for row in rows if row["player_id"] == player]
            for row, following in zip(decisions, decisions[1:]):
                self.assertEqual(row["next_state"], following["state"])
                self.assertEqual(row["next_legal_actions"], following["legal_actions"])
                self.assertEqual(row["steps_to_next"], following["turn_id"] - row["turn_id"])
                self.assertEqual(row["reward"], 0)
                self.assertFalse(row["done"])
            last = decisions[-1]
            self.assertTrue(last["done"])
            self.assertFalse(last["truncated"])
            self.assertEqual(last["next_legal_actions"], [])
            self.assertEqual(last["reward"], 1 if self.game["winner"] == player else -1)

    def test_cutoff_is_not_terminal_or_draw(self):
        game = generate_game(123, weights={"random": 1}, max_steps=2)
        self.assertIsNone(game["winner"])
        self.assertTrue(game["truncated"])
        for row in game["transitions"]:
            self.assertFalse(row["done"])
            self.assertTrue(row["truncated"])
            self.assertEqual(row["reward"], 0)

    def test_known_cards_from_take_and_bottom_trump(self):
        game = position([["6D"], ["7C"]], table=[{"attack": "6C", "defense": None}],
                        deck=["AS"], taking=True, attack_limit=1)
        before = game.snapshot()
        memory = PublicMemory()
        game.apply({"type": "pass"})
        after = game.snapshot()
        memory.update(before, {"type": "pass"}, after)
        self.assertEqual(memory.known, [{"AS"}, {"6C"}])
        self.assertEqual(memory.observation(after, 0)["known_opponent"], ["6C"])
        self.assertEqual(memory.observation(after, 1)["known_opponent"], ["AS"])

    def test_bottom_trump_drawn_by_defender(self):
        game = position([["6D", "7D", "8D", "9D", "10D", "JD"], ["7C"]],
                        table=[{"attack": "6C", "defense": "8C"}], deck=["AS"], attack_limit=2)
        before = game.snapshot()
        game.apply({"type": "pass"})
        memory = PublicMemory()
        memory.update(before, {"type": "pass"}, game.snapshot())
        self.assertEqual(memory.known, [set(), {"AS"}])

    def test_hidden_world_does_not_change_observation(self):
        first = _native.Game.new_game(22).snapshot()
        second = copy.deepcopy(first)
        second["hands"][1][0], second["deck"][0] = second["deck"][0], second["hands"][1][0]
        memory = PublicMemory()
        self.assertEqual(memory.observation(first, 0), memory.observation(second, 0))
        self.assertEqual(memory.observation(first, 0)["known_opponent"], [])

    def test_bad_weights(self):
        for items in (["random:nan"], ["mcts:-1"], ["missing"], [], ["random", "random"]):
            with self.assertRaises(ValueError):
                parse_strategies(items)


class BehaviourCloningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.game = generate_game(901, iterations=24, rollout_depth=48, max_steps=1000)

    def test_split_by_game_id_including_duplicate_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(5):
                game = {**self.game, "game_id": f"game-{i // 2}"}
                (root / f"{i}.json").write_text(json.dumps(game))
            train, validation = split_games(root, 0.4, 7)
            ids = lambda paths: {json.loads(path.read_text())["game_id"] for path in paths}
            self.assertTrue(ids(train).isdisjoint(ids(validation)))
            self.assertEqual(set(train + validation), set(root.glob("*.json")))
            self.assertEqual((train, validation), split_games(root, 0.4, 7))
            self.assertEqual(len(list(GamesDataset(train))), len(train) * len(self.game["transitions"]))

    def test_padding_loss_and_gradient_are_finite(self):
        rows = self.game["transitions"]
        smallest = min(rows, key=lambda row: len(row["legal_actions"]))
        largest = max(rows, key=lambda row: len(row["legal_actions"]))
        batch = collate_examples([encode_transition(smallest, "scores"), encode_transition(largest, "scores")])
        model = BehaviourCloningPolicy(32)
        logits = model(batch)
        self.assertTrue(torch.isneginf(logits[~batch["action_mask"]]).all())
        self.assertTrue((logits.softmax(-1)[~batch["action_mask"]] == 0).all())
        loss = batch_loss(logits, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_set_order_and_action_order(self):
        row = copy.deepcopy(self.game["transitions"][0])
        state, actions = row["state"], row["legal_actions"]
        model = BehaviourCloningPolicy(32).eval()
        original = model(collate_examples([encode_state(state, actions)]))
        state["hand"].reverse()
        state["discard"].reverse()
        reordered = model(collate_examples([encode_state(state, list(reversed(actions)))]))
        torch.testing.assert_close(original, reordered.flip(-1), atol=1e-6, rtol=1e-5)

    def test_defense_target_is_part_of_action(self):
        game = position([["7C", "8C"], ["AS"]],
                        table=[{"attack": "6C", "defense": None}, {"attack": "6D", "defense": None}],
                        attacker=1, turn=0, attack_limit=2)
        # Один козырь бьёт обе цели: их embeddings обязаны различаться.
        full = game.snapshot()
        full["hands"] = [["7S", "8C"], ["AS"]]
        full["discard"].remove("7S")
        full["discard"].append("7C")
        game = _native.Game(full)
        state = PublicMemory().observation(game.snapshot(), 0)
        encoded = encode_state(state, game.legal_moves())["actions"]
        defended = [a for a in encoded if a[0] == 1 and a[1] == 7]
        self.assertEqual(len(defended), 2)
        self.assertNotEqual(defended[0][2], defended[1][2])

    def test_reject_invalid_label_distribution_and_illegal_actions(self):
        row = copy.deepcopy(self.game["transitions"][0])
        bad = copy.deepcopy(row)
        bad["action"] = {"type": "take"}
        with self.assertRaises(ValueError):
            encode_transition(bad)
        for values in ([1.0], [float("nan")] * len(row["legal_actions"]), [-1.0] * len(row["legal_actions"])):
            bad = {**row, "action_probabilities": values}
            with self.assertRaises(ValueError):
                encode_transition(bad, "scores")
        with self.assertRaises(ValueError):
            encode_state(row["state"], [{"type": "take"}])

    def test_fit_small_batch_checkpoint_and_inference(self):
        torch.manual_seed(7)
        row = copy.deepcopy(self.game["transitions"][0])
        batch = collate_examples([encode_transition(row)])
        model = BehaviourCloningPolicy(32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = batch_loss(model(batch), batch).item()
        for _ in range(30):
            optimizer.zero_grad()
            loss = batch_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
        self.assertLess(batch_loss(model(batch), batch).item(), initial * 0.5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.pt"
            torch.save({"schema_version": 1, "model_config": {"hidden_size": 32},
                        "model_state_dict": model.state_dict()}, path)
            restored = load_policy(path)
            torch.testing.assert_close(model(batch), restored(batch))
            result = restored.suggest(row["state"])
            self.assertIn(result["action"], row["legal_actions"])
            self.assertAlmostEqual(sum(a["probability"] for a in result["moves"]), 1, places=6)
            self.assertEqual(result["action"], row["action"])

    def test_training_entry_point_with_and_without_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            for i in range(2):
                game = {**self.game, "game_id": f"smoke-{i}", "transitions": self.game["transitions"][:4]}
                (dataset / f"{i}.json").write_text(json.dumps(game))
            for fraction, target in ((0.5, "action"), (0, "scores")):
                with self.subTest(fraction=fraction), redirect_stdout(io.StringIO()):
                    output = root / target
                    self.assertEqual(main([
                        "train", "--dataset", str(dataset), "--output", str(output),
                        "--epochs", "1", "--hidden-size", "32", "--threads", "1",
                        "--target", target, "--validation-fraction", str(fraction),
                    ]), 0)
                    model = load_policy(output / "best.pt")
                    self.assertTrue((output / "last.pt").is_file())
                    metrics = json.loads((output / "metrics.jsonl").read_text())
                    self.assertEqual(metrics["validation"] is None, fraction == 0)
                    self.assertIn(model.suggest(self.game["transitions"][0]["state"])["action"],
                                  self.game["transitions"][0]["legal_actions"])


if __name__ == "__main__":
    unittest.main()
