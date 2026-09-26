"""Budgets, deterministic reduction, validation and Python-to-native wiring."""

import copy
import unittest
from unittest.mock import Mock, patch

from game_engine import DurakEngine, _native
from test_engine import observation, position


class ParallelSearchTests(unittest.TestCase):
    def setUp(self):
        self.obs = observation(_native.Game.new_game(17, attacker=0).snapshot())
        self.obs["known_opponent"] = []

    def test_deals_budget_and_reproducibility_across_threads(self):
        before = copy.deepcopy(self.obs)
        results = [_native.analyze(self.obs, rollouts=37, deals=7, threads=threads,
                                   time_limit_ms=0, seed=21, exploration=0.8)
                   for threads in (1, 2, 4, 12)]
        for threads, result in zip((1, 2, 4, 7), results):
            self.assertEqual(result["threads"], threads)
            self.assertEqual(result["iterations"], 37 * 7)
            self.assertEqual(result["deals_started"], 7)
            self.assertEqual(result["deals_completed"], 7)
            self.assertEqual(result["search_mode"], "determinized")
            self.assertEqual(sum(m["visits"] for m in result["moves"]), 37 * 7)
            self.assertEqual(result["terminal_rollouts"] + result["cutoff_rollouts"], 37 * 7)
            self.assertEqual(result["moves"], results[0]["moves"])
            self.assertEqual(result["action"], results[0]["action"])
            self.assertEqual(result["moves"][0]["value"], max(m["value"] for m in result["moves"]))
        self.assertEqual(self.obs, before)

    def test_legacy_parallel_budget_is_not_multiplied(self):
        for budget, threads in ((103, 4), (2, 8)):
            a = _native.analyze(self.obs, iterations=budget, threads=threads, time_limit_ms=0, seed=3)
            b = _native.analyze(self.obs, iterations=budget, threads=threads, time_limit_ms=0, seed=3)
            self.assertEqual(a["iterations"], budget)
            self.assertEqual(sum(m["visits"] for m in a["moves"]), budget)
            self.assertEqual(a["threads"], min(budget, threads))
            self.assertEqual(a["deals_started"], 0)
            self.assertEqual(a["search_mode"], "ismcts")
            self.assertEqual(a["moves"], b["moves"])

    def test_determinized_search_finds_winning_defense(self):
        obs = observation(position([["7C"], ["8D"]], table=[{"attack": "6C"}],
                                   attacker=1, turn=0, attack_limit=1))
        for threads in (1, 3):
            result = _native.analyze(obs, rollouts=100, deals=5, threads=threads,
                                     time_limit_ms=0, seed=8)
            self.assertEqual(result["action"], dict(type="defend", card="7C", target=0))
            self.assertEqual(result["iterations"], 500)

    def test_merged_scores_are_weighted_by_visits(self):
        seed = 21
        mask = (1 << 64) - 1
        totals = {}
        for task in range(3):
            task_seed = seed
            if task:
                z = (seed + 0x9e3779b97f4a7c15 * task) & mask
                z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
                z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
                task_seed = z ^ (z >> 31)
            part = _native.analyze(self.obs, rollouts=45, deals=1, seed=task_seed, time_limit_ms=0)
            for move in part["moves"]:
                key = (move["type"], move.get("card"), move.get("target"))
                count, score = totals.get(key, (0, 0.0))
                totals[key] = count + move["visits"], score + (move["value"] or 0) * move["visits"]
        combined = _native.analyze(self.obs, rollouts=45, deals=3, threads=2, seed=seed, time_limit_ms=0)
        for move in combined["moves"]:
            count, score = totals[(move["type"], move.get("card"), move.get("target"))]
            self.assertEqual(move["visits"], count)
            self.assertAlmostEqual(move["value"], score / count)

    def test_shared_deadline_and_at_least_one_simulation(self):
        for limit in (0.000001, 10):
            result = _native.analyze(self.obs, rollouts=100000, deals=100,
                                     threads=4, time_limit_ms=limit, seed=9)
            self.assertGreaterEqual(result["iterations"], 1)
            self.assertLess(result["iterations"], 10000000)
            self.assertLess(result["deals_completed"], 100)
            self.assertLess(result["elapsed_ms"], 2000)
            self.assertEqual(sum(m["visits"] for m in result["moves"]), result["iterations"])

    def test_invalid_native_parameters(self):
        for options in ({"rollouts": -1}, {"deals": 0}, {"deals": 2},
                        {"rollouts": 100001, "deals": 100}, {"threads": 0},
                        {"threads": 257}, {"exploration": -1}, {"exploration": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                _native.analyze(self.obs, **options)

    def test_invalid_observation_with_threads_raises(self):
        self.obs["opponent_count"] += 1
        with self.assertRaises(ValueError):
            _native.analyze(self.obs, rollouts=10, deals=4, threads=4)

    def test_python_validates_before_loading_native(self):
        for options in ({"rollouts": 0}, {"deals": 3}, {"threads": 1.5},
                        {"threads": False}, {"exploration": float("inf")},
                        {"rollouts": 500000, "deals": 100}):
            with self.subTest(options=options), patch("game_engine.game_engine._native_module") as load:
                with self.assertRaises(ValueError):
                    DurakEngine(**options)
                load.assert_not_called()

    def test_python_forwards_all_settings(self):
        native = Mock()
        native.analyze.return_value = {"action": {"type": "attack", "card": "6C"}, "moves": []}
        with patch("game_engine.game_engine._native_module", return_value=native), patch(
            "game_engine.game_engine.observation_from_state", return_value={"turn": 0}
        ):
            engine = DurakEngine("S", rollouts=50, deals=7, exploration=0.8, threads=4)
            engine.suggest({"phase": "your_turn"})
        options = native.analyze.call_args.kwargs
        for name, value in dict(rollouts=50, deals=7, exploration=0.8, threads=4).items():
            self.assertEqual(options[name], value)


if __name__ == "__main__":
    unittest.main()
