import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from behaviour_cloning import BehaviourCloningEngine, BehaviourCloningPolicy, CARDS
from game_state.bot import format_state, main, recommendation_overlay, state_snapshot
from game_state.engine_process import _initialize_engine, _suggest
from game_state.game import DurakGameState


def snapshot(hand, enemy, table=(), *, phase="your_turn", button="yourturn"):
    layout = []
    for attack, defense in table:
        root = len(layout)
        layout.append({"card": attack, "covers": None})
        if defense:
            layout.append({"card": defense, "covers": root})
    field = [item["card"] for item in layout]
    result = state_snapshot(DurakGameState())
    result.update(phase=phase, button=button, trump="S", hand_cards=hand,
                  opponent_card_count=len(enemy), known_opponent_cards=enemy,
                  deck_remaining=0, field_cards=field, field_layout=layout,
                  out_cards=sorted(set(CARDS) - set(hand + enemy + field)),
                  unknown_opponent_cards=[])
    return result


class BcBotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.checkpoint = Path(cls.temporary.name) / "bc.pt"
        model = BehaviourCloningPolicy(32)
        # Равные logits: воспроизводимо выбирается первое легальное действие.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        torch.save({"schema_version": 1, "model_config": {"hidden_size": 32},
                    "model_state_dict": model.state_dict()}, cls.checkpoint)
        cls.engine = BehaviourCloningEngine(cls.checkpoint)

    def test_defense_has_target_for_phone_and_console(self):
        state = snapshot(["7C"], ["8D"], [("6C", None)], phase="defend_or_take", button="itake")
        result = self.engine.suggest(state)
        self.assertEqual(result["action"]["type"], "defend")
        self.assertEqual(result["action"]["target_card"], "6C")
        self.assertEqual(result["moves"][0]["target_card"], "6C")
        self.assertEqual(result["policy_probability"], 0.5)
        self.assertIn("Defend 6C with 7C", recommendation_overlay(result))
        panel = format_state({**state, "recommendation": result})
        self.assertIn("BC:", panel)
        self.assertIn("50.0%", panel)
        self.assertNotIn("симуляций", panel)

    def test_pass_button_resolves_throw_in_even_during_opponent_phase(self):
        for button, defense in (("pass", None), ("bat", "7C")):
            with self.subTest(button=button):
                state = snapshot(["8D"], ["AS"], [("6C", defense)], phase="opponent_turn", button=button)
                if button == "pass":
                    state["opponent"] = "itake"
                result = self.engine.suggest(state)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["action"]["type"], "pass")
                self.assertEqual(result["action"]["button"].lower(), button)

    def test_waiting_and_bad_recognition(self):
        for state in ({"phase": "ready"}, {"phase": "opponent_turn"}, {"phase": "your_turn"}):
            self.assertEqual(self.engine.suggest(state)["status"], "waiting")
        state = snapshot(["6C"], ["7C"])
        state["opponent_card_count"] = 9
        with patch("game_state.engine_process._engine", self.engine):
            self.assertEqual(_suggest(state)["status"], "invalid_state")

    def test_worker_loads_bc_once_and_keeps_options(self):
        options = {"bc_checkpoint": str(self.checkpoint), "device": "cpu", "trump": "S"}
        original = dict(options)
        with patch("game_state.engine_process._engine"), patch("game_engine.DurakEngine") as mcts:
            _initialize_engine(options)
            self.assertEqual(_suggest(snapshot(["6C"], ["7C"]))["engine"], "bc")
            mcts.assert_not_called()
        self.assertEqual(options, original)

    def test_cli_passes_bc_options_without_connecting(self):
        with patch("game_state.bot.preload_models"), patch("iphone_screen.iphone_client_v2.IPhoneRemote"), patch(
            "game_state.bot.run_bot"
        ) as run:
            self.assertEqual(main(["--bc-checkpoint", str(self.checkpoint), "--bc-device", "cpu", "--trump", "S"]), 0)
        self.assertEqual(run.call_args.kwargs["engine_options"],
                         {"bc_checkpoint": str(self.checkpoint.resolve()), "device": "cpu",
                          "trump": "S", "bottom_trump": None})

    def test_missing_checkpoint_fails_before_phone_connection(self):
        with contextlib.redirect_stderr(io.StringIO()), patch("iphone_screen.iphone_client_v2.IPhoneRemote") as phone:
            with self.assertRaises(SystemExit) as result:
                main(["--bc-checkpoint", str(self.checkpoint.parent / "missing.pt")])
            self.assertEqual(result.exception.code, 2)
            phone.assert_not_called()


if __name__ == "__main__":
    unittest.main()
