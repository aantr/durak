import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from game_engine.game_engine import DurakEngine
from game_state.bot import format_state, state_snapshot
from game_state.game import DurakGameState, _default_trump


class TrumpTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.detector = Mock(return_value="S")
        self.state = DurakGameState(detectors={
            "button": lambda _: "your turn", "opponent": lambda _: "",
            "mine": lambda _: "", "deque": lambda _: 24,
            "hand": lambda _: ["6C"], "field": lambda _: [],
            "trump": self.detector,
        }, deck_confirmation_frames=1)  # Проверяем подтверждение козыря отдельно от счётчика.

    def test_confirm_and_remember_trump(self):
        self.state.update(self.frame)
        self.assertIsNone(self.state.trump)
        self.state.update(self.frame)
        self.assertEqual(self.state.trump, "S")
        self.detector.return_value = "H"
        self.state.update(self.frame)
        self.assertEqual(self.state.trump, "S")
        self.assertEqual(self.detector.call_count, 2)
        snapshot = state_snapshot(self.state)
        self.assertEqual(snapshot["trump"], "S")
        self.assertIn("Козырь:   ♠", format_state(snapshot))

    def test_missing_result_breaks_confirmation(self):
        self.detector.side_effect = ["S", None, "S", "H", "H"]
        for _ in range(4):
            self.state.update(self.frame)
            self.assertIsNone(self.state.trump)
        self.state.update(self.frame)
        self.assertEqual(self.state.trump, "H")

    def test_manual_trump_skips_detection(self):
        self.state.trump = "D"
        self.state.update(self.frame)
        self.detector.assert_not_called()

    def test_crop_rotation_and_class_mapping(self):
        frame = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        model = Mock()
        result = SimpleNamespace(probs=SimpleNamespace(top1=0, top1conf=0.95), names={0: "clubs"})
        model.predict.return_value = [result]
        with patch("constants.constants.CROP_TRUMP", {(4, 4): (1, 0, 3, 3)}), patch(
            "game_state.game._models", return_value=(None, model)
        ):
            self.assertEqual(_default_trump(frame), "C")
            np.testing.assert_array_equal(
                model.predict.call_args.kwargs["source"],
                cv2.rotate(frame[0:3, 1:3], cv2.ROTATE_90_COUNTERCLOCKWISE),
            )
            result.probs.top1conf = 0.5
            self.assertIsNone(_default_trump(frame))

    def test_unknown_resolution_skips_model(self):
        with patch("game_state.game._models") as models:
            self.assertIsNone(_default_trump(self.frame))
            models.assert_not_called()

    def test_engine_waits_then_uses_detected_trump(self):
        native = Mock()
        native.analyze.return_value = {"action": {"type": "attack", "card": "6C"}, "moves": []}
        with patch("game_engine.game_engine._native_module", return_value=native):
            engine = DurakEngine()
            self.state.update(self.frame)
            self.assertEqual(engine.suggest(self.state)["status"], "waiting")
            native.analyze.assert_not_called()
            self.state.update(self.frame)
            self.assertEqual(engine.suggest(self.state)["status"], "ok")
            self.assertEqual(native.analyze.call_args.args[0]["trump"], "S")
            engine.trump = "D"
            engine.suggest(self.state)
            self.assertEqual(native.analyze.call_args.args[0]["trump"], "D")


if __name__ == "__main__":
    unittest.main()
