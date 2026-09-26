"""Исчезновение счётчика пустой колоды и временные пропуски OCR."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from game_state.game import DurakGameState, _default_deque, _ocr_lines
from game_state.visualization import DetectionVisualization


class DeckCountTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.values = {
            "button": "your turn", "opponent": "", "mine": "",
            "deque": 2, "hand": ["6C"], "field": [],
        }
        self.state = DurakGameState(detectors={
            name: lambda image, name=name: self.values[name] for name in self.values
        }, deck_confirmation_frames=1)  # Здесь проверяется исчезновение надписи.

    def update(self, count):
        self.values["deque"] = count
        self.state.update(self.frame)

    def test_missing_count_confirms_zero_and_refreshes_opponent_count(self):
        self.update(2)
        for _ in range(2):
            self.update(None)
            self.assertEqual(self.state.deck_remaining, 2)
        self.update(None)
        self.assertEqual(self.state.deck_remaining, 0)
        self.assertEqual(self.state.opponent_card_count, 35)

    def test_recognized_count_resets_missing_streak(self):
        for count in (2, None, None, 1, None, None):
            self.update(count)
        self.assertEqual(self.state.deck_remaining, 1)
        self.update(None)
        self.assertEqual(self.state.deck_remaining, 0)
        self.update(24)
        self.assertEqual(self.state.deck_remaining, 24)

    def test_empty_count_at_startup(self):
        for count in (None, "", " \n\t"):
            self.update(count)
        self.assertEqual(self.state.deck_remaining, 0)

    def test_explicit_zero_is_immediate(self):
        self.update(2)
        self.update(0)
        self.assertEqual(self.state.deck_remaining, 0)

    def test_invalid_nonempty_count_breaks_missing_streak(self):
        for invalid in ("garbage", -1, 37):
            for count in (2, None, None, invalid, None, None):
                self.update(count)
            self.assertEqual(self.state.deck_remaining, 2)

    def test_confirmation_threshold_is_configurable(self):
        self.state.deck_empty_confirmation_frames = 1
        self.update(None)
        self.assertEqual(self.state.deck_remaining, 0)

    def test_invalid_confirmation_threshold(self):
        for threshold in (0, -1, 1.5, None):
            with self.assertRaises(ValueError):
                DurakGameState(deck_empty_confirmation_frames=threshold)


class DeckConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.count = 9
        self.state = DurakGameState(deck_remaining=9, detectors={
            "button": lambda _: "your turn", "opponent": lambda _: "",
            "mine": lambda _: "", "deque": lambda _: self.count,
            "hand": lambda _: [], "field": lambda _: [],
        })

    def update(self, count):
        self.count = count
        self.state.update(self.frame)

    def test_initial_count_requires_confirmation(self):
        self.state.deck_remaining = None
        for _ in range(2):
            self.update(6)
            self.assertIsNone(self.state.deck_remaining)
            self.assertIsNone(self.state.opponent_card_count)
        self.update(6)
        self.assertEqual(self.state.deck_remaining, 6)
        self.assertEqual(self.state.opponent_card_count, 30)

    def test_six_nine_flicker_does_not_change_confirmed_count(self):
        for count in (6, 9, 6, 6, 9, 6, 9):
            self.update(count)
            self.assertEqual(self.state.deck_remaining, 9)
        for _ in range(3):
            self.update(6)
        self.assertEqual(self.state.deck_remaining, 6)
        for count in (9, 6, 9, 9, 6):
            self.update(count)
            self.assertEqual(self.state.deck_remaining, 6)

    def test_stable_correction_is_not_permanently_locked_out(self):
        self.state.deck_remaining = 6
        for _ in range(3):
            self.update(9)
        self.assertEqual(self.state.deck_remaining, 9)

    def test_missing_invalid_and_current_count_break_confirmation(self):
        for interruption in (None, "", "bad", -1, 37, 9, float("inf")):
            with self.subTest(interruption=interruption):
                self.state.deck_remaining = 9
                for count in (6, 6, interruption, 6, 6):
                    self.update(count)
                self.assertEqual(self.state.deck_remaining, 9)
                self.update(6)
                self.assertEqual(self.state.deck_remaining, 6)

    def test_empty_deck_clears_pending_number_and_can_recover(self):
        for count in (6, 6, None, None, None):
            self.update(count)
        self.assertEqual(self.state.deck_remaining, 0)
        for _ in range(2):
            self.update(6)
            self.assertEqual(self.state.deck_remaining, 0)
        self.update(6)
        self.assertEqual(self.state.deck_remaining, 6)

    def test_reset_preserves_threshold_but_discards_candidate(self):
        self.state.deck_confirmation_frames = 2
        self.update(6)
        self.state.reset()
        self.assertEqual(self.state.deck_confirmation_frames, 2)
        self.assertIsNone(self.state._deck_candidate)
        self.update(6)
        self.assertIsNone(self.state.deck_remaining)
        self.update(6)
        self.assertEqual(self.state.deck_remaining, 6)

    def test_invalid_threshold(self):
        for threshold in (0, -1, 1.5, None, True):
            with self.assertRaises(ValueError):
                DurakGameState(deck_confirmation_frames=threshold)


class DeckOcrTests(unittest.TestCase):
    def test_counter_disables_rotation_and_keeps_crop_overlay(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        visualization = DetectionVisualization(frame)
        ocr = Mock()
        ocr.predict.return_value = [{
            "rec_texts": [" 6 "], "rec_scores": [.99],
            "rec_polys": [[[5, 20], [15, 20], [15, 30], [5, 30]]],
        }]
        with patch("constants.constants.CROP_DEQUE_LOST_CARDS", {(100, 80): (10, 10, 60, 70)}), patch(
            "game_state.game._ocr", return_value=ocr
        ):
            self.assertEqual(_default_deque(frame, visualization=visualization), 6)
        self.assertEqual(ocr.predict.call_count, 1)
        self.assertEqual(ocr.predict.call_args.args[0].shape, (60, 50, 3))
        self.assertEqual(ocr.predict.call_args.kwargs, dict(
            use_textline_orientation=False, use_doc_orientation_classify=False, use_doc_unwarping=False))
        annotated = visualization.render()
        self.assertTrue(annotated[10:70, 10:60].any())
        annotated[10:70, 10:60] = 0
        self.assertFalse(annotated.any())
        self.assertFalse(frame.any())

    def test_other_ocr_keeps_default_options(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        ocr = Mock()
        ocr.predict.return_value = [{"rec_texts": ["Pass"]}]
        with patch("game_state.game._ocr", return_value=ocr):
            self.assertEqual(_ocr_lines(frame, {}, lambda image, _: image), ["Pass"])
        ocr.predict.assert_called_once_with(frame)


if __name__ == "__main__":
    unittest.main()
