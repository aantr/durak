import unittest
from unittest.mock import Mock, patch

import numpy as np

from game_state import game
from game_state.game import DurakGameState
from game_state.benchmark import measure
from game_state.bot import main, run_bot


class MultirateTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((6, 6, 3), dtype=np.uint8)
        self.values = dict(mine="", opponent="", button="I take", deque=24,
                           hand=["AS"], field=["6C", "7C"], trump="S")
        self.order = []

        def detect(name):
            self.order.append(name)
            return self.values[name]

        self.detectors = {name: Mock(side_effect=lambda _, name=name: detect(name))
                          for name in self.values}
        self.state = DurakGameState(trump="S", detectors=self.detectors)

    def update(self, **values):
        self.values.update(values)
        return self.state.update(self.frame, slow_every=2)

    def test_texts_every_frame_and_others_every_second_frame(self):
        for _ in range(5):
            self.update()
        for key in ("mine", "opponent"):
            self.assertEqual(self.detectors[key].call_count, 5)
        for key in ("button", "deque", "hand", "field"):
            self.assertEqual(self.detectors[key].call_count, 3)
        self.assertEqual(self.order[:2], ["mine", "opponent"])

    def test_fast_frame_does_not_confirm_cached_cards_or_count(self):
        self.update()
        self.assertEqual(self.state._table_candidate_frames, 1)
        self.assertEqual(self.state._deck_candidate_frames, 1)
        with patch.object(self.state, "_apply_table_events", wraps=self.state._apply_table_events) as events:
            self.update()
        events.assert_not_called()
        self.assertFalse(self.state._confirmed_table)
        self.assertEqual(self.state._deck_candidate_frames, 1)
        self.assertIsNone(self.state.deck_remaining)
        self.update()
        self.assertEqual(self.state._confirmed_table, {"6C", "7C"})
        self.assertIsNone(self.state.deck_remaining)
        self.update()
        self.update()
        self.assertEqual(self.state.deck_remaining, 24)

    def test_skipped_deck_detection_is_not_missing_text(self):
        self.state.deck_remaining = 9
        for _ in range(4):
            self.update(deque=None)
            self.assertEqual(self.state.deck_remaining, 9)
        self.assertEqual(self.state._deck_missing_frames, 2)
        self.update(deque=None)
        self.assertEqual(self.state.deck_remaining, 0)

    def test_bat_on_fast_frame_refreshes_real_table_and_transfers_once(self):
        self.update()
        self.update(mine=" B a t ")
        self.assertEqual(self.detectors["field"].call_count, 2)
        self.assertEqual(self.state.out_cards, {"6C", "7C"})
        for _ in range(4):
            self.update()
        self.assertEqual(self.state.out_cards, {"6C", "7C"})

    def test_opponent_take_on_fast_frame_reads_fresh_button(self):
        self.update(button="your turn")
        self.update(opponent="I take", button="Pass")
        self.assertEqual(self.state.phase, "throw_in")
        self.assertEqual(self.detectors["button"].call_count, 2)
        self.assertEqual(self.state.field_cards, {"6C", "7C"})
        self.assertFalse(self.state.known_opponent_cards)

    def test_text_clearing_forces_refresh_but_whitespace_does_not(self):
        self.update(mine="Pass")
        self.update(mine=" P a s s ")
        self.assertEqual(self.detectors["field"].call_count, 1)
        self.update()
        self.update(mine="")
        self.assertEqual(self.detectors["field"].call_count, 3)

    def test_resolution_change_forces_full_pass(self):
        self.update()
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.update()
        self.assertEqual(self.detectors["field"].call_count, 2)

    def test_reset_starts_with_full_pass(self):
        self.update()
        self.state.reset(trump="S")
        self.update()
        self.assertEqual(self.detectors["field"].call_count, 2)

    def test_default_state_api_still_runs_all_detectors(self):
        for _ in range(2):
            self.state.update(self.frame)
        self.assertEqual(self.detectors["field"].call_count, 2)

    def test_fast_overlay_uses_current_frame_without_old_card_pixels(self):
        def mark(key, image, *, visualization=None):
            if visualization is not None:
                bounds = (0, 0, 2, 2) if key in ("mine", "opponent") else (3, 3, 5, 5)
                visualization.add_crop({(6, 6): bounds})[0, 0] = 255
            return self.values[key]

        defaults = {key: (lambda image, key=key, **kw: mark(key, image, **kw)) for key in self.values}
        with patch.dict(game._DEFAULT_DETECTORS, defaults):
            state = DurakGameState(trump="S")
            state.update(self.frame, slow_every=5, draw_detections=True)
            self.assertTrue(state.annotated_frame[3, 3].all())
            for level in range(1, 5):
                current = np.full_like(self.frame, level)
                state.update(current, slow_every=5, draw_detections=True)
                expected = current.copy()
                expected[0, 0] = expected[3, 3] = 255
                np.testing.assert_array_equal(state.annotated_frame, expected)
                np.testing.assert_array_equal(current, np.full_like(current, level))
        self.assertFalse(self.frame.any())

    def test_empty_detection_clears_only_its_own_overlay(self):
        visible = {"mine", "field"}
        calls = []

        def mark(key, image, *, visualization=None):
            calls.append(key)
            if visualization is not None:
                # Перекрывающиеся кропы не должны стирать чужие подписи.
                crop = visualization.add_crop({})
                if key in visible:
                    position = 0 if key == "mine" else 3
                    crop[position, position] = 255
            return self.values[key]

        defaults = {key: (lambda image, key=key, **kw: mark(key, image, **kw)) for key in self.values}
        with patch.dict(game._DEFAULT_DETECTORS, defaults):
            state = DurakGameState(trump="S")
            state.update(self.frame, slow_every=5, draw_detections=True)
            visible.remove("field")
            for _ in range(4):
                state.update(self.frame, slow_every=5, draw_detections=True)
                self.assertTrue(state.annotated_frame[3, 3].all())
            self.assertEqual(calls.count("field"), 1)
            state.update(self.frame, slow_every=5, draw_detections=True)
            self.assertFalse(state.annotated_frame[3, 3].any())
            self.assertTrue(state.annotated_frame[0, 0].all())
            visible.clear()
            state.update(self.frame, slow_every=5, draw_detections=True)
            self.assertFalse(state.annotated_frame.any())
            self.assertEqual(calls.count("field"), 2)

    def test_overlay_cache_replaced_and_cleared_on_resize_reset_and_toggle(self):
        position = 4
        calls = []

        def mark(key, image, *, visualization=None):
            calls.append(key)
            if visualization is not None and key == "field":
                visualization.add_crop({})[position, position] = 255
            return self.values[key]

        defaults = {key: (lambda image, key=key, **kw: mark(key, image, **kw)) for key in self.values}
        with patch.dict(game._DEFAULT_DETECTORS, defaults):
            state = DurakGameState(trump="S")
            state.update(self.frame, slow_every=5, draw_detections=True)
            position = 1
            state.update(self.frame, draw_detections=True)
            self.assertFalse(state.annotated_frame[4, 4].any())
            self.assertTrue(state.annotated_frame[1, 1].all())
            position = 0
            small = np.zeros((1, 1, 3), dtype=np.uint8)
            state.update(small, slow_every=5, draw_detections=True)
            self.assertTrue(state.annotated_frame[0, 0].all())
            state.update(small, slow_every=5, draw_detections=False)
            self.assertFalse(state._detection_overlays)
            self.assertIsNone(state.annotated_frame)
            count = calls.count("field")
            state.update(small, slow_every=5, draw_detections=True)
            self.assertEqual(calls.count("field"), count + 1)
            self.assertTrue(state.annotated_frame[0, 0].all())
            state.reset(trump="S")
            self.assertFalse(state._detection_overlays)
            self.assertIsNone(state.annotated_frame)

    def test_invalid_interval_does_not_run_detectors(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.state.update(self.frame, slow_every=value)
                with self.assertRaises(ValueError):
                    run_bot(Mock(), slow_every=value)
        self.assertFalse(self.order)

    def test_cli_forwards_interval(self):
        with patch("game_state.bot.preload_models"), patch("iphone_screen.iphone_client_v2.IPhoneRemote"), patch("game_state.bot.run_bot") as run:
            main(["--slow-every", "3"])
        self.assertEqual(run.call_args.kwargs["slow_every"], 3)

    def test_benchmark_records_real_calls_not_cached_results(self):
        detectors = {key: Mock(return_value=value) for key, value in self.values.items()}
        with patch.dict(game._DEFAULT_DETECTORS, detectors):
            result = measure([("fake.png", self.frame)], repeats=4, warmup=0, draw=False, slow_every=2)
        self.assertEqual(result["summary"]["mine"]["count"], 4)
        self.assertEqual(result["summary"]["field"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
