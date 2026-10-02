"""После конечной надписи поле собирается по нескольким свежим кадрам."""

import unittest
from unittest.mock import Mock

import numpy as np

from game_state.game import DurakGameState, TERMINAL_FIELD_FRAMES


class TerminalFieldWindowTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.values = dict(button="", mine="", opponent="", deque=24,
                           hand=["6C"], field=["8H"])
        self.detectors = {
            key: Mock(side_effect=lambda image, key=key: self.values[key])
            for key in self.values
        }
        self.state = DurakGameState(trump="S", detectors=self.detectors)
        self.assertEqual(TERMINAL_FIELD_FRAMES, 5)

    def update(self, **values):
        self.values.update(values)
        self.state.update(self.frame, slow_every=100)

    def test_bat_reads_five_consecutive_frames_and_adds_late_confirmed_card(self):
        self.state.update(self.frame)
        self.state.update(self.frame)
        self.assertEqual(self.state._confirmed_table, {"8H"})

        self.update(mine="Bat", field=["8H", "9H", "AS"])
        self.assertTrue(self.state.terminal_pending)
        self.assertFalse(self.state.out_cards)
        self.update(mine="", field=["8H", "9H"])
        self.update(field=["8H", "9H"])
        self.update(field=[])
        self.update(field=[])
        self.assertFalse(self.state.terminal_pending)
        self.assertEqual(self.state.out_cards, {"8H", "9H"})
        self.assertEqual(self.state.last_out_cards, {"8H", "9H"})
        self.assertEqual(self.detectors["field"].call_count, 2 + TERMINAL_FIELD_FRAMES)
        self.update()
        self.assertEqual(self.detectors["field"].call_count, 2 + TERMINAL_FIELD_FRAMES)

    def test_opponent_itake_uses_cards_seen_after_text_disappears(self):
        self.state.update(self.frame)
        self.update(opponent="I take", field=["8H"])
        self.update(opponent="", field=["8H", "9H"])
        self.update(field=["8H", "9H"])
        self.update(field=[])
        self.update(field=[])
        self.assertEqual(self.state.known_opponent_cards, {"8H", "9H"})
        self.assertFalse(self.state.field_cards)
        self.assertEqual(self.detectors["field"].call_count, 1 + TERMINAL_FIELD_FRAMES)

    def test_opponent_bat_starts_window_too(self):
        self.state.update(self.frame)
        self.update(opponent="Bat", field=["8H"])
        for _ in range(TERMINAL_FIELD_FRAMES - 1):
            self.update(opponent="", field=["8H"])
        self.assertEqual(self.state.out_cards, {"8H"})
        self.assertEqual(self.detectors["field"].call_count, 1 + TERMINAL_FIELD_FRAMES)

    def test_bat_does_not_move_cards_from_next_round(self):
        self.state.update(self.frame)
        self.state.update(self.frame)
        self.update(mine="Bat", field=["8H"])
        self.update(mine="", field=[])
        for _ in range(TERMINAL_FIELD_FRAMES - 2):
            self.update(field=["9D"])
        self.assertEqual(self.state.out_cards, {"8H"})
        self.assertEqual(self.state.field_cards, {"9D"})

    def test_opponent_itake_with_pass_can_collect_after_empty_frame(self):
        self.state.update(self.frame)
        self.update(opponent="I take", button="Pass", field=["8H"])
        self.update(opponent="", field=[])
        self.update(field=["9H"])
        self.update(field=["9H"])
        self.update(field=[])
        self.assertFalse(self.state.known_opponent_cards)
        self.assertEqual(self.state.field_cards, {"8H", "9H"})

    def test_bat_replaces_wrong_suit_at_same_position(self):
        self.state.trump = "D"
        def layout(cover):
            return {"cards": ["6H", cover], "layout": [
                {"card": "6H", "bbox": (10, 10, 30, 90), "covers": None},
                {"card": cover, "bbox": (80, 15, 100, 95), "covers": 0},
            ]}
        self.values["field"] = layout("JH")
        self.state.update(self.frame)
        self.state.update(self.frame)
        self.assertEqual(self.state._confirmed_table, {"6H", "JH"})
        self.update(mine="Bat", field=layout("JD"))
        for _ in range(TERMINAL_FIELD_FRAMES - 1):
            self.update(field=layout("JD"))
        self.assertEqual(self.state.out_cards, {"6H", "JD"})
        self.assertNotIn("JH", self.state.out_cards)

    def test_bat_rejects_illegal_cover_with_geometry(self):
        self.state.trump = "D"
        observed = {"cards": ["6H", "7S"], "layout": [
            {"card": "6H", "bbox": (10, 10, 30, 90), "covers": None},
            {"card": "7S", "bbox": (80, 15, 100, 95), "covers": 0},
        ]}
        self.values["field"] = observed
        self.state.update(self.frame)
        self.state.update(self.frame)
        self.update(mine="Bat")
        for _ in range(TERMINAL_FIELD_FRAMES - 1):
            self.update()
        self.assertFalse(self.state.out_cards)

    def test_late_stable_correction_replaces_earlier_majority(self):
        self.state.trump = "D"
        def layout(cover):
            return {"cards": ["6H", cover], "layout": [
                {"card": "6H", "bbox": (10, 10, 30, 90), "covers": None},
                {"card": cover, "bbox": (80, 15, 100, 95), "covers": 0},
            ]}
        self.values["field"] = layout("JH")
        self.state.update(self.frame)
        self.state.update(self.frame)
        self.update(mine="Bat")
        self.update()
        self.update()
        self.update(field=layout("JD"))
        self.update(field=layout("JD"))
        self.assertEqual(self.state.out_cards, {"6H", "JD"})

    def test_bat_includes_late_confirmed_pair(self):
        self.state.trump = "D"
        first = [
            {"card": "6H", "bbox": (10, 10, 30, 90), "covers": None},
            {"card": "JH", "bbox": (80, 15, 100, 95), "covers": 0},
        ]
        second = [
            {"card": "8S", "bbox": (210, 10, 230, 90), "covers": None},
            {"card": "9S", "bbox": (280, 15, 300, 95), "covers": 2},
        ]
        self.values["field"] = {"cards": ["6H", "JH"], "layout": first}
        self.state.update(self.frame)
        self.state.update(self.frame)
        self.update(mine="Bat")
        self.update()
        self.update()
        full = {"cards": ["6H", "JH", "8S", "9S"], "layout": first + second}
        self.update(field=full)
        self.update(field=full)
        self.assertEqual(self.state.out_cards, {"6H", "JH", "8S", "9S"})

    def test_new_round_can_reuse_same_table_positions(self):
        def layout(attack, cover):
            return {"cards": [attack, cover], "layout": [
                {"card": attack, "bbox": (10, 10, 30, 90), "covers": None},
                {"card": cover, "bbox": (80, 15, 100, 95), "covers": 0},
            ]}
        self.values["field"] = layout("6H", "7H")
        self.state.update(self.frame)
        self.state.update(self.frame)
        self.update(mine="Bat")
        self.update(mine="", field=[])
        self.update(field=layout("8H", "9H"))
        self.update()
        self.update()
        self.assertEqual(self.state.out_cards, {"6H", "7H"})
        self.assertEqual(self.state.field_cards, {"8H", "9H"})

    def test_reset_discards_window_and_keeps_limit(self):
        self.update(mine="Bat")
        self.assertTrue(self.state.terminal_pending)
        self.state.reset(trump="S")
        self.assertFalse(self.state.terminal_pending)
        self.assertEqual(self.state.terminal_field_frames, TERMINAL_FIELD_FRAMES)

    def test_invalid_window_length(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DurakGameState(terminal_field_frames=value)


if __name__ == "__main__":
    unittest.main()
