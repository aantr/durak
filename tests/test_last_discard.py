import unittest

import numpy as np

from game_state.bot import format_state, state_snapshot
from game_state.game import DurakGameState


class LastDiscardTests(unittest.TestCase):
    def setUp(self):
        self.values = dict(button="I take", mine="", opponent="", deque=24,
                           hand=["6C"], field=["8H", "9H"])
        self.state = DurakGameState(
            trump="S", field_confirmation_frames=1, terminal_field_frames=1,
            detectors={key: lambda image, key=key: self.values[key] for key in self.values})
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def update(self, **values):
        self.values.update(values)
        self.state.update(self.frame)

    def test_empty_at_start_and_after_reset(self):
        self.assertIn("Последняя бита [0]:  —", format_state(state_snapshot(self.state)))
        self.update(mine="Bat")
        self.state.reset(trump="S")
        self.assertEqual(self.state.last_out_cards, set())

    def test_last_discard_is_replaced_not_accumulated(self):
        self.update(mine="Bat")
        self.assertEqual(self.state.last_out_cards, {"8H", "9H"})
        snapshot = state_snapshot(self.state)
        self.update(mine="", field=["8D", "9D"])
        self.assertEqual(self.state.last_out_cards, {"8H", "9H"})
        self.update(opponent="Bat", field=[])
        self.assertEqual(self.state.last_out_cards, {"8D", "9D"})
        self.assertEqual(self.state.out_cards, {"8H", "9H", "8D", "9D"})
        self.assertEqual(snapshot["last_out_cards"], ["8H", "9H"])
        self.assertIn("Последняя бита [2]:  8♦  9♦", format_state(state_snapshot(self.state)))

    def test_repeated_bat_does_not_erase_last_discard(self):
        self.update(mine="Bat")
        for _ in range(3):
            self.update(opponent="Bat", field=[])
            self.assertEqual(self.state.last_out_cards, {"8H", "9H"})

    def test_take_by_either_player_preserves_last_discard(self):
        for actor in ("mine", "opponent"):
            with self.subTest(actor=actor):
                self.setUp()
                self.update(mine="Bat")
                self.update(mine="", field=["8D", "9D"])
                self.update(**{actor: "I take", "field": []})
                self.assertEqual(self.state.last_out_cards, {"8H", "9H"})

    def test_last_discard_contains_only_actually_transferred_cards(self):
        self.update(field=["6C", "8H", "9H"], mine="Bat")
        self.assertEqual(self.state.last_out_cards, {"8H", "9H"})
        self.assertIsNot(self.state.last_out_cards, self.state.out_cards)
        self.assertTrue(self.state.last_out_cards.isdisjoint(self.state.unknown_opponent_cards))
        self.update(mine="", field=["8D", "9D"])
        self.state.field_confirmation_frames = 3
        self.update(field=[], opponent="Bat")
        # Таблица уже подтверждена предыдущим кадром с порогом 1.
        self.assertEqual(self.state.last_out_cards, {"8D", "9D"})

    def test_empty_transfer_replaces_previous_discard_with_empty_set(self):
        self.update(mine="Bat")
        self.state.field_confirmation_frames = 3
        self.update(mine="", field=["8D", "9D"])
        self.update(opponent="Bat", field=[])
        self.assertEqual(self.state.last_out_cards, set())
        self.assertEqual(self.state.out_cards, {"8H", "9H"})


if __name__ == "__main__":
    unittest.main()
