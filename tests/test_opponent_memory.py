"""Известная рука соперника не забывается после одиночной ошибки детекции."""

import unittest

import numpy as np

from game_state.game import DurakGameState


class OpponentMemoryTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.values = dict(button="", mine="", opponent="", deque=12,
                           hand=["6C"], field=[])
        self.state = DurakGameState(
            trump="S", known_opponent_cards={"AS", "KH"},
            detectors={key: lambda image, key=key: self.values[key] for key in self.values})

    def update(self, *, slow_every=1, **values):
        self.values.update(values)
        self.state.update(self.frame, slow_every=slow_every)
        self.assertTrue(self.state.known_opponent_cards.isdisjoint(self.state.hand_cards))
        self.assertTrue(self.state.known_opponent_cards.isdisjoint(self.state.field_cards))
        self.assertTrue(self.state.known_opponent_cards.isdisjoint(self.state.out_cards))

    def assert_remembered(self):
        self.assertIn("AS", self.state.known_opponent_cards)
        self.assertNotIn("AS", self.state.unknown_opponent_cards)

    def test_one_false_field_detection_does_not_forget_card(self):
        self.update(field=["AS"])
        self.assert_remembered()
        self.update(field=[])
        self.assert_remembered()

    def test_one_false_hand_detection_does_not_forget_card(self):
        self.update(hand=["6C", "AS"])
        self.assert_remembered()
        self.update(hand=["6C"])
        self.assert_remembered()

    def test_confirmed_play_removes_only_played_card(self):
        self.update(field=["AS"])
        self.update(field=["AS", "6D"])
        self.assertEqual(self.state.known_opponent_cards, {"KH"})
        self.assertEqual(self.state.field_cards, {"AS", "6D"})
        self.assertNotIn("AS", self.state.unknown_opponent_cards)

    def test_confirmed_hand_observation_transfers_card(self):
        self.update(hand=["6C", "AS"])
        self.update(hand=["7C", "AS"])
        self.assertEqual(self.state.known_opponent_cards, {"KH"})
        self.assertEqual(self.state.hand_cards, {"7C", "AS"})

    def test_interrupted_and_alternating_zones_do_not_confirm(self):
        for values in (
            dict(field=["AS"]), dict(field=[]), dict(field=["AS"]),
            dict(field=[], hand=["6C", "AS"]), dict(hand=["6C"], field=["AS"]),
        ):
            self.update(**values)
            self.assert_remembered()

    def test_ambiguous_observations_reset_confirmation(self):
        self.update(field=["AS"])
        self.update(field=["AS"], hand=["AS", "6C"])
        self.assert_remembered()
        self.update(hand=["6C"])
        self.assert_remembered()
        self.update()
        self.assertNotIn("AS", self.state.known_opponent_cards)

    def test_skipped_detectors_do_not_confirm_or_clear_candidates(self):
        self.update(field=["AS"], slow_every=5)
        for _ in range(4):
            self.update(slow_every=5)
            self.assert_remembered()
        self.update(slow_every=5)
        self.assertNotIn("AS", self.state.known_opponent_cards)

    def test_layout_keeps_indices_and_does_not_mutate_detector_result(self):
        layout = [{"card": "AS", "covers": None}, {"card": "6D", "covers": 0}]
        hand_layout = [{"card": "KH", "bbox": (0, 0, 1, 1)}]
        self.update(field={"cards": ["AS", "6D"], "layout": layout},
                    hand={"cards": ["6C", "KH"], "layout": hand_layout})
        self.assertIsNone(self.state.field_layout[0]["card"])
        self.assertEqual(self.state.field_layout[1]["covers"], 0)
        self.assertIsNone(self.state.hand_layout[0]["card"])
        self.assertEqual(layout[0]["card"], "AS")
        self.assertEqual(hand_layout[0]["card"], "KH")
        self.update()
        self.assertEqual(self.state.field_layout, layout)
        self.assertEqual(self.state.hand_layout, hand_layout)

    def test_bat_noise_does_not_forget_unplayed_known_card(self):
        self.update(field=["8D", "9D"])
        self.update()
        self.update(mine="Bat", field=["8D", "9D", "AS"])
        self.assert_remembered()
        self.assertEqual(self.state.out_cards, {"8D", "9D"})
        self.update(field=[])
        self.assert_remembered()

    def test_confirmed_card_can_go_to_discard_or_be_taken(self):
        for actor, action in (("mine", "Bat"), ("mine", "I take"), ("opponent", "I take")):
            with self.subTest(actor=actor, action=action):
                self.setUp()
                for _ in range(2):
                    self.update(field=["AS", "6D"])
                self.update(field=[], **{actor: action})
                target = (self.state.out_cards if action == "Bat" else
                          self.state.hand_cards if actor == "mine" else self.state.known_opponent_cards)
                self.assertIn("AS", target)
                self.assertNotIn("AS", self.state.unknown_opponent_cards)

    def test_threshold_one_preserves_immediate_transition(self):
        self.state.field_confirmation_frames = 1
        self.update(field=["AS"])
        self.assertNotIn("AS", self.state.known_opponent_cards)

    def test_replay_after_opponent_take_confirms_without_extra_delay(self):
        self.state.known_opponent_cards.clear()
        self.update(field=["AS", "6D"])
        self.update()
        self.update(opponent="I take", field=[])
        self.assert_remembered()
        self.update(opponent="", field=["AS"])
        self.assert_remembered()
        self.update()
        self.update(mine="Bat", field=[])
        self.assertEqual(self.state.last_out_cards, {"AS"})
        self.assertEqual(self.state.known_opponent_cards, {"6D"})

    def test_known_card_played_after_bat_confirms_without_extra_delay(self):
        self.update(field=["8D", "9D"])
        self.update()
        self.update(mine="Bat", field=[])
        self.update(mine="", field=["AS"])
        self.assert_remembered()
        self.update()
        self.update(mine="Bat", field=[])
        self.assertEqual(self.state.last_out_cards, {"AS"})

    def test_resize_and_reset_clear_pending_observations(self):
        self.update(field=["AS"])
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.update()
        self.assert_remembered()
        self.state.reset(trump="S")
        self.assertFalse(self.state._opponent_move_candidates)


if __name__ == "__main__":
    unittest.main()
