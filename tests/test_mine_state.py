"""Переходы состояния по надписям игроков без GPU и подключения iPhone."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from game_state.bot import format_state, state_snapshot
from game_state.game import DurakGameState, _default_mine, normalize_text
from game_state.visualization import DetectionVisualization


class MineStateTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.values = {
            "button": "I take", "opponent": "", "mine": "", "deque": 24,
            "hand": ["6C"], "field": ["8H", "9H"],
        }
        self.state = DurakGameState(detectors={
            name: lambda image, name=name: self.values[name] for name in self.values
        }, field_confirmation_frames=1)  # Здесь проверяются события без задержки подтверждения.

    def update(self, **values):
        self.values.update(values)
        return self.state.update(self.frame)

    def test_button_is_not_a_take_event(self):
        self.update()
        self.assertEqual(self.state.field_cards, {"8H", "9H"})
        self.assertEqual(self.state.hand_cards, {"6C"})

    def test_my_bat_moves_accumulated_table_even_after_it_disappears(self):
        self.update()
        self.update(field=[])
        self.update(mine=" B a t ")
        self.assertEqual(self.state.out_cards, {"8H", "9H"})
        self.assertFalse(self.state.field_cards)
        self.assertFalse(self.state.known_opponent_cards)

    def test_both_bat_labels_resolve_once(self):
        self.update()
        for _ in range(5):
            self.update(mine="Bat", opponent="Bat")
            self.assertEqual(self.state.out_cards, {"8H", "9H"})
            self.assertFalse(self.state.field_cards)

    def test_my_take_waits_for_hand_detection(self):
        self.update()
        self.update(mine="I take", field=[])
        for _ in range(3):
            self.update()
            self.assertEqual(self.state.hand_cards, {"6C", "8H", "9H"})
        self.assertFalse(self.state.out_cards)
        self.assertFalse(self.state.known_opponent_cards)
        self.update(hand=["6C", "8H", "9H"])
        self.update(mine="", hand=["6C", "9H"], field=["8H"])
        self.assertEqual(self.state.hand_cards, {"6C", "9H"})
        self.assertEqual(self.state.field_cards, {"8H"})

    def test_repeated_take_does_not_resurrect_confirmed_cards(self):
        self.update(mine="I take")
        self.update(hand=["6C", "8H", "9H"], field=[])
        self.update(hand=["6C"])
        self.assertEqual(self.state.hand_cards, {"6C"})

    def test_additional_cards_during_take(self):
        self.update(mine="I take")
        self.update(field=["8H", "9H", "8D"])
        self.assertEqual(self.state.hand_cards, {"6C", "8H", "9H", "8D"})
        self.assertFalse(self.state.field_cards)

    def test_opponent_take_and_later_replay(self):
        self.update()
        self.update(field=[], opponent="I take")
        self.assertEqual(self.state.known_opponent_cards, {"8H", "9H"})
        self.assertEqual(self.state.hand_cards, {"6C"})
        self.update(opponent="", field=["8H"])
        self.assertEqual(self.state.known_opponent_cards, {"9H"})

    def test_pass_from_either_player_leaves_cards_on_table(self):
        self.update(mine="Pass", opponent="Pass")
        self.assertEqual(self.state.field_cards, {"8H", "9H"})
        self.assertFalse(self.state.out_cards)
        self.assertEqual(self.state.hand_cards, {"6C"})

    def test_following_round_can_end_in_bat_again(self):
        self.update(mine="Bat")
        self.update(mine="", field=[])
        self.update(field=["8D", "9D"])
        self.update(opponent="Bat", field=[])
        self.assertEqual(self.state.out_cards, {"8H", "9H", "8D", "9D"})

    def test_conflicting_labels_do_not_assign_two_destinations(self):
        self.update(mine="Bat", opponent="I take")
        self.assertFalse(self.state.out_cards)
        self.assertFalse(self.state.known_opponent_cards)

    def test_mine_is_in_console_output(self):
        self.update(mine=" P a s s ")
        snapshot = state_snapshot(self.state)
        self.assertEqual(snapshot["mine"], "pass")
        self.assertIn("Больше не подкидываю", format_state(snapshot))

    def test_text_fragments_and_ocr_records(self):
        self.assertEqual(normalize_text([" I ", "\tTake\n"]), "itake")
        self.assertEqual(normalize_text(["Your", " ", "turn"]), "yourturn")
        self.assertEqual(normalize_text([([], " B a t ", .9)]), "bat")

    def test_mine_crop_and_overlay_use_one_ocr_call(self):
        # Module import is replaced only to avoid a PaddleOCR dependency here.
        from types import ModuleType
        from constants.constants import CROP_MINE

        module = ModuleType("detect.detect_mine")
        frame = np.zeros((2622, 1206, 3), dtype=np.uint8)
        x1, y1, x2, y2 = CROP_MINE[(1206, 2622)]
        module.crop_by_size = lambda image, crop: image[y1:y2, x1:x2]
        ocr = Mock()
        ocr.predict.return_value = [{
            "rec_texts": ["I", "take"], "rec_scores": [.95, .96],
            "rec_polys": [[[5, 30], [30, 30], [30, 50], [5, 50]],
                          [[40, 30], [100, 30], [100, 50], [40, 50]]],
        }]
        visualization = DetectionVisualization(frame)
        with patch.dict("sys.modules", {"detect.detect_mine": module}), patch("game_state.game._ocr", return_value=ocr):
            self.assertEqual(_default_mine(frame, visualization=visualization), "itake")
        self.assertEqual(ocr.predict.call_count, 1)
        self.assertEqual(ocr.predict.call_args.args[0].shape, (y2-y1, x2-x1, 3))
        annotated = visualization.render()
        self.assertTrue(annotated[y1:y2, x1:x2].any())
        annotated[y1:y2, x1:x2] = 0
        self.assertFalse(annotated.any())
        self.assertFalse(frame.any())


class DiscardConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.values = {
            "button": "I take", "opponent": "", "mine": "", "deque": 24,
            "hand": ["6C"], "field": ["8H", "9H"],
        }
        self.state = DurakGameState(detectors={
            name: lambda image, name=name: self.values[name] for name in self.values
        })

    def update(self, **values):
        self.values.update(values)
        return self.state.update(self.frame)

    def confirm(self, **values):
        self.update(**values)
        self.update()

    def test_one_frame_false_positive_does_not_enter_discard(self):
        self.update(field=["8H", "9H", "AS"])
        self.confirm(field=["8H", "9H"])
        self.update(mine="Bat", field=[])
        self.assertEqual(self.state.out_cards, {"8H", "9H"})

    def test_corrected_stable_snapshot_replaces_old_card(self):
        self.confirm(field=["8H", "9S"])
        self.confirm(field=["8H", "9H"])
        self.update(mine="Bat", field=[])
        self.assertEqual(self.state.out_cards, {"8H", "9H"})

    def test_noise_in_bat_frame_does_not_change_confirmed_table(self):
        self.confirm()
        self.update(mine="Bat", field=["8H", "9H", "AS"])
        self.assertEqual(self.state.out_cards, {"8H", "9H"})

    def test_animation_and_next_round_do_not_extend_old_bat(self):
        self.confirm()
        self.update(mine="Bat")
        self.update(field=["8H", "9H", "AS"])
        self.update(opponent="Bat", field=["8D", "9D"])
        self.update(mine="", field=["8D", "9D"])
        self.assertEqual(self.state.out_cards, {"8H", "9H"})
        self.confirm(opponent="", field=["8D", "9D"])
        self.update(mine="Bat", field=[])
        self.assertEqual(self.state.out_cards, {"8H", "9H", "8D", "9D"})

    def test_unconfirmed_cards_are_not_discarded(self):
        self.update(mine="Bat")
        for _ in range(3):
            self.update()
        self.assertFalse(self.state.out_cards)

    def test_visible_hand_card_is_not_discarded(self):
        self.confirm(field=["6C", "8H", "9H"])
        self.update(mine="Bat", field=[])
        self.assertEqual(self.state.out_cards, {"8H", "9H"})
        self.assertEqual(self.state.hand_cards, {"6C"})

    def test_unresolved_previous_round_does_not_leak_into_new_round(self):
        self.confirm()
        self.update(field=[])
        self.update(field=["8D", "9D"])
        self.update(mine="Bat", field=[])
        self.assertFalse(self.state.out_cards)


if __name__ == "__main__":
    unittest.main()
