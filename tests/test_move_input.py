import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from game_state.move_input import execute_move


class MoveInputTests(unittest.TestCase):
    def setUp(self):
        self.iphone = Mock()
        self.snapshot = {"button": "pass", "hand_cards": ["6C"]}
        self.geometry = {
            "frame_size": (1206, 2622),
            "hand_layout": [{"card": "6C", "bbox": (20, 30, 40, 70)}],
            "field_layout": [{"card": "6D", "bbox": (100, 100, 140, 180), "covers": None}],
        }

    def execute(self, action):
        return execute_move(self.iphone, {"status": "ok", "action": action}, self.snapshot, self.geometry)

    def test_pass_bat_and_take_tap_button_center(self):
        for button, action in (("pass", {"type": "pass", "button": "Pass"}),
                               ("bat", {"type": "pass", "button": "Bat"}),
                               ("itake", {"type": "take"})):
            self.snapshot["button"] = button
            result = self.execute(action)
            self.iphone.send_tap_async.assert_called_with(250, 2422)
            self.assertIs(result, self.iphone.send_tap_async.return_value)
        self.iphone.swipe_async.assert_not_called()

    def test_attack_uses_hand_crop_offset_and_field_center(self):
        self.execute({"type": "attack", "card": "6C"})
        self.iphone.swipe_async.assert_called_once_with(
            30, 2016, 603, 1310.5, duration=0.3, frame_size=(1206, 2622))

    def test_defend_uses_target_card_and_field_crop_offset(self):
        self.snapshot["button"] = "itake"
        self.execute({"type": "defend", "card": "6C", "target_card": "6D"})
        self.iphone.swipe_async.assert_called_once_with(
            30, 2016, 120, 795, duration=0.3, frame_size=(1206, 2622))

    def test_previous_move_cannot_be_executed_during_calculation(self):
        with self.assertRaises(ValueError):
            execute_move(self.iphone, {"status": "calculating", "last_move": {
                "status": "ok", "action": {"type": "take"}}}, self.snapshot, self.geometry)
        self.assertFalse(self.iphone.mock_calls)

    def test_wrong_button_and_missing_or_covered_card_do_not_send_input(self):
        actions = [{"type": "take"}, {"type": "pass", "button": "Bat"},
                   {"type": "attack", "card": "AS"}]
        for action in actions:
            with self.assertRaises(ValueError):
                self.execute(action)
        self.snapshot["button"] = "itake"
        self.geometry["field_layout"].append({"card": "7D", "covers": 0, "bbox": (120, 120, 140, 180)})
        with self.assertRaises(ValueError):
            self.execute({"type": "defend", "card": "6C", "target_card": "6D"})
        self.snapshot["button"] = "pass"
        self.geometry["hand_layout"] = []
        with self.assertRaises(ValueError):
            self.execute({"type": "attack", "card": "6C"})
        self.assertFalse(self.iphone.mock_calls)

    def test_card_move_rejected_after_turn_changes(self):
        self.snapshot["button"] = ""
        for action in ({"type": "attack", "card": "6C"},
                       {"type": "defend", "card": "6C", "target_card": "6D"}):
            with self.assertRaises(ValueError):
                self.execute(action)
        self.assertFalse(self.iphone.mock_calls)

    def test_swipe_converts_pixels_in_input_worker_and_preserves_logical_api(self):
        # Video/network dependencies are unused: all device calls are mocked.
        with patch.dict(sys.modules, {"av": Mock(), "requests": Mock(), "websocket": Mock()}):
            from iphone_screen.iphone_client_v2 import IPhoneRemote
        client = IPhoneRemote.__new__(IPhoneRemote)
        client._screen_width, client._screen_height = 402, 874
        client.swipe = Mock(return_value="ok")
        with ThreadPoolExecutor(max_workers=1) as executor:
            client._input_executor = executor
            self.assertEqual(client.swipe_async(300, 600, 600, 1200, frame_size=(1206, 2622)).result(timeout=1), "ok")
            client.swipe.assert_called_with(100, 200, 200, 400, 0.1)
            client.swipe_async(300, 600, 600, 1200).result(timeout=1)
            client.swipe.assert_called_with(300, 600, 600, 1200, 0.1)
            client.swipe_async(600, 300, 1200, 600, frame_size=(2622, 1206)).result(timeout=1)
            client.swipe.assert_called_with(200, 100, 400, 200, 0.1)


if __name__ == "__main__":
    unittest.main()
