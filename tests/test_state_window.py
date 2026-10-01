"""Окно состояния использует изображения карт и работает без GUI в тестах."""

import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from game_state.bot import STATE_WINDOW_NAME, run_bot
from game_state.game import DurakGameState
from game_state.state_window import StateWindowRenderer


class StateWindowTests(unittest.TestCase):
    def test_card_uses_rank_and_suit_assets(self):
        renderer = StateWindowRenderer()
        tile = renderer.card("6H")
        self.assertEqual(tile.size, (60, 83))
        self.assertIsNotNone(renderer._ranks["6R"])
        self.assertIsNotNone(renderer._suits["H"])
        self.assertIs(tile, renderer.card("6H"))
        self.assertNotEqual(tile.tobytes(), renderer.card("6S").tobytes())

    def test_full_table_and_missing_assets_have_readable_fallback(self):
        snapshot = dict(phase="your_turn", trump="S", deck_remaining=12,
                        opponent_card_count=5, hand_cards=["6H", "AS"],
                        field_cards=["8H", "9H"],
                        field_layout=[{"card": "8H", "covers": None},
                                      {"card": "9H", "covers": 0}],
                        known_opponent_cards=["KC"], out_cards=["6D"],
                        last_out_cards=["6D"], unknown_opponent_cards=["7S"] * 36,
                        last_evaluation=0.5)
        image = StateWindowRenderer().render(snapshot)
        self.assertEqual(image.shape[1], 1160)
        self.assertGreaterEqual(image.shape[0], 770)
        self.assertTrue(np.any(image))
        self.assertEqual(image.shape[2], 3)
        with self.subTest("fallback without dataset"):
            fallback = StateWindowRenderer(Path("/tmp/nonexistent-durak-assets"))
            self.assertEqual(fallback.render(snapshot).shape, image.shape)
            self.assertIsNone(fallback._ranks["6R"])

    def test_bot_opens_state_window_instead_of_logging_table(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        iphone = Mock()
        iphone.get_screen.side_effect = [frame, KeyboardInterrupt()]
        state = DurakGameState()
        state.update = Mock()

        def completed(fn, image):
            future = Future()
            future.set_result(fn(image))
            return future

        with patch("game_state.bot.ThreadPoolExecutor") as pool, patch(
            "game_state.bot.EngineProcess"
        ) as engine, patch("game_state.bot.cv2.namedWindow") as named, patch(
            "game_state.bot.cv2.resizeWindow"
        ), patch("game_state.bot.cv2.setMouseCallback"), patch(
            "game_state.bot.cv2.imshow"
        ) as show, patch("game_state.bot.cv2.waitKeyEx", return_value=-1), patch(
            "game_state.bot.cv2.getWindowProperty", return_value=1
        ), patch("game_state.bot.cv2.destroyWindow"), patch(
            "game_state.bot.logger.info"
        ) as log:
            pool.return_value.submit.side_effect = completed
            engine.return_value.last_evaluation = None
            engine.return_value.poll.return_value = None
            run_bot(iphone, state=state, engine_options={})

        named.assert_any_call(STATE_WINDOW_NAME, 0)
        self.assertTrue(any(call.args[0] == STATE_WINDOW_NAME for call in show.call_args_list))
        self.assertFalse(any(call.args and call.args[0] in ("\n%s", "Состояние: %s")
                             for call in log.call_args_list))


if __name__ == "__main__":
    unittest.main()
