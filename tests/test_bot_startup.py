import unittest
from unittest.mock import patch

from game_state.bot import main


class BotStartupTests(unittest.TestCase):
    def test_connection_failure_skips_model_loading(self):
        with patch("iphone_screen.iphone_client_v2.IPhoneRemote", side_effect=ConnectionError("bridge unavailable")), patch(
            "game_state.bot.preload_models"
        ) as preload, patch("game_state.bot.run_bot") as run, self.assertLogs("game_state.bot", level="ERROR"):
            self.assertEqual(main([]), 1)
        preload.assert_not_called()
        run.assert_not_called()

    def test_model_loading_requires_connected_client_and_closes_on_failure(self):
        with patch("iphone_screen.iphone_client_v2.IPhoneRemote") as remote, patch(
            "game_state.bot.preload_models"
        ) as preload, patch("game_state.bot.run_bot") as run, self.assertLogs("game_state.bot", level="ERROR"):
            def fail_loading():
                remote.return_value.__enter__.assert_called_once()
                remote.return_value.__exit__.assert_not_called()
                raise RuntimeError("model loading failed")

            preload.side_effect = fail_loading
            self.assertEqual(main([]), 1)
        remote.return_value.__exit__.assert_called_once()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
