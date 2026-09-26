import contextlib
import io
import unittest
from unittest.mock import patch

from game_state.bot import main
from game_state.engine_process import _initialize_engine


class MctsOptionsTests(unittest.TestCase):
    def test_cli_forwards_search_options_without_connecting_to_phone(self):
        with patch("iphone_screen.iphone_client_v2.IPhoneRemote"), patch("game_state.bot.run_bot") as run:
            main(["--suggest-moves", "--mcts-rollouts", "5000", "--mcts-deals", "24",
                  "--mcts-exploration", "0.8", "--mcts-threads", "4", "--mcts-ms", "0"])
        options = run.call_args.kwargs["engine_options"]
        for key, value in dict(rollouts=5000, deals=24, exploration=0.8, threads=4, time_limit_ms=0).items():
            self.assertEqual(options[key], value)

    def test_invalid_cli_options_rejected_before_phone_connection(self):
        for args in (["--mcts-deals", "5"], ["--mcts-threads", "0"],
                     ["--mcts-rollouts", "0"], ["--mcts-exploration", "nan"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), patch(
                "iphone_screen.iphone_client_v2.IPhoneRemote"
            ) as phone:
                with self.assertRaises(SystemExit) as result:
                    main(args)
                self.assertEqual(result.exception.code, 2)
                phone.assert_not_called()

    def test_worker_initializer_forwards_options(self):
        options = dict(rollouts=5000, deals=24, exploration=0.8, threads=4)
        with patch("game_state.engine_process._engine"), patch("game_engine.DurakEngine") as engine:
            _initialize_engine(options)
            engine.assert_called_once_with(**options)


if __name__ == "__main__":
    unittest.main()
