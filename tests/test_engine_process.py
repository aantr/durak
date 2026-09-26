import unittest
from concurrent.futures import Future
from unittest.mock import Mock, patch

import numpy as np

from game_state.engine_process import EngineProcess, _suggest
from game_state.bot import format_state, state_snapshot, run_bot, recommendation_overlay
from game_state.game import DurakGameState
from game_engine.game_engine import observation_from_state


class EngineProcessTests(unittest.TestCase):
    def setUp(self):
        self.pool_patch = patch("game_state.engine_process.ProcessPoolExecutor")
        self.pool = self.pool_patch.start()
        self.addCleanup(self.pool_patch.stop)
        self.executor = self.pool.return_value
        self.futures = []

        def submit(*args):
            future = Future()
            self.futures.append(future)
            return future

        self.executor.submit.side_effect = submit
        self.worker = EngineProcess({"iterations": 100})
        self.addCleanup(self.worker.close)

    def test_spawn_and_single_worker(self):
        self.pool.assert_not_called()
        self.worker.request({})
        kwargs = self.pool.call_args.kwargs
        self.assertEqual(kwargs["max_workers"], 1)
        self.assertEqual(kwargs["mp_context"].get_start_method(), "spawn")
        self.assertEqual(kwargs["initargs"], ({"iterations": 100},))

    def test_pending_calculation_does_not_block_or_resubmit(self):
        self.worker.request({"hand_cards": ["6C"]})
        for _ in range(3):
            result = self.worker.poll()
            self.assertEqual(result["reason"], "Calculating...")
        self.executor.submit.assert_called_once()
        expected = {"status": "waiting", "reason": "Ход соперника", "action": None}
        self.futures[0].set_result(expected)
        self.assertEqual(self.worker.poll(), expected)
        self.assertEqual(self.worker.poll(), expected)
        self.executor.submit.assert_called_once()

    def test_request_freezes_snapshot_and_does_not_queue_more_work(self):
        snapshot = {"hand_cards": ["6C"]}
        self.assertTrue(self.worker.request(snapshot))
        snapshot["hand_cards"].append("7C")
        self.assertFalse(self.worker.request({"hand_cards": ["8C"]}))
        self.assertEqual(len(self.futures), 1)
        self.assertEqual(self.executor.submit.call_args.args[1], {"hand_cards": ["6C"]})
        expected = {"status": "ok", "action": "saved"}
        self.futures[0].set_result(expected)
        for _ in range(3):
            self.assertEqual(self.worker.poll(), expected)
        self.executor.submit.assert_called_once()

    def test_worker_errors_not_hidden(self):
        self.worker.request({"trump": "S"})
        self.futures[0].set_exception(RuntimeError("worker failed"))
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            self.worker.poll()

    def test_previous_move_shown_with_calculating_status(self):
        snapshot = state_snapshot(DurakGameState())
        move = {"status": "ok", "action": {"type": "attack", "card": "6C"},
                "moves": [{"value": 0.7}], "iterations": 100, "elapsed_ms": 20}
        self.worker.request(snapshot)
        self.futures[0].set_result(move)
        self.assertEqual(self.worker.poll(), move)
        next_snapshot = {**snapshot, "hand_cards": ["7C"]}
        self.worker.request(next_snapshot)
        recommendation = self.worker.poll()
        self.assertEqual(recommendation["last_move"], move)
        self.assertEqual(recommendation["status"], "calculating")
        self.assertEqual(recommendation_overlay(recommendation), "Previous: Play 6C | Calculating...")
        panel = format_state({**next_snapshot, "recommendation": recommendation})
        self.assertIn("Предыдущий ход: Походить 6♣ | Calculating...", panel)
        self.assertEqual(recommendation_overlay(move), "Play 6C | Ready")

    def test_reset_discards_running_result_even_for_identical_snapshot(self):
        snapshot = {"trump": "S"}
        self.worker.request(snapshot)
        self.futures[0].set_running_or_notify_cancel()
        self.worker.reset()
        self.futures[0].set_result({"status": "ok", "action": "old game"})
        result = self.worker.poll()
        self.assertEqual(result["status"], "idle")
        self.assertNotIn("last_move", result)
        self.assertEqual(len(self.futures), 1)
        self.worker.request(snapshot)
        self.assertEqual(len(self.futures), 2)

    def test_reset_clears_game_memory_and_keeps_configuration(self):
        detector = Mock(return_value="S")
        state = DurakGameState(detectors={"trump": detector}, trump="S",
                               field_confirmation_frames=3, deck_empty_confirmation_frames=4)
        state.out_cards.add("6C")
        state.hand_cards.add("7C")
        state._pending_hand_cards.add("8C")
        state._round_cards.add("9C")
        state._trump_candidate = "S"
        state.frame_number = 10
        state.reset()
        self.assertEqual(state_snapshot(state), state_snapshot(DurakGameState()))
        self.assertFalse(state._pending_hand_cards)
        self.assertFalse(state._round_cards)
        self.assertIsNone(state._trump_candidate)
        self.assertEqual(state.frame_number, 0)
        self.assertEqual(state.field_confirmation_frames, 3)
        self.assertEqual(state.deck_empty_confirmation_frames, 4)
        self.assertIs(state.detectors["trump"], detector)
        state.reset(trump="H")
        self.assertEqual(state.trump, "H")

    def test_invalid_observation_returns_status(self):
        engine = Mock()
        engine.suggest.side_effect = ValueError("missing cards")
        with patch("game_state.engine_process._engine", engine):
            self.assertEqual(_suggest({})["status"], "invalid_state")

    def test_inconsistent_opponent_memory_has_readable_counts(self):
        snapshot = state_snapshot(DurakGameState())
        snapshot.update(phase="your_turn", hand_cards=["6C"], deck_remaining=30,
                        opponent_card_count=5, known_opponent_cards=["6D", "7D", "8D", "9D", "10D", "JD"])
        with self.assertRaisesRegex(ValueError, "известных карт соперника 6, всего по подсчёту 5"):
            observation_from_state(snapshot, trump="S")

    def test_calculating_in_console(self):
        snapshot = state_snapshot(DurakGameState())
        self.worker.request(snapshot)
        snapshot["recommendation"] = self.worker.poll()
        self.assertIn("Calculating...", format_state(snapshot))

    def test_bot_recognizes_without_starting_engine_before_space(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        iphone = Mock()
        iphone.get_screen.side_effect = [frame, frame, frame, KeyboardInterrupt()]
        state = DurakGameState()
        state.update = Mock()

        def recognize(fn, image):
            future = Future()
            future.set_result(fn(image))
            return future

        with patch("game_state.bot.ThreadPoolExecutor") as thread_pool, patch(
            "game_state.bot.time.monotonic", side_effect=range(100)
        ), patch("game_state.bot.cv2.namedWindow"), patch(
            "game_state.bot.cv2.resizeWindow"
        ), patch("game_state.bot.cv2.setMouseCallback"), patch(
            "game_state.bot.cv2.imshow"
        ) as show, patch("game_state.bot.cv2.waitKeyEx", return_value=-1), patch(
            "game_state.bot.cv2.getWindowProperty", return_value=1
        ), patch("game_state.bot.cv2.destroyWindow"):
            thread_pool.return_value.submit.side_effect = recognize
            run_bot(iphone, state=state, engine_options={})
        self.assertEqual(state.update.call_count, 3)
        self.pool.assert_not_called()
        self.executor.submit.assert_not_called()
        self.assertTrue(np.any(show.call_args.args[1]))
        self.assertFalse(np.any(frame))

    def test_reset_key_waits_for_active_recognition(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        iphone = Mock()
        iphone.get_screen.return_value = frame
        state = DurakGameState(out_cards={"6C"})
        active = Future()
        next_frame = Future()
        keys = iter((ord("r"), -1, ord("q")))

        def keyboard(_):
            key = next(keys)
            if key == -1:
                self.assertEqual(state.out_cards, {"6C"})
                active.set_result(None)
            return key

        with patch("game_state.bot.ThreadPoolExecutor") as thread_pool, patch(
            "game_state.bot.EngineProcess"
        ) as engine, patch("game_state.bot.time.monotonic", side_effect=range(100)), patch(
            "game_state.bot.cv2.namedWindow"
        ), patch("game_state.bot.cv2.resizeWindow"), patch(
            "game_state.bot.cv2.setMouseCallback"
        ), patch("game_state.bot.cv2.imshow"), patch(
            "game_state.bot.cv2.waitKeyEx", side_effect=keyboard
        ), patch("game_state.bot.cv2.getWindowProperty", return_value=1), patch(
            "game_state.bot.cv2.destroyWindow"
        ):
            thread_pool.return_value.submit.side_effect = [active, next_frame]
            thread_pool.return_value.shutdown.side_effect = lambda **kwargs: next_frame.cancel()
            returned = run_bot(iphone, state=state, engine_options={})
        self.assertIs(returned, state)
        self.assertFalse(state.out_cards)
        engine.return_value.reset.assert_called_once()
        engine.return_value.poll.assert_not_called()

    def test_space_calculates_but_never_sends_input(self):
        iphone = self.run_key_sequence([-1, ord(" "), -1, -1, ord("q")])
        self.executor.submit.assert_called_once()
        iphone.send_tap_async.assert_not_called()

    def test_up_sends_last_calculated_move_once(self):
        for key in (16777235, 65362, 2490368, 63232):
            with self.subTest(key=key):
                iphone = self.run_key_sequence([-1, ord(" "), -1, key, key, ord("q")])
                iphone.send_tap_async.assert_called_once_with(40, 70)
                self.executor.submit.assert_called_once()

    def test_up_before_result_does_not_queue_execution(self):
        iphone = self.run_key_sequence([-1, ord(" "), 65362, -1, ord("q")])
        iphone.send_tap_async.assert_not_called()

    def test_ready_survives_new_frames_without_automatic_calculation(self):
        iphone = self.run_key_sequence([-1, ord(" "), -1, -1, -1, 65362, ord("q")])
        self.executor.submit.assert_called_once()
        iphone.send_tap_async.assert_called_once_with(40, 70)

    def test_last_move_can_be_played_during_explicit_recalculation(self):
        iphone = self.run_key_sequence([-1, ord(" "), -1, ord(" "), 65362, ord("q")])
        self.assertEqual(self.executor.submit.call_count, 2)
        self.assertFalse(self.futures[1].done())
        iphone.send_tap_async.assert_called_once_with(40, 70)

    def test_repeated_space_while_busy_does_not_queue_calculations(self):
        self.run_key_sequence([-1, ord(" "), ord(" "), -1, ord("q")], complete_on=(4,))
        self.executor.submit.assert_called_once()

    def test_reset_discards_result_and_does_not_restart_engine(self):
        iphone = self.run_key_sequence([-1, ord(" "), ord("r"), -1, 65362, ord("q")], complete_on=(4,))
        self.executor.submit.assert_called_once()
        iphone.send_tap_async.assert_not_called()

    def test_changed_button_prevents_executing_old_move(self):
        iphone = self.run_key_sequence([-1, ord(" "), -1, 65362, ord("q")],
                                       buttons=["pass", "pass", "", "", ""])
        iphone.send_tap_async.assert_not_called()

    def run_key_sequence(self, keys, *, complete_on=(3,), buttons=None):
        self.executor.reset_mock()
        self.futures.clear()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        iphone = Mock()
        iphone.get_screen.return_value = frame
        sent = Future()
        sent.set_result(None)
        iphone.send_tap_async.return_value = sent
        button_values = iter(buttons or ["pass"] * len(keys))
        state = DurakGameState(detectors={
            "button": lambda _: next(button_values), "opponent": lambda _: "",
            "mine": lambda _: "", "deque": lambda _: 24,
            "hand": lambda _: [], "field": lambda _: [],
        })
        recommendation = {"status": "ok", "action": {"type": "pass", "button": "Pass"},
                          "moves": [{"value": 0.5}], "iterations": 1, "elapsed_ms": 1}
        key_events = iter(enumerate(keys, 1))

        def keyboard(_):
            iteration, key = next(key_events)
            if self.futures and not self.futures[-1].done() and not self.futures[-1].running():
                self.futures[-1].set_running_or_notify_cancel()
            if iteration in complete_on and self.futures:
                self.futures[-1].set_result(recommendation)
            return key

        def recognize(fn, image):
            done = Future()
            done.set_result(fn(image))
            return done

        with patch("game_state.bot.ThreadPoolExecutor") as thread_pool, patch(
            "game_state.bot.time.monotonic", side_effect=range(100)
        ), patch("game_state.bot.cv2.namedWindow"), patch(
            "game_state.bot.cv2.resizeWindow"
        ), patch("game_state.bot.cv2.setMouseCallback"), patch(
            "game_state.bot.cv2.imshow"
        ), patch("game_state.bot.cv2.waitKeyEx", side_effect=keyboard), patch(
            "game_state.bot.cv2.getWindowProperty", return_value=1
        ), patch("game_state.bot.cv2.destroyWindow"), patch(
            "game_state.move_input.CROP_BUTTON", {(200, 100): (10, 50, 70, 90)}
        ):
            thread_pool.return_value.submit.side_effect = recognize
            run_bot(iphone, state=state, engine_options={})
        self.assertEqual(state.frame_number, len(keys) if ord("r") not in keys else 3)
        return iphone


if __name__ == "__main__":
    unittest.main()
