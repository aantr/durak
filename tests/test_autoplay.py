import unittest
from concurrent.futures import Future
from unittest.mock import Mock

from game_state.autoplay import AutoPlay, position_key
from game_state.engine_process import EngineProcess


class AutoPlayTests(unittest.TestCase):
    def setUp(self):
        self.engine = EngineProcess({})
        self.executor = Mock()
        self.futures = []
        def submit(*args):
            future = Future()
            self.futures.append(future)
            return future
        self.executor.submit.side_effect = submit
        self.engine.executor = self.executor
        self.auto = AutoPlay(self.engine)
        self.send = Mock()
        self.snapshot = dict(phase='your_turn', button='yourturn', mine='', opponent='',
                             hand_cards=['6C'], field_cards=[])
        self.move = dict(status='ok', action=dict(type='attack', card='6C'), moves=[{'value': 0.625}])

    def step(self, now, snapshot=None, **kwargs):
        self.auto.step(self.snapshot if snapshot is None else snapshot, now,
                       execute=self.send, **kwargs)

    def start_calculation(self):
        self.auto.toggle()
        self.step(0)
        self.step(1)
        self.futures[-1].set_running_or_notify_cancel()

    def test_disabled_by_default(self):
        self.step(0)
        self.step(10)
        self.executor.submit.assert_not_called()
        self.send.assert_not_called()

    def test_waits_one_second_then_sends_once_on_completion(self):
        self.auto.toggle()
        self.step(10)
        self.step(10.999)
        self.executor.submit.assert_not_called()
        self.step(11)
        self.step(11.1)
        self.executor.submit.assert_called_once()
        self.send.assert_not_called()
        self.futures[0].set_result(self.move)
        self.step(11.2)
        self.step(20)
        self.send.assert_called_once_with(self.move)
        self.executor.submit.assert_called_once()
        self.assertEqual(self.engine.last_evaluation, 0.625)
        self.auto.toggle()
        self.assertEqual(self.engine.last_evaluation, 0.625)
        self.auto.reset(clear_evaluation=True)
        self.assertIsNone(self.engine.last_evaluation)

    def test_custom_delay(self):
        self.auto.delay = 0.25
        self.auto.toggle()
        self.step(0)
        self.step(0.24)
        self.executor.submit.assert_not_called()
        self.step(0.25)
        self.executor.submit.assert_called_once()

    def test_changes_before_calculation_do_not_restart_delay(self):
        self.auto.toggle()
        self.step(0)
        changed = {**self.snapshot, 'hand_cards': ['6C', '7C']}
        self.step(0.9, changed)
        self.step(1.1, changed)
        self.executor.submit.assert_called_once()
        self.assertEqual(self.executor.submit.call_args.args[1], changed)

    def test_changed_state_discards_running_result_and_recalculates(self):
        self.start_calculation()
        changed = {**self.snapshot, 'hand_cards': ['7C']}
        self.step(1.1, changed)
        self.assertEqual(self.engine.generation, self.engine.submitted_generation)
        self.futures[0].set_result(self.move)
        self.step(1.2, changed)
        self.send.assert_not_called()
        self.step(2.2, changed)
        self.assertEqual(len(self.futures), 2)
        new_move = dict(status='ok', action=dict(type='attack', card='7C'))
        self.futures[1].set_result(new_move)
        self.step(2.3, changed)
        self.send.assert_called_once_with(new_move)

    def test_toggle_off_discards_result_even_after_reenable_same_state(self):
        self.start_calculation()
        self.auto.toggle()
        self.futures[0].set_result(self.move)
        self.step(2)
        self.send.assert_not_called()
        self.auto.toggle()
        self.step(3)
        self.step(4)
        self.assertEqual(len(self.futures), 2)
        self.send.assert_not_called()

    def test_leaving_our_turn_cancels_delayed_start(self):
        self.auto.toggle()
        self.step(0)
        self.step(0.8, {**self.snapshot, 'phase': 'opponent_turn', 'button': ''})
        self.step(2, {**self.snapshot, 'phase': 'opponent_turn', 'button': ''})
        self.executor.submit.assert_not_called()

    def test_round_transition_and_already_taken_hand_are_not_turns(self):
        for changes in ({'phase': 'ready', 'button': 'ready'}, {'mine': 'bat'},
                        {'opponent': 'bat'}, {'mine': 'itake'}, {'mine': 'pass'}):
            with self.subTest(changes=changes):
                self.assertFalse(AutoPlay.our_turn({**self.snapshot, **changes}))
        for phase, button in (('your_turn', 'yourturn'), ('throw_in', 'pass'),
                              ('throw_in', 'bat'), ('defend_or_take', 'itake')):
            self.assertTrue(AutoPlay.our_turn({**self.snapshot, 'phase': phase, 'button': button}))

    def test_busy_input_prevents_request_and_execution(self):
        self.auto.toggle()
        self.step(0)
        self.step(1, input_busy=True)
        self.executor.submit.assert_not_called()
        self.step(2)
        self.futures[0].set_result(self.move)
        self.step(3, input_busy=True)
        self.send.assert_not_called()
        self.step(4)
        self.send.assert_called_once()

    def test_invalid_result_never_uses_previous_move_and_retries_after_delay(self):
        self.start_calculation()
        self.engine.last_move = self.move
        self.futures[0].set_result({'status': 'invalid_state', 'action': None})
        self.step(2)
        self.step(2.99)
        self.send.assert_not_called()
        self.executor.submit.assert_called_once()
        self.step(3)
        self.assertEqual(len(self.futures), 2)
        self.futures[1].set_result(self.move)
        self.step(3.1)
        self.send.assert_called_once_with(self.move)

    def test_send_failure_does_not_retry_potentially_partial_input(self):
        self.start_calculation()
        self.futures[0].set_result(self.move)
        self.send.side_effect = RuntimeError('partial input')
        with self.assertRaises(RuntimeError):
            self.step(2)
        self.step(3)
        self.send.assert_called_once()

    def test_detection_order_and_non_action_ocr_do_not_discard_result(self):
        self.snapshot.update(hand_cards=['6C', '7C'], field_cards=['8C', '9C'],
                             field_layout=[{'card': '8C', 'covers': None}, {'card': '9C', 'covers': 0}])
        self.start_calculation()
        reordered = {**self.snapshot, 'hand_cards': ['7C', '6C'], 'field_cards': ['9C', '8C'],
                     'field_layout': [{'card': '9C', 'covers': 1}, {'card': '8C', 'covers': None}],
                     'mine': 'ocr noise', 'opponent': 'player name',
                     'last_out_cards': ['AS'], 'unknown_opponent_cards': ['AD']}
        self.assertEqual(position_key(self.snapshot), position_key(reordered))
        self.step(1.1, reordered)
        self.futures[0].set_result(self.move)
        self.step(1.2, reordered)
        self.send.assert_called_once_with(self.move)
        self.step(5, {**reordered, 'opponent': 'other noise'})
        self.executor.submit.assert_called_once()
        self.send.assert_called_once()

    def test_changed_cover_relationship_is_a_different_position(self):
        first = {**self.snapshot, 'field_layout': [
            {'card': '6C', 'covers': None}, {'card': '7C', 'covers': None},
            {'card': '8C', 'covers': 0}]}
        second = {**first, 'field_layout': [
            {'card': '6C', 'covers': None}, {'card': '7C', 'covers': None},
            {'card': '8C', 'covers': 1}]}
        self.assertNotEqual(position_key(first), position_key(second))

    def test_video_noise_continues_during_calculation_without_cancellation(self):
        self.start_calculation()
        for index in range(10):
            noisy = {**self.snapshot, 'mine': f'noise{index}'}
            self.step(1.1 + index / 10, noisy)
        self.executor.submit.assert_called_once()
        self.assertEqual(self.engine.generation, self.engine.submitted_generation)
        self.futures[0].set_result(self.move)
        self.step(2.5)
        self.send.assert_called_once_with(self.move)

    def test_invalid_delay(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AutoPlay(self.engine, value)


if __name__ == '__main__':
    unittest.main()
