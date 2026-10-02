"""Журнал содержит только изменения распознанного состояния."""

from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from game_state.bot import run_bot, state_snapshot
from game_state.game import DurakGameState
from game_state.recognition_log import RecognitionChangeLog


class RecognitionLogTests(unittest.TestCase):
    def test_records_only_state_changes_with_detector_details(self):
        state = DurakGameState(trump="S")
        state.frame_number = 1
        state.frame_size = (200, 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "recognition.jsonl"
            journal = RecognitionChangeLog(path)
            self.assertTrue(journal.record(state, state_snapshot(state)))
            state.frame_number = 2
            self.assertFalse(journal.record(state, state_snapshot(state)))
            state.frame_number = 3
            state.field_cards.add("6H")
            state.field_layout = [{"card": "6H", "covers": None,
                                   "bbox": (1, 2, 3, 4), "confidence": 0.91}]
            self.assertTrue(journal.record(state, state_snapshot(state)))
            journal.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(len(records), 2)
        self.assertEqual([item["frame"] for item in records], [1, 3])
        self.assertIn("field_cards", records[1]["changed"])
        self.assertEqual(records[1]["state"]["field_cards"], ["6H"])
        self.assertEqual(records[1]["detection"]["field_layout"][0]["confidence"], 0.91)
        self.assertEqual(records[1]["frame_size"], [200, 100])
        self.assertEqual(records[0]["session"], records[1]["session"])

    def test_bot_writes_changes_without_recommendation_noise(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        iphone = Mock()
        iphone.get_screen.side_effect = [frame, frame, frame, frame, KeyboardInterrupt()]
        state = DurakGameState(trump="S")
        updates = 0

        def update(_image, **_kwargs):
            nonlocal updates
            updates += 1
            state.frame_number = updates
            state.frame_size = (4, 4)
            state._terminal_frames_left = 1 if updates == 1 else 0
            if updates == 3:
                state.hand_cards.add("6H")

        state.update = Mock(side_effect=update)

        def completed(fn, image):
            future = Future()
            future.set_result(fn(image))
            return future

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recognition.jsonl"
            with patch("game_state.bot.ThreadPoolExecutor") as pool, patch(
                "game_state.bot.EngineProcess"
            ) as engine:
                pool.return_value.submit.side_effect = completed
                engine.return_value.last_evaluation = 0.5
                engine.return_value.poll.side_effect = [None, {"status": "calculating", "reason": "идёт расчёт"},
                                                        None, {"status": "idle", "reason": "ожидание"}]
                run_bot(iphone, state=state, show_window=False, engine_options={},
                        recognition_log_path=path)
            records = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(len(records), 2)
        self.assertEqual([item["frame"] for item in records], [1, 3])
        self.assertNotIn("recommendation", records[1]["state"])
        self.assertNotIn("last_evaluation", records[1]["state"])


if __name__ == "__main__":
    unittest.main()
