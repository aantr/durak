"""Получение свежего кадра без ожидания следующего и без повторов."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from iphone_screen.iphone_client_v2 import IPhoneRemote


class LiveFramesTests(unittest.TestCase):
    def setUp(self):
        self.old = np.zeros((4, 4, 3), dtype=np.uint8)
        self.new = self.old.copy()  # Даже одинаковые пиксели могут быть новым кадром.
        self.client = SimpleNamespace(_frame_cond=threading.Condition(),
                                      _frame_id=2, _latest_frame=self.new,
                                      _video_error=None)

    def test_already_received_new_frame_is_returned_without_waiting(self):
        with patch.object(self.client._frame_cond, "wait") as wait:
            frame = IPhoneRemote.get_screen(self.client, after_frame=self.old, timeout=0)
        self.assertIs(frame, self.new)
        wait.assert_not_called()

    def test_processed_frame_is_not_returned_again(self):
        with self.assertRaises(TimeoutError):
            IPhoneRemote.get_screen(self.client, after_frame=self.new, timeout=0)

    def test_waiting_returns_frame_when_decoder_publishes_it(self):
        self.client._latest_frame = self.old
        def publish(_):
            self.client._latest_frame = self.new
            self.client._frame_id += 1
        with patch.object(self.client._frame_cond, "wait", side_effect=publish) as wait:
            frame = IPhoneRemote.get_screen(self.client, after_frame=self.old)
        self.assertIs(frame, self.new)
        wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
