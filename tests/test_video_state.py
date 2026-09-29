import time
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from game_state.game import DurakGameState
from game_state.video_test import VideoInfo, TimestampVideoReader, frame_at_time, playback, run_video, stop_worker, _open_log, main


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Capture:
    def __init__(self):
        self.position = 0
        self.read_indices = []

    def get(self, prop):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        return self.position

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        return True

    def read(self):
        index = self.position
        self.position += 1
        self.read_indices.append(index)
        return True, np.full((4, 4, 3), index, dtype=np.uint8)


def slow_worker(control, output, path, options):
    # Настоящий spawn/IPC, но без GPU. Родитель не должен ждать эти 5 секунд.
    control.send({"type": "ready", "video": VideoInfo(20, 3, 4, 4), "startup_seconds": 0})
    control.recv()
    time.sleep(5)


def fast_worker(control, output, path, options):
    control.send({"type": "ready", "video": VideoInfo(10, 100, 4, 4), "startup_seconds": 0})
    control.recv()
    for index in range(40):
        output.put(({"type": "state", "frame_index": index, "video_time": index / 10,
                     "recognition_ms": 0, "state": {}}, None))
    control.send({"type": "done", "processed_frames": 40})
    control.close()


class VideoStateTests(unittest.TestCase):
    def test_sampling_uses_last_tick_not_future_frame(self):
        video = VideoInfo(30, 300, 4, 4)
        self.assertEqual(frame_at_time(0, 10, video), 0)
        self.assertEqual(frame_at_time(0.29, 10, video), 6)
        self.assertEqual(frame_at_time(0.31, 10, video), 9)
        self.assertIsNone(frame_at_time(10, 10, video))

    def run_playback(self, *, inference_time, sample_fps=10, video=None, realtime=True):
        clock = Clock()
        capture = Capture()
        state = DurakGameState()
        state.update = Mock(side_effect=lambda *args, **kwargs: clock.sleep(inference_time))
        records = []
        playback(capture, state, video or VideoInfo(10, 10, 4, 4), fps=sample_fps,
                 slow_every=2, draw=False, start=0,
                 emit=lambda record, _: records.append(record), clock=clock, sleep=clock.sleep,
                 realtime=realtime)
        return records, capture, state

    def test_slow_recognition_skips_old_frames_and_drops_late_result(self):
        records, capture, state = self.run_playback(inference_time=0.35)
        self.assertEqual(capture.read_indices, [0, 3, 7])
        self.assertEqual([r["frame_index"] for r in records], [0, 3])
        self.assertTrue(all(r["finished_at"] < 1 for r in records))
        self.assertEqual(state.update.call_args.kwargs, dict(slow_every=2, draw_detections=False))

    def test_fast_recognition_waits_for_sampling_rate(self):
        records, capture, _ = self.run_playback(inference_time=0.01, sample_fps=5)
        self.assertEqual(capture.read_indices, [0, 2, 4, 6, 8])
        self.assertEqual(len(records), 5)

    def test_no_duplicates_when_requested_fps_exceeds_source(self):
        records, capture, _ = self.run_playback(inference_time=0.01, sample_fps=30,
                                               video=VideoInfo(5, 3, 4, 4))
        self.assertEqual(capture.read_indices, [0, 1, 2])
        self.assertEqual(len(records), 3)

    def test_acceleration_preserves_frames_states_and_late_result_cutoff(self):
        for duration in (0, 0.01, 0.09, 0.15, 0.35, 1.1):
            for sample_fps in (5, 10, 30):
                with self.subTest(duration=duration, fps=sample_fps):
                    realtime, real_capture, _ = self.run_playback(
                        inference_time=duration, sample_fps=sample_fps)
                    accelerated, fast_capture, _ = self.run_playback(
                        inference_time=duration, sample_fps=sample_fps, realtime=False)
                    self.assertEqual(real_capture.read_indices, fast_capture.read_indices)
                    self.assertEqual(len(realtime), len(accelerated))
                    for before, after in zip(realtime, accelerated):
                        for key in ("frame_index", "video_time", "sample_time", "state"):
                            self.assertEqual(before[key], after[key])
                        # Прежний sleep округляет короткие ожидания вверх до 0.1 мс.
                        self.assertAlmostEqual(before["finished_at"], after["finished_at"], delta=0.001)
                        self.assertAlmostEqual(before["recognition_ms"], after["recognition_ms"])

    def test_acceleration_skips_vfr_gaps_without_sleep_or_duplicate_updates(self):
        def run(realtime):
            clock = Clock()
            reader = object.__new__(TimestampVideoReader)
            timestamps = (0.15, 0.55, 0.85)
            def read_at(seconds):
                available = [t for t in timestamps if t <= seconds + 1e-9]
                return (available[-1], np.zeros((4, 4, 3), dtype=np.uint8)) if available else None
            reader.read_at = read_at
            state = DurakGameState()
            def update(*args, **kwargs):
                state.frame_number += 1
                clock.sleep(0.01)
            state.update = Mock(side_effect=update)
            sleep = Mock(side_effect=clock.sleep)
            records = []
            playback(reader, state, VideoInfo(10, 10, 4, 4), fps=30, slow_every=2,
                     draw=False, start=0, emit=lambda record, _: records.append(record),
                     clock=clock, sleep=sleep, realtime=realtime)
            return records, clock.now, sleep
        before, real_elapsed, real_sleep = run(True)
        after, fast_elapsed, fast_sleep = run(False)
        self.assertEqual([r["video_time"] for r in after], [0.15, 0.55, 0.85])
        self.assertEqual([r["state"] for r in before], [r["state"] for r in after])
        self.assertEqual([r["sample_time"] for r in before], [r["sample_time"] for r in after])
        self.assertGreaterEqual(real_elapsed, 1)
        self.assertAlmostEqual(fast_elapsed, 0.03)
        fast_sleep.assert_not_called()
        self.assertTrue(real_sleep.called)

    def test_parent_drains_all_accelerated_results_before_returning(self):
        with patch("game_state.video_test._worker", fast_worker):
            summary = run_video("unused.mp4", window=False, state_format="json", startup_timeout=10)
        self.assertEqual(summary["delivered_frames"], 40)
        self.assertEqual(summary["last_video_time"], 3.9)
        self.assertLess(summary["playback_seconds"], 2)

    def test_unreadable_frame_fails_instead_of_repeating_previous(self):
        capture = Capture()
        capture.read = Mock(return_value=(False, None))
        with self.assertRaisesRegex(RuntimeError, "декодировать"):
            playback(capture, Mock(), VideoInfo(10, 10, 4, 4), fps=10, slow_every=2,
                     draw=False, start=0, emit=Mock(), clock=lambda: 0)

    def test_missing_video_metadata_is_rejected(self):
        capture = Mock()
        capture.get.return_value = 0
        with self.assertRaises(ValueError):
            VideoInfo.read(capture)

    def test_stop_does_not_join_without_timeout(self):
        process = Mock()
        process.is_alive.side_effect = [True, True]
        stop_worker(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertTrue(all(call.kwargs.get("timeout") for call in process.join.call_args_list))

    def test_parent_deadline_interrupts_unfinished_inference(self):
        start = time.monotonic()
        with patch("game_state.video_test._worker", slow_worker):
            summary = run_video("unused.mp4", window=False, startup_timeout=10)
        self.assertLess(time.monotonic() - start, 3)
        self.assertLess(summary["playback_seconds"], 0.5)
        self.assertEqual(summary["delivered_frames"], 0)
        self.assertIsNone(summary["last_state"])

    def test_invalid_options_rejected_before_starting_models(self):
        for options in ({"fps": 0}, {"fps": float("nan")}, {"slow_every": 0},
                        {"warmup": -1}, {"startup_timeout": 0}, {"resize": (0, 10)}):
            with self.subTest(options=options), patch("game_state.video_test.multiprocessing.get_context") as context:
                with self.assertRaises(ValueError):
                    run_video("unused.mp4", **options)
                context.assert_not_called()

    def test_log_requires_explicit_overwrite(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            output = Path(directory) / "states.jsonl"
            output.write_text("old log", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _open_log(video, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "old log")
            with _open_log(video, output, overwrite=True) as stream:
                stream.write("new log")
            self.assertEqual(output.read_text(encoding="utf-8"), "new log")

    def test_video_is_protected_from_overwrite_including_links(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"video data")
            symbolic = Path(directory) / "symbolic.jsonl"
            symbolic.symlink_to(video)
            hard = Path(directory) / "hard.jsonl"
            os.link(video, hard)
            for output in (video, symbolic, hard):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    _open_log(video, output, overwrite=True)
            self.assertEqual(video.read_bytes(), b"video data")

    def test_existing_log_has_friendly_cli_error(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            output = Path(directory) / "states.jsonl"
            video.write_bytes(b"video data")
            output.write_text("old log", encoding="utf-8")
            with self.assertLogs("game_state.video_test", level="ERROR") as logs:
                self.assertEqual(main([str(video), "--output", str(output), "--no-window"]), 1)
            self.assertIn("--overwrite", "\n".join(logs.output))
            self.assertNotIn("Traceback", "\n".join(logs.output))
            self.assertEqual(output.read_text(encoding="utf-8"), "old log")

    def test_cli_forwards_explicit_overwrite(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.touch()
            with patch("game_state.video_test.run_video") as run:
                self.assertEqual(main([str(video), "--output", "states.jsonl", "--overwrite", "--realtime"]), 0)
                self.assertTrue(run.call_args.kwargs["overwrite"])
                self.assertTrue(run.call_args.kwargs["realtime"])


if __name__ == "__main__":
    unittest.main()
