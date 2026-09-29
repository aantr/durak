import unittest
from collections import defaultdict
from unittest.mock import Mock, patch

import numpy as np

from game_state import game
from game_state.benchmark import instrument, measure, summarize


class BenchmarkTests(unittest.TestCase):
    def test_summary(self):
        result = summarize([1, 2, 3, 4])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mean_ms"], 2.5)
        self.assertEqual(result["median_ms"], 2.5)
        self.assertAlmostEqual(result["p95_ms"], 3.85)

    def test_warmup_excluded_and_visualization_forwarded(self):
        for draw in (True, False):
            with self.subTest(draw=draw):
                values = dict(button="your turn", opponent="", mine="", deque=24,
                              hand=[], field=[], trump=None)
                detectors = {name: Mock(return_value=value) for name, value in values.items()}
                frame = np.zeros((4, 4, 3), dtype=np.uint8)
                with patch.dict(game._DEFAULT_DETECTORS, detectors):
                    result = measure([("fake.png", frame)], repeats=3, warmup=2, draw=draw)
                    self.assertIs(game._DEFAULT_DETECTORS["button"], detectors["button"])
                self.assertEqual(len(result["frames"]), 3)
                self.assertEqual(result["summary"]["total"]["count"], 3)
                self.assertEqual(result["summary"]["button"]["count"], 3)
                self.assertEqual(detectors["button"].call_count, 6)
                self.assertEqual("render" in result["summary"], draw)
                self.assertEqual("visualization" in detectors["button"].call_args.kwargs, draw)
                self.assertNotIn("total", result["frames"][0]["stages_ms"])
                self.assertFalse(frame.any())

    def test_legacy_comparison_overrides_only_temporary_ocr_call(self):
        original = Mock(return_value=["6"])
        with patch.object(game, "_ocr_lines", original):
            with instrument(defaultdict(list), legacy_orientation=True):
                game._ocr_lines(None, {}, None, fixed_orientation=True)
            self.assertIs(game._ocr_lines, original)
        self.assertFalse(original.call_args.kwargs["fixed_orientation"])


if __name__ == "__main__":
    unittest.main()
