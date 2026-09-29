"""Отдельные замеры детекторов на сохранённых кадрах, без iPhone, окна и MCTS.

python -m game_state.benchmark --images screenshots/IMG_1417.PNG \
    --repeats 10 --compare-orientation --output /tmp/durak-benchmark.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np

from game_state import game
from game_state.visualization import DetectionVisualization


@lru_cache(maxsize=1)
def _legacy_ocr():
    """Только для явного сравнения со старым режимом автоповорота."""
    from paddleocr import PaddleOCR
    return PaddleOCR(use_textline_orientation=True, use_doc_orientation_classify=True,
                     lang="en", device="gpu")


def summarize(values):
    return {"count": len(values), "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(max(values))}


@contextmanager
def instrument(timings, *, legacy_orientation=False):
    """Временные обёртки только в этом процессе; исходники бота не меняются."""
    def timed(name, function):
        def call(*args, **kwargs):
            start = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                timings[name].append((perf_counter() - start) * 1000)
        return call

    detectors = {key: timed(key, detector) for key, detector in game._DEFAULT_DETECTORS.items()}
    original_ocr_lines = game._ocr_lines

    def ocr_lines(*args, **kwargs):
        if legacy_orientation:
            kwargs["fixed_orientation"] = False
        return original_ocr_lines(*args, **kwargs)

    # Подмена самого реестра сохраняет передачу visualization в update().
    with patch.dict(game._DEFAULT_DETECTORS, detectors), patch.object(
        DetectionVisualization, "render", timed("render", DetectionVisualization.render)
    ), patch.object(game, "_ocr_lines", ocr_lines), patch.object(
        game, "_ocr", _legacy_ocr if legacy_orientation else game._ocr
    ):
        yield


def measure(images, *, repeats=10, warmup=2, draw=True, legacy_orientation=False, slow_every=1):
    timings = defaultdict(list)
    rows = []
    with instrument(timings, legacy_orientation=legacy_orientation):
        state = game.DurakGameState()

        def update(frame):
            start = perf_counter()
            state.update(frame, draw_detections=draw, slow_every=slow_every)
            return (perf_counter() - start) * 1000

        # Первый кадр и прогрев не попадают в steady-state статистику.
        first_ms = update(images[0][1])
        first_stages = {key: sum(values) for key, values in timings.items()}
        for _ in range(warmup):
            for _, frame in images:
                update(frame)
        timings.clear()
        for _ in range(repeats):
            for path, frame in images:
                previous_counts = {key: len(values) for key, values in timings.items()}
                elapsed = update(frame)
                stages = {key: sum(values[previous_counts.get(key, 0):])
                          for key, values in timings.items() if key not in ("total", "other")}
                other = max(0.0, elapsed - sum(stages.values()))
                timings["other"].append(other)
                timings["total"].append(elapsed)
                rows.append({"image": str(path), "total_ms": elapsed,
                             "stages_ms": {**stages, "other": other}})
    summary = {key: summarize(values) for key, values in timings.items() if values}
    return {"draw_detections": draw, "legacy_orientation": legacy_orientation,
            "slow_every": slow_every,
            "first_frame_ms": first_ms, "first_frame_stages_ms": first_stages,
            "warmup_cycles": warmup, "repeats": repeats, "summary": summary, "frames": rows}


def print_report(result):
    mode = "legacy OCR orientation" if result["legacy_orientation"] else "current OCR orientation"
    print(f"\n{mode}; draw_detections={result['draw_detections']}; slow_every={result['slow_every']}", flush=True)
    print(f"First frame (excluded): {result['first_frame_ms']:.1f} ms", flush=True)
    print(f"{'stage':12} {'calls':>7} {'mean ms':>10} {'median':>10} {'p95':>10} {'max':>10}", flush=True)
    for key, stats in result["summary"].items():
        print(f"{key:12} {stats['count']:7} {stats['mean_ms']:10.2f} {stats['median_ms']:10.2f} "
              f"{stats['p95_ms']:10.2f} {stats['max_ms']:10.2f}", flush=True)
    print(f"Recognition throughput: {1000 / result['summary']['total']['mean_ms']:.2f} FPS", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--slow-every", type=int, default=1, help="Частота редких детекторов в обработанных кадрах")
    parser.add_argument("--compare-orientation", action="store_true",
                        help="Сравнить текущие настройки OCR с прежним автоповоротом всех надписей")
    parser.add_argument("--output", type=Path, help="Сохранить JSON со статистикой и каждым замером")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats >= 1, warmup >= 0")
    if args.slow_every < 1:
        parser.error("slow_every >= 1")
    from constants.constants import CROP_DEQUE_LOST_CARDS
    images = []
    for path in args.images:
        frame = cv2.imread(str(path))
        if frame is None:
            parser.error(f"Не удалось прочитать {path}")
        size = (frame.shape[1], frame.shape[0])
        if size not in CROP_DEQUE_LOST_CARDS:
            parser.error(f"Нет настроек кропа для размера {size}: {path}")
        images.append((path, frame))
    results = []
    for legacy in ([False, True] if args.compare_orientation else [False]):
        for draw in (True, False):
            print(f"Running: legacy_orientation={legacy}, draw={draw}; loading/warming models...", flush=True)
            result = measure(images, repeats=args.repeats, warmup=args.warmup,
                             draw=draw, legacy_orientation=legacy, slow_every=args.slow_every)
            results.append(result)
            print_report(result)
    report = {"images": [str(path) for path, _ in images], "results": results,
              "notes": "Wall-clock durations; first frame and warmup excluded. No live video, GUI or MCTS. "
                       "Models are cached between scenarios. GPU results are materialized by detectors. "
                       "Other = state bookkeeping and instrumentation overhead."}
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
