"""Проверка DurakGameState на видео с пропуском ожиданий между кадрами.

python -m game_state.video_test game.mp4 --fps 30 --draw-detections
python -m game_state.video_test game.mp4 --fps 10 --no-window --output states.jsonl

Загрузка/прогрев моделей выполняются до запуска часов видео. --realtime
возвращает ожидание кадров по реальным часам. Время распознавания учитывается
в обоих режимах; общий лимит работы процесса равен длительности видео.
Окно закрывается автоматически. Space не нужен; ввод на iPhone не отправляется.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import multiprocessing
from pathlib import Path
from queue import Empty
import signal
import sys
import time

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game_state.bot import format_state, state_snapshot
from game_state.game import DurakGameState

logger = logging.getLogger(__name__)
WINDOW = "Durak - video test"


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float | None = None

    @property
    def duration(self):
        return self.duration_seconds if self.duration_seconds is not None else self.frame_count / self.fps

    @classmethod
    def read(cls, capture):
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError("Видео должно иметь известную частоту и число кадров")
        return cls(fps, int(count), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


class TimestampVideoReader:
    """Выбор последнего кадра с PTS <= времени видео, включая VFR и start_time.

    Декодируем последовательно до ближайшего будущего кадра. При большом
    отставании перематываем назад к ключевому кадру по timestamp, не по FPS.
    В памяти хранятся только текущий и один следующий кадр.
    """
    def __init__(self, path):
        import av
        self.container = av.open(str(path))
        try:
            if not self.container.streams.video:
                raise ValueError("В файле нет видеопотока")
            self.stream = self.container.streams.video[0]
            self.decoder = iter(self.container.decode(self.stream))
            self.current = next(self.decoder, None)
            if self.current is None or self.current.time is None:
                raise ValueError("Первый видеокадр не содержит временной метки")
            self.origin = float(self.stream.start_time * self.stream.time_base) if self.stream.start_time is not None else self.current.time
            if self.stream.duration is not None:
                duration = float(self.stream.duration * self.stream.time_base)
            elif self.container.duration is not None:
                container_start = (self.container.start_time or 0) / av.time_base
                duration = container_start + self.container.duration / av.time_base - self.origin
            else:
                raise ValueError("Неизвестна длительность видео")
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Некорректная длительность видео")
            rate = self.stream.average_rate or self.stream.guessed_rate or self.stream.base_rate
            fps = float(rate) if rate else 30.0
            self.info = VideoInfo(fps, self.stream.frames or math.ceil(duration * fps),
                                  self.current.width, self.current.height, duration)
            self.current_time = self.current.time - self.origin
            self.next = None
            self.eof = False
            self._image = None
        except BaseException:
            self.container.close()
            raise

    def _advance(self):
        self.next = next(self.decoder, None)
        if self.next is None:
            self.eof = True
        elif self.next.time is None:
            raise ValueError("Видеокадр не содержит PTS")

    def read_at(self, seconds):
        if seconds - self.current_time > 1.0 and not self.eof:
            timestamp = int((seconds + self.origin) / self.stream.time_base)
            self.container.seek(timestamp, stream=self.stream, backward=True, any_frame=False)
            self.decoder = iter(self.container.decode(self.stream))
            self.next = None
            # current сохраняется как допустимый предыдущий кадр при EOF seek.
            self._advance()
        if self.next is None and not self.eof:
            self._advance()
        while self.next is not None and self.next.time - self.origin <= seconds + 1e-9:
            candidate_time = self.next.time - self.origin
            if candidate_time >= self.current_time:
                self.current = self.next
                self.current_time = candidate_time
                self._image = None
            self._advance()
        if self.current_time > seconds + 1e-9:
            return None  # В этом месте временной шкалы ещё нет кадра.
        if self._image is None:
            self._image = self.current.to_ndarray(format="bgr24")
        return self.current_time, self._image

    def first_frame(self):
        return self.current.to_ndarray(format="bgr24")

    def release(self):
        self.container.close()


def frame_at_time(elapsed, sample_fps, video):
    """Последний доступный кадр сетки sample_fps, никогда не будущий кадр."""
    if elapsed >= video.duration:
        return None
    sample_time = math.floor(max(0.0, elapsed) * sample_fps) / sample_fps
    return min(video.frame_count - 1, int(sample_time * video.fps + 1e-9))


def playback(capture, state, video, *, fps, slow_every, draw, start, emit,
             resize=None, clock=time.monotonic, sleep=time.sleep, realtime=False):
    """После каждого update выбирает индекс по часам, а не предыдущий + 1.

    По умолчанию ожидание следующей выборки заменяется сдвигом часов видео.
    Декодирование и распознавание по-прежнему расходуют время видео.
    Работает в дочернем процессе: родитель может прервать зависший декодер
    или долгий вызов модели по общему дедлайну.
    """
    skipped_wait = 0.0
    last_index = -1
    last_timestamp = None
    last_tick = -1
    timestamp_reader = isinstance(capture, TimestampVideoReader)
    while True:
        elapsed = clock() - start + skipped_wait
        index = frame_at_time(elapsed, fps, video)
        if index is None:
            return
        tick = math.floor(elapsed * fps)
        if tick == last_tick or (not timestamp_reader and index == last_index):
            next_tick = (math.floor(elapsed * fps) + 1) / fps
            if realtime:
                sleep(min(0.01, max(0.0001, next_tick - elapsed), video.duration - elapsed))
            else:
                # nextafter не даёт округлению оставить нас на прежнем tick.
                target = min(math.nextafter(next_tick, math.inf), video.duration)
                skipped_wait += target - elapsed
            continue
        last_tick = tick
        if timestamp_reader:
            sample = capture.read_at(tick / fps)
            if sample is None or sample[0] == last_timestamp:
                continue
            timestamp, frame = sample
            index = None  # В VFR нельзя восстановить порядковый номер из среднего FPS.
        else:
            # Совместимость с CFR VideoCapture / простыми тестовыми источниками.
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise RuntimeError(f"Не удалось перейти к кадру {index}")
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Не удалось декодировать кадр {index}")
            timestamp = index / video.fps
        if resize:
            frame = cv2.resize(frame, resize)
        if clock() - start + skipped_wait >= video.duration:
            return
        inference_start = clock()
        state.update(frame, slow_every=slow_every, draw_detections=draw)
        finished = clock() - start + skipped_wait
        if finished >= video.duration:
            return  # Поздний результат не продлевает видео и не попадает в журнал.
        record = {
            "type": "state", "frame_index": index, "video_time": timestamp,
            "sample_time": tick / fps,
            "finished_at": finished, "recognition_ms": (clock() - inference_start) * 1000,
            "state": state_snapshot(state),
        }
        emit(record, state.annotated_frame if draw else frame)
        last_index = index
        last_timestamp = timestamp


def _worker(control, output, path, options):
    capture = None
    try:
        startup = time.monotonic()
        capture = TimestampVideoReader(path)
        video = capture.info
        first = capture.first_frame()
        resize = options["resize"]
        if resize:
            first = cv2.resize(first, resize)
        from constants.constants import CROP_FIELD
        size = (first.shape[1], first.shape[0])
        if size not in CROP_FIELD:
            raise ValueError(f"Нет областей детекции для {size}. Укажите --resize WIDTH HEIGHT "
                             f"для одного из размеров {list(CROP_FIELD)}")
        state = DurakGameState(trump=options["trump"])
        for _ in range(options["warmup"]):
            state.update(first, draw_detections=options["draw"], slow_every=1)
        state.reset(trump=options["trump"])  # Прогрев не меняет историю партии.
        # Paddle устанавливает обработчик SIGTERM с аварийной трассировкой.
        # Здесь остановка worker по дедлайну — штатное завершение, не ошибка GPU.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        control.send({"type": "ready", "video": video, "startup_seconds": time.monotonic() - startup})
        start = control.recv()
        processed = 0

        def emit(record, image):
            nonlocal processed
            # В том числе после ленивой загрузки моделей при --warmup 0.
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            processed += 1
            record["processed_frames"] = processed
            preview = None
            if options["window"]:
                width = min(460, image.shape[1])
                preview = cv2.resize(image, (width, max(1, round(image.shape[0] * width / image.shape[1]))))
            # Не запускаем следующий inference, пока родитель не забрал результат.
            # В очереди не бывает пачки старых кадров; после отправки снова смотрим часы.
            output.put((record, preview))

        playback(capture, state, video, fps=options["fps"], slow_every=options["slow_every"],
                 draw=options["draw"], start=start, emit=emit, resize=resize,
                 realtime=options["realtime"])
        control.send({"type": "done", "processed_frames": processed})
    except BaseException as exc:
        try:
            control.send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if capture is not None:
            capture.release()
        control.close()


def stop_worker(process):
    """Не ждём окончания тяжёлого inference после конца видео."""
    if process.is_alive():
        process.terminate()
    process.join(timeout=0.5)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.5)


def _open_log(video, output, *, overwrite=False):
    video, output = Path(video), Path(output)
    if video.resolve() == output.resolve() or (
        video.exists() and output.exists() and video.samefile(output)
    ):
        raise ValueError("Журнал --output не должен совпадать с исходным видео")
    return output.open("w" if overwrite else "x", encoding="utf-8")


def run_video(path, *, fps=30.0, slow_every=2, draw=False, window=True,
              state_format="pretty", warmup=2, resize=None, trump=None, output=None,
              startup_timeout=180.0, overwrite=False, realtime=False):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps должен быть конечным числом > 0")
    if isinstance(slow_every, bool) or not isinstance(slow_every, int) or slow_every < 1:
        raise ValueError("slow_every должен быть положительным целым числом")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup должен быть целым числом >= 0")
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError("startup_timeout должен быть конечным числом > 0")
    if state_format not in ("pretty", "json"):
        raise ValueError("state_format: pretty или json")
    if resize is not None and (len(resize) != 2 or any(not isinstance(n, int) or n < 1 for n in resize)):
        raise ValueError("resize: два положительных целых размера")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    results = context.Queue(maxsize=1)
    options = dict(fps=fps, slow_every=slow_every, draw=draw, window=window,
                   warmup=warmup, resize=resize, trump=trump, realtime=realtime)
    process = context.Process(target=_worker, args=(child, results, str(path), options), daemon=True)
    stream = None
    window_created = False
    started = False
    try:
        # Перезапись журнала разрешена только явно; исходное видео защищено всегда.
        if output:
            stream = _open_log(path, output, overwrite=overwrite)
        process.start()
        started = True
        child.close()
        logger.info("Загрузка и прогрев моделей; часы видео ещё не запущены")
        startup_deadline = time.monotonic() + startup_timeout
        while not parent.poll(0.05):
            if not process.is_alive():
                raise RuntimeError("Процесс распознавания завершился до запуска видео")
            if time.monotonic() >= startup_deadline:
                raise TimeoutError("Истёк лимит загрузки/прогрева моделей")
        ready = parent.recv()
        if ready["type"] != "ready":
            raise RuntimeError(ready.get("message", "Ошибка запуска видео"))
        video = ready["video"]
        metadata = {"type": "metadata", "path": str(path), "duration": video.duration,
                    "source_fps": video.fps, "source_frames": video.frame_count,
                    "frame_selection": "pts", "realtime": realtime,
                    "sample_fps": fps, "slow_every": slow_every,
                    "startup_seconds": ready["startup_seconds"]}
        if stream:
            stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        logger.info("Видео %.3f с, %.3f FPS; выборка до %.3f FPS. Прогрев %.2f с",
                    video.duration, video.fps, fps, ready["startup_seconds"])
        if window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(WINDOW, 460, 900)
        start = time.monotonic()
        parent.send(start)
        previous = None
        latest = None
        delivered = 0
        reason = "end_of_video"
        expected_frames = None
        while time.monotonic() - start < video.duration:
            remaining = video.duration - (time.monotonic() - start)
            try:
                record, preview = results.get(timeout=max(0, min(0.005, remaining)))
            except Empty:
                pass
            else:
                if time.monotonic() - start >= video.duration:
                    break
                latest = record
                delivered += 1
                if stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                if record["state"] != previous:
                    if state_format == "json":
                        logger.info("%s", json.dumps(record, ensure_ascii=False))
                    else:
                        logger.info("Кадр %s · видео %.3f с · распознавание %.1f мс\n%s",
                                    record.get("processed_frames", record["frame_index"]), record["video_time"], record["recognition_ms"],
                                    format_state(record["state"]))
                    previous = record["state"]
                if window and preview is not None:
                    cv2.imshow(WINDOW, preview)
            if expected_frames is None and parent.poll():
                message = parent.recv()
                if message["type"] == "error":
                    raise RuntimeError(message["message"])
                if message["type"] == "done":
                    expected_frames = message["processed_frames"]
            # Pipe может доставить done раньше последнего элемента Queue.
            if expected_frames is not None and delivered >= expected_frames:
                break
            if window:
                if cv2.waitKeyEx(1) in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    reason = "user_stop"
                    break
        elapsed = time.monotonic() - start
        stop_worker(process)
        summary = {"type": "summary", "reason": reason, "duration": video.duration,
                   "playback_seconds": elapsed, "delivered_frames": delivered,
                   "last_video_time": latest["video_time"] if latest else None,
                   "last_state": latest["state"] if latest else None}
        if stream:
            stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
        logger.info("Готово: %.3f с воспроизведения / %.3f с видео, %d кадров распознано",
                    elapsed, video.duration, delivered)
        return summary
    finally:
        if started:
            stop_worker(process)
        else:
            child.close()
        parent.close()
        results.close()
        if stream:
            stream.close()
        if window_created:
            cv2.destroyWindow(WINDOW)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--fps", type=float, default=30.0, help="Частота выборки кадров из видео")
    parser.add_argument("--slow-every", type=int, default=2, help="Как в bot.py: остальные детекторы раз в N кадров")
    parser.add_argument("--draw-detections", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Ждать кадры по реальным часам, без ускорения пауз")
    parser.add_argument("--state-format", choices=("pretty", "json"), default="pretty")
    parser.add_argument("--warmup", type=int, default=2, help="Прогрев до запуска часов видео (0 — без прогрева)")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--trump", choices=("C", "D", "H", "S"))
    parser.add_argument("--output", type=Path, help="Новый JSONL: временные метки, состояния и итоги")
    parser.add_argument("--overwrite", action="store_true", help="Перезаписать существующий журнал --output")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error("Укажите существующий видеофайл")
    if args.overwrite and args.output is None:
        parser.error("--overwrite используется вместе с --output")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps должен быть > 0")
    if args.slow_every < 1 or args.warmup < 0:
        parser.error("--slow-every >= 1, --warmup >= 0")
    if args.resize and min(args.resize) < 1:
        parser.error("Размеры --resize должны быть > 0")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("--startup-timeout должен быть > 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_video(args.video, fps=args.fps, slow_every=args.slow_every, draw=args.draw_detections,
                  window=not args.no_window, state_format=args.state_format, warmup=args.warmup,
                  resize=tuple(args.resize) if args.resize else None, trump=args.trump,
                  output=args.output, startup_timeout=args.startup_timeout, overwrite=args.overwrite,
                  realtime=args.realtime)
    except KeyboardInterrupt:
        logger.info("Остановлено пользователем")
    except FileExistsError as exc:
        logger.error("Файл %s уже существует. Укажите другое имя --output или добавьте "
                     "--overwrite, чтобы заменить старый журнал.", exc.filename)
        return 1
    except Exception:
        logger.exception("Ошибка обработки видео")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
