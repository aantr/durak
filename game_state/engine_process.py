"""Расчёт подсказки в отдельном процессе по явному запросу пользователя."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing


_engine = None


def _initialize_engine(options):
    global _engine
    from game_engine import DurakEngine
    _engine = DurakEngine(**options)


def _suggest(snapshot):
    try:
        return _engine.suggest(snapshot)
    except ValueError as exc:
        return {"status": "invalid_state", "reason": str(exc), "action": None}


class EngineProcess:
    def __init__(self, options):
        self.options = dict(options)
        self.executor = None
        self.pending = None
        self.recommendation = None
        self.last_move = None
        self.generation = 0
        self.submitted_generation = 0

    def reset(self):
        self.generation += 1
        self.recommendation = None
        self.last_move = None
        if self.pending is not None and self.pending.cancel():
            self.pending = None

    def request(self, snapshot):
        """Запускает один расчёт; повторное нажатие во время расчёта не ставит очередь."""
        self.poll()
        if self.pending is not None:
            return False
        if self.executor is None:
            self.executor = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_engine, initargs=(self.options,),
            )
        self.submitted_generation = self.generation
        self.pending = self.executor.submit(_suggest, deepcopy(snapshot))
        self.recommendation = None
        return True

    def poll(self):
        """Получает результат, не запуская новых расчётов при изменении кадра."""
        if self.pending is not None and self.pending.done():
            result = self.pending.result()
            self.pending = None
            if self.submitted_generation == self.generation:
                if result.get("status") == "ok":
                    self.last_move = result
                self.recommendation = result
        result = self.recommendation
        if result is None:
            calculating = self.pending is not None and self.submitted_generation == self.generation
            result = {"status": "calculating" if calculating else "idle",
                      "reason": "Calculating..." if calculating else "Space — рассчитать ход",
                      "action": None}
        if result.get("status") != "ok" and self.last_move is not None:
            return {**result, "last_move": self.last_move}
        return result

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
