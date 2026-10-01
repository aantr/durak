"""Автоматический расчёт и однократное выполнение хода по свежему состоянию."""

from copy import deepcopy
import logging
import math

logger = logging.getLogger(__name__)


def position_key(snapshot):
    """Игровая позиция без порядка детекций, координат и произвольного OCR-текста."""
    layout = snapshot.get("field_layout", [])
    table = []
    for item in layout:
        parent = item.get("covers")
        if parent is None:
            covered = ""
        elif type(parent) is int and 0 <= parent < len(layout):
            covered = str(layout[parent].get("card"))
        else:
            covered = f"invalid:{parent!r}"
        table.append((str(item.get("card")), covered))
    actions = ("bat", "pass", "itake")
    return (
        *(snapshot.get(key) for key in ("phase", "button", "trump", "deck_remaining", "opponent_card_count")),
        *(tuple(sorted(snapshot.get(key, []))) for key in
          ("hand_cards", "field_cards", "out_cards", "known_opponent_cards")),
        tuple(sorted(table)),
        *(snapshot.get(key) if snapshot.get(key) in actions else "" for key in ("mine", "opponent")),
    )


class AutoPlay:
    def __init__(self, engine, delay=1.0):
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("auto_delay должен быть конечным числом >= 0")
        self.engine = engine
        self.delay = delay
        self.enabled = False
        self.snapshot = None
        self.deadline = None
        self.requested = False
        self.sent_key = None
        self.status = "OFF"

    def _clear_calculation(self, *, clear_evaluation=False):
        self.engine.reset(clear_evaluation=clear_evaluation)
        self.snapshot = None
        self.deadline = None
        self.requested = False

    def reset(self, *, clear_evaluation=False):
        self._clear_calculation(clear_evaluation=clear_evaluation)
        self.sent_key = None
        self.status = "Waiting for turn" if self.enabled else "OFF"

    def toggle(self):
        self.enabled = not self.enabled
        self.reset()
        return self.enabled

    @staticmethod
    def our_turn(snapshot):
        return (snapshot.get("phase") in ("your_turn", "throw_in", "defend_or_take")
                and snapshot.get("button") in ("yourturn", "pass", "bat", "itake")
                and snapshot.get("mine") not in ("bat", "itake", "pass")
                and snapshot.get("opponent") != "bat")

    def step(self, snapshot, now, *, input_busy=False, execute):
        """Вызывать после свежего распознавания и обработки клавиатуры.

        Таймер отсчитывается от появления своего хода. Новые кадры уточняют
        позицию, но не прерывают расчёт; его результат проверяется перед вводом.
        """
        if not self.enabled or snapshot is None:
            return
        if not self.our_turn(snapshot):
            if self.deadline is not None or self.requested:
                self._clear_calculation()
                logger.info("Автоигра: ожидание своего хода")
            self.status = "Waiting for turn"
            return
        key = position_key(snapshot)
        if key == self.sent_key:
            if self.deadline is not None or self.requested:
                self._clear_calculation()
            self.status = "Waiting for move confirmation"
            return
        if self.deadline is None:
            self.deadline = now + self.delay
            logger.info("Автоигра: свой ход, расчёт через %g с", self.delay)
        if input_busy:
            self.status = "Waiting for input"
            return
        if now < self.deadline:
            self.status = f"Calculate in {self.deadline - now:.1f}s"
            return
        if not self.requested:
            if not self.engine.request(snapshot):
                self.status = "Waiting for previous calculation"
                return
            self.snapshot = deepcopy(snapshot)
            self.requested = True
            logger.info("Автоигра: расчёт запущен")
        result = self.engine.poll()
        if result.get("status") in ("calculating", "idle"):
            self.status = "Calculating..."
            return
        if result.get("status") != "ok":
            logger.info("Автоигра: расчёт без хода (%s); будет повторная попытка",
                        result.get("reason", result.get("status")))
            self._clear_calculation()
            self.deadline = now + max(self.delay, 0.1)
            self.status = "Waiting for valid state; retry scheduled"
            return
        if key != position_key(self.snapshot):
            logger.info("Автоигра: позиция изменилась за время расчёта; пересчитываем")
            self._clear_calculation()
            self.deadline = now
            self.status = "Position changed; recalculating"
            return
        # Включая ошибку отправки: частично выполненный ввод нельзя повторять.
        self.sent_key = key
        self._clear_calculation()
        self.status = "Waiting for move confirmation"
        execute(result)
