"""Выполнение подтверждённой пользователем подсказки в координатах кадра."""

from constants.constants import CROP_BUTTON, CROP_CARDS, CROP_FIELD


def _center(bounds, frame_size):
    x1, y1, x2, y2 = bounds
    width, height = frame_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Область хода выходит за границы кадра")
    return (x1 + x2) / 2, (y1 + y2) / 2


def _card_center(layout, code, crop, frame_size, *, uncovered=False):
    matches = [(i, item) for i, item in enumerate(layout) if item.get("card") == code]
    if len(matches) != 1 or "bbox" not in matches[0][1]:
        raise ValueError(f"Нет однозначных координат карты {code}")
    index, item = matches[0]
    if uncovered and (item.get("covers") is not None or any(c.get("covers") == index for c in layout)):
        raise ValueError("Карта для защиты уже накрыта")
    ox, oy, _, _ = crop[frame_size]
    x1, y1, x2, y2 = item["bbox"]
    return _center((x1 + ox, y1 + oy, x2 + ox, y2 + oy), frame_size)


def execute_move(iphone, recommendation, snapshot, geometry):
    if not recommendation or recommendation.get("status") != "ok":
        raise ValueError("Дождитесь актуального рассчитанного хода")
    frame_size = geometry.get("frame_size")
    action = recommendation["action"]
    kind = action["type"]
    if kind in ("pass", "take"):
        if frame_size not in CROP_BUTTON:
            raise ValueError("Не задана область кнопки для этого размера кадра")
        button = snapshot["button"]
        if kind == "take" and button != "itake":
            raise ValueError("Кнопка взятия карт не распознана")
        if kind == "pass" and (button not in ("pass", "bat") or action.get("button", button).lower() != button):
            raise ValueError("Кнопка завершения хода изменилась")
        return iphone.send_tap_async(*_center(CROP_BUTTON[frame_size], frame_size))
    if kind not in ("attack", "defend"):
        raise ValueError("Неизвестный тип хода")
    if kind == "attack" and snapshot["button"] not in ("yourturn", "pass", "bat"):
        raise ValueError("Сейчас нельзя ходить или подкидывать")
    if kind == "defend" and snapshot["button"] != "itake":
        raise ValueError("Сейчас нет хода защиты")
    if frame_size not in CROP_CARDS or frame_size not in CROP_FIELD:
        raise ValueError("Не заданы области карт для этого размера кадра")
    if action["card"] not in snapshot["hand_cards"]:
        raise ValueError("Карты хода больше нет в руке")
    source = _card_center(geometry["hand_layout"], action["card"], CROP_CARDS, frame_size)
    if kind == "defend":
        target = _card_center(geometry["field_layout"], action["target_card"], CROP_FIELD,
                              frame_size, uncovered=True)
    else:
        target = _center(CROP_FIELD[frame_size], frame_size)
    return iphone.swipe_async(*source, *target, duration=0.3, frame_size=frame_size)
