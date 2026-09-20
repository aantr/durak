#!/usr/bin/env python3
"""
crop_cards.py — обрезка изображений по заранее заданным размерам.

Принимает на вход пути к изображениям и изменяет их же (in-place).

Логика:
    CROP_CARDS — словарь вида {(width, height): (x1, y1, x2, y2)}.
    Для каждого входного изображения берём его реальные (width, height),
    ищем подходящий ключ в CROP_CARDS и обрезаем по указанному прямоугольнику.
    Если ключа нет — изображение пропускается.

Пример:
    python crop_cards.py img1.png img2.png img3.png
    python crop_cards.py ./images/*.png
"""

import sys
from pathlib import Path

import cv2

from constants import CROP_CARDS


def crop_image(path: Path) -> bool:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[!] Не удалось прочитать: {path}")
        return False

    h, w = img.shape[:2]
    key = (w, h)
    if key not in CROP_CARDS:
        print(f"[=] Пропуск (нет правила для {w}x{h}): {path}")
        return False

    x1, y1, x2, y2 = CROP_CARDS[key]
    # защита от выхода за границы
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        print(f"[!] Некорректная область обрезки для {path}: "
              f"{(x1, y1, x2, y2)}")
        return False

    cropped = img[y1:y2, x1:x2]
    if not cv2.imwrite(str(path), cropped):
        print(f"[!] Не удалось записать: {path}")
        return False

    print(f"[+] {path}: {w}x{h} -> {cropped.shape[1]}x{cropped.shape[0]} "
          f"(crop {x1},{y1},{x2},{y2})")
    return True


def main():
    if len(sys.argv) < 2:
        print("Использование: python crop_cards.py <image1> [image2 ...]")
        sys.exit(1)

    ok = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.is_file():
            print(f"[!] Не файл: {p}")
            continue
        if crop_image(p):
            ok += 1

    print(f"Готово. Обработано: {ok}")


if __name__ == "__main__":
    main()