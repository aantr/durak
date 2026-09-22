#!/usr/bin/env python3
"""
Генератор датасета для YOLO-классификации.

- Итоговая картинка ВСЕГДА size×size.
- Фон НЕ масштабируется (кроме случая, когда он меньше size —
  тогда увеличиваем, чтобы закрыть кадр), кроп по центру.
- Объект НЕ масштабируется — вставляется как есть, строго по центру фона.
- Классы = имена папок в --src (кроме backgrounds).

Пример:
    python gen_dataset.py --src dataset_suit --dst dataset_cls \
        --size 224 --per-class 300 --val-ratio 0.2 --augment
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
BACKGROUND_DIR = "backgrounds"


def load_images(folder: Path):
    return [p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXT]


def fit_background(bg: Image.Image, size: int) -> Image.Image:
    """
    Приводит фон к size×size.
    - Если фон >= size по обеим сторонам: центральный кроп.
    - Если хотя бы по одной стороне меньше: увеличиваем с сохранением
      пропорций до покрытия size×size, затем центральный кроп.
    Никакого уменьшения фона, если он больше — только кроп.
    """
    W, H = bg.size
    if W < size or H < size:
        scale = max(size / W, size / H)
        nw, nh = int(W * scale + 0.5), int(H * scale + 0.5)
        bg = bg.resize((nw, nh), Image.LANCZOS)
        W, H = bg.size

    left = (W - size) // 2
    top = (H - size) // 2
    return bg.crop((left, top, left + size, top + size))


def augment_object(obj: Image.Image, args) -> Image.Image:
    if args.aug_rotate > 0:
        angle = random.uniform(-args.aug_rotate, args.aug_rotate)
        obj = obj.rotate(angle, resample=Image.BICUBIC, expand=True)

    if args.aug_brightness > 0:
        f = 1.0 + random.uniform(-args.aug_brightness, args.aug_brightness)
        obj = ImageEnhance.Brightness(obj).enhance(f)
    if args.aug_contrast > 0:
        f = 1.0 + random.uniform(-args.aug_contrast, args.aug_contrast)
        obj = ImageEnhance.Contrast(obj).enhance(f)
    if args.aug_color > 0:
        f = 1.0 + random.uniform(-args.aug_color, args.aug_color)
        obj = ImageEnhance.Color(obj).enhance(f)
    return obj


def augment_background(bg: Image.Image, args) -> Image.Image:
    if args.aug_brightness > 0:
        f = 1.0 + random.uniform(-args.aug_brightness, args.aug_brightness)
        bg = ImageEnhance.Brightness(bg).enhance(f)
    if args.aug_contrast > 0:
        f = 1.0 + random.uniform(-args.aug_contrast, args.aug_contrast)
        bg = ImageEnhance.Contrast(bg).enhance(f)
    if args.aug_blur > 0 and random.random() < 0.3:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=args.aug_blur))
    return bg


def add_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0, sigma * 255, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def paste_center(bg: Image.Image, obj: Image.Image) -> Image.Image:
    """
    Вставляет объект строго по центру фона, без масштабирования.
    Если объект больше фона — центрируем и обрезаем по границам кадра
    (иначе PIL упадёт).
    """
    W, H = bg.size
    ow, oh = obj.size

    x = (W - ow) // 2
    y = (H - oh) // 2

    bg_rgba = bg.convert("RGBA")

    if ow <= W and oh <= H:
        bg_rgba.alpha_composite(obj, (x, y))
    else:
        # объект больше фона: создаём холст size×size и центрируем объект
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        canvas.alpha_composite(obj, (x, y))
        bg_rgba = Image.alpha_composite(bg_rgba, canvas)

    return bg_rgba.convert("RGB")


def generate_one(backgrounds, class_objects, args) -> Image.Image:
    bg = Image.open(random.choice(backgrounds)).convert("RGB")
    bg = fit_background(bg, args.size)
    if args.augment:
        bg = augment_background(bg, args)

    obj = Image.open(random.choice(class_objects)).convert("RGBA")
    if args.augment:
        obj = augment_object(obj, args)

    img = paste_center(bg, obj)

    if args.augment and args.aug_noise > 0:
        img = add_noise(img, args.aug_noise)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_suit")
    ap.add_argument("--dst", default="dataset_cls")
    ap.add_argument("--size", type=int, default=224,
                    help="сторона итоговой картинки (всегда size×size)")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jpeg-quality", type=int, default=92)

    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--aug-rotate", type=float, default=15.0)
    ap.add_argument("--aug-brightness", type=float, default=0.2)
    ap.add_argument("--aug-contrast", type=float, default=0.2)
    ap.add_argument("--aug-color", type=float, default=0.2)
    ap.add_argument("--aug-blur", type=float, default=1.2)
    ap.add_argument("--aug-noise", type=float, default=0.02)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)

    bg_dir = src / BACKGROUND_DIR
    if not bg_dir.is_dir():
        raise SystemExit(f"Не найдена папка фонов: {bg_dir}")

    backgrounds = load_images(bg_dir)
    if not backgrounds:
        raise SystemExit(f"В {bg_dir} нет картинок")

    classes = sorted(
        p.name for p in src.iterdir()
        if p.is_dir() and p.name != BACKGROUND_DIR
    )
    if not classes:
        raise SystemExit(f"В {src} нет папок классов")

    print(f"Фонов: {len(backgrounds)}")
    print(f"Классы: {classes}")
    print(f"Размер итоговой картинки: {args.size}x{args.size}")

    for split in ("train", "val"):
        for cls in classes:
            (dst / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_dir = src / cls
        objects = load_images(cls_dir)
        if not objects:
            print(f"[skip] нет картинок в {cls_dir}")
            continue

        print(f"[{cls}] объектов: {len(objects)}, генерирую {args.per_class}...")

        n_val = int(args.per_class * args.val_ratio)

        for i in range(args.per_class):
            img = generate_one(backgrounds, objects, args)
            split = "val" if i < n_val else "train"
            out = dst / split / cls / f"{cls}_{i:05d}.jpg"
            img.save(out, "JPEG", quality=args.jpeg_quality)

    print("Готово.")
    print(f"Датасет: {dst.resolve()}")
    print(f'Обучение: yolo classify train data="{dst}" '
          f'model=yolo11n-cls.pt imgsz={args.size} epochs=100')


if __name__ == "__main__":
    main()