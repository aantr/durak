#!/usr/bin/env python3
"""
Генератор датасета для YOLO.

- backgrounds всегда внутри --dataset (папка dataset/backgrounds)
- размер выходного изображения = размер выбранного фона
- объекты классов вставляются с исходным размером (с возможностью поворота и масштаба)
- в аннотации попадают ТОЛЬКО классы, перечисленные в --yaml
- классы из dataset, которых нет в yaml, тоже загружаются, но не размечаются
- --weights: коэффициенты частоты выбора классов (влияют на частоту).
  Теперь применяются и к background-items тоже.
- --max-angle: максимальный угол поворота (общий или per-class)
- --scale / --scale-per-class: диапазон масштабирования (min,max) для классов
- --background-items: классы, которые размещаются ПЕРВЫМИ на фоне
  (например: tree,grass). Остальные объекты рисуются ПОСЛЕ них и могут
  перекрывать их сверху. Классы из background-items МОГУТ выбираться
  случайно (с учётом весов) наравне с остальными, но всегда кладутся
  в нижний слой.
- генерируется generation/dataset.yaml

Пример:
    python generate.py \
        --dataset ./dataset \
        --yaml dataset.yaml \
        --output ./generation \
        --num-images 500 \
        --area 0.1,0.1,0.9,0.9 \
        --area-per-class "cat=0.05,0.05,0.45,0.45;dog=0.55,0.55,0.95,0.95" \
        --weights "cat=3;dog=1;tree=2;grass=4" \
        --max-angle 30 \
        --max-angle-per-class "cat=15;tree=180" \
        --scale "cat=0.7,1.3;dog=0.5,1.0" \
        --background-items "tree,grass" \
        --objects-per-image 4 \
        --val-split 0.2 \
        --seed 42
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKGROUNDS_DIRNAME = "backgrounds"


# ---------- парсеры CLI ----------

def parse_area(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ожидается 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise argparse.ArgumentTypeError("area в [0..1], x1<x2, y1<y2")
    return (x1, y1, x2, y2)


def parse_area_list(s: str):
    result = {}
    if not s:
        return result
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(
                f"Неверный формат области: '{chunk}', ожидается class=x1,y1,x2,y2"
            )
        name, coords = chunk.split("=", 1)
        result[name.strip()] = parse_area(coords)
    return result


def parse_scale(s: str):
    """'0.5,1.5' -> (0.5, 1.5)"""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("ожидается 'min,max'")
    lo, hi = parts
    if lo <= 0 or hi <= 0:
        raise argparse.ArgumentTypeError("scale должен быть > 0")
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def parse_scale_list(s: str):
    """'cat=0.5,1.5;dog=1.0,1.0' -> {'cat': (0.5,1.5), 'dog': (1.0,1.0)}"""
    result = {}
    if not s:
        return result
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(
                f"Неверный формат scale: '{chunk}', ожидается class=min,max"
            )
        name, coords = chunk.split("=", 1)
        result[name.strip()] = parse_scale(coords)
    return result


def parse_weights(s: str):
    """'cat=3;dog=1;tree=0.5' -> {'cat':3.0, 'dog':1.0, 'tree':0.5}"""
    result = {}
    if not s:
        return result
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(
                f"Неверный формат веса: '{chunk}', ожидается class=value"
            )
        name, val = chunk.split("=", 1)
        try:
            v = float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Вес не число: '{val}'")
        if v < 0:
            raise argparse.ArgumentTypeError(f"Вес должен быть >= 0: '{chunk}'")
        result[name.strip()] = v
    return result


def parse_angles(s: str):
    """'cat=15;tree=180' -> {'cat':15.0, 'tree':180.0}"""
    return parse_weights(s)


def parse_csv_list(s: str):
    """'a,b,c' -> ['a','b','c']"""
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


# ---------- загрузка метаданных ----------

def load_dataset_classes(dataset_dir: Path):
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Папка dataset не найдена: {dataset_dir}")
    classes = sorted(
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and d.name != BACKGROUNDS_DIRNAME
    )
    if not classes:
        raise RuntimeError(f"В dataset нет подпапок с классами: {dataset_dir}")
    return classes


def load_yaml(yaml_path: Path):
    if yaml_path is None:
        return {}
    if not yaml_path.exists():
        print(f"[!] yaml не найден: {yaml_path}")
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_yaml_names(yaml_data):
    names = yaml_data.get("names") if isinstance(yaml_data, dict) else None
    if names is None:
        return []
    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
        return [str(v) for _, v in items]
    if isinstance(names, list):
        return [str(v) for v in names]
    return []


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS]


# ---------- загрузка объектов ----------

def load_image_with_alpha(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    elif img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255
        img = bgr
    else:
        img = img[:, :, :3]
        white = np.all(img > 245, axis=2)
        mask = np.where(white, 0, 255).astype(np.uint8)
    return img, mask


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_to_mask(img, mask):
    b = bbox_from_mask(mask)
    if b is None:
        return None, None
    x1, y1, x2, y2 = b
    return img[y1:y2 + 1, x1:x2 + 1].copy(), mask[y1:y2 + 1, x1:x2 + 1].copy()


def load_objects(dataset_dir: Path, class_names):
    objects = {}
    for name in class_names:
        folder = dataset_dir / name
        loaded = []
        for p in list_images(folder):
            img, mask = load_image_with_alpha(p)
            if img is None:
                print(f"[!] Не удалось прочитать: {p}")
                continue
            img_c, mask_c = crop_to_mask(img, mask)
            if img_c is None:
                continue
            loaded.append((img_c, mask_c))
        objects[name] = loaded
        print(f"  {name}: {len(loaded)} объектов")
    return objects


# ---------- трансформации ----------

def scale_object(img, mask, scale):
    """Масштабирует объект и маску с сохранением пропорций."""
    if abs(scale - 1.0) < 1e-3:
        return img, mask
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    img_s = cv2.resize(img, (new_w, new_h), interpolation=interp)
    mask_s = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_s, mask_s


def rotate_object(img, mask, angle_deg):
    """Поворачивает объект и маску вокруг центра. Возвращает обрезанные по маске."""
    if abs(angle_deg) < 0.01:
        return img, mask
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    diag = int(np.ceil(np.sqrt(w * w + h * h)))
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += diag / 2 - cx
    M[1, 2] += diag / 2 - cy

    img_r = cv2.warpAffine(
        img, M, (diag, diag),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask_r = cv2.warpAffine(
        mask, M, (diag, diag),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    img_c, mask_c = crop_to_mask(img_r, mask_r)
    if img_c is None:
        return img, mask
    return img_c, mask_c


# ---------- вставка ----------

def paste_object(bg, obj, mask, x, y):
    oh, ow = obj.shape[:2]
    bh, bw = bg.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + ow)
    y2 = min(bh, y + oh)
    if x1 >= x2 or y1 >= y2:
        return None

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    obj_crop = obj[oy1:oy2, ox1:ox2]
    mask_crop = mask[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0
    mask_crop = mask_crop[..., None]

    roi = bg[y1:y2, x1:x2].astype(np.float32)
    blended = obj_crop.astype(np.float32) * mask_crop + roi * (1.0 - mask_crop)
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)

    return (x1, y1, x2, y2)


def clip_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2.0 / w,
        (y1 + y2) / 2.0 / h,
        (x2 - x1) / w,
        (y2 - y1) / h,
    )


def weighted_choice(items, weights):
    """items: list[str], weights: list[float] (>=0)"""
    total = sum(weights)
    if total <= 0:
        return random.choice(items)
    r = random.uniform(0, total)
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Генератор датасета для YOLO")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--yaml", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("generation"))
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--area", type=parse_area, default=(0.0, 0.0, 1.0, 1.0),
                    help="Общая область: x1,y1,x2,y2 в [0..1]")
    ap.add_argument("--area-per-class", type=parse_area_list, default={},
                    help="Области по классам: 'cat=0.1,0.1,0.5,0.5;dog=...'")
    ap.add_argument("--objects-per-image", type=int, default=3,
                    help="Макс. количество объектов на изображение")
    ap.add_argument("--weights", type=parse_weights, default={},
                    help="Коэффициенты частоты классов: 'cat=3;dog=1;tree=2'. "
                         "Чем больше — тем чаще класс выбирается для генерации. "
                         "Применяется и к background-items.")
    ap.add_argument("--max-angle", type=float, default=0.0,
                    help="Общий максимальный угол поворота (град). "
                         "Случайный угол в [-max, max].")
    ap.add_argument("--max-angle-per-class", type=parse_angles, default={},
                    help="Максимальный угол по классам: 'cat=15;tree=180'. "
                         "Переопределяет --max-angle.")
    ap.add_argument("--scale", type=parse_scale, default=(1.0, 1.0),
                    help="Общий диапазон масштаба: 'min,max' (например 0.5,1.5). "
                         "По умолчанию 1.0,1.0 (без изменений).")
    ap.add_argument("--scale-per-class", type=parse_scale_list, default={},
                    help="Диапазон масштаба по классам: "
                         "'cat=0.7,1.3;dog=0.5,1.0'. Переопределяет --scale.")
    ap.add_argument("--background-items", type=parse_csv_list, default=[],
                    help="Классы, которые размещаются ПЕРВЫМИ на фоне "
                         "(например: tree,grass). Через запятую. "
                         "Остальные объекты рисуются ПОСЛЕ и могут "
                         "перекрывать их сверху. Эти классы МОГУТ выбираться "
                         "случайно (с учётом весов), но всегда кладутся "
                         "в нижний слой.")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = args.dataset
    backgrounds_dir = dataset_dir / BACKGROUNDS_DIRNAME
    if not backgrounds_dir.is_dir():
        print(f"[!] Нет папки backgrounds: {backgrounds_dir}")
        sys.exit(1)

    # 1. Классы
    dataset_classes = load_dataset_classes(dataset_dir)
    print(f"Классы в dataset: {dataset_classes}")

    yaml_data = load_yaml(args.yaml)
    yaml_names = parse_yaml_names(yaml_data)
    if not yaml_names:
        print("[!] В yaml нет names. Будут размечены все классы dataset.")
        yaml_names = dataset_classes

    name_to_id = {n: i for i, n in enumerate(yaml_names)}
    print(f"Классы для разметки (из yaml): {yaml_names}")

    # 2. Объекты
    print("Загрузка объектов...")
    objects = load_objects(dataset_dir, dataset_classes)

    available = [c for c in dataset_classes if objects.get(c)]
    missing = [c for c in dataset_classes if not objects.get(c)]
    if missing:
        print(f"[!] Классы без объектов (будут проигнорированы): {missing}")
    if not available:
        print("[!] Нет ни одного объекта. Выход.")
        sys.exit(1)

    # 3. Background-items (только те, что реально есть в dataset)
    bg_items = [c for c in args.background_items if c in available]
    unknown_bg_items = [c for c in args.background_items if c not in available]
    if unknown_bg_items:
        print(f"[!] background-items не найдены в dataset: {unknown_bg_items}")
    if bg_items:
        print(f"Background-items (нижний слой, веса учитываются): {bg_items}")

    # 4. Веса (теперь для ВСЕХ классов, включая background-items)
    weights = {c: max(0.0, args.weights.get(c, 1.0)) for c in available}
    weighted_all = [c for c in available if weights[c] > 0]
    if not weighted_all:
        print("[!] Все веса нулевые. Использую равномерное распределение.")
        weighted_all = available
        weights = {c: 1.0 for c in available}
    w_list = [weights[c] for c in weighted_all]
    print("Веса классов: " + ", ".join(
        f"{c}={weights[c]:g}" for c in weighted_all))

    # 5. Углы
    def get_max_angle(cls):
        if cls in args.max_angle_per_class:
            return args.max_angle_per_class[cls]
        return args.max_angle

    if args.max_angle_per_class:
        print("Макс. углы по классам: " + ", ".join(
            f"{k}={v:g}" for k, v in args.max_angle_per_class.items()))
    if args.max_angle:
        print(f"Общий макс. угол: {args.max_angle:g}")

    # 5b. Масштабы
    def get_scale_range(cls):
        if cls in args.scale_per_class:
            return args.scale_per_class[cls]
        return args.scale

    if args.scale_per_class:
        print("Масштаб по классам: " + ", ".join(
            f"{k}={v[0]:g}..{v[1]:g}" for k, v in args.scale_per_class.items()))
    if args.scale != (1.0, 1.0):
        print(f"Общий масштаб: {args.scale[0]:g}..{args.scale[1]:g}")

    # 6. Фоны
    bg_files = list_images(backgrounds_dir)
    if not bg_files:
        print(f"[!] В {backgrounds_dir} нет изображений")
        sys.exit(1)
    print(f"Фонов: {len(bg_files)}")

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # 7. Генерация
    for i in range(args.num_images):
        bg_path = random.choice(bg_files)
        bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if bg is None:
            print(f"[!] Не удалось прочитать фон: {bg_path}")
            continue
        H, W = bg.shape[:2]

        labels = []
        n_objects = random.randint(1, max(1, args.objects_per_image))

        # --- Выбор классов с учётом весов ---
        # Каждый слот тянется из общего распределения по весам
        # (background-items тоже участвуют).
        chosen_all = []
        for _ in range(n_objects):
            chosen_all.append(weighted_choice(weighted_all, w_list))

        # Разделяем на слои:
        # - bg_layer: классы из --background-items (кладутся первыми, снизу)
        # - fg_layer: все остальные (кладутся вторыми, сверху)
        bg_layer = [c for c in chosen_all if c in bg_items]
        fg_layer = [c for c in chosen_all if c not in bg_items]

        # Внутри слоёв перемешиваем, чтобы порядок наложения был случайным
        random.shuffle(bg_layer)
        random.shuffle(fg_layer)

        # Итоговый порядок вставки: сначала нижний слой, потом верхний
        chosen = bg_layer + fg_layer

        for cls in chosen:
            img_c, mask_c = random.choice(objects[cls])

            # Масштаб
            lo, hi = get_scale_range(cls)
            if (lo, hi) != (1.0, 1.0) and hi > 0:
                scale = random.uniform(lo, hi)
                img_c, mask_c = scale_object(img_c, mask_c, scale)

            # Поворот
            max_ang = get_max_angle(cls)
            if max_ang > 0:
                angle = random.uniform(-max_ang, max_ang)
                img_c, mask_c = rotate_object(img_c, mask_c, angle)
                if img_c is None:
                    continue

            oh, ow = img_c.shape[:2]

            area = args.area_per_class.get(cls, args.area)
            ax1, ay1, ax2, ay2 = area
            area_x1 = int(round(ax1 * W))
            area_y1 = int(round(ay1 * H))
            area_x2 = int(round(ax2 * W))
            area_y2 = int(round(ay2 * H))
            area_w = area_x2 - area_x1
            area_h = area_y2 - area_y1
            if area_w <= 0 or area_h <= 0:
                continue

            max_dx = max(0, area_w - ow)
            max_dy = max(0, area_h - oh)
            x = area_x1 + (random.randint(0, max_dx) if max_dx > 0 else 0)
            y = area_y1 + (random.randint(0, max_dy) if max_dy > 0 else 0)

            visible = paste_object(bg, img_c, mask_c, x, y)
            if visible is None:
                continue

            # Разметка: только если класс есть в yaml
            if cls not in name_to_id:
                continue

            vx1, vy1, vx2, vy2 = visible
            bx1 = max(vx1, area_x1)
            by1 = max(vy1, area_y1)
            bx2 = min(vx2, area_x2)
            by2 = min(vy2, area_y2)
            if bx2 <= bx1 or by2 <= by1:
                continue
            bx = clip_xyxy((bx1, by1, bx2, by2), W, H)
            if bx is None:
                continue

            cx, cy, bw, bh = to_yolo(bx, W, H)
            labels.append(f"{name_to_id[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        base = f"gen_cards0_{i:03d}"
        img_path = out_dir / f"{base}.png"
        lbl_path = out_dir / f"{base}.txt"

        cv2.imwrite(str(img_path), bg)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))
        generated.append((img_path, lbl_path))

    # 8. train/val
    random.shuffle(generated)
    n_val = int(len(generated) * args.val_split)
    val_set = generated[:n_val]
    train_set = generated[n_val:]

    with open(out_dir / "train.txt", "w", encoding="utf-8") as f:
        for p, _ in train_set:
            f.write(str(p.resolve()) + "\n")
    with open(out_dir / "val.txt", "w", encoding="utf-8") as f:
        for p, _ in val_set:
            f.write(str(p.resolve()) + "\n")

    with open(out_dir / "yolo_annotations.txt", "w", encoding="utf-8") as f:
        for p, lp in generated:
            f.write(f"{p.resolve()} {lp.resolve()}\n")

    final_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(yaml_names),
        "names": yaml_names,
    }
    with open(out_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(final_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"Готово: {len(generated)} изображений → {out_dir}")
    print(f"train: {len(train_set)}, val: {len(val_set)}")


if __name__ == "__main__":
    main()