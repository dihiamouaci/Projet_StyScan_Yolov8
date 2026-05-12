import sys
import ast
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import RAW_DIR, PROCESSED_DIR, DATASET_YAML

DATASET_DIR = RAW_DIR / "deepfashion2"

TRAIN_CSV = DATASET_DIR / "input" / "train.csv"
VAL_CSV = DATASET_DIR / "input" / "test.csv"

TRAIN_IMG_DIR = DATASET_DIR / "train"
VAL_IMG_DIR = DATASET_DIR / "test"

MAX_IMAGES_PER_CLASS = 150

CLASS_MAP = {
    "short sleeve top": "Short Sleeve Top",
    "long sleeve top": "Long Sleeve Top",
    "short sleeve outwear": "Short Sleeve Outwear",
    "long sleeve outwear": "Long Sleeve Outwear",
    "vest": "Vest",
    "sling": "Sling",
    "shorts": "Shorts",
    "trousers": "Trousers",
    "skirt": "Skirt",
    "short sleeve dress": "Short Sleeve Dress",
    "long sleeve dress": "Long Sleeve Dress",
    "vest dress": "Vest Dress",
    "sling dress": "Sling Dress",
}

YOLO_CLASSES = [
    "Short Sleeve Top",
    "Long Sleeve Top",
    "Short Sleeve Outwear",
    "Long Sleeve Outwear",
    "Vest",
    "Sling",
    "Shorts",
    "Trousers",
    "Skirt",
    "Short Sleeve Dress",
    "Long Sleeve Dress",
    "Vest Dress",
    "Sling Dress",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(YOLO_CLASSES)}



def reset_processed():
    for split in ["train", "val"]:

        images_dir = PROCESSED_DIR / split / "images"
        labels_dir = PROCESSED_DIR / split / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for folder in [images_dir, labels_dir]:

            for file in folder.glob("*"):

                try:
                    file.unlink()

                except Exception as e:
                    print(f"Impossible de supprimer {file}: {e}")
                    
def parse_bbox(value):
    if isinstance(value, list):
        return value

    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def bbox_to_yolo(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox

    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return x_center, y_center, width, height


def image_name_from_path(path_value):
    return Path(str(path_value)).name


def process_csv(csv_path, img_dir, split):
    print(f"Traitement {split} depuis {csv_path}")

    if not csv_path.exists():
        print(f"CSV introuvable : {csv_path}")
        return

    if not img_dir.exists():
        print(f"Dossier images introuvable : {img_dir}")
        return

    df = pd.read_csv(csv_path)

    counters = {cls: 0 for cls in YOLO_CLASSES}
    grouped_labels = {}
    copied_images = set()

    for _, row in df.iterrows():
        category_name = str(row["category_name"]).strip()

        simple_class = CLASS_MAP.get(category_name)

        if simple_class not in YOLO_CLASSES:
            continue

        if counters[simple_class] >= MAX_IMAGES_PER_CLASS:
            continue

        bbox = parse_bbox(row["b_box"])

        if bbox is None or len(bbox) != 4:
            continue

        img_width = int(row["img_width"])
        img_height = int(row["img_height"])

        x, y, w, h = bbox_to_yolo(bbox, img_width, img_height)

        if w <= 0 or h <= 0:
            continue

        image_name = image_name_from_path(row["path"])

        src_img = img_dir / image_name

        if not src_img.exists():
            continue

        dst_img = PROCESSED_DIR / split / "images" / image_name

        dst_label = (
            PROCESSED_DIR
            / split
            / "labels"
            / image_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        class_id = CLASS_TO_ID[simple_class]

        label_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

        grouped_labels.setdefault(dst_label, []).append(label_line)

        if dst_img not in copied_images:
            shutil.copy2(src_img, dst_img)
            copied_images.add(dst_img)

        counters[simple_class] += 1

    for label_path, lines in grouped_labels.items():
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"{split}: {len(copied_images)} images copiées")
    print(counters)


def write_dataset_yaml():
    content = f"""path: {PROCESSED_DIR.as_posix()}

train: train/images
val: val/images

names:
"""

    for idx, name in enumerate(YOLO_CLASSES):
        content += f"  {idx}: {name}\n"

    DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)

    with open(DATASET_YAML, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"dataset.yaml créé : {DATASET_YAML}")


def main():
    print("Préparation DeepFashion2 CSV vers YOLO...")

    reset_processed()

    # On utilise train.csv pour créer train + val
    process_csv(TRAIN_CSV, TRAIN_IMG_DIR, "train")
    process_csv(TRAIN_CSV, TRAIN_IMG_DIR, "val")

    write_dataset_yaml()

    print("Préparation terminée.")
    print("Next: python scripts/train.py")


if __name__ == "__main__":
    main()