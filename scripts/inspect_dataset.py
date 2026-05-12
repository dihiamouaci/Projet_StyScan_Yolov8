import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import RAW_DIR


def print_tree(path: Path, max_depth=3, level=0):
    if not path.exists():
        print(f"Dossier introuvable : {path}")
        return

    if level > max_depth:
        return

    items = list(path.iterdir())

    for item in items[:50]:
        print("  " * level + f"- {item.name}")
        if item.is_dir():
            print_tree(item, max_depth, level + 1)

    if len(items) > 50:
        print("  " * level + f"... ({len(items) - 50} éléments masqués)")


if __name__ == "__main__":
    print(f"Inspection de : {RAW_DIR}")
    print_tree(RAW_DIR)
