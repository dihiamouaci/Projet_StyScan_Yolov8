import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ultralytics import YOLO
from config import DATASET_YAML


def main():
    print("Chargement YOLOv8n...")

    model = YOLO("yolov8n.pt")

    print("Début training...")

    model.train(
        data=str(DATASET_YAML),
        epochs=5,
        imgsz=416,
        batch=2,
        workers=0,
        device="cpu",
        name="stylescan_df2",
        save=True
    )

    print("Training terminé.")
    print("Export ONNX...")

    model.export(format="onnx")

    print("Export terminé.")
    print("Modèle disponible dans : runs/detect/stylescan_df2/weights/")


if __name__ == "__main__":
    main()
