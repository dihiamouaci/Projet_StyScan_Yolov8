import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ultralytics import YOLO
from utils.color_extractor import extract_color

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "stylescan_df2-3" / "weights" / "best.pt"

model = None


def get_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {MODEL_PATH}. Lancez d'abord scripts/train.py"
            )

        model = YOLO(str(MODEL_PATH))

    return model


def predict(frame):
    model = get_model()
    results = model(frame)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]
            color = extract_color(crop)

            detections.append({
                "category": model.names[cls],
                "color": color,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

    return detections
