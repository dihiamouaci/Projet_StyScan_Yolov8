import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

from api.predictor import predict

app = FastAPI(title="StyleScan DeepFashion2 API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = predict(img)

    return {
        "filename": file.filename,
        "detections": detections
    }
