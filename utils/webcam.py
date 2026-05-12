import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import cv2
from api.predictor import predict


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam introuvable")
        return

    prev = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Impossible de lire la webcam")
            break

        detections = predict(frame)

        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        for item in detections:
            x1, y1, x2, y2 = item["bbox"]

            label = f'{item["category"]} | {item["color"]} | {item["confidence"]}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("StyleScan DeepFashion2", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
