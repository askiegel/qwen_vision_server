from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("yolov8n.pt")


@app.get("/")
def root():
    return {"status": "Vision server running", "endpoint": "/detect"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()

    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "detections": [],
            "objects": [],
            "description": "No valid image received."
        }

    height, width = frame.shape[:2]

    results = model(frame, verbose=False)

    detections = []
    labels = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            confidence = float(box.conf[0])

            if confidence < 0.40:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = {
                "label": label,
                "confidence": round(confidence, 3),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "center_x": int((x1 + x2) / 2),
                "center_y": int((y1 + y2) / 2),
                "image_width": width,
                "image_height": height,
            }

            detections.append(detection)
            labels.append(label)

    unique_objects = sorted(list(set(labels)))

    if unique_objects:
        description = "I see " + ", ".join(unique_objects) + "."
    else:
        description = "I do not recognize any common objects."

    return {
        "detections": detections,
        "objects": unique_objects,
        "description": description
    }
