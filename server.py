from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime


app = FastAPI()

MODEL_PATH = os.getenv("VISION_MODEL", "yolov8n.pt")
CAMERA_INDEX = int(os.getenv("VISION_CAMERA_INDEX", "0"))
CONFIDENCE_THRESHOLD = float(os.getenv("VISION_CONFIDENCE", "0.40"))

model = YOLO(MODEL_PATH)

latest_frame = None
latest_detections = []
latest_description = "No frame processed yet."
latest_timestamp = None
camera_running = False
lock = threading.Lock()


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def run_yolo(frame):
    height, width = frame.shape[:2]
    results = model(frame, verbose=False)

    detections = []
    labels = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
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

    return detections, unique_objects, description


def camera_loop():
    global latest_frame
    global latest_detections
    global latest_description
    global latest_timestamp
    global camera_running

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        camera_running = False
        print(f"Could not open camera index {CAMERA_INDEX}")
        return

    camera_running = True
    print(f"Vision camera started on index {CAMERA_INDEX}")

    while camera_running:
        ok, frame = cap.read()

        if not ok:
            time.sleep(0.2)
            continue

        detections, objects, description = run_yolo(frame)
        timestamp = now_iso()

        with lock:
            latest_frame = frame.copy()
            latest_detections = detections
            latest_description = description
            latest_timestamp = timestamp

        time.sleep(0.1)

    cap.release()


@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()


@app.get("/")
def root():
    return {
        "status": "Vision server running",
        "mode": "perception_hub",
        "camera_running": camera_running,
        "endpoints": [
            "/detect",
            "/detections/latest",
            "/description",
            "/frame"
        ]
    }


@app.get("/detections/latest")
def detections_latest():
    with lock:
        return {
            "timestamp": latest_timestamp,
            "detections": latest_detections,
            "description": latest_description,
            "camera_running": camera_running
        }


@app.get("/description")
def description():
    with lock:
        return {
            "timestamp": latest_timestamp,
            "description": latest_description
        }


@app.get("/frame")
def frame():
    def generate():
        while True:
            with lock:
                if latest_frame is None:
                    time.sleep(0.2)
                    continue

                frame_copy = latest_frame.copy()

            ok, buffer = cv2.imencode(".jpg", frame_copy)

            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )

            time.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


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

    detections, objects, description = run_yolo(frame)

    return {
        "detections": detections,
        "objects": objects,
        "description": description
    }
