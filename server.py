#!/usr/bin/env python3

import os
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO


app = FastAPI()

MODEL_PATH = os.getenv("VISION_MODEL", "yolov8n.pt")
CAMERA_URL = os.getenv(
    "VISION_CAMERA_URL",
    "http://192.168.68.124:8091/camera/latest.jpg",
)
CONFIDENCE_THRESHOLD = float(
    os.getenv("VISION_CONFIDENCE", "0.40")
)
POLL_INTERVAL = float(
    os.getenv("VISION_POLL_INTERVAL", "0.20")
)

model = YOLO(MODEL_PATH)

latest_frame = None
latest_detections = []
latest_description = "No frame processed yet."
latest_timestamp = None
camera_running = False
last_error = None

lock = threading.Lock()
shutdown_event = threading.Event()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def run_yolo(frame):
    height, width = frame.shape[:2]
    results = model(frame, verbose=False)

    detections = []
    labels = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = str(result.names[cls_id])
            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = max(0, int(x2 - x1))
            box_height = max(0, int(y2 - y1))

            detections.append(
                {
                    "label": label,
                    "confidence": round(confidence, 3),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "width": box_width,
                    "height": box_height,
                    "center_x": int((x1 + x2) / 2),
                    "center_y": int((y1 + y2) / 2),
                    "area": box_width * box_height,
                    "image_width": width,
                    "image_height": height,
                }
            )

            labels.append(label)

    unique_objects = sorted(set(labels))

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
    global last_error

    session = requests.Session()

    while not shutdown_event.is_set():
        try:
            response = session.get(
                CAMERA_URL,
                timeout=5,
                headers={"Cache-Control": "no-cache"},
            )
            response.raise_for_status()

            image_data = np.frombuffer(
                response.content,
                dtype=np.uint8,
            )

            frame = cv2.imdecode(
                image_data,
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                raise RuntimeError(
                    "Camera relay returned an invalid JPEG."
                )

            detections, objects, description = run_yolo(frame)
            timestamp = now_iso()

            with lock:
                latest_frame = frame.copy()
                latest_detections = detections
                latest_description = description
                latest_timestamp = timestamp
                camera_running = True
                last_error = None

        except Exception as exc:
            with lock:
                camera_running = False
                last_error = str(exc)

        time.sleep(POLL_INTERVAL)


@app.on_event("startup")
def startup_event():
    shutdown_event.clear()

    thread = threading.Thread(
        target=camera_loop,
        daemon=True,
        name="http-camera-loop",
    )
    thread.start()

    print(f"Vision model: {MODEL_PATH}")
    print(f"Camera URL: {CAMERA_URL}")


@app.on_event("shutdown")
def shutdown_handler():
    shutdown_event.set()


@app.get("/")
def root():
    with lock:
        error = last_error

    return {
        "status": "Vision server running",
        "mode": "http_camera",
        "camera_url": CAMERA_URL,
        "camera_running": camera_running,
        "last_error": error,
        "endpoints": [
            "/detect",
            "/detections/latest",
            "/description",
            "/frame",
        ],
    }


@app.get("/detections/latest")
def detections_latest():
    with lock:
        return {
            "timestamp": latest_timestamp,
            "detections": list(latest_detections),
            "description": latest_description,
            "camera_running": camera_running,
            "camera_url": CAMERA_URL,
            "last_error": last_error,
        }


@app.get("/description")
def description():
    with lock:
        return {
            "timestamp": latest_timestamp,
            "description": latest_description,
            "camera_running": camera_running,
        }


@app.get("/frame")
def frame():
    def generate():
        while not shutdown_event.is_set():
            with lock:
                frame_copy = (
                    latest_frame.copy()
                    if latest_frame is not None
                    else None
                )

            if frame_copy is None:
                time.sleep(0.20)
                continue

            ok, buffer = cv2.imencode(".jpg", frame_copy)

            if not ok:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

            time.sleep(0.10)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image_data = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_data,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        return {
            "detections": [],
            "objects": [],
            "description": "No valid image received.",
        }

    detections, objects, description = run_yolo(frame)

    return {
        "detections": detections,
        "objects": objects,
        "description": description,
    }
