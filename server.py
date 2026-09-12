#!/usr/bin/env python3

import os
import logging
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
CANDIDATE_CONFIDENCE = float(
    os.getenv("VISION_CANDIDATE_CONFIDENCE", "0.05")
)
POLL_INTERVAL = float(
    os.getenv("VISION_POLL_INTERVAL", "0.20")
)
TRACKER_CONFIG = os.getenv(
    "VISION_TRACKER",
    os.path.join(
        os.path.dirname(__file__),
        "botsort_reid.yaml",
    ),
)
PERSON_MATCH_IOU_THRESHOLD = 0.50


def _validate_confidence_configuration():
    if not (
        0.0 < CANDIDATE_CONFIDENCE
        <= CONFIDENCE_THRESHOLD
        <= 1.0
    ):
        raise ValueError(
            "VISION_CANDIDATE_CONFIDENCE must satisfy "
            "0.0 < candidate <= VISION_CONFIDENCE <= 1.0 "
            f"(got candidate={CANDIDATE_CONFIDENCE!r}, "
            f"publish={CONFIDENCE_THRESHOLD!r})."
        )


_validate_confidence_configuration()

model = YOLO(MODEL_PATH)
person_tracker_model = YOLO(MODEL_PATH)

logger = logging.getLogger(__name__)

latest_frame = None
latest_detections = []
latest_candidate_detections = []
latest_description = "No frame processed yet."
latest_timestamp = None
camera_running = False
last_error = None

lock = threading.Lock()
shutdown_event = threading.Event()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _detection_from_box(result, box, width, height):
    cls_id = int(box.cls[0])
    label = str(result.names[cls_id])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    box_width = max(0, int(x2 - x1))
    box_height = max(0, int(y2 - y1))

    return {
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


def _box_iou(left, right):
    x1 = max(left["x1"], right["x1"])
    y1 = max(left["y1"], right["y1"])
    x2 = min(left["x2"], right["x2"])
    y2 = min(left["y2"], right["y2"])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    union = left["area"] + right["area"] - intersection
    return intersection / union if union else 0.0


def _merge_person_track_ids(detections, tracker_results, width, height):
    tracker_boxes = []
    for result in tracker_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0 or box.id is None:
                continue
            tracked = _detection_from_box(
                result,
                box,
                width,
                height,
            )
            tracked["track_id"] = int(box.id[0])
            tracker_boxes.append(tracked)

    candidates = [
        (index, detection)
        for index, detection in enumerate(detections)
        if detection["label"] == "person"
    ]
    matches = []
    for detection_index, detection in candidates:
        for tracker_index, tracked in enumerate(tracker_boxes):
            matches.append(
                (_box_iou(detection, tracked), detection_index, tracker_index)
            )

    used_detections = set()
    used_trackers = set()
    for iou, detection_index, tracker_index in sorted(
        matches,
        key=lambda item: (-item[0], item[1], item[2]),
    ):
        if (
            iou < PERSON_MATCH_IOU_THRESHOLD
            or detection_index in used_detections
            or tracker_index in used_trackers
        ):
            continue
        detections[detection_index]["track_id"] = tracker_boxes[tracker_index][
            "track_id"
        ]
        used_detections.add(detection_index)
        used_trackers.add(tracker_index)


def run_yolo(frame, return_candidates=False):
    height, width = frame.shape[:2]
    general_results = model(
        frame,
        conf=CANDIDATE_CONFIDENCE,
        verbose=False,
    )

    candidate_detections = []
    candidate_confidences = []

    for result in general_results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < CANDIDATE_CONFIDENCE:
                continue
            detection = _detection_from_box(result, box, width, height)
            candidate_detections.append(detection)
            candidate_confidences.append(confidence)

    try:
        tracker_results = person_tracker_model.track(
            frame,
            classes=[0],
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False,
        )
        _merge_person_track_ids(
            candidate_detections,
            tracker_results,
            width,
            height,
        )
    except Exception:
        logger.exception("Person tracking failed; retaining general detections.")

    detections = [
        detection
        for detection, confidence in zip(
            candidate_detections,
            candidate_confidences,
        )
        if confidence >= CONFIDENCE_THRESHOLD
    ]
    labels = [detection["label"] for detection in detections]
    unique_objects = sorted(set(labels))

    if unique_objects:
        description = "I see " + ", ".join(unique_objects) + "."
    else:
        description = "I do not recognize any common objects."

    if return_candidates:
        return (
            detections,
            unique_objects,
            description,
            candidate_detections,
        )
    return detections, unique_objects, description


def camera_loop():
    global latest_frame
    global latest_detections
    global latest_candidate_detections
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

            (
                detections,
                objects,
                description,
                candidate_detections,
            ) = run_yolo(frame, return_candidates=True)
            timestamp = now_iso()

            with lock:
                latest_frame = frame.copy()
                latest_detections = detections
                latest_candidate_detections = candidate_detections
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
            "/detections/target/latest",
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


@app.get("/detections/target/latest")
def detections_target_latest(label: str):
    normalized_label = label.casefold()
    with lock:
        detections = [
            dict(detection)
            for detection in latest_candidate_detections
            if str(detection.get("label", "")).casefold()
            == normalized_label
        ]
        timestamp = latest_timestamp
        running = camera_running
        error = last_error

    detections.sort(
        key=lambda detection: detection.get("confidence", 0.0),
        reverse=True,
    )
    return {
        "timestamp": timestamp,
        "label": label,
        "found": bool(detections),
        "best_detection": detections[0] if detections else None,
        "detections": detections,
        "camera_running": running,
        "camera_url": CAMERA_URL,
        "last_error": error,
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
