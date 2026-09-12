#!/usr/bin/env python3

import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np

import server


class Vector:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def tolist(self):
        return list(self.values)


class Box:
    def __init__(self, cls_id, confidence, coords, track_id=None):
        self.cls = Vector([cls_id])
        self.conf = Vector([confidence])
        self.xyxy = [Vector(coords)]
        self.id = Vector([track_id]) if track_id is not None else None


class Result:
    names = {
        0: "person",
        24: "backpack",
        25: "umbrella",
        56: "chair",
    }

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        return self.results

    def track(self, frame, **kwargs):
        if self.error:
            raise self.error
        return self.results


def main():
    general = FakeModel([
        Result([
            Box(25, 0.60, [0, 0, 100, 100]),
            Box(24, 0.09, [190, 160, 490, 478]),
            Box(24, 0.08, [200, 170, 480, 470]),
            Box(24, 0.06, [210, 180, 470, 460]),
            Box(24, 0.04, [220, 190, 460, 450]),
            Box(56, 0.45, [300, 20, 500, 200]),
            Box(0, 0.95, [100, 100, 300, 450]),
        ])
    ])
    tracker = FakeModel([
        Result([Box(0, 0.95, [100, 100, 300, 450], track_id=7)])
    ])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch.object(server, "model", general), patch.object(
        server, "person_tracker_model", tracker
    ):
        published, labels, description, candidates = server.run_yolo(
            frame,
            return_candidates=True,
        )

    assert general.calls[0]["conf"] == server.CANDIDATE_CONFIDENCE
    assert len(candidates) == 6
    assert {item["label"] for item in published} == {
        "umbrella", "chair", "person"
    }
    assert not any(item["label"] == "backpack" for item in published)
    print("PASS: candidate floor is passed and publication remains at 0.40.")

    with server.lock:
        server.latest_candidate_detections = candidates
        server.latest_timestamp = "test-time"
        server.camera_running = True
        server.last_error = None
    response = server.detections_target_latest("BaCkPaCk")
    assert response["found"] is True
    assert [item["confidence"] for item in response["detections"]] == [
        0.09, 0.08, 0.06
    ]
    assert response["best_detection"]["label"] == "backpack"
    assert response["best_detection"]["x1"] == 190
    print("PASS: backpack query returns only matching candidates, sorted.")

    below = FakeModel([Result([Box(24, 0.04, [0, 0, 10, 10])])])
    with patch.object(server, "model", below), patch.object(
        server, "person_tracker_model", FakeModel()
    ):
        _, _, _, below_candidates = server.run_yolo(
            frame,
            return_candidates=True,
        )
    assert below_candidates == []
    print("PASS: detections below the candidate floor are unavailable.")

    failing_tracker = FakeModel(error=RuntimeError("tracker unavailable"))
    with patch.object(server, "model", general), patch.object(
        server, "person_tracker_model", failing_tracker
    ):
        _, _, _, preserved = server.run_yolo(frame, return_candidates=True)
    assert len(preserved) == 6
    print("PASS: tracker failure preserves candidate detections.")

    invalid_env = dict(os.environ)
    invalid_env["VISION_CANDIDATE_CONFIDENCE"] = "0.50"
    invalid_env["VISION_CONFIDENCE"] = "0.40"
    result = subprocess.run(
        [sys.executable, "-c", "import server"],
        cwd=os.path.dirname(__file__),
        env=invalid_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "VISION_CANDIDATE_CONFIDENCE" in result.stderr
    print("PASS: invalid candidate/publication configuration is rejected.")

    print("\nTarget-candidate test passed.")


if __name__ == "__main__":
    main()
