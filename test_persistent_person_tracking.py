#!/usr/bin/env python3

from unittest.mock import patch

import numpy as np

import server


class FakeVector:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def tolist(self):
        return list(self.values)


class FakeBox:
    def __init__(self, cls_id, confidence, coordinates, track_id=None):
        self.cls = FakeVector([cls_id])
        self.conf = FakeVector([confidence])
        self.xyxy = [FakeVector(coordinates)]
        self.id = FakeVector([track_id]) if track_id is not None else None


class FakeResult:
    names = {0: "person", 24: "backpack", 25: "umbrella", 56: "chair", 62: "tv"}

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.track_calls = []
        self.predict_calls = []

    def __call__(self, frame, **kwargs):
        self.predict_calls.append(kwargs)
        return self.results

    def track(self, frame, **kwargs):
        self.track_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results


def run_with_models(general, tracker):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch.object(server, "model", general), patch.object(
        server, "person_tracker_model", tracker
    ):
        return server.run_yolo(frame)


def main():
    general = FakeModel([
        FakeResult([
            FakeBox(24, 0.91, [40, 40, 180, 200]),
            FakeBox(56, 0.88, [200, 50, 350, 260]),
            FakeBox(62, 0.82, [360, 60, 600, 250]),
            FakeBox(0, 0.93, [100, 100, 300, 450]),
        ])
    ])
    umbrella_tracker = FakeModel([
        FakeResult([FakeBox(25, 0.99, [0, 0, 640, 480], track_id=7)])
    ])
    detections, labels, _ = run_with_models(general, umbrella_tracker)
    assert {item["label"] for item in detections} == {"backpack", "chair", "tv", "person"}
    assert "track_id" not in next(item for item in detections if item["label"] == "person")
    assert umbrella_tracker.track_calls[0]["persist"] is True
    assert umbrella_tracker.track_calls[0]["tracker"] == server.TRACKER_CONFIG
    assert umbrella_tracker.track_calls[0]["classes"] == [0]
    assert labels == ["backpack", "chair", "person", "tv"]
    print("PASS: all general detections survive a non-person tracker result.")

    person_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.91, [100, 100, 300, 450], track_id=42)])
    ])
    detections, _, _ = run_with_models(general, person_tracker)
    person = next(item for item in detections if item["label"] == "person")
    assert person["track_id"] == 42
    assert all(
        field in person
        for field in (
            "label", "confidence", "x1", "y1", "x2", "y2",
            "center_x", "area", "image_width", "image_height",
            "track_id",
        )
    )
    print("PASS: matched person receives a persistent track_id.")

    second_detections, _, _ = run_with_models(general, person_tracker)
    second_person = next(
        item for item in second_detections if item["label"] == "person"
    )
    assert second_person["track_id"] == person["track_id"]
    assert len(person_tracker.track_calls) == 2
    assert all(call["persist"] is True for call in person_tracker.track_calls)
    print("PASS: sequential tracked frames preserve the person track_id.")

    unmatched_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.91, [500, 100, 620, 300], track_id=99)])
    ])
    detections, _, _ = run_with_models(general, unmatched_tracker)
    person = next(item for item in detections if item["label"] == "person")
    assert "track_id" not in person
    print("PASS: unmatched person remains valid without track_id.")

    failing_tracker = FakeModel(error=RuntimeError("tracker unavailable"))
    detections, _, _ = run_with_models(general, failing_tracker)
    assert len(detections) == 4
    assert {item["label"] for item in detections} == {"backpack", "chair", "tv", "person"}
    print("PASS: tracker failure preserves general detections.")

    no_id_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.91, [100, 100, 300, 450])])
    ])
    detections, _, _ = run_with_models(general, no_id_tracker)
    assert "track_id" not in next(item for item in detections if item["label"] == "person")
    print("PASS: detections without tracker IDs remain valid.")
    print("\nPersistent person-tracking and multi-object test passed.")


if __name__ == "__main__":
    main()
