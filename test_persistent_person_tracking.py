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
    def __init__(self, track_id):
        self.cls = FakeVector([0])
        self.conf = FakeVector([0.91])
        self.xyxy = [
            FakeVector([100, 80, 260, 430])
        ]

        self.id = (
            FakeVector([track_id])
            if track_id is not None
            else None
        )


class FakeResult:
    names = {0: "person"}

    def __init__(self, track_id):
        self.boxes = [FakeBox(track_id)]


class FakeModel:
    def __init__(self, track_id):
        self.track_id = track_id
        self.track_calls = []
        self.predict_calls = []

    def __call__(self, frame, **kwargs):
        self.predict_calls.append(kwargs)
        return [FakeResult(self.track_id)]

    def track(self, frame, **kwargs):
        self.track_calls.append(kwargs)
        return [FakeResult(self.track_id)]


def main():
    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    tracked_model = FakeModel(track_id=42)

    with patch.object(
        server,
        "model",
        tracked_model,
    ):
        detections, _, _ = server.run_yolo(
            frame
        )

    assert tracked_model.track_calls, (
        "run_yolo did not invoke model.track()."
    )

    assert not tracked_model.predict_calls, (
        "run_yolo still invoked ordinary prediction."
    )

    call = tracked_model.track_calls[0]

    assert call.get("persist") is True, (
        "ByteTrack state was not persisted across frames."
    )

    assert call.get("tracker") == server.TRACKER_CONFIG, (
        "run_yolo did not select the configured tracker."
    )

    assert detections[0]["track_id"] == 42, (
        "Ultralytics track ID was not added to the detection."
    )

    print("PASS: run_yolo invokes persistent person tracking.")
    print("PASS: tracked detection includes track_id.")

    untracked_model = FakeModel(track_id=None)

    with patch.object(
        server,
        "model",
        untracked_model,
    ):
        detections, _, _ = server.run_yolo(
            frame
        )

    assert "track_id" not in detections[0], (
        "Detection without a tracker ID received an invalid track_id."
    )

    print(
        "PASS: detections without tracker IDs remain valid."
    )
    print()
    print("Persistent person-tracking test passed.")


if __name__ == "__main__":
    main()
