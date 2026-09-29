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
    names = {
        0: "person", 24: "backpack", 25: "umbrella", 56: "chair",
        62: "tv", 77: "teddy bear",
    }

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


def run_with_models(general, tracker, continuity_tracker=None):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    continuity_tracker = continuity_tracker or FakeModel([])
    with patch.object(server, "model", general), patch.object(
        server, "person_tracker_model", tracker
    ), patch.object(
        server, "marvin_continuity_tracker_model", continuity_tracker
    ):
        return server.run_yolo(
            frame,
            return_candidates=True,
            return_proposals=True,
        )


def continuity_metadata(tracker_id):
    return {
        "tracker_id": tracker_id,
        "tracker_source": "marvin_continuity_botsort",
        "tracker_generation": server.MARVIN_CONTINUITY_TRACKER_GENERATION,
    }


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
    detections, labels, _, _candidates, _proposals = run_with_models(
        general, umbrella_tracker,
    )
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
    detections, _, _, _candidates, _proposals = run_with_models(
        general, person_tracker,
    )
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

    second_detections, _, _, _candidates, _proposals = run_with_models(
        general, person_tracker,
    )
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
    detections, _, _, _candidates, _proposals = run_with_models(
        general, unmatched_tracker,
    )
    person = next(item for item in detections if item["label"] == "person")
    assert "track_id" not in person
    print("PASS: unmatched person remains valid without track_id.")

    failing_tracker = FakeModel(error=RuntimeError("tracker unavailable"))
    detections, _, _, _candidates, _proposals = run_with_models(
        general, failing_tracker,
    )
    assert len(detections) == 4
    assert {item["label"] for item in detections} == {"backpack", "chair", "tv", "person"}
    print("PASS: tracker failure preserves general detections.")

    no_id_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.91, [100, 100, 300, 450])])
    ])
    detections, _, _, _candidates, _proposals = run_with_models(
        general, no_id_tracker,
    )
    assert "track_id" not in next(item for item in detections if item["label"] == "person")
    print("PASS: detections without tracker IDs remain valid.")

    low_confidence_person = FakeModel([
        FakeResult([FakeBox(0, 0.03, [100, 100, 300, 450])])
    ])
    low_confidence_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.91, [100, 100, 300, 450], track_id=73)])
    ])
    _detections, _, _, candidates, proposals = run_with_models(
        low_confidence_person, low_confidence_tracker,
    )
    assert candidates == []
    assert len(proposals) == 1
    assert proposals[0]["label"] == "person"
    assert proposals[0]["confidence"] == 0.03
    assert proposals[0]["track_id"] == 73
    print("PASS: low-confidence proposals preserve existing tracker IDs.")

    no_id_proposal_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.91, [100, 100, 300, 450])])
    ])
    _detections, _, _, candidates, proposals = run_with_models(
        low_confidence_person, no_id_proposal_tracker,
    )
    assert candidates == []
    assert len(proposals) == 1
    assert "track_id" not in proposals[0]
    print("PASS: proposal tracker IDs are never fabricated.")

    continuity_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.03, [100, 100, 300, 450], track_id=84)])
    ])
    _detections, _, _, candidates, proposals = run_with_models(
        low_confidence_person,
        FakeModel([]),
        continuity_tracker,
    )
    assert candidates == []
    assert proposals[0]["marvin_continuity"] == continuity_metadata(84)
    assert not {"identity_id", "entity_id", "target_lock"} & set(proposals[0])
    assert continuity_tracker.track_calls[0]["conf"] == server.PROPOSAL_CONFIDENCE
    assert continuity_tracker.track_calls[0]["tracker"] == server.MARVIN_CONTINUITY_TRACKER_CONFIG
    assert "classes" not in continuity_tracker.track_calls[0]
    _detections, _, _, _, next_proposals = run_with_models(
        low_confidence_person,
        FakeModel([]),
        continuity_tracker,
    )
    assert next_proposals[0]["marvin_continuity"] == continuity_metadata(84)
    print("PASS: diagnostic continuity metadata is isolated from identity.")

    low_confidence_teddy_bear = FakeModel([
        FakeResult([FakeBox(77, 0.03, [100, 100, 300, 450])])
    ])
    teddy_continuity_tracker = FakeModel([
        FakeResult([FakeBox(77, 0.03, [100, 100, 300, 450], track_id=86)])
    ])
    _detections, _, _, candidates, proposals = run_with_models(
        low_confidence_teddy_bear,
        FakeModel([]),
        teddy_continuity_tracker,
    )
    assert candidates == []
    assert proposals[0]["label"] == "teddy bear"
    assert proposals[0]["marvin_continuity"] == continuity_metadata(86)
    assert not {
        "identity_id", "entity_id", "target_lock", "world_model",
        "controller", "execution",
    } & set(proposals[0])
    print("PASS: teddy bear proposals receive diagnostic continuity metadata.")

    mixed_low_confidence = FakeModel([
        FakeResult([
            FakeBox(0, 0.03, [100, 100, 300, 450]),
            FakeBox(77, 0.03, [320, 100, 520, 450]),
        ])
    ])
    mixed_continuity_tracker = FakeModel([
        FakeResult([
            FakeBox(0, 0.03, [100, 100, 300, 450], track_id=87),
            FakeBox(77, 0.03, [320, 100, 520, 450], track_id=88),
        ])
    ])
    _detections, _, _, candidates, proposals = run_with_models(
        mixed_low_confidence,
        FakeModel([]),
        mixed_continuity_tracker,
    )
    assert candidates == []
    assert {
        proposal["label"]: proposal["marvin_continuity"]["tracker_id"]
        for proposal in proposals
    } == {"person": 87, "teddy bear": 88}
    assert {
        proposal["marvin_continuity"]["tracker_generation"]
        for proposal in proposals
    } == {server.MARVIN_CONTINUITY_TRACKER_GENERATION}
    print("PASS: different proposal classes retain independent continuity IDs.")

    overlapping_cross_label_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.03, [100, 100, 300, 450], track_id=89)])
    ])
    _detections, _, _, candidates, proposals = run_with_models(
        low_confidence_teddy_bear,
        FakeModel([]),
        overlapping_cross_label_tracker,
    )
    assert candidates == []
    assert "marvin_continuity" not in proposals[0]
    print("PASS: overlapping cross-label boxes do not receive continuity metadata.")

    no_id_continuity_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.03, [100, 100, 300, 450])])
    ])
    _detections, _, _, candidates, proposals = run_with_models(
        low_confidence_person,
        FakeModel([]),
        no_id_continuity_tracker,
    )
    assert candidates == []
    assert "marvin_continuity" not in proposals[0]
    print("PASS: missing continuity tracker IDs fail closed.")

    unmatched_continuity_tracker = FakeModel([
        FakeResult([FakeBox(0, 0.03, [400, 100, 600, 450], track_id=85)])
    ])
    _detections, _, _, candidates, proposals = run_with_models(
        low_confidence_person,
        FakeModel([]),
        unmatched_continuity_tracker,
    )
    assert candidates == []
    assert "marvin_continuity" not in proposals[0]
    print("PASS: unmatched continuity tracker metadata fails closed.")

    first_generation = server._new_marvin_continuity_generation()
    second_generation = server._new_marvin_continuity_generation()
    assert isinstance(first_generation, str) and first_generation
    assert isinstance(second_generation, str) and second_generation
    assert first_generation != second_generation
    print("PASS: new continuity tracker generations are opaque and distinct.")
    print("\nPersistent person-tracking and multi-object test passed.")


if __name__ == "__main__":
    main()
