import numpy as np

import server


def test_source_stamp_parser_fails_closed_on_missing_or_invalid_headers():
    assert server.source_frame_stamp_ns_from_headers({
        "X-Mayday-Source-Stamp-Sec": "12",
        "X-Mayday-Source-Stamp-Nanosec": "34",
    }) == 12_000_000_034
    assert server.source_frame_stamp_ns_from_headers({}) is None
    assert server.source_frame_stamp_ns_from_headers({
        "X-Mayday-Source-Stamp-Sec": "12",
        "X-Mayday-Source-Stamp-Nanosec": "1000000000",
    }) is None


def test_camera_loop_binds_each_inference_to_its_response_stamp(monkeypatch):
    responses = [
        type("Response", (), {
            "headers": {
                "X-Mayday-Source-Stamp-Sec": "1",
                "X-Mayday-Source-Stamp-Nanosec": "10",
            },
            "content": b"A-frame",
            "raise_for_status": lambda self: None,
        })(),
        type("Response", (), {
            "headers": {
                "X-Mayday-Source-Stamp-Sec": "2",
                "X-Mayday-Source-Stamp-Nanosec": "20",
            },
            "content": b"B-frame",
            "raise_for_status": lambda self: None,
        })(),
    ]

    class Session:
        def get(self, *_args, **_kwargs):
            return responses.pop(0)

    monkeypatch.setattr(server.requests, "Session", Session)
    monkeypatch.setattr(
        server.cv2,
        "imdecode",
        lambda data, _mode: np.full(
            (2, 3, 3), int(data[0]), dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(
        server,
        "run_yolo",
        lambda frame, **_kwargs: (
            [{"frame_marker": int(frame[0, 0, 0])}],
            [],
            "description",
            [],
            [],
        ),
    )
    timestamps = iter(["inference-a", "inference-b"])
    monkeypatch.setattr(server, "now_iso", lambda: next(timestamps))

    published = []
    original_publish = server._publish_inference_result

    def capture_publish(*args):
        published.append((args[1][0]["frame_marker"], args[-1]))
        original_publish(*args)

    monkeypatch.setattr(server, "_publish_inference_result", capture_publish)
    sleep_count = 0

    def stop_after_two_frames(_seconds):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            server.shutdown_event.set()

    monkeypatch.setattr(server.time, "sleep", stop_after_two_frames)
    server.shutdown_event.clear()
    try:
        server.camera_loop()
    finally:
        server.shutdown_event.clear()

    assert published == [
        (ord("A"), 1_000_000_010),
        (ord("B"), 2_000_000_020),
    ]
    latest = server.detections_latest()
    assert latest["timestamp"] == "inference-b"
    assert latest["source_frame_stamp_ns"] == 2_000_000_020
    assert latest["detections"] == [{"frame_marker": ord("B")}]


def test_candidate_endpoint_exposes_same_published_source_stamp():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    server._publish_inference_result(
        frame,
        [],
        [],
        [{"label": "chair"}],
        "description",
        "inference-time",
        7_000_000_008,
    )
    payload = server.detections_candidates_latest()
    assert payload["timestamp"] == "inference-time"
    assert payload["source_frame_stamp_ns"] == 7_000_000_008
    assert payload["detections"] == [{"label": "chair"}]
