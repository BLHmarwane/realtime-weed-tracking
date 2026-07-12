"""Tests de robustesse du pipeline vidéo."""

import sys

import pytest

from weedtrack import pipeline


class FakeCapture:
    def __init__(self, *, opened=True, fps=25.0, total_frames=0):
        self.opened = opened
        self.fps = fps
        self.total_frames = total_frames
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        if property_id == FakeCv2.CAP_PROP_FPS:
            return self.fps
        if property_id == FakeCv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        raise AssertionError(f"Unexpected capture property: {property_id}")

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, *, opened=True):
        self.opened = opened
        self.released = False
        self.frames = []

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class FakeCv2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7

    def __init__(self, *, capture=None, writer=None):
        self.capture = capture
        self.writer = writer
        self.capture_sources = []
        self.writer_calls = []
        self.fourcc_calls = []

    def VideoCapture(self, source):
        self.capture_sources.append(source)
        return self.capture

    def VideoWriter_fourcc(self, *codec):
        self.fourcc_calls.append(codec)
        return codec

    def VideoWriter(self, output_path, fourcc, fps, size):
        self.writer_calls.append((output_path, fourcc, fps, size))
        return self.writer


class FakeFrame:
    shape = (480, 640, 3)


class FakeBoxes:
    xyxy = []
    cls = []
    conf = []
    id = None

    def __len__(self):
        return 0


class FakeResult:
    boxes = FakeBoxes()
    names = {}
    orig_img = FakeFrame()


def test_video_metadata_rejects_unreadable_source_and_releases():
    capture = FakeCapture(opened=False)
    cv2_module = FakeCv2(capture=capture)
    with pytest.raises(pipeline.VideoPipelineError, match="Cannot open source video"):
        pipeline._video_metadata(cv2_module, "missing.mp4")
    assert capture.released is True
    assert cv2_module.capture_sources == ["missing.mp4"]


def test_open_video_writer_rejects_failure_and_releases():
    writer = FakeWriter(opened=False)
    cv2_module = FakeCv2(writer=writer)
    with pytest.raises(pipeline.VideoPipelineError, match="Cannot open output video"):
        pipeline._open_video_writer(cv2_module, "out.mp4", 25.0, 640, 480)
    assert writer.released is True
    assert cv2_module.fourcc_calls == [("m", "p", "4", "v")]
    assert cv2_module.writer_calls == [
        ("out.mp4", ("m", "p", "4", "v"), 25.0, (640, 480))
    ]


def test_run_on_video_uses_full_run_timing_and_releases_writer(monkeypatch):
    capture = FakeCapture(fps=30.0, total_frames=2)
    writer = FakeWriter()
    cv2_module = FakeCv2(capture=capture, writer=writer)

    def two_results(*args, **kwargs):
        yield FakeResult()
        yield FakeResult()

    timestamps = iter([10.0, 12.0])
    monkeypatch.setitem(sys.modules, "cv2", cv2_module)
    monkeypatch.setattr(pipeline, "iter_tracked_results", two_results)
    monkeypatch.setattr(pipeline, "annotate_frame", lambda *args: FakeFrame())
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: next(timestamps))

    summary = pipeline.run_on_video("in.mp4", "out.mp4", "weights.pt")

    assert summary["frames"] == 2
    assert summary["pipeline_fps"] == 1.0
    assert capture.released is True
    assert writer.released is True
    assert cv2_module.capture_sources == ["in.mp4"]
    assert cv2_module.writer_calls == [
        ("out.mp4", ("m", "p", "4", "v"), 30.0, (640, 480))
    ]


def test_run_on_video_releases_writer_when_tracking_fails(monkeypatch):
    capture = FakeCapture(fps=30.0, total_frames=2)
    writer = FakeWriter()
    cv2_module = FakeCv2(capture=capture, writer=writer)

    def failing_results(*args, **kwargs):
        yield FakeResult()
        raise RuntimeError("tracking failed")

    monkeypatch.setitem(sys.modules, "cv2", cv2_module)
    monkeypatch.setattr(pipeline, "iter_tracked_results", failing_results)
    monkeypatch.setattr(pipeline, "annotate_frame", lambda *args: FakeFrame())

    with pytest.raises(RuntimeError, match="tracking failed"):
        pipeline.run_on_video("in.mp4", "out.mp4", "weights.pt")

    assert writer.released is True
