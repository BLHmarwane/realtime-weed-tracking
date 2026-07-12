import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from weedtrack import demo
from weedtrack.demo import process_demo_video, reencode_h264, temporary_upload

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_demo_renders_preview_before_metrics_and_json():
    source = (PROJECT_ROOT / "app" / "streamlit_app.py").read_text()

    progress = source.index('progress.progress(1.0, text="Complete")')
    preview = source.index("if result.browser_preview:")
    metrics = source.index("col1, col2, col3, col4 = st.columns(4)")
    raw_json = source.index('with st.expander("Raw statistics (JSON)")')

    assert progress < preview < metrics < raw_json


def test_streamlit_requirement_supports_max_upload_size():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
    dependency_specs = {line.split("#", 1)[0].strip() for line in requirements}

    assert "streamlit>=1.59,<2" in dependency_specs


def test_temporary_upload_removes_file_after_context():
    with temporary_upload("clip.mov", b"video") as path:
        assert path.read_bytes() == b"video"
        saved_path = path

    assert not saved_path.exists()


def test_temporary_upload_removes_file_when_context_raises():
    with pytest.raises(RuntimeError, match="processing failed"):
        with temporary_upload("clip.mov", b"video") as path:
            saved_path = path
            raise RuntimeError("processing failed")

    assert not saved_path.exists()


def test_process_demo_video_returns_encoded_bytes_after_cleanup(tmp_path):
    seen_output = None

    def runner(**kwargs):
        nonlocal seen_output
        seen_output = Path(kwargs["output_path"])
        seen_output.write_bytes(b"raw")
        return {"frames": 2, "pipeline_fps": 1.0, "unique_tracks": {}}

    def encoder(source, destination):
        destination.write_bytes(b"h264")
        return True

    config = {
        "model": {"conf_threshold": 0.25, "imgsz": 640},
        "tracking": {"tracker": "bytetrack.yaml"},
    }
    result = process_demo_video(
        tmp_path / "in.mp4",
        Path("weights.pt"),
        config,
        runner=runner,
        encoder=encoder,
    )

    assert result.video_bytes == b"h264"
    assert result.browser_preview is True
    assert seen_output is not None and not seen_output.exists()


def test_process_demo_video_returns_raw_bytes_when_encoding_fails(tmp_path):
    seen_output = None

    def runner(**kwargs):
        nonlocal seen_output
        seen_output = Path(kwargs["output_path"])
        seen_output.write_bytes(b"raw")
        return {"frames": 2, "pipeline_fps": 1.0, "unique_tracks": {}}

    def encoder(source, destination):
        return False

    config = {
        "model": {"conf_threshold": 0.25, "imgsz": 640},
        "tracking": {"tracker": "bytetrack.yaml"},
    }
    result = process_demo_video(
        tmp_path / "in.mp4",
        Path("weights.pt"),
        config,
        runner=runner,
        encoder=encoder,
    )

    assert result.video_bytes == b"raw"
    assert result.browser_preview is False
    assert seen_output is not None and not seen_output.exists()


def test_reencode_h264_returns_false_for_operational_failures(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg"),
    )

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(demo.subprocess, "run", fail)

    assert reencode_h264(tmp_path / "source.mp4", tmp_path / "dest.mp4") is False


def test_reencode_h264_does_not_hide_unexpected_failures(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg"),
    )

    def fail(*args, **kwargs):
        raise ValueError("invalid command")

    monkeypatch.setattr(demo.subprocess, "run", fail)

    with pytest.raises(ValueError, match="invalid command"):
        reencode_h264(tmp_path / "source.mp4", tmp_path / "dest.mp4")
