"""Runtime helpers for the interactive video demo."""

import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .pipeline import run_on_video


@dataclass(frozen=True)
class DemoOutput:
    """Video-processing result that remains valid after temporary cleanup."""

    summary: dict
    video_bytes: bytes
    browser_preview: bool


@contextmanager
def temporary_upload(name: str, data: bytes):
    """Persist an upload for processing and always remove it afterwards."""

    path = None
    try:
        suffix = Path(name).suffix or ".mp4"
        with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            path = Path(handle.name)
            handle.write(data)
        yield path
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def reencode_h264(source: Path, dest: Path) -> bool:
    """Re-encode an MP4 to browser-friendly H.264 when ffmpeg is available."""

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except (ImportError, OSError, subprocess.SubprocessError):
        return False


def process_demo_video(
    source_path: Path,
    weights_path: Path,
    config: dict,
    *,
    device: str | None = None,
    conf: float | None = None,
    on_frame=None,
    runner=run_on_video,
    encoder=reencode_h264,
) -> DemoOutput:
    """Run the demo pipeline and return output safe from temporary cleanup."""

    with TemporaryDirectory(prefix="weedtrack_") as directory:
        raw = Path(directory) / "annotated_raw.mp4"
        encoded = Path(directory) / "annotated_h264.mp4"
        summary = runner(
            source=str(source_path),
            output_path=str(raw),
            weights=str(weights_path),
            tracker=config["tracking"]["tracker"],
            conf=conf if conf is not None else config["model"]["conf_threshold"],
            imgsz=config["model"]["imgsz"],
            device=device,
            on_frame=on_frame,
        )
        browser_preview = encoder(raw, encoded)
        chosen = encoded if browser_preview else raw
        return DemoOutput(summary, chosen.read_bytes(), browser_preview)
