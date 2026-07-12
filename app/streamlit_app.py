"""Streamlit demo: upload a video and receive annotated tracking output.

Run locally:
    .venv/bin/streamlit run app/streamlit_app.py

Docker (one command after the image is built):
    docker build -f docker/Dockerfile -t weedtrack-demo .
    docker run -p 8501:8501 weedtrack-demo
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from weedtrack import __version__  # noqa: E402
from weedtrack.config import load_config  # noqa: E402
from weedtrack.demo import process_demo_video, temporary_upload  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "models" / "best.pt"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_video.mp4"


def available_devices() -> list[str]:
    """Return available devices, best first, while importing torch lazily."""
    devices = ["cpu"]
    try:
        import torch

        if torch.backends.mps.is_available():
            devices.insert(0, "mps")
        elif torch.cuda.is_available():
            devices.insert(0, "0")
    except Exception:
        pass
    return devices


def analyse(source_path: Path, device: str, conf: float, config: dict) -> None:
    """Run the pipeline and display metrics plus the annotated video."""
    progress = st.progress(0.0, text="Detecting and tracking...")

    def on_frame(index: int, total: int) -> None:
        if total > 0:
            progress.progress(min(index / total, 1.0), text=f"Frame {index}/{total}")

    result = process_demo_video(
        source_path,
        WEIGHTS_PATH,
        config,
        conf=conf,
        device=device,
        on_frame=on_frame,
    )
    progress.progress(1.0, text="Complete")

    summary = result.summary
    tracks = summary.get("unique_tracks", {})

    if result.browser_preview:
        st.video(result.video_bytes)
    else:
        st.info(
            "Browser preview is unavailable. Download the annotated video below."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", summary["frames"])
    col2.metric("Pipeline FPS", summary["pipeline_fps"])
    col3.metric("Short-term weed tracks", tracks.get("weed", 0))
    col4.metric("Short-term crop tracks", tracks.get("crop", 0))

    st.download_button(
        "Download the annotated video (MP4)",
        data=result.video_bytes,
        file_name="weedtrack_annotated.mp4",
        mime="video/mp4",
    )
    with st.expander("Raw statistics (JSON)"):
        st.code(json.dumps(summary, indent=2, ensure_ascii=False), language="json")


st.set_page_config(page_title="Weed Detection & Tracking", page_icon="🌱", layout="wide")

st.title("🌱 Real-Time Weed Detection & Tracking")
st.caption(
    f"weedtrack v{__version__} — fine-tuned YOLO11n (mAP@50 0.80) + ByteTrack. "
    "Red boxes mark weeds and green boxes mark crops. Track IDs are short-term "
    "identifiers within the analysed clip."
)

if not WEIGHTS_PATH.exists():
    st.error(
        "Model weights were not found at `models/best.pt`. "
        "Train them first with `python scripts/train.py` (see README)."
    )
    st.stop()

config = load_config()

with st.sidebar:
    st.header("Settings")
    device = st.selectbox("Device", available_devices())
    conf = st.slider(
        "Confidence threshold",
        0.05,
        0.90,
        float(config["model"]["conf_threshold"]),
        0.05,
    )
    st.divider()
    st.markdown(
        "**Model**: YOLO11n fine-tuned on the "
        "[Sudars et al. 2020](https://data.mendeley.com/datasets/nj4vtk4tt6/1) "
        "dataset (CC BY 4.0).\n\n"
        "[Source code ↗](https://github.com/BLHmarwane/realtime-weed-tracking)"
    )

uploaded = st.file_uploader(
    "Video to analyse (MP4, AVI or MOV)",
    type=["mp4", "avi", "mov"],
    max_upload_size=int(config["demo"]["max_upload_mb"]),
)

if uploaded is not None:
    if st.button("Analyse the video", type="primary"):
        try:
            with temporary_upload(uploaded.name, uploaded.getvalue()) as source_path:
                analyse(source_path, device, conf, config)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Video processing failed: {error}")
elif SAMPLE_PATH.exists():
    st.caption("No video to upload?")
    if st.button("▶ Try the sample video", type="primary"):
        try:
            analyse(SAMPLE_PATH, device, conf, config)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Video processing failed: {error}")
