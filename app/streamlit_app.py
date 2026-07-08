"""Démo Streamlit : une vidéo en entrée → détection + tracking annotés.

Lancement local :
    .venv/bin/streamlit run app/streamlit_app.py

Docker (une commande, après build) :
    docker build -f docker/Dockerfile -t weedtrack-demo .
    docker run -p 8501:8501 weedtrack-demo
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from weedtrack import __version__  # noqa: E402
from weedtrack.config import load_config  # noqa: E402
from weedtrack.pipeline import run_on_video  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "models" / "best.pt"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_video.mp4"


def available_devices() -> list[str]:
    """Devices utilisables, le meilleur en premier (import torch paresseux)."""
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


def reencode_h264(source: Path, dest: Path) -> bool:
    """Ré-encode en H.264 pour lecture dans le navigateur.

    Le VideoWriter OpenCV produit du MPEG-4 part 2 (mp4v), que les
    navigateurs ne lisent pas. On repasse par le ffmpeg embarqué
    d'imageio-ffmpeg ; en cas d'échec la démo propose le téléchargement.
    """
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(source),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(dest),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False


def analyse(source_path: Path, device: str, conf: float, config: dict) -> None:
    """Exécute le pipeline et affiche stats + vidéo annotée."""
    workdir = Path(tempfile.mkdtemp(prefix="weedtrack_"))
    raw_output = workdir / "annotated_raw.mp4"

    progress = st.progress(0.0, text="Détection + tracking en cours…")

    def on_frame(index: int, total: int) -> None:
        if total > 0:
            progress.progress(min(index / total, 1.0), text=f"Frame {index}/{total}")

    summary = run_on_video(
        source=str(source_path),
        output_path=str(raw_output),
        weights=str(WEIGHTS_PATH),
        tracker=config["tracking"]["tracker"],
        conf=conf,
        imgsz=config["model"]["imgsz"],
        device=device,
        on_frame=on_frame,
    )
    progress.progress(1.0, text="Terminé")

    tracks = summary.get("unique_tracks", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", summary["frames"])
    col2.metric("FPS pipeline", summary["pipeline_fps"])
    col3.metric("Pistes weed", tracks.get("weed", 0))
    col4.metric("Pistes crop", tracks.get("crop", 0))

    display_output = workdir / "annotated_h264.mp4"
    video_bytes = None
    if reencode_h264(raw_output, display_output):
        video_bytes = display_output.read_bytes()
        st.video(video_bytes)
    else:
        st.info("Aperçu navigateur indisponible — téléchargez la vidéo annotée ci-dessous.")

    st.download_button(
        "Télécharger la vidéo annotée (MP4)",
        data=video_bytes or raw_output.read_bytes(),
        file_name="weedtrack_annotated.mp4",
        mime="video/mp4",
    )
    with st.expander("Statistiques brutes (JSON)"):
        st.code(json.dumps(summary, indent=2, ensure_ascii=False), language="json")


st.set_page_config(page_title="Weed Detection & Tracking", page_icon="🌱", layout="wide")

st.title("🌱 Real-Time Weed Detection & Tracking")
st.caption(
    f"weedtrack v{__version__} — YOLO11n fine-tuné (mAP@50 0.80) + ByteTrack. "
    "Boîtes rouges : adventices (weed) · boîtes vertes : cultures (crop), avec ID de piste stable."
)

if not WEIGHTS_PATH.exists():
    st.error(
        "Poids du modèle introuvables (`models/best.pt`). "
        "Entraîner d'abord : `python scripts/train.py` (voir README)."
    )
    st.stop()

config = load_config()

with st.sidebar:
    st.header("Réglages")
    device = st.selectbox("Device", available_devices())
    conf = st.slider(
        "Seuil de confiance", 0.05, 0.90, float(config["model"]["conf_threshold"]), 0.05
    )
    st.divider()
    st.markdown(
        "**Modèle** : YOLO11n fine-tuné sur le dataset "
        "[Sudars et al. 2020](https://data.mendeley.com/datasets/nj4vtk4tt6/1) "
        "(CC BY 4.0).\n\n"
        "[Code source ↗](https://github.com/BLHmarwane/realtime-weed-tracking)"
    )

uploaded = st.file_uploader(
    "Vidéo à analyser (mp4, avi, mov)", type=["mp4", "avi", "mov"]
)

if uploaded is not None:
    if st.button("Analyser la vidéo", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(uploaded.getbuffer())
        analyse(Path(handle.name), device, conf, config)
elif SAMPLE_PATH.exists():
    st.caption("Pas de vidéo sous la main ?")
    if st.button("▶ Essayer avec la vidéo d'exemple", type="primary"):
        analyse(SAMPLE_PATH, device, conf, config)
