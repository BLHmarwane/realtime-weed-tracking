"""Démo Streamlit : uploader une vidéo → détection + tracking annotés.

Prévu au milestone M4 (une fois le pipeline M3 validé). Squelette minimal pour
vérifier dès maintenant que l'app démarre dans le venv du projet :

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from weedtrack import __version__  # noqa: E402

st.set_page_config(page_title="Weed Detection & Tracking", page_icon="🌱")

st.title("🌱 Real-Time Weed Detection & Tracking")
st.caption(f"weedtrack v{__version__} — démo en construction (milestone M4)")

st.info(
    "La démo arrive au milestone M4 : upload d'une vidéo de parcelle, "
    "détection cultures/adventices (YOLO fine-tuné) et suivi ByteTrack avec "
    "IDs stables, rendu annoté en direct."
)

st.file_uploader(
    "Vidéo à analyser (bientôt actif)",
    type=["mp4", "avi", "mov"],
    disabled=True,
)
