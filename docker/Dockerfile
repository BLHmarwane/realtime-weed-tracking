# Image de démo — objectif : `docker run -p 8501:8501 weedtrack-demo` → démo utilisable.
#
# Build (depuis la racine du projet ; nécessite models/best.pt entraîné à M2
# et data/sample_video.mp4 généré par scripts/make_test_video.py) :
#   docker build -f docker/Dockerfile -t weedtrack-demo .

FROM python:3.12-slim

# OpenCV a besoin de quelques libs système même en headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dossiers d'écriture d'ultralytics/streamlit + pas de télémétrie
ENV YOLO_CONFIG_DIR=/tmp/ultralytics \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .
# torch CPU d'abord (roues légères) : sinon pip tire les roues CUDA multi-Go
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY weedtrack/ weedtrack/
COPY configs/ configs/
COPY app/ app/
# Poids fine-tunés + vidéo d'exemple : hors Git, présents localement au build
# (à M5 : publiés en release GitHub pour un build reproductible partout)
COPY models/best.pt models/best.pt
COPY data/sample_video.mp4 data/sample_video.mp4

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
