# Image locale de la démo Streamlit.
#
# Build depuis la racine après avoir placé les poids dans models/best.pt et
# généré data/sample_video.mp4 avec scripts/make_test_video.py :
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
RUN pip install --no-cache-dir "torch>=2.12,<3" "torchvision>=0.27,<1" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY weedtrack/ weedtrack/
COPY configs/ configs/
COPY app/ app/
# Ces deux artefacts sont hors Git et doivent être présents localement au build.
COPY models/best.pt models/best.pt
COPY data/sample_video.mp4 data/sample_video.mp4

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /tmp/ultralytics /tmp/.streamlit \
    && chown -R appuser:appuser /tmp/ultralytics /tmp/.streamlit /app
ENV HOME=/tmp
USER 10001:10001

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
