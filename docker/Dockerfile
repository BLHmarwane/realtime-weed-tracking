# Image de démo — draft, finalisée au milestone M4.
# Objectif : `docker run -p 8501:8501 <image>` → démo Streamlit utilisable.
#
# Build (depuis la racine du projet) :
#   docker build -f docker/Dockerfile -t weedtrack-demo .

FROM python:3.12-slim

# OpenCV a besoin de quelques libs système même en headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY weedtrack/ weedtrack/
COPY configs/ configs/
COPY app/ app/
# TODO(M4) : COPY models/best.pt models/  (poids fine-tunés, hors Git —
# fournis au build ou téléchargés au démarrage depuis une release GitHub)

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
