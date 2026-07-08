# Real-Time Weed Detection & Tracking

![tests](https://github.com/BLHmarwane/realtime-weed-tracking/actions/workflows/tests.yml/badge.svg)

Detect crops vs. weeds in video and give every plant a **stable tracking ID** —
so a precision-weeding robot never treats the same weed twice.

![Detection and tracking demo — red boxes are weeds, with stable track IDs](assets/demo.gif)

## The story

During my Master's I prototyped real-time weed detection with **classical
computer vision** (C++/OpenCV — grayscale, filtering, morphology) for a real
use case brought by a research engineer from INRAE Clermont-Ferrand.

This project revisits the same problem with a **modern deep-learning stack**:

- fine-tuned **YOLO** detector (Ultralytics),
- **ByteTrack** multi-object tracking on video streams,
- measured metrics (mAP, FPS) reproducible from scripts,
- a live **Streamlit** demo, packaged as a **one-command Docker image**.

Same problem, two eras of computer vision — and I can explain both without a
black box.

## Planned stack

| Piece | Choice | Why |
|---|---|---|
| Detection | Ultralytics YOLO (nano) | Solid fine-tuning workflow, real-time capable on CPU |
| Tracking | ByteTrack | Robust ID assignment, standard in modern MOT |
| Video I/O | OpenCV | Frame-level control, annotation overlays |
| Demo | Streamlit | Upload a video → watch detections + tracks live |
| Packaging | Docker | `docker run` → demo up, zero setup |

## Milestones

| # | Deliverable | Success criteria | Status |
|---|---|---|---|
| M0 | Scaffold, docs, method | Smoke tests green, independent audit passed | ✅ |
| M1 | Dataset selected (license checked) + EDA | Valid `data.yaml`, class stats documented | ✅ |
| M2 | Fine-tuned baseline | mAP@50 measured on val split, archived in `metrics/` | ✅ |
| M3 | Video pipeline: detection + tracking | FPS benchmark + annotated video with stable IDs | ✅ |
| M4 | Streamlit demo + Docker image | One-command `docker run` → usable demo | ✅ |
| M5 | Public release | Public repo, English README, demo GIF, metrics table | ✅ |

## Metrics

*Every number below is produced by `scripts/evaluate.py` and archived in
[`metrics/`](metrics/) — nothing is hand-written.*

Fine-tuned **YOLO11n** (2.58 M fused parameters, as reported by the model
summary in `scripts/evaluate.py`), 60 epochs at 640 px.
Validation split: 235 images / 1 483 boxes.

| Metric | all | crop | weed |
|---|---|---|---|
| mAP@50 | **0.802** | 0.793 | 0.812 |
| mAP@50-95 | 0.522 | 0.522 | 0.522 |

> Note: the val split contains only 56 `crop` boxes (the dataset is heavily
> weed-dominated), so per-class `crop` numbers carry statistical noise.

Measured single-image inference speed (100 val images, 640 px, Apple M1 Pro):

| Device | FPS |
|---|---|
| CPU | 22.3 |
| Apple MPS | 25.3 |

End-to-end **video pipeline** (detection + ByteTrack tracking + annotation +
video writing), 640×640 test video, 600 frames — measured by
`scripts/track_video.py`:

| Device | Pipeline FPS |
|---|---|
| CPU | 16.2 |
| Apple MPS | 37.1 |

53 unique track IDs over 6 simulated camera passes (~29 frames per track on
average — stable IDs, no fragmentation), identical tracking results on CPU
and MPS.

Sources: [`metrics/baseline.json`](metrics/baseline.json),
[`metrics/baseline_cpu.json`](metrics/baseline_cpu.json),
[`metrics/tracking.json`](metrics/tracking.json),
[`metrics/tracking_cpu.json`](metrics/tracking_cpu.json),
[`metrics/dataset_stats.json`](metrics/dataset_stats.json).

## Test video

`scripts/make_test_video.py` builds a deterministic test video from val
images (same CC BY 4.0 source): a 640×640 window pans across each field
image, simulating a weeding-robot camera pass — plants enter and leave the
frame continuously, which is exactly what the tracker must handle. Stock
footage was deliberately rejected: drone shots of mature crops are
out-of-distribution for a model trained on top-down seedling images.

## Quickstart

### Local

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Either: reproduce everything from scratch
.venv/bin/python scripts/download_dataset.py      # dataset (CC BY 4.0, SHA-256 checked)
.venv/bin/python scripts/train.py --device mps    # or cpu → produces models/best.pt
.venv/bin/python scripts/make_test_video.py --num-images 2 --seconds 5 --out data/sample_video.mp4

# Or: skip training, grab the released weights + sample video
curl -L -o models/best.pt https://github.com/BLHmarwane/realtime-weed-tracking/releases/download/v0.1.0/best.pt
curl -L -o data/sample_video.mp4 https://github.com/BLHmarwane/realtime-weed-tracking/releases/download/v0.1.0/sample_video.mp4

.venv/bin/streamlit run app/streamlit_app.py      # → http://localhost:8501
```

The demo ships with a **built-in sample video button** — no file needed to try
it. Upload your own mp4/avi/mov to analyse it; the annotated video (H.264,
plays in the browser) and the tracking stats are displayed and downloadable.

### Docker (one command)

```bash
docker build -f docker/Dockerfile -t weedtrack-demo .   # needs models/best.pt + sample video (built above)
docker run -p 8501:8501 weedtrack-demo                  # → http://localhost:8501
```

The image bundles the fine-tuned weights and the sample video, uses CPU-only
PyTorch wheels (small image, no CUDA), and exposes a Docker `HEALTHCHECK` on
Streamlit's health endpoint.

## Repository layout

```
weedtrack/        core package: detection, tracking, video pipeline
configs/          single YAML config driving train / eval / demo
scripts/          download_dataset, train, evaluate (reproducible metrics)
app/              Streamlit demo
docker/           Dockerfile
tests/            smoke tests (run without the heavy ML dependencies)
data/, models/    git-ignored: datasets and weights never enter the repo
```

## Dataset

Training uses the **Dataset of annotated food crops and weed images for
robotic computer vision control** (Sudars et al., 2020) — 1 176 field and
controlled-environment images, 6 crop and 8 weed species (7 853 annotations),
mapped to two classes (`crop` / `weed`) for this project.
License: **CC BY 4.0** — [data.mendeley.com/datasets/nj4vtk4tt6/1](https://data.mendeley.com/datasets/nj4vtk4tt6/1).

## License

This repository is released under **AGPL-3.0** (see [LICENSE](LICENSE)),
required by its dependency on
[Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
