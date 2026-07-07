# Real-Time Weed Detection & Tracking

![tests](https://github.com/BLHmarwane/realtime-weed-tracking/actions/workflows/tests.yml/badge.svg)

> 🚧 **Work in progress** — project scaffolded, training pipeline landing soon.
> Follow the milestones table below.

Detect crops vs. weeds in video and give every plant a **stable tracking ID** —
so a precision-weeding robot never treats the same weed twice.

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
| M2 | Fine-tuned baseline | mAP@50 measured on val split, archived in `metrics/` | ⬜ |
| M3 | Video pipeline: detection + tracking | FPS benchmark + annotated video with stable IDs | ⬜ |
| M4 | Streamlit demo + Docker image | One-command `docker run` → usable demo | ⬜ |
| M5 | Public release | Public repo, English README, demo GIF, metrics table | ⬜ |

## Metrics

*Coming at M2/M3 — every number in this table is produced by
`scripts/evaluate.py` and archived in `metrics/`.*

## Quickstart

*Coming at M4. Target:*

```bash
docker run -p 8501:8501 <image>   # then open http://localhost:8501
```

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
