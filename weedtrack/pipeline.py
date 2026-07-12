"""Pipeline vidéo : lecture, annotation, statistiques et boucle complète.

Contient :
- `FpsMeter` : mesure du débit réel (pur Python) ;
- `TrackingStats` : accumulateur de statistiques de tracking (pur Python) ;
- `annotate_frame` : dessin des boîtes/IDs sur une frame (OpenCV, import paresseux) ;
- `run_on_video` : boucle complète détection + tracking → vidéo annotée + stats.
"""

import time

from .detect import Detection
from .track import iter_tracked_results

# Couleurs BGR par classe (OpenCV) : crop en vert, weed en rouge.
CLASS_COLORS = {"crop": (80, 200, 60), "weed": (60, 60, 230)}
DEFAULT_COLOR = (200, 200, 200)


class VideoPipelineError(RuntimeError):
    """Raised when the source or destination video cannot be opened."""


def _video_metadata(cv2_module, source: str) -> tuple[float, int]:
    capture = cv2_module.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise VideoPipelineError(f"Cannot open source video: {source}")
        source_fps = capture.get(cv2_module.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
        return float(source_fps), total_frames
    finally:
        capture.release()


def _open_video_writer(
    cv2_module, output_path: str, fps: float, width: int, height: int
):
    fourcc = cv2_module.VideoWriter_fourcc(*"mp4v")
    writer = cv2_module.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise VideoPipelineError(f"Cannot open output video: {output_path}")
    return writer


class FpsMeter:
    """Mesure un débit en frames/seconde sur une fenêtre glissante.

    Sert au benchmark honnête du pipeline : le FPS annoncé dans le README
    devra sortir de cette classe, pas d'une estimation.
    """

    def __init__(self, window: int = 30):
        if window < 1:
            raise ValueError("window doit être >= 1")
        self._window = window
        self._timestamps: list[float] = []

    def tick(self) -> None:
        """À appeler une fois par frame traitée."""
        self._timestamps.append(time.perf_counter())
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)

    @property
    def fps(self) -> float:
        """FPS moyen sur la fenêtre courante (0.0 tant qu'il manque de données)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


class TrackingStats:
    """Accumule les statistiques d'un run de tracking (pur Python, testé).

    Un « track » est compté une seule fois par classe, quel que soit le nombre
    de frames où il apparaît. Les détections sans ID (le tracker n'a pas
    encore confirmé la piste) sont comptées à part.
    """

    def __init__(self):
        self._frames = 0
        self._detections = 0
        self._untracked = 0
        self._tracks: dict[str, set[int]] = {}

    def start_frame(self) -> None:
        self._frames += 1

    def add(self, class_name: str, track_id: int | None) -> None:
        self._detections += 1
        if track_id is None:
            self._untracked += 1
        else:
            self._tracks.setdefault(class_name, set()).add(int(track_id))

    def summary(self) -> dict:
        return {
            "frames": self._frames,
            "detections": self._detections,
            "detections_per_frame": (
                round(self._detections / self._frames, 2) if self._frames else 0.0
            ),
            "unique_tracks": {
                name: len(ids) for name, ids in sorted(self._tracks.items())
            },
            "untracked_detections": self._untracked,
        }


def annotate_frame(frame, detections, track_ids=None):
    """Dessine boîtes, classes, scores (et IDs de piste si fournis) sur la frame.

    Couleur par classe : crop vert, weed rouge — lisible en démo.
    """
    import cv2

    annotated = frame.copy()
    for index, det in enumerate(detections):
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(annotated, p1, p2, color, 2)
        label = f"{det.class_name} {det.score:.2f}"
        if track_ids is not None and index < len(track_ids) and track_ids[index] is not None:
            label = f"#{track_ids[index]} {label}"
        cv2.putText(
            annotated, label, (p1[0], max(12, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return annotated


def run_on_video(
    source: str,
    output_path: str,
    weights: str,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    on_frame=None,
) -> dict:
    """Boucle complète : vidéo → détection + tracking → vidéo annotée + stats.

    Retourne le résumé `TrackingStats` enrichi du FPS pipeline mesuré.
    Attention à la comparaison : ce `pipeline_fps` couvre TOUT le tuyau
    (détection + tracking + annotation + écriture vidéo), contrairement au
    FPS de `scripts/evaluate.py` qui mesure l'inférence image seule.
    """
    import cv2

    source_fps, total_frames = _video_metadata(cv2, source)

    writer = None
    stats = TrackingStats()
    started_at = time.perf_counter()
    try:
        for result in iter_tracked_results(
            str(source), weights, tracker=tracker, conf=conf, imgsz=imgsz, device=device
        ):
            boxes = result.boxes
            names = result.names
            detections: list[Detection] = []
            track_ids: list[int | None] = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
                class_id = int(boxes.cls[i])
                detections.append(
                    Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        score=float(boxes.conf[i]),
                        class_id=class_id,
                        class_name=names.get(class_id, str(class_id)),
                    )
                )
                track_ids.append(int(boxes.id[i]) if boxes.id is not None else None)

            stats.start_frame()
            for det, track_id in zip(detections, track_ids):
                stats.add(det.class_name, track_id)
            if on_frame is not None:
                on_frame(stats.summary()["frames"], total_frames)

            annotated = annotate_frame(result.orig_img, detections, track_ids)
            if writer is None:
                height, width = annotated.shape[:2]
                writer = _open_video_writer(
                    cv2, output_path, source_fps, width, height
                )
            writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started_at
    summary = stats.summary()
    summary["pipeline_fps"] = (
        round(summary["frames"] / elapsed, 2) if elapsed > 0 else 0.0
    )
    summary["source_fps"] = round(source_fps, 2)
    return summary
