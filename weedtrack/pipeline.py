"""Pipeline vidéo : lecture, annotation et mesure de performance.

Contient les briques indépendantes du modèle :
- `FpsMeter` : mesure du débit réel (pur Python, testé dès M0) ;
- `annotate_frame` : dessin des boîtes/IDs sur une frame (OpenCV, import paresseux) ;
- `run_on_video` : boucle complète détection+tracking → vidéo annotée (M3).
"""

import time


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


def annotate_frame(frame, detections, track_ids=None):
    """Dessine boîtes, classes, scores (et IDs de piste si fournis) sur la frame.

    TODO(M3) : couleur par classe (crop vs weed), épaisseur lisible en démo.
    """
    import cv2

    annotated = frame.copy()
    for index, det in enumerate(detections):
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
        label = f"{det.class_name} {det.score:.2f}"
        if track_ids is not None and index < len(track_ids):
            label = f"#{track_ids[index]} {label}"
        cv2.putText(
            annotated, label, (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return annotated


def run_on_video(source: str, output_path: str, config: dict) -> dict:
    """Boucle complète : vidéo → détection + tracking → vidéo annotée + stats.

    Retournera un dict de stats (fps moyen, nombre de pistes, durée).

    TODO(M3) : implémenter avec `weedtrack.track.iter_tracked_results`,
    `annotate_frame` et `FpsMeter`, puis archiver les stats dans `metrics/`.
    """
    raise NotImplementedError("Prévu au milestone M3 — voir STATE.md.")
