"""Détection : enveloppe fine autour d'un modèle YOLO Ultralytics.

Le reste du pipeline ne parle jamais à ultralytics directement — uniquement à
`Detector` et à des objets `Detection` simples. Avantages :

- les smoke tests importent ce module sans ultralytics installé ;
- si on change de détecteur un jour (RT-DETR, autre lib), seul ce fichier bouge.
"""

from dataclasses import dataclass


@dataclass
class Detection:
    """Une boîte détectée sur une frame, en pixels."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


class Detector:
    """Charge des poids YOLO et prédit des `Detection` sur une image."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # message actionnable plutôt qu'un traceback nu
            raise ImportError(
                "ultralytics n'est pas installé. Créer le venv du projet puis "
                "`pip install -r requirements.txt` (voir TUTORIAL.md §7)."
            ) from exc

        self._model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, frame) -> list[Detection]:
        """Détecte sur une frame BGR (ndarray OpenCV) et retourne les boîtes.

        TODO(M2) : valider seuils et classes sur le dataset réel, ajouter un
        test d'intégration sur une image d'exemple.
        """
        results = self._model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                class_id = int(box.cls[0])
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        score=float(box.conf[0]),
                        class_id=class_id,
                        class_name=names.get(class_id, str(class_id)),
                    )
                )
        return detections
