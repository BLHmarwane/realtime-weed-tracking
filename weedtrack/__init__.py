"""weedtrack — détection + tracking temps réel de cultures/adventices.

Package coeur du projet : détection (YOLO), tracking (ByteTrack) et pipeline
vidéo. Les dépendances lourdes (ultralytics, cv2) sont importées paresseusement
dans les fonctions, jamais au niveau module : le package reste importable
partout, y compris pour les smoke tests sans GPU ni installation ML.
"""

__version__ = "0.1.0"
