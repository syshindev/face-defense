import os
import json

import cv2
import numpy as np


# MediaPipe eye landmark indices for blink detection
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


class FaceDatabase:
    def __init__(self, db_path="face_db"):
        self.db_path = db_path
        self.meta_path = os.path.join(db_path, "meta.json")
        self.users = {}
        os.makedirs(db_path, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.users = json.load(f)

    def _save(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)

    def register(self, name, embedding):
        emb_file = f"{name}.npy"
        np.save(os.path.join(self.db_path, emb_file), embedding)
        self.users[name] = emb_file
        self._save()

    def recognize(self, embedding, threshold=0.4):
        best_name = None
        best_sim = -1
        for name, emb_file in self.users.items():
            stored = np.load(os.path.join(self.db_path, emb_file))
            sim = np.dot(embedding, stored) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored) + 1e-8
            )
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name, float(best_sim)

    def delete(self, name):
        if name not in self.users:
            return None
        emb_path = os.path.join(self.db_path, self.users[name])
        if os.path.exists(emb_path):
            os.remove(emb_path)
        del self.users[name]
        self._save()
        return name

    def count(self):
        return len(self.users)


def compute_ear(landmarks, eye_indices):
    """Compute Eye Aspect Ratio for blink detection."""
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def crop_face(frame, bbox, margin=0.2):
    """Crop face from frame with margin."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    H, W = frame.shape[:2]
    x1 = max(0, int(x1) - mx)
    y1 = max(0, int(y1) - my)
    x2 = min(W, int(x2) + mx)
    y2 = min(H, int(y2) + my)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
