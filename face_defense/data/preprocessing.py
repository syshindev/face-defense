from typing import List, Optional, Tuple

import cv2
import numpy as np


class FacePreprocessor:
    # Face detection, alignment, and cropping
    def __init__(self, face_size: int = 256, margin: float = 0.3):
        self.face_size = face_size
        self.margin = margin
        self._detector = None

    def _init_detector(self):
        # Lazy-load face detector to avoid import overhead
        if self._detector is None:
            from insightface.app import FaceAnalysis
            self._detector = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._detector.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        # Detect faces and return list of {bbox, landmarks, score}
        self._init_detector()
        faces = self._detector.get(image)
        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int),
                "landmarks": face.kps,
                "score": float(face.det_score),
            })
        return results
    
    def crop_face(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        # Crop and resize a face region with margin
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        margin_h = int(h * self.margin)
        margin_w = int(w * self.margin)

        # Clamp to image bounds
        img_h, img_w = image.shape[:2]
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(img_w, x2 + margin_w)
        y2 = min(img_h, y2 + margin_h)

        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (self.face_size, self.face_size))
        return face
    
    def process(self, image: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
        # Detect faces and return cropped face images with metadata
        detections = self.detect_faces(image)
        results = []
        for det in detections:
            face = self.crop_face(image, det["bbox"])
            results.append((face, det))
        return results
    