import cv2
import numpy as np
import mediapipe as mp
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("depth_estimator")
class DepthEstimator(BaseDetector):
    # Depth estimation using MediaPipe face mesh Z-coordinates

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return DetectorResult(
                is_real=False, score=0.0,
                attack_type="unknown", confidence=0.0,
                metadata={"detector": "depth_estimator", "error": "no_face"},
            )

        landmarks = results.multi_face_landmarks[0].landmark
        z_values = np.array([lm.z for lm in landmarks])

        # Real faces have significant Z variance (nose sticks out)
        # Flat photos have minimal Z variance
        z_range = float(z_values.max() - z_values.min())
        z_std = float(z_values.std())

        # Normalize: real face z_std is typically > 0.02
        score = min(z_std / 0.03, 1.0)
        is_real = score >= 0.5

        return DetectorResult(
            is_real=is_real,
            score=float(score),
            attack_type=None if is_real else "print",
            confidence=abs(score - 0.5) * 2,
            metadata={
                "detector": "depth_estimator",
                "z_range": z_range,
                "z_std": z_std,
            },
        )

    def get_name(self) -> str:
        return "depth_estimator"

    def get_attack_types(self):
        return ["print", "3d_mask"]
