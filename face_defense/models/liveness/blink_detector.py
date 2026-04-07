import numpy as np
import cv2
import mediapipe as mp
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY

# MediaPipe eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def compute_ear(landmarks, eye_indices) -> float:
    # Compute Eye Aspect Ratio from 6 landmark points
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


@DETECTOR_REGISTRY.register("blink_detector")
class BlinkDetector(BaseDetector):
    # Eye blink detection using MediaPipe EAR (Eye Aspect Ratio)

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.ear_threshold = config.get("ear_threshold", 0.21)
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
                metadata={"detector": "blink_detector", "error": "no_face"},
            )
        
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = compute_ear(landmarks, LEFT_EYE)
        right_ear = compute_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        # High EAR = eyes open = likely real (for single frame)
        # Low EAR = eyes closed or flat image artifact
        score = min(avg_ear / 0.3, 1.0)
        is_real = avg_ear > self.ear_threshold

        return DetectorResult(
            is_real=is_real,
            score=float(score),
            attack_type=None if is_real else "print",
            confidence=abs(avg_ear - self.ear_threshold) / 0.3,
            metadata={
                "detector": "blink_detector",
                "left_ear": float(left_ear),
                "right_ear": float(right_ear),
                "avg_ear": float(avg_ear),
            },
        )

    def get_name(self) -> str:
        return "blink_detector"

    def get_attack_types(self):
        return ["print", "replay"]
