import numpy as np
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("blink_detector")
class BlinkDetector(BaseDetector):
    # Eye blink detection using MediaPipe EAR (Eye Aspect Ratio)

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.ear_threshold = config.get("ear_threshold", 0.21)
        self.consec_frames = config.get("consec_frames", 3)
        self.blink_count = 0
        self.frame_counter = 0

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        # TODO: Extract eye landmarks via MediaPipe, compute EAR
        return DetectorResult(
            is_real=True,
            score=1.0,
            attack_type=None,
            confidence=0.0,
            metadata={"detector": "blink_detector", "blink_count": self.blink_count},
        )

    def get_name(self) -> str:
        return "blink_detector"

    def get_attack_types(self):
        return ["print", "replay"]
