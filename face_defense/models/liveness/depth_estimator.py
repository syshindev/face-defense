import numpy as np
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("depth_estimator")
class DepthEstimator(BaseDetector):
    # Depth estimation using MediaPipe face mesh Z-coordinates

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        # TODO: Extract 3D landmarks, analyze Z-variance
        return DetectorResult(
            is_real=True,
            score=1.0,
            attack_type=None,
            confidence=0.0,
            metadata={"detector": "depth_estimator"},
        )

    def get_name(self) -> str:
        return "depth_estimator"

    def get_attack_types(self):
        return ["print", "3d_mask"]
