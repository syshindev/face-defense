import numpy as np
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("deepface_spoof")
class DeepFaceSpoofDetector(BaseDetector):
    # deepface library anti-spoofing wrapper

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        # TODO: Call DeepFace.extract_faces(img, anti_spoofing=True)
        return DetectorResult(
            is_real=True,
            score=1.0,
            attack_type=None,
            confidence=0.0,
            metadata={"detector": "deepface_spoof"},
        )

    def get_name(self) -> str:
        return "deepface_spoof"

    def get_attack_types(self):
        return ["print", "replay"]
