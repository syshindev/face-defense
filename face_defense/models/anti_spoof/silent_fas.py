import numpy as np
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY

@DETECTOR_REGISTRY.register("silent_fas")
class SilentFASDetector(BaseDetector):
    # Silent Face Anti-Spoofing detector wrapper
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model = None

    def _load_model(self):
        # TODO: Load MiniFASNet from third_party/Silent-Face-Anti-Spoofing
        pass

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        if self.model is None:
            self._load_model()
        # TODO: Run multi-scale inference and return result
        return DetectorResult(
            is_real=True,
            score=1.0,
            attack_type=None,
            confidence=0.0,
            metadata={"detector": "silent_fas"},
        )
    
    def get_name(self) -> str:
        return "silent_fas"
    
    def get_attack_types(self):
        return ["print", "replay"]
    