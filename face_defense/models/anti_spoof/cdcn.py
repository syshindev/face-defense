import numpy as np
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("cdcn")
class CDCNDetector(BaseDetector):
    # CDCN (Central Difference Convolutional Network) detector

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model = None

    def _load_model(self):
        # TODO: Load CDCN model from weights checkpoint
        pass

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        if self.model is None:
            self._load_model()
        # TODO: Predict depth map and compute realness score
        return DetectorResult(
            is_real=True,
            score=1.0,
            attack_type=None,
            confidence=0.0,
            metadata={"detector": "cdcn"},
        )

    def get_name(self) -> str:
        return "cdcn"

    def get_attack_types(self):
        return ["print", "replay", "3d_mask"]
