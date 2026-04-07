import os
import sys

import numpy as np
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY

# Add Silent-FAS repo to Python path
_SILENT_FAS_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "third_party", "Silent-Face-Anti-Spoofing"
)


@DETECTOR_REGISTRY.register("silent_fas")
class SilentFASDetector(BaseDetector):
    # Silent Face Anti-Spoofing detector wrapper

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model_dir = os.path.join(_SILENT_FAS_ROOT, "resources", "anti_spoof_models")
        self.threshold = config.get("threshold", 0.5)
        self._predictor = None

    def _load_model(self):
        # Add repo to path so we can import its modules
        if _SILENT_FAS_ROOT not in sys.path:
            sys.path.insert(0, _SILENT_FAS_ROOT)
        from src.anti_spoof_predict import AntiSpoofPredict
        self._predictor = AntiSpoofPredict(device_id=0)

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        if self._predictor is None:
            self._load_model()

        try:
            # Get model paths
            model_paths = sorted(
                [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir) if f.endswith(".pth")]
            )

            # Run each model and sum scores
            total_score = 0.0
            for model_path in model_paths:
                result = self._predictor.predict(face_image, model_path)
                total_score += result[0][1]  # result[0][1] = real probability

            # Average score across models
            score = total_score / max(len(model_paths), 1)
            is_real = score >= self.threshold

            return DetectorResult(
                is_real=is_real,
                score=float(score),
                attack_type=None if is_real else "spoof",
                confidence=abs(score - 0.5) * 2,
                metadata={"detector": "silent_fas", "num_models": len(model_paths)},
            )

        except Exception as e:
            return DetectorResult(
                is_real=False, score=0.0,
                attack_type="unknown", confidence=0.0,
                metadata={"detector": "silent_fas", "error": str(e)},
            )

    def get_name(self) -> str:
        return "silent_fas"

    def get_attack_types(self):
        return ["print", "replay"]
