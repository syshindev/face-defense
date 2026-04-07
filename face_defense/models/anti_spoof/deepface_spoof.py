import numpy as np
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("deepface_spoof")
class DeepFaceSpoofDetector(BaseDetector):
    # Deepface library anti-spoofing wrapper

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.detector_backend = config.get("detector_backend", "skip")
        self.threshold = config.get("threshold", 0.5)

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        from deepface import DeepFace

        try:
            results = DeepFace.extract_faces(
                face_image,
                anti_spoofing=True,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )

            if not results:
                return DetectorResult(
                    is_real=False, score=0.0,
                    attack_type="unknown", confidence=0.0,
                    metadata={"detector": "deepface_spoof", "error": "no_face"},
                )

            face = results[0]
            is_real = face.get("is_real", False)
            score = float(face.get("antispoof_score", 0.0))

            return DetectorResult(
                is_real=is_real,
                score=score,
                attack_type=None if is_real else "spoof",
                confidence=abs(score - 0.5) * 2,
                metadata={"detector": "deepface_spoof"},
            )

        except Exception as e:
            return DetectorResult(
                is_real=False, score=0.0,
                attack_type="unknown", confidence=0.0,
                metadata={"detector": "deepface_spoof", "error": str(e)},
            )

    def get_name(self) -> str:
        return "deepface_spoof"

    def get_attack_types(self):
        return ["print", "replay"]
