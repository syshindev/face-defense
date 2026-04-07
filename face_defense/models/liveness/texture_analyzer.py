import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("texture_analyzer")
class TextureAnalyzer(BaseDetector):
    # Texture analysis using LBP and Laplacian variance

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.lbp_radius = config.get("lbp_radius", 1)
        self.lbp_points = config.get("lbp_points", 8)
        self.laplacian_threshold = config.get("laplacian_threshold", 100.0)

    def _compute_lbp_score(self, gray: np.ndarray) -> float:
        # LBP histogram variance — real skin has richer texture
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method="uniform")
        hist, _ = np.histogram(lbp, bins=self.lbp_points + 2, density=True)
        return float(np.var(hist))

    def _compute_laplacian_score(self, gray: np.ndarray) -> float:
        # Laplacian variance — screens/prints have different frequency response
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        lbp_var = self._compute_lbp_score(gray)
        lap_var = self._compute_laplacian_score(gray)

        # Normalize laplacian score (higher = sharper = more likely real)
        lap_score = min(lap_var / self.laplacian_threshold, 1.0)

        # Combine scores
        score = (lbp_var * 1000 + lap_score) / 2.0
        score = min(max(score, 0.0), 1.0)
        is_real = score >= 0.5

        return DetectorResult(
            is_real=is_real,
            score=float(score),
            attack_type=None if is_real else "print",
            confidence=abs(score - 0.5) * 2,
            metadata={
                "detector": "texture_analyzer",
                "lbp_variance": float(lbp_var),
                "laplacian_variance": float(lap_var),
            },
        )

    def get_name(self) -> str:
        return "texture_analyzer"

    def get_attack_types(self):
        return ["print", "replay"]
