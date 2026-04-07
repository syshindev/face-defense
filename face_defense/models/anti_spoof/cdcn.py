import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY
from face_defense.models.anti_spoof.cdcn_model import CDCN


@DETECTOR_REGISTRY.register("cdcn")
class CDCNDetector(BaseDetector):
    # CDCN (Central Difference Convolutional Network) detector

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model = None
        self.input_size = tuple(config.get("input_size", [256, 256]))
        self.threshold = config.get("threshold", 0.5)

    def _load_model(self):
        # Load CDCN model from weights checkpoint
        self.model = CDCN(in_channels=3, theta=0.7)
        checkpoint = self.config.get("checkpoint", None)
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location=self._device)
            self.model.load_state_dict(state_dict)
        self.model.to(self._device)
        self.model.eval()

    def _preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        # Resize and normalize face image to tensor
        img = cv2.resize(face_image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device)

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        if self.model is None:
            self._load_model()

        tensor = self._preprocess(face_image)

        with torch.no_grad():
            depth_map = self.model(tensor)

        # Compute realness score from depth map mean
        score = depth_map.mean().item()
        is_real = score >= self.threshold

        return DetectorResult(
            is_real=is_real,
            score=score,
            attack_type=None if is_real else "spoof",
            confidence=abs(score - self.threshold) / self.threshold,
            metadata={"detector": "cdcn", "depth_map_mean": score},
        )

    def get_name(self) -> str:
        return "cdcn"

    def get_attack_types(self):
        return ["print", "replay", "3d_mask"]
