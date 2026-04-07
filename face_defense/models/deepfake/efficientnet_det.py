import os

import cv2
import numpy as np
import torch
import timm
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult
from face_defense.core.registry import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("efficientnet")
class EfficientNetDetector(BaseDetector):
    # EfficientNet-B4 deepfake detector

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.variant = config.get("variant", "efficientnet_b4")
        self.input_size = config.get("input_size", [380, 380])
        self.threshold = config.get("threshold", 0.5)
        self.model = None

    def _load_model(self):
        # Load EfficientNet with ImageNet pretrained weights
        self.model = timm.create_model(self.variant, pretrained=True, num_classes=2)

        # Load fine-tuned checkpoint if available
        checkpoint = self.config.get("weights", {}).get("checkpoint", None)
        if checkpoint and os.path.exists(checkpoint):
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        img = cv2.resize(face_image, tuple(self.input_size))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, face_image: np.ndarray) -> DetectorResult:
        if self.model is None:
            self._load_model()

        try:
            tensor = self._preprocess(face_image)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]

            real_prob = float(probs[1])
            is_real = real_prob >= self.threshold

            return DetectorResult(
                is_real=is_real,
                score=real_prob,
                attack_type=None if is_real else "deepfake",
                confidence=abs(real_prob - 0.5) * 2,
                metadata={"detector": "efficientnet", "fake_prob": float(probs[0]), "real_prob": real_prob},
            )

        except Exception as e:
            return DetectorResult(
                is_real=False, score=0.0,
                attack_type="unknown", confidence=0.0,
                metadata={"detector": "efficientnet", "error": str(e)},
            )

    def get_name(self) -> str:
        return "efficientnet"

    def get_attack_types(self):
        return ["deepfake", "faceswap"]
