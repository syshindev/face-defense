import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from omegaconf import DictConfig

from face_defense.data.base_dataset import BaseDataset


class DeepfakeDataset(BaseDataset):
    # Dataset for deepfake detection (FaceForensics++, Celeb-DF, DFDC)

    def __init__(self, config: DictConfig, split: str = "train"):
        super().__init__(config, split)
        self.root = config.get("root", "")
        self.methods = config.get("methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
        self.transform = None
        self.load_samples()

    def load_samples(self):
        # Load samples from dataset directory
        # Expected structure: root/original/ and root/{method}/
        original_dir = os.path.join(self.root, "original")
        if os.path.exists(original_dir):
            for fname in sorted(os.listdir(original_dir)):
                if fname.lower().endswith((".jpg", ".png", ".bmp")):
                    self.samples.append({
                        "path": os.path.join(original_dir, fname),
                        "label": 1,
                        "method": "original",
                    })

        for method in self.methods:
            method_dir = os.path.join(self.root, method)
            if os.path.exists(method_dir):
                for fname in sorted(os.listdir(method_dir)):
                    if fname.lower().endswith((".jpg", ".png", ".bmp")):
                        self.samples.append({
                            "path": os.path.join(method_dir, fname),
                            "label": 0,
                            "method": method,
                        })

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        sample = self.samples[index]
        image = cv2.imread(sample["path"])

        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        metadata = {"path": sample["path"], "method": sample["method"]}
        return image, sample["label"], metadata
