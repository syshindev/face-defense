import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from omegaconf import DictConfig

from face_defense.data.base_dataset import BaseDataset


class AntiSpoofDataset(BaseDataset):
    # Dataset for anti-spoofing (OULU-NPU, CASIA-FASD, Replay-Attack)

    def __init__(self, config: DictConfig, split: str = "train"):
        super().__init__(config, split)
        self.root = config.get("root", "")
        self.transform = None
        self.load_samples()

    def load_samples(self):
        # Load samples from dataset directory
        # Expected structure: root/real/ and root/spoof/
        real_dir = os.path.join(self.root, "real")
        spoof_dir = os.path.join(self.root, "spoof")

        if os.path.exists(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if fname.lower().endswith((".jpg", ".png", ".bmp")):
                    self.samples.append({
                        "path": os.path.join(real_dir, fname),
                        "label": 1,
                        "attack_type": None,
                    })

        if os.path.exists(spoof_dir):
            for fname in sorted(os.listdir(spoof_dir)):
                if fname.lower().endswith((".jpg", ".png", ".bmp")):
                    self.samples.append({
                        "path": os.path.join(spoof_dir, fname),
                        "label": 0,
                        "attack_type": "spoof",
                    })

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        sample = self.samples[index]
        image = cv2.imread(sample["path"])

        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        metadata = {"path": sample["path"], "attack_type": sample["attack_type"]}
        return image, sample["label"], metadata
