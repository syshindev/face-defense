import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CelebASpoofDataset(Dataset):
    # CelebA-Spoof dataset loader
    # Folder structure: Data/{split}/{subject_id}/live/ or spoof/
    # Label: 0 = live (real), 1 = spoof (fake)

    def __init__(self, root: str, split: str = "train", image_size: int = 256, transform=None):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        # Scan folder structure for images and labels
        split_dir = os.path.join(self.root, "Data", self.split)

        for subject_id in sorted(os.listdir(split_dir)):
            subject_dir = os.path.join(split_dir, subject_id)
            if not os.path.isdir(subject_dir):
                continue

            # Live images (label = 0)
            live_dir = os.path.join(subject_dir, "live")
            if os.path.exists(live_dir):
                for fname in os.listdir(live_dir):
                    if fname.lower().endswith((".jpg", ".png")):
                        self.samples.append({
                            "path": os.path.join(live_dir, fname),
                            "label": 0,
                        })

            # Spoof images (label = 1)
            spoof_dir = os.path.join(subject_dir, "spoof")
            if os.path.exists(spoof_dir):
                for fname in os.listdir(spoof_dir):
                    if fname.lower().endswith((".jpg", ".png")):
                        self.samples.append({
                            "path": os.path.join(spoof_dir, fname),
                            "label": 1,
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = cv2.imread(sample["path"])

        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Normalize to [0, 1] and convert to CHW tensor
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        return image, sample["label"]
