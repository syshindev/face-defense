import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CelebASpoofDataset(Dataset):
    # CelebA-Spoof dataset loader
    # Annotation index 40: 0 = live (real), 1 = spoof (fake)

    def __init__(self, root: str, split: str = "train", image_size: int = 256, transform=None):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.samples = []
        self._load_annotations()

    def _load_annotations(self):
        # Parse annotation file from metas/
        if self.split == "train":
            meta_path = os.path.join(self.root, "metas", "intra_test", "train_label.txt")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(self.root, "metas", "intra_train", "items.txt")
        else:
            meta_path = os.path.join(self.root, "metas", "intra_test", "test_label.txt")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(self.root, "metas", "intra_test", "items.txt")

        with open(meta_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 41:
                    continue
                img_path = parts[0]
                label = int(parts[40])  # 0 = live, 1 = spoof
                self.samples.append({
                    "path": os.path.join(self.root, "Data", img_path),
                    "label": label,
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
