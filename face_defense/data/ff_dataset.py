import os
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


REAL_CLASSES = {"original"}
FAKE_CLASSES = {
    "Deepfakes", "Face2Face", "FaceSwap",
    "NeuralTextures", "FaceShifter", "DeepFakeDetection",
}

NUM_CLASSES = 2  # 0=real, 1=fake


class FFDataset(Dataset):
    # FaceForensics++ binary classifier (real vs fake)
    # Folder structure: {class_name}/*.jpg
    # Stratified 80/20 split per class with fixed seed

    def __init__(self, root: str, split: str = "train", image_size: int = 299,
                 transform=None, train_ratio: float = 0.8, seed: int = 42):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.train_ratio = train_ratio
        self.seed = seed
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        rng = random.Random(self.seed)

        for class_name in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue

            if class_name in REAL_CLASSES:
                label = 0
            elif class_name in FAKE_CLASSES:
                label = 1
            else:
                continue

            files = [
                f for f in sorted(os.listdir(class_dir))
                if f.lower().endswith((".jpg", ".png"))
            ]
            rng.shuffle(files)
            split_idx = int(len(files) * self.train_ratio)
            chosen = files[:split_idx] if self.split == "train" else files[split_idx:]

            for fname in chosen:
                self.samples.append({
                    "path": os.path.join(class_dir, fname),
                    "label": label,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = cv2.imread(sample["path"])

        if image is None:
            print(f"WARN: unreadable image, substituting blank: {sample['path']}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)
        else:
            image = image.astype(np.float32) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, sample["label"]
