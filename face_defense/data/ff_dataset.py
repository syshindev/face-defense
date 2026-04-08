import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Multi-class labels for FF++ dataset
CLASS_MAP = {
    "original": 0,
    "Deepfakes": 1,
    "Face2Face": 2,
    "FaceSwap": 3,
    "NeuralTextures": 4,
    "FaceShifter": 5,
    "DeepFakeDetection": 6,
}

NUM_CLASSES = len(CLASS_MAP)


class FFDataset(Dataset):
    # FaceForensics++ dataset loader (extracted frames)
    # Folder structure: {class_name}/*.jpg
    # Labels: 0=real, 1-6=fake types

    def __init__(self, root: str, split: str = "train", image_size: int = 299,
                 transform=None, train_ratio: float = 0.8):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.train_ratio = train_ratio
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        # Scan each class folder for images
        for class_name, label in CLASS_MAP.items():
            class_dir = os.path.join(self.root, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".png")):
                    self.samples.append({
                        "path": os.path.join(class_dir, fname),
                        "label": label,
                    })

        # Split into train/test
        all_samples = sorted(self.samples, key=lambda x: x["path"])
        split_idx = int(len(all_samples) * self.train_ratio)

        if self.split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

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

        # Normalize and convert to CHW tensor
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, sample["label"]
