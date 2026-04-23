import os
import random
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REAL_CLASSES = {"original"}
FAKE_CLASSES = {
    "Deepfakes", "Face2Face", "FaceSwap",
    "NeuralTextures", "FaceShifter", "DeepFakeDetection",
}

NUM_CLASSES = 2  # 0=real, 1=fake

KAGGLE_PREFIX = "/kaggle/input/ff-andcelebdf-frame-dataset-by-wish/"


def _load_and_preprocess(path, image_size, transform):
    image = cv2.imread(path)
    if image is None:
        print(f"WARN: unreadable image, substituting blank: {path}")
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    if transform:
        return transform(image)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(image).permute(2, 0, 1).float()


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
        image = _load_and_preprocess(sample["path"], self.image_size, self.transform)
        return image, sample["label"]


class CSVFrameDataset(Dataset):
    # CSV-driven dataset for ff-celebdf-frames-style layouts.
    # CSV columns: filepath, source, label, video, frame, split, det_box, det_prob
    # Paths in CSV are Kaggle-absolute; translated to data_root at load time.

    def __init__(self, csv_path: str, data_root: str, image_size: int = 299,
                 transform=None, kaggle_prefix: str = KAGGLE_PREFIX):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        self.kaggle_prefix = kaggle_prefix
        self.labels = self.df["label"].astype(int).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        rel = row["filepath"].replace(self.kaggle_prefix, "")
        path = os.path.join(self.data_root, rel)
        image = _load_and_preprocess(path, self.image_size, self.transform)
        return image, int(row["label"])
