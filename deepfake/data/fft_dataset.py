import os
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 2


def compute_fft_spectrum(img_bgr, image_size=299):
    """Convert face image to FFT magnitude spectrum.

    GAN/deepfake artifacts leave distinctive patterns in the frequency domain
    that survive video compression better than spatial artifacts.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (image_size, image_size))

    # 2D FFT → shift zero-frequency to center
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Log magnitude spectrum (compress dynamic range)
    magnitude = np.log1p(np.abs(fshift))

    # Normalize to 0-1
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    # Convert to 3-channel float32 for CNN
    spectrum = np.stack([magnitude, magnitude, magnitude], axis=2)
    return spectrum.astype(np.float32)


class FFTDataset(Dataset):
    """Dataset that converts face images to FFT spectrums.

    Reads images from directories, applies FFT, returns spectrum + label.
    Compatible with existing train_deepfake.py CSV format.
    """

    def __init__(self, csv_path: str, data_root: str, image_size: int = 299,
                 kaggle_prefix: str = "/kaggle/input/ff-andcelebdf-frame-dataset-by-wish/"):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.image_size = image_size
        self.kaggle_prefix = kaggle_prefix
        self.labels = self.df["label"].astype(int).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        filepath = row["filepath"]
        if isinstance(filepath, str) and filepath.startswith(self.kaggle_prefix):
            filepath = filepath[len(self.kaggle_prefix):]
        path = os.path.join(self.data_root, filepath)

        img = cv2.imread(path)
        if img is None:
            # Blank spectrum
            spectrum = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        else:
            spectrum = compute_fft_spectrum(img, self.image_size)

        # Normalize with ImageNet stats (spectrum is already 0-1)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        spectrum = (spectrum - mean) / std

        tensor = torch.from_numpy(spectrum).permute(2, 0, 1).float()
        return tensor, int(row["label"])
