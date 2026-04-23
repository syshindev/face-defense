import os
import random
import re
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VideoSequenceDataset(Dataset):
    """Build frame sequences from pre-extracted face crops.

    Expects files named: {video_name}_f{frame_idx}.jpg
    Groups by video_name, sorts by frame_idx, builds consecutive sequences.

    Each sample = (sequence of N face crops, label).
    """

    def __init__(self, real_dir: str, fake_dir: str, split: str = "train",
                 seq_len: int = 10, image_size: int = 299,
                 train_ratio: float = 0.8, max_seqs_per_video: int = 3,
                 seed: int = 42, transform=None):
        self.seq_len = seq_len
        self.image_size = image_size
        self.transform = transform
        self.samples = []  # list of (list_of_paths, label)

        rng = random.Random(seed)

        for directory, label in [(real_dir, 0), (fake_dir, 1)]:
            if not os.path.isdir(directory):
                print(f"  SKIP {directory} (not found)")
                continue

            # Group files by video name
            videos = defaultdict(list)
            for fname in os.listdir(directory):
                if not fname.lower().endswith((".jpg", ".png")):
                    continue
                # Extract video name and frame index
                match = re.match(r"(.+)_f(\d+)\.", fname)
                if match:
                    video_name = match.group(1)
                    frame_idx = int(match.group(2))
                    videos[video_name].append((frame_idx, os.path.join(directory, fname)))

            # Sort each video's frames by index
            for vname in videos:
                videos[vname].sort(key=lambda x: x[0])

            # Split videos into train/test
            video_names = sorted(videos.keys())
            rng.shuffle(video_names)
            split_idx = int(len(video_names) * train_ratio)
            chosen = video_names[:split_idx] if split == "train" else video_names[split_idx:]

            # Build sequences
            for vname in chosen:
                frames = videos[vname]
                if len(frames) < seq_len:
                    continue
                # Slide window
                seqs_added = 0
                for start in range(0, len(frames) - seq_len + 1, seq_len):
                    if seqs_added >= max_seqs_per_video:
                        break
                    seq_paths = [frames[start + i][1] for i in range(seq_len)]
                    self.samples.append((seq_paths, label))
                    seqs_added += 1

        rng.shuffle(self.samples)
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"VideoSequenceDataset [{split}]: {len(self.samples)} sequences "
              f"(real={n_real}, fake={n_fake}), seq_len={seq_len}")

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, path):
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        if self.transform:
            return self.transform(img)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(img).permute(2, 0, 1).float()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        paths, label = self.samples[index]
        frames = [self._load_frame(p) for p in paths]
        sequence = torch.stack(frames, dim=0)  # (seq_len, C, H, W)
        return sequence, label
