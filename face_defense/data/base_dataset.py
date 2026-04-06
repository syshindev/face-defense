from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig


class BaseDataset(Dataset, ABC):
    # Abstract base class for all face defense datasets
    def __init__(self, config: DictConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.samples: List[Dict[str, Any]] = []

    @abstractmethod
    def load_samples(self):
        # Load dataset file paths and labels into self.samples
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        # Return (image, label, metadata) for a given index, label: 0 = fake/spoof, 1 = real
        pass

    def __len__(self) -> int:
        return len(self.samples)
    