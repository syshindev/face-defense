from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig


@dataclass
class DetectorResult:
    # Standardized output from any detector
    is_real: bool
    score: float
    attack_type: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    # Abstract base class that all detectors must implement
    def __init__(self, config: DictConfig):
        self.config = config
        self._device = config.get("device", "cpu")


    @abstractmethod
    def predict(self, face_image: np.ndarray) -> DetectorResult:
        # Run inference on a single aligned face image
        pass


    def predict_batch(self, images: List[np.ndarray]) -> List[DetectorResult]:
        # Run inference on a batch, Override for GPU batching
        return [self.predict(img) for img in images]


    @abstractmethod
    def get_name(self) -> str:
        # Return a unique identifier for this detector
        pass


    def get_attack_types(self) -> List[str]:
        # Return attack types this detector can catch
        return []


    @property
    def device(self) -> str:
        return self._device


    @device.setter
    def device(self, value: str):
        self._device = value
