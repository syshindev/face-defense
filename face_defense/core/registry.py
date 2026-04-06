from typing import Callable, Dict, Type
from face_defense.core.base_detector import BaseDetector


class Registry:
    # Registry that maps string names to detector classes
    def __init__(self):
        self._registry: Dict[str, Type[BaseDetector]] = {}

    
    def register(self, name: str) -> Callable:
        def wrapper(cls: Type[BaseDetector]) -> Type[BaseDetector]:
            if name in self._registry:
                raise ValueError(f"Detector '{name}' is already registered")
            self._registry[name] = cls
            return cls
        return wrapper
    

    def build(self, name: str, config) -> BaseDetector:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Detector '{name}' not found, available: [{available}]")
        return self._registry[name](config)
    

    def list_available(self):
        return sorted(self._registry.keys())
    

DETECTOR_REGISTRY = Registry()
