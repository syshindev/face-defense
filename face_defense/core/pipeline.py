import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from face_defense.core.base_detector import BaseDetector, DetectorResult

@dataclass
class PipelineResult:
    # Final verdict from the full defense pipeline
    is_real: bool
    confidence: float
    attack_type: Optional[str] = None
    stage_results: Dict[str, List[DetectorResult]] = field(default_factory=dict)
    exited_early: bool = False
    exit_stage: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class StageConfig:
    # Configuration for a single pipeline stage
    name: str
    detectors: List[BaseDetector]
    weights: List[float]
    early_exit: bool = False
    early_exit_threshold: float = 0.15


class DefensePipeline:
    # Multi-stage cascade pipeline with optional early exit
    def __init__(self, config: DictConfig):
        self.config = config
        self.stages: List[StageConfig] = []
        self.fusion_method: str = config.get("fusion", {}).get("method", "weighted_average")
        self.final_threshold: float = config.get("fusion", {}).get("final_threshold", 0.5)

    
    def add_stage(self, stage: StageConfig):
        self.stages.append(stage)

    
    def run(self, face_image: np.ndarray) -> PipelineResult:
        start = time.perf_counter()
        all_stage_results: Dict[str, List[DetectorResult]] = {}
        all_scores: List[float] = []
        all_weights: List[float] = []

        for stage in self.stages:
            results = [det.predict(face_image) for det in stage.detectors]
            all_stage_results[stage.name] = results

            # Weighted score for this stage
            stage_score = sum(r.score * w for r, w in zip(results, stage.weights)) / max(sum(stage.weights), 1e-8)
            all_scores.append(stage_score)
            all_weights.append(1.0)

            # Early exit check
            if stage.early_exit and stage_score < stage.early_exit_threshold:
                attack = self._pick_attack_type(results)
                elapsed = (time.perf_counter() - start) * 1000
                return PipelineResult(
                    is_real=False,
                    confidence=1.0 - stage_score,
                    attack_type=attack,
                    stage_results=all_stage_results,
                    exited_early=True,
                    exit_stage=stage.name,
                    latency_ms=elapsed,
                )
        
        # Final fusion
        final_score = self._fuse_scores(all_scores, all_weights)
        is_real = final_score >= self.final_threshold
        attack = None if is_real else self._pick_attack_type_from_stages(all_stage_results)
        elapsed = (time.perf_counter() - start) * 1000

        return PipelineResult(
            is_real=is_real,
            confidence=final_score if is_real else 1.0 - final_score,
            attack_type=attack,
            stage_results=all_stage_results,
            latency_ms=elapsed,
        )
    

    def _fuse_scores(self, scores: List[float], weights: List[float]) -> float:
        total_w = sum(weights)
        if total_w == 0:
            return 0.5
        return sum(s * w for s, w in zip(scores, weights)) / total_w
    
    def _pick_attack_type(self, results: List[DetectorResult]) -> Optional[str]:
        fake_results = [r for r in results if not r.is_real and r.attack_type]
        if not fake_results:
            return None
        return min(fake_results, key=lambda r: r.score).attack_type
    
    def _pick_attack_type_from_stages(self, stage_results: Dict[str, List[DetectorResult]]) -> Optional[str]:
        all_results = [r for results in stage_results.values() for r in results]
        return self._pick_attack_type(all_results)
    