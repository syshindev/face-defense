from typing import List

import numpy as np


class ScoreFusion:
    # Fuse scores from multiple detectors

    def __init__(self, method: str = "weighted_average"):
        self.method = method

    def fuse(self, scores: List[float], weights: List[float]) -> float:
        # Combine multiple scores into a single realness score
        if self.method == "weighted_average":
            total_w = sum(weights)
            if total_w == 0:
                return 0.5
            return sum(s * w for s, w in zip(scores, weights)) / total_w

        elif self.method == "voting":
            votes = sum(1 for s in scores if s >= 0.5)
            return votes / max(len(scores), 1)

        return sum(scores) / max(len(scores), 1)
