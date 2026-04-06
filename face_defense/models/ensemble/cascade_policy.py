
class CascadePolicy:
    # Early-exit policy for cascade pipeline stages

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def should_exit(self, stage_score: float) -> bool:
        # Return True if the score is low enough to classify as spoof
        return stage_score < self.threshold
