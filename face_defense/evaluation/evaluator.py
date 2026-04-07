import numpy as np
from typing import Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from face_defense.core.pipeline import DefensePipeline
from face_defense.evaluation.metrics import (
    compute_auc, compute_eer, compute_acer, compute_apcer, compute_bpcer,
)


class Evaluator:
    # Batch evaluation runner for the defense pipeline

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(self, pipeline: DefensePipeline, dataloader: DataLoader) -> Dict:
        all_labels = []
        all_scores = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            images, labels, metadata = batch

            for i in range(len(images)):
                image = images[i].numpy() if hasattr(images[i], "numpy") else images[i]
                result = pipeline.run(image)
                all_labels.append(labels[i].item() if hasattr(labels[i], "item") else labels[i])
                all_scores.append(result.score)

        labels = np.array(all_labels)
        scores = np.array(all_scores)
        preds = (scores >= self.threshold).astype(int)

        auc = compute_auc(labels, scores)
        eer = compute_eer(labels, scores)
        apcer = compute_apcer(labels, preds)
        bpcer = compute_bpcer(labels, preds)
        acer = compute_acer(apcer, bpcer)

        results = {
            "auc": auc,
            "eer": eer,
            "apcer": apcer,
            "bpcer": bpcer,
            "acer": acer,
            "num_samples": len(labels),
            "num_real": int((labels == 1).sum()),
            "num_fake": int((labels == 0).sum()),
        }

        print(f"AUC: {auc:.4f} | EER: {eer:.4f} | ACER: {acer:.4f}")
        print(f"APCER: {apcer:.4f} | BPCER: {bpcer:.4f}")
        print(f"Samples: {len(labels)} (Real: {results['num_real']}, Fake: {results['num_fake']})")

        return results
