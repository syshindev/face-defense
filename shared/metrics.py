import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    # Area Under ROC Curve
    return roc_auc_score(labels, scores)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    # Equal Error Rate: where FAR == FRR
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(fpr[idx])


def compute_acer(apcer: float, bpcer: float) -> float:
    # Average Classification Error Rate
    return (apcer + bpcer) / 2.0


def compute_apcer(labels: np.ndarray, preds: np.ndarray) -> float:
    # Attack Presentation Classification Error Rate
    # False acceptance of attacks
    attack_mask = labels == 0
    if attack_mask.sum() == 0:
        return 0.0
    return float((preds[attack_mask] == 1).sum() / attack_mask.sum())


def compute_bpcer(labels: np.ndarray, preds: np.ndarray) -> float:
    # Bona fide Presentation Classification Error Rate
    # False rejection of real faces
    real_mask = labels == 1
    if real_mask.sum() == 0:
        return 0.0
    return float((preds[real_mask] == 0).sum() / real_mask.sum())
