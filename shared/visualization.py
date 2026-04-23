import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, save_path: str = None):
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b-", linewidth=2)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, save_path: str = None):
    # Plot confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fake", "Real"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_score_distribution(labels: np.ndarray, scores: np.ndarray, save_path: str = None):
    # Plot score distributions for real vs fake
    real_scores = scores[labels == 1]
    fake_scores = scores[labels == 0]

    plt.figure(figsize=(8, 6))
    plt.hist(real_scores, bins=50, alpha=0.6, label="Real", color="green")
    plt.hist(fake_scores, bins=50, alpha=0.6, label="Fake", color="red")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
