import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from antispoof.models.cdcn_model import CDCN
from shared.metrics import (
    compute_auc, compute_eer, compute_apcer, compute_bpcer, compute_acer,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = CDCN(in_channels=3, theta=0.7)
    state_dict = torch.load("checkpoints/cdcn_best.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # CelebA-Spoof test set
    test_root = "data/CelebA_Spoof/Data/test"
    labels = []
    scores = []

    for subject_id in tqdm(sorted(os.listdir(test_root)), desc="Subjects"):
        subject_dir = os.path.join(test_root, subject_id)
        if not os.path.isdir(subject_dir):
            continue

        for category, label in [("live", 1), ("spoof", 0)]:
            cat_dir = os.path.join(subject_dir, category)
            if not os.path.exists(cat_dir):
                continue
            for fname in os.listdir(cat_dir):
                if not fname.lower().endswith((".jpg", ".png")):
                    continue
                img_path = os.path.join(cat_dir, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                img = img.astype(np.float32) / 255.0
                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    depth_map = model(tensor)
                score = depth_map.mean().item()
                labels.append(label)
                scores.append(score)

    labels = np.array(labels)
    scores = np.array(scores)
    print(f"\nEvaluated: {len(labels)} images "
          f"({(labels == 1).sum()} real, {(labels == 0).sum()} spoof)")
    print(f"Real scores  - mean: {scores[labels == 1].mean():.4f}, "
          f"std: {scores[labels == 1].std():.4f}")
    print(f"Spoof scores - mean: {scores[labels == 0].mean():.4f}, "
          f"std: {scores[labels == 0].std():.4f}")

    # Determine score direction
    real_mean = scores[labels == 1].mean()
    spoof_mean = scores[labels == 0].mean()
    if real_mean < spoof_mean:
        print("\nScore inverted: applying 1 - score")
        scores = 1 - scores

    preds = (scores >= 0.5).astype(int)
    auc = compute_auc(labels, scores)
    eer = compute_eer(labels, scores)
    apcer = compute_apcer(labels, preds)
    bpcer = compute_bpcer(labels, preds)
    acer = compute_acer(apcer, bpcer)

    print(f"\nAUC:   {auc:.4f}")
    print(f"EER:   {eer:.4f}")
    print(f"APCER: {apcer:.4f}")
    print(f"BPCER: {bpcer:.4f}")
    print(f"ACER:  {acer:.4f}")


if __name__ == "__main__":
    main()
