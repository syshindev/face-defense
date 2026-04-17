import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_defense.evaluation.metrics import compute_auc, compute_eer


KAGGLE_PREFIX = "/kaggle/input/ff-andcelebdf-frame-dataset-by-wish/"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def source_category(source):
    s = source.lower()
    if s.startswith("ffpp") or s.startswith("celeb_fake") or s.startswith("celeb_real"):
        return "face_swap"
    if "stylegan" in s:
        return "gan"
    if "diffusion" in s or "sdxl" in s or "flux" in s:
        return "diffusion"
    if "ffhq" in s or s == "real":
        return "real"
    return "other"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark deepfake detector on ff-celebdf-frames")
    parser.add_argument("--data_root", type=str, default="data/ff-celebdf-frames")
    parser.add_argument("--csv", type=str, default="data/ff-celebdf-frames/test_labels.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/legacy_xception_best.pth")
    parser.add_argument("--model", type=str, default="legacy_xception")
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="If set, save ROC curve + score histogram plots here")
    parser.add_argument("--plot_tag", type=str, default="",
                        help="Suffix for plot filenames, e.g. '_v2'")
    parser.add_argument("--compare_checkpoint", type=str, default=None,
                        help="If set, evaluate a second checkpoint on the same data "
                             "and print a side-by-side per-category comparison")
    parser.add_argument("--compare_label", type=str, default="v2",
                        help="Short label for the comparison checkpoint (e.g. 'v2')")
    parser.add_argument("--main_label", type=str, default="v3",
                        help="Short label for the main checkpoint")
    return parser.parse_args()


def translate_path(kaggle_path: str, data_root: str) -> str:
    rel = kaggle_path.replace(KAGGLE_PREFIX, "")
    return os.path.join(data_root, rel)


def load_image(path: str, image_size: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def evaluate(model, df, data_root, image_size, batch_size, device):
    labels_all = []
    scores_all = []
    sources_all = []

    buffer_tensors = []
    buffer_labels = []
    buffer_sources = []
    missing = 0

    def flush():
        if not buffer_tensors:
            return
        batch = torch.from_numpy(np.stack(buffer_tensors)).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        scores_all.extend(probs.tolist())
        labels_all.extend(buffer_labels)
        sources_all.extend(buffer_sources)
        buffer_tensors.clear()
        buffer_labels.clear()
        buffer_sources.clear()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        path = translate_path(row["filepath"], data_root)
        img = load_image(path, image_size)
        if img is None:
            missing += 1
            continue
        buffer_tensors.append(img)
        buffer_labels.append(int(row["label"]))
        buffer_sources.append(str(row["source"]))
        if len(buffer_tensors) >= batch_size:
            flush()
    flush()

    if missing > 0:
        print(f"Warning: {missing} images missing")

    return np.array(labels_all), np.array(scores_all), np.array(sources_all)


def plot_combined_roc(splits, save_path):
    plt.figure(figsize=(7, 6))
    for name, labels, scores in splits:
        if len(labels) < 2 or len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = compute_auc(labels, scores)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Deepfake ROC — overall / per-domain")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_hist(labels, scores, save_path):
    plt.figure(figsize=(7, 5))
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="Real", color="green")
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="Fake", color="red")
    plt.xlabel("P(fake)")
    plt.ylabel("Count")
    plt.title("Score distribution (overall)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def report(name, labels, scores):
    if len(labels) == 0:
        print(f"[{name}] no samples")
        return
    preds = (scores >= 0.5).astype(int)
    acc = (preds == labels).mean()
    try:
        auc = compute_auc(labels, scores)
        eer = compute_eer(labels, scores)
    except Exception:
        auc = eer = float("nan")
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    print(f"[{name}] n={len(labels)} (real={n_real}, fake={n_fake}) "
          f"ACC={acc:.4f} AUC={auc:.4f} EER={eer:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Ckpt:   {args.checkpoint}")

    model = timm.create_model(args.model, pretrained=False, num_classes=2).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    df = pd.read_csv(args.csv)
    print(f"Loaded CSV: {len(df)} rows")
    print(df["source"].value_counts())

    labels, scores, sources = evaluate(
        model, df, args.data_root, args.image_size, args.batch_size, device
    )

    print("\n=== Overall ===")
    report("overall", labels, scores)

    print("\n=== Per-domain (legacy ffpp / celebdf) ===")
    ffpp_mask = np.array([s.startswith("ffpp") for s in sources])
    celeb_mask = np.array([s.startswith("celeb") for s in sources])
    report("ffpp", labels[ffpp_mask], scores[ffpp_mask])
    report("celebdf", labels[celeb_mask], scores[celeb_mask])

    categories = np.array([source_category(s) for s in sources])
    unique_cats = [c for c in ["face_swap", "gan", "diffusion", "real", "other"] if c in categories]
    if len(unique_cats) > 2:
        print("\n=== Per-category ===")
        for cat in unique_cats:
            mask = categories == cat
            report(cat, labels[mask], scores[mask])

    print("\n=== Per-source ===")
    for src in np.unique(sources):
        mask = sources == src
        report(src, labels[mask], scores[mask])

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        splits = [("overall", labels, scores)]
        if ffpp_mask.any():
            splits.append(("ffpp", labels[ffpp_mask], scores[ffpp_mask]))
        if celeb_mask.any():
            splits.append(("celebdf", labels[celeb_mask], scores[celeb_mask]))
        for cat in unique_cats:
            if cat == "real":
                continue
            mask = categories == cat
            if len(np.unique(labels[mask])) == 2:
                splits.append((cat, labels[mask], scores[mask]))
        roc_path = os.path.join(args.plot_dir, f"roc{args.plot_tag}.png")
        hist_path = os.path.join(args.plot_dir, f"score_hist{args.plot_tag}.png")
        plot_combined_roc(splits, roc_path)
        plot_score_hist(labels, scores, hist_path)
        print(f"\nSaved plots: {roc_path}, {hist_path}")

    if args.compare_checkpoint:
        print(f"\n=== Comparing checkpoints: {args.main_label} vs {args.compare_label} ===")
        model2 = timm.create_model(args.model, pretrained=False, num_classes=2).to(device)
        state2 = torch.load(args.compare_checkpoint, map_location=device)
        model2.load_state_dict(state2)
        model2.eval()
        labels2, scores2, sources2 = evaluate(
            model2, df, args.data_root, args.image_size, args.batch_size, device
        )
        categories2 = np.array([source_category(s) for s in sources2])

        def stats(lbl, scr):
            if len(lbl) == 0:
                return float("nan"), float("nan")
            try:
                auc = compute_auc(lbl, scr)
            except Exception:
                auc = float("nan")
            acc = ((scr >= 0.5).astype(int) == lbl).mean()
            return acc, auc

        print(f"{'Category':<14} {'N':>6} "
              f"{args.main_label+' ACC':>9} {args.main_label+' AUC':>9} "
              f"{args.compare_label+' ACC':>9} {args.compare_label+' AUC':>9}  Δ AUC")
        for cat in unique_cats:
            m = categories == cat
            a1, u1 = stats(labels[m], scores[m])
            m2 = categories2 == cat
            a2, u2 = stats(labels2[m2], scores2[m2])
            delta = u1 - u2 if not (np.isnan(u1) or np.isnan(u2)) else float("nan")
            print(f"{cat:<14} {int(m.sum()):>6} "
                  f"{a1:>9.4f} {u1:>9.4f} {a2:>9.4f} {u2:>9.4f}  {delta:+.4f}")
        a1, u1 = stats(labels, scores)
        a2, u2 = stats(labels2, scores2)
        delta = u1 - u2 if not (np.isnan(u1) or np.isnan(u2)) else float("nan")
        print(f"{'overall':<14} {len(labels):>6} "
              f"{a1:>9.4f} {u1:>9.4f} {a2:>9.4f} {u2:>9.4f}  {delta:+.4f}")


if __name__ == "__main__":
    main()
