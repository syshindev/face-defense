import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_defense.evaluation.metrics import compute_auc, compute_eer


KAGGLE_PREFIX = "/kaggle/input/ff-andcelebdf-frame-dataset-by-wish/"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark deepfake detector on ff-celebdf-frames")
    parser.add_argument("--data_root", type=str, default="data/ff-celebdf-frames")
    parser.add_argument("--csv", type=str, default="data/ff-celebdf-frames/test_labels.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/legacy_xception_best.pth")
    parser.add_argument("--model", type=str, default="legacy_xception")
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int, default=32)
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

    print("\n=== Per-domain ===")
    ffpp_mask = np.array([s.startswith("ffpp") for s in sources])
    celeb_mask = np.array([s.startswith("celeb") for s in sources])
    report("ffpp (in-domain)", labels[ffpp_mask], scores[ffpp_mask])
    report("celebdf (cross-domain)", labels[celeb_mask], scores[celeb_mask])

    print("\n=== Per-source ===")
    for src in np.unique(sources):
        mask = sources == src
        report(src, labels[mask], scores[mask])


if __name__ == "__main__":
    main()
