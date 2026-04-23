import argparse
import os
import subprocess
import sys
import time
import urllib.request

STYLEGAN3_REPO = "https://github.com/NVlabs/stylegan3.git"
PRETRAINED = {
    # NVlabs NVIDIA research, public download
    "ffhq-1024-t": (
        "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"
        "versions/1/files/stylegan3-t-ffhq-1024x1024.pkl"
    ),
    "ffhq-1024-r": (
        "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"
        "versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"
    ),
    "ffhqu-256-t": (
        "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"
        "versions/1/files/stylegan3-t-ffhqu-256x256.pkl"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate StyleGAN3 faces locally using NVlabs pretrained models"
    )
    parser.add_argument("--out_dir", type=str, default="data/extra/stylegan")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--model", type=str, default="ffhq-1024-t",
                        choices=list(PRETRAINED.keys()),
                        help="Pretrained variant (1024 is slower but higher quality)")
    parser.add_argument("--repo_dir", type=str, default="third_party/stylegan3")
    parser.add_argument("--weights_dir", type=str, default="third_party/stylegan3_weights")
    parser.add_argument("--trunc", type=float, default=0.7,
                        help="Truncation psi (lower = more typical, higher = more diverse)")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    return parser.parse_args()


def ensure_repo(repo_dir):
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        return
    os.makedirs(os.path.dirname(repo_dir) or ".", exist_ok=True)
    print(f"Cloning {STYLEGAN3_REPO} -> {repo_dir}")
    subprocess.check_call(["git", "clone", "--depth", "1", STYLEGAN3_REPO, repo_dir])


def ensure_weights(weights_dir, model):
    os.makedirs(weights_dir, exist_ok=True)
    url = PRETRAINED[model]
    path = os.path.join(weights_dir, os.path.basename(url))
    if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
        return path
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ensure_repo(args.repo_dir)
    weights_path = ensure_weights(args.weights_dir, args.model)

    sys.path.insert(0, os.path.abspath(args.repo_dir))

    # Force pure-PyTorch fallback for ALL custom ops — skip CUDA compilation
    from torch_utils.ops import bias_act, upfirdn2d, filtered_lrelu
    bias_act._init = lambda: False
    upfirdn2d._init = lambda: False
    filtered_lrelu._init = lambda: False

    import pickle
    import numpy as np
    import torch
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(weights_path, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device).eval()

    label_dim = getattr(G, "c_dim", 0)
    generated = 0
    skipped = 0
    start = time.time()

    while generated + skipped < args.count:
        batch = min(args.batch, args.count - generated - skipped)
        seeds = list(range(args.seed_start + generated + skipped,
                           args.seed_start + generated + skipped + batch))

        # Skip any existing files in this batch
        targets = []
        for s in seeds:
            p = os.path.join(args.out_dir, f"stylegan_{s:06d}.jpg")
            if os.path.exists(p):
                skipped += 1
            else:
                targets.append((s, p))
        if not targets:
            continue

        z = torch.randn([len(targets), G.z_dim], device=device)
        c = torch.zeros([len(targets), label_dim], device=device) if label_dim > 0 else None
        with torch.no_grad():
            imgs = G(z, c, truncation_psi=args.trunc, noise_mode="const")
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        for (seed, path), img in zip(targets, imgs):
            Image.fromarray(img).save(path, quality=92)
            generated += 1

        if generated % 50 == 0:
            elapsed = time.time() - start
            rate = generated / max(elapsed, 1)
            eta = (args.count - generated - skipped) / max(rate, 1e-6)
            print(f"  generated={generated} skipped={skipped} "
                  f"rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)

    total = time.time() - start
    print(f"\nDone. generated={generated} skipped={skipped} total={total:.0f}s")


if __name__ == "__main__":
    main()
