import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_defense.data.fft_dataset import FFTDataset, NUM_CLASSES
from face_defense.evaluation.metrics import compute_auc


def parse_args():
    parser = argparse.ArgumentParser(description="Train deepfake detector on FFT frequency spectrums")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--model", type=str, default="legacy_xception")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--early_stop", type=int, default=7)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (spectrums, labels) in enumerate(loader):
        spectrums = spectrums.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(spectrums)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {correct/total:.4f}")

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for spectrums, labels in loader:
            spectrums = spectrums.to(device)
            labels = labels.to(device)

            output = model(spectrums)
            loss = criterion(output, labels)

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs = F.softmax(output, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    try:
        auc = compute_auc(np.array(all_labels), np.array(all_scores))
    except ValueError:
        auc = float("nan")
    return avg_loss, accuracy, auc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Mode: FFT frequency domain")

    # Dataset
    print("Loading training FFT data...")
    train_dataset = FFTDataset(
        csv_path=args.train_csv, data_root=args.data_root,
        image_size=args.image_size,
    )
    print(f"Training samples: {len(train_dataset)}")

    print("Loading validation FFT data...")
    val_dataset = FFTDataset(
        csv_path=args.val_csv, data_root=args.data_root,
        image_size=args.image_size,
    )
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model — fresh pretrained (NOT using v3.4 weights, frequency domain is different input)
    model = timm.create_model(args.model, pretrained=True, num_classes=NUM_CLASSES).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    train_labels = train_dataset.labels
    n_real = int((train_labels == 0).sum())
    n_fake = int((train_labels == 1).sum())
    total_n = n_real + n_fake
    w_real = total_n / (2 * max(n_real, 1))
    w_fake = total_n / (2 * max(n_fake, 1))
    class_weights = torch.tensor([w_real, w_fake], device=device)
    print(f"Class counts: real={n_real} fake={n_fake} | weights=[{w_real:.3f}, {w_fake:.3f}]")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)

    best_auc = 0.0
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
        )
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_since_improve = 0
            save_path = os.path.join(args.save_dir, "fft_xception_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (AUC: {best_auc:.4f})")
        else:
            epochs_since_improve += 1

        save_path = os.path.join(args.save_dir, "fft_xception_latest.pth")
        torch.save(model.state_dict(), save_path)

        if args.early_stop > 0 and epochs_since_improve >= args.early_stop:
            print(f"\nEarly stop: no AUC improvement for {args.early_stop} epochs.")
            break

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
