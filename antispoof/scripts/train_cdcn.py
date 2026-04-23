import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from antispoof.models.cdcn_model import CDCN
from antispoof.data.celeba_spoof_dataset import CelebASpoofDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train CDCN on CelebA-Spoof")
    parser.add_argument("--data_root", type=str, required=True, help="Path to CelebA-Spoof root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def generate_depth_label(label, map_size=32):
    # live(0) -> depth map all 1s, spoof(1) -> depth map all 0s
    if label == 0:
        return torch.ones(1, map_size, map_size)
    else:
        return torch.zeros(1, map_size, map_size)


def collate_fn(batch):
    # Custom collate to generate depth map labels
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Generate depth map targets
    depth_targets = torch.stack([generate_depth_label(l) for l in labels])

    return images, labels_tensor, depth_targets


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels, depth_targets) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        depth_targets = depth_targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        depth_pred = model(images)

        # Resize prediction to match target size
        if depth_pred.shape != depth_targets.shape:
            depth_pred = nn.functional.interpolate(
                depth_pred, size=depth_targets.shape[2:], mode="bilinear", align_corners=True
            )

        # Depth map loss
        loss = criterion(depth_pred, depth_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy from depth map mean
        with torch.no_grad():
            scores = depth_pred.mean(dim=[1, 2, 3])
            # high depth score -> live(0), low -> spoof(1)
            pred_labels = (scores < 0.5).long()
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {correct/total:.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, depth_targets in loader:
            images = images.to(device)
            labels = labels.to(device)
            depth_targets = depth_targets.to(device)

            depth_pred = model(images)

            if depth_pred.shape != depth_targets.shape:
                depth_pred = nn.functional.interpolate(
                    depth_pred, size=depth_targets.shape[2:], mode="bilinear", align_corners=True
                )

            loss = criterion(depth_pred, depth_targets)
            total_loss += loss.item()

            scores = depth_pred.mean(dim=[1, 2, 3])
            # high depth score -> live(0), low -> spoof(1)
            pred_labels = (scores < 0.5).long()
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    print("Loading training data...")
    train_dataset = CelebASpoofDataset(
        root=args.data_root, split="train", image_size=args.image_size
    )
    print(f"Training samples: {len(train_dataset)}")

    print("Loading test data...")
    test_dataset = CelebASpoofDataset(
        root=args.data_root, split="test", image_size=args.image_size
    )
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # Model
    model = CDCN(in_channels=3, theta=0.7).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, "cdcn_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (Acc: {best_acc:.4f})")

        # Save latest
        save_path = os.path.join(args.save_dir, "cdcn_latest.pth")
        torch.save(model.state_dict(), save_path)

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
