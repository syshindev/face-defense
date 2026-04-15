import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from face_defense.data.ff_dataset import FFDataset, CSVFrameDataset, NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Train deepfake detector (FF++ folder mode or CSV mode)")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root (folder layout root or CSV data_root)")
    parser.add_argument("--train_csv", type=str, default=None, help="CSV mode: training labels CSV")
    parser.add_argument("--val_csv", type=str, default=None, help="CSV mode: validation labels CSV")
    parser.add_argument("--model", type=str, default="legacy_xception", choices=["legacy_xception", "xception41", "efficientnet_b4"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_suffix", type=str, default="", help="Suffix for checkpoint filename, e.g. '_v2'")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
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

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # Adjust image size per model
    if args.model == "efficientnet_b4":
        args.image_size = 380

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Dataset
    csv_mode = args.train_csv is not None
    print(f"Mode: {'CSV' if csv_mode else 'folder'}")

    if csv_mode:
        print("Loading training data from CSV...")
        train_dataset = CSVFrameDataset(
            csv_path=args.train_csv, data_root=args.data_root,
            image_size=args.image_size, transform=train_transform,
        )
        print(f"Training samples: {len(train_dataset)}")

        print("Loading validation data from CSV...")
        test_dataset = CSVFrameDataset(
            csv_path=args.val_csv, data_root=args.data_root,
            image_size=args.image_size, transform=test_transform,
        )
        print(f"Validation samples: {len(test_dataset)}")
    else:
        print("Loading training data (folder mode)...")
        train_dataset = FFDataset(
            root=args.data_root, split="train", image_size=args.image_size,
            transform=train_transform,
        )
        print(f"Training samples: {len(train_dataset)}")

        print("Loading test data (folder mode)...")
        test_dataset = FFDataset(
            root=args.data_root, split="test", image_size=args.image_size,
            transform=test_transform,
        )
        print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model = timm.create_model(args.model, pretrained=True, num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    # Inverse-frequency class weights computed from the training split
    if csv_mode:
        train_labels = train_dataset.labels
    else:
        train_labels = np.array([s["label"] for s in train_dataset.samples])
    n_real = int((train_labels == 0).sum())
    n_fake = int((train_labels == 1).sum())
    total = n_real + n_fake
    w_real = total / (2 * max(n_real, 1))
    w_fake = total / (2 * max(n_fake, 1))
    class_weights = torch.tensor([w_real, w_fake], device=device)
    print(f"Class counts: real={n_real} fake={n_fake} | weights=[{w_real:.3f}, {w_fake:.3f}]")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
            save_path = os.path.join(args.save_dir, f"{args.model}{args.ckpt_suffix}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (Acc: {best_acc:.4f})")

        # Save latest
        save_path = os.path.join(args.save_dir, f"{args.model}{args.ckpt_suffix}_latest.pth")
        torch.save(model.state_dict(), save_path)

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
