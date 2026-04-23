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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from deepfake.data.video_sequence_dataset import VideoSequenceDataset
from shared.metrics import compute_auc


def parse_args():
    parser = argparse.ArgumentParser(description="Train temporal deepfake detector (XceptionNet + LSTM)")
    parser.add_argument("--real_dir", type=str, default="data/extra/ff_video_crops/real",
                        help="Directory of pre-extracted real face crops")
    parser.add_argument("--fake_dir", type=str, default="data/extra/ff_video_crops/fake",
                        help="Directory of pre-extracted fake face crops")
    parser.add_argument("--backbone", type=str, default="legacy_xception")
    parser.add_argument("--backbone_ckpt", type=str, default=None,
                        help="Pretrained backbone checkpoint (e.g. v3.4). If None, uses ImageNet pretrained.")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Freeze XceptionNet weights (train LSTM only)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--early_stop", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


class TemporalDeepfakeDetector(nn.Module):
    """XceptionNet (frozen feature extractor) + LSTM temporal head.

    Input: (batch, seq_len, C, H, W)
    Output: (batch, 2) logits
    """

    def __init__(self, backbone_name, hidden_dim=256, lstm_layers=2,
                 backbone_ckpt=None, freeze_backbone=True):
        super().__init__()

        # Backbone: XceptionNet without final classifier
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        if backbone_ckpt:
            state = torch.load(backbone_ckpt, map_location="cpu")
            # Load matching keys only (skip classifier head)
            backbone_state = {k: v for k, v in state.items()
                             if k in self.backbone.state_dict()
                             and v.shape == self.backbone.state_dict()[k].shape}
            self.backbone.load_state_dict(backbone_state, strict=False)
            print(f"Loaded {len(backbone_state)} backbone weights from {backbone_ckpt}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.freeze_backbone = freeze_backbone

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 299, 299)
            feat_dim = self.backbone(dummy).shape[1]
        print(f"Backbone feature dim: {feat_dim}")

        # Temporal head
        self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers=lstm_layers,
                           batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        # x: (batch, seq_len, C, H, W)
        B, T, C, H, W = x.shape

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                x = x.view(B * T, C, H, W)
                features = self.backbone(x)  # (B*T, feat_dim)
        else:
            x = x.view(B * T, C, H, W)
            features = self.backbone(x)

        features = features.view(B, T, -1)  # (B, T, feat_dim)
        lstm_out, _ = self.lstm(features)   # (B, T, hidden_dim)
        last_hidden = lstm_out[:, -1, :]     # (B, hidden_dim) — last timestep
        logits = self.classifier(last_hidden)  # (B, 2)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (sequences, labels) in enumerate(loader):
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(sequences)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
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
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            output = model(sequences)
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
    print(f"Backbone: {args.backbone}")

    # Dataset — uses pre-extracted face crops (fast I/O)
    print("Loading training sequences...")
    train_dataset = VideoSequenceDataset(
        real_dir=args.real_dir, fake_dir=args.fake_dir,
        split="train", seq_len=args.seq_len,
    )
    print("Loading validation sequences...")
    val_dataset = VideoSequenceDataset(
        real_dir=args.real_dir, fake_dir=args.fake_dir,
        split="test", seq_len=args.seq_len,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = TemporalDeepfakeDetector(
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        backbone_ckpt=args.backbone_ckpt,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # Class weights for real:fake imbalance
    n_real = sum(1 for _, l in train_dataset.samples if l == 0)
    n_fake = sum(1 for _, l in train_dataset.samples if l == 1)
    total_samples = n_real + n_fake
    w_real = total_samples / (2 * max(n_real, 1))
    w_fake = total_samples / (2 * max(n_fake, 1))
    class_weights = torch.tensor([w_real, w_fake], device=device)
    print(f"Class weights: real={w_real:.3f}, fake={w_fake:.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
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
            save_path = os.path.join(args.save_dir, "temporal_lstm_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (AUC: {best_auc:.4f})")
        else:
            epochs_since_improve += 1

        if args.early_stop > 0 and epochs_since_improve >= args.early_stop:
            print(f"\nEarly stop: no AUC improvement for {args.early_stop} epochs.")
            break

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
