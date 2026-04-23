import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from antispoof.models.cdcn_model import CDCN


class NUAADataset(Dataset):
    # NUAA dataset loader using txt split files
    # ClientRaw = real (label 0), ImposterRaw = spoof (label 1)

    def __init__(self, data_root, client_txt, imposter_txt, image_size=256):
        self.data_root = data_root
        self.image_size = image_size
        self.samples = []

        # Parse client (real) paths
        with open(os.path.join(data_root, client_txt), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Convert kaggle path to local path
                fname = line.split("raw/")[-1]
                path = os.path.join(data_root, fname)
                if os.path.exists(path):
                    self.samples.append({"path": path, "label": 0})

        # Parse imposter (spoof) paths
        with open(os.path.join(data_root, imposter_txt), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname = line.split("raw/")[-1]
                path = os.path.join(data_root, fname)
                if os.path.exists(path):
                    self.samples.append({"path": path, "label": 1})

        print(f"Loaded {len(self.samples)} samples "
              f"({sum(1 for s in self.samples if s['label']==0)} real, "
              f"{sum(1 for s in self.samples if s['label']==1)} spoof)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["path"])
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)

        # Depth map target: real -> ones, spoof -> zeros
        depth_size = self.image_size // 4
        if sample["label"] == 0:
            depth = torch.ones(1, depth_size, depth_size)
        else:
            depth = torch.zeros(1, depth_size, depth_size)

        return tensor, depth, sample["label"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained CDCN
    model = CDCN(in_channels=3, theta=0.7)
    state_dict = torch.load("checkpoints/cdcn_best.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Pretrained CDCN loaded")

    data_root = "C:/Users/gmission/Downloads/test-data"

    # Train dataset
    train_dataset = NUAADataset(
        data_root=data_root,
        client_txt="client_train_raw.txt",
        imposter_txt="imposter_train_raw.txt",
    )

    # Test dataset
    test_dataset = NUAADataset(
        data_root=data_root,
        client_txt="client_test_raw.txt",
        imposter_txt="imposter_test_raw.txt",
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Fine-tuning settings (lower lr than training from scratch)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    epochs = 10

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for images, depth_targets, labels in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            images = images.to(device)
            depth_targets = depth_targets.to(device)

            optimizer.zero_grad()
            depth_pred = model(images)

            # Resize target to match prediction size
            if depth_pred.shape != depth_targets.shape:
                depth_targets = nn.functional.interpolate(
                    depth_targets, size=depth_pred.shape[2:], mode="bilinear", align_corners=True
                )

            loss = criterion(depth_pred, depth_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, depth_targets, labels in test_loader:
                images = images.to(device)
                labels = labels.numpy()
                depth_pred = model(images)
                scores = depth_pred.mean(dim=[1, 2, 3]).cpu().numpy()

                for score, label in zip(scores, labels):
                    pred = 0 if score >= 0.5 else 1
                    if pred == label:
                        correct += 1
                    total += 1

        acc = correct / total
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "checkpoints/cdcn_nuaa_best.pth")
            print(f"  -> Best model saved (Acc: {best_acc:.4f})")

        torch.save(model.state_dict(), "checkpoints/cdcn_nuaa_latest.pth")

    print(f"\nFine-tuning complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
