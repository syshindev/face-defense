from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

class Trainer:
    # Generic training loop for face defense models
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.get("device", "cuda:0"))
        self.epochs = config.get("epochs", 50)
        self.lr = config.get("lr", 0.0001)

    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

            if val_loader is not None:
                self._validate(model, val_loader, epoch)

    def _validate(self, model: nn.Module, val_loader: DataLoader, epoch: int):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / max(total, 1)
        print(f"Val Accuracy: {acc:.4f}")
        