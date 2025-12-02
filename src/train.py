import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.model import MultimodalEngagementNet
from src.dataset import MultiModalDataset
from src.db_client import users_df, movie_ids

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(loader):

        # Move tensors to device
        for key in batch:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()

        outputs = model(batch)
        targets = batch["target"]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"\nEpoch Completed — Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------------------------------
    # Build Dataset and DataLoader
    # ---------------------------------------
    users = list(users_df["user_id"])
    movies = movie_ids

    dataset = MultiModalDataset(users, movies, length=2000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ---------------------------------------
    # Initialize Model
    # ---------------------------------------
    model = MultimodalEngagementNet().to(device)

    # Loss: MSE between predicted score and synthetic target
    criterion = nn.MSELoss()

    # Optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---------------------------------------
    # Train for 1 epoch (as required)
    # ---------------------------------------
    train_one_epoch(model, loader, optimizer, criterion, device)

    # Optional: save the model
    torch.save(model.state_dict(), "model_checkpoint.pth")
    print("\nModel saved as model_checkpoint.pth")
