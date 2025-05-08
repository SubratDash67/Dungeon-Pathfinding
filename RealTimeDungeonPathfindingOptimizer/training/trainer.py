import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt


class GNNTrainer:
    def __init__(self, model, device=None, checkpoint_dir="checkpoints"):
        self.model = model
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.best_loss = float("inf")
        self.epoch = 0
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []

    def configure_optimizer(self, lr=0.001, weight_decay=1e-5):
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self, train_loader, criterion):
        self.model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            path_scores, _ = self.model(data)

            # Calculate loss
            loss = criterion(path_scores, data.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.num_graphs

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)

                # Forward pass
                path_scores, _ = self.model(data)

                # Calculate loss
                loss = criterion(path_scores, data.y)

                total_loss += loss.item() * data.num_graphs

        return total_loss / len(val_loader.dataset)

    def train(self, train_loader, val_loader, criterion, num_epochs=100, patience=10):
        """Train the model with early stopping."""
        if self.optimizer is None:
            self.configure_optimizer()

        no_improve_count = 0
        start_time = time.time()

        for epoch in range(self.epoch, self.epoch + num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader, criterion)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Print progress
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{self.epoch + num_epochs}, "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Time: {elapsed:.2f}s"
            )

            # Check for improvement
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                no_improve_count = 0
            else:
                no_improve_count += 1
                self.save_checkpoint(epoch)

            # Early stopping
            if no_improve_count >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

        self.epoch = epoch + 1
        self.plot_training_history()

        # Load best model
        self.load_checkpoint(best=True)
        return self.train_losses, self.val_losses

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        # Save regular checkpoint
        torch.save(
            checkpoint, os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        )

        # Save best checkpoint
        if is_best:
            torch.save(
                checkpoint, os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            )

    def load_checkpoint(self, path=None, best=False):
        """Load model checkpoint."""
        if path is None:
            if best:
                path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            else:
                path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")

        if not os.path.exists(path):
            print(f"Checkpoint file {path} does not exist. Starting from scratch.")
            return False

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint["best_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return True

    def plot_training_history(self):
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.checkpoint_dir, "training_history.png"))
        plt.close()

    def predict(self, data):
        """Make predictions with the model."""
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            path_scores, edge_scores = self.model(data)

        return path_scores, edge_scores
