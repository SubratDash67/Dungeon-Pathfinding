import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_generator import DungeonPathfindingDataset
from models.gnn_model import PathfindingGNN
from training.trainer import GNNTrainer
from utils.logging_utils import Logger
from utils.custom_loss import RecallFocusedLoss, F1Loss
from utils.data_augmentation import DungeonDataAugmentation


def main():
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"enhanced_training_{timestamp}"

    # Create directories
    checkpoint_dir = f"checkpoints/{run_name}"
    log_dir = f"logs/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = Logger(log_dir=log_dir, experiment_name=run_name)
    logger.logger.info(f"Starting enhanced training run: {run_name}")

    # Create a larger dataset with more samples
    logger.logger.info("Generating dataset...")
    dataset = DungeonPathfindingDataset(
        root="enhanced_dataset",
        num_samples=300,  # Increased dataset size
        width_range=(10, 25),  # Wider range of dungeon sizes
        height_range=(10, 25),
        room_density=0.6,
        trap_probability=0.15,  # Increased trap probability
        guard_count_range=(2, 6),  # More guards
        force_reload=True,
    )

    # Apply data augmentation to increase dataset diversity
    logger.logger.info("Applying data augmentation...")
    augmenter = DungeonDataAugmentation()
    augmented_dataset = augmenter.generate_augmented_dataset(
        dataset, num_augmentations_per_sample=2
    )
    logger.logger.info(f"Dataset size after augmentation: {len(augmented_dataset)}")

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(augmented_dataset))
    val_size = int(0.15 * len(augmented_dataset))
    test_size = len(augmented_dataset) - train_size - val_size

    train_dataset = augmented_dataset[:train_size]
    val_dataset = augmented_dataset[train_size : train_size + val_size]
    test_dataset = augmented_dataset[train_size + val_size :]

    logger.logger.info(
        f"Dataset split: {train_size} train, {val_size} validation, {test_size} test"
    )

    # Create data loaders with smaller batch size for better generalization
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12)
    test_loader = DataLoader(test_dataset, batch_size=12)

    # Create a larger model with more capacity
    model = PathfindingGNN(
        node_features=dataset.num_features,
        hidden_channels=128,  # Increased from 64
        num_layers=4,  # Increased from 3
    )

    # Log model architecture
    logger.log_model_summary(model)

    # Create trainer with custom loss function
    trainer = EnhancedGNNTrainer(
        model,
        checkpoint_dir=checkpoint_dir,
        recall_focused=True,  # Use recall-focused loss
    )

    # Configure optimizer with lower learning rate and weight decay
    trainer.configure_optimizer(lr=0.0005, weight_decay=1e-4)

    # Train for more epochs with lower threshold for early stopping
    logger.logger.info("Training model with enhanced parameters...")
    start_time = time.time()

    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        num_epochs=100,  # Increased from 50
        patience=20,  # Increased from 15
        threshold_adjustment=0.3,  # Lower threshold for path prediction
    )

    training_time = time.time() - start_time
    logger.logger.info(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on test set
    logger.logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.logger.info(f"Test metrics: {test_metrics}")

    # Save final model
    final_checkpoint = {
        "epoch": trainer.epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_metrics": test_metrics,
    }

    torch.save(final_checkpoint, os.path.join(checkpoint_dir, "final_model.pt"))
    logger.logger.info(
        f"Final model saved at: {os.path.join(checkpoint_dir, 'final_model.pt')}"
    )

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "training_history.png"))

    logger.logger.info("Enhanced training complete!")
    logger.logger.info(
        f"Best model saved at: {os.path.join(checkpoint_dir, 'best_checkpoint.pt')}"
    )
    logger.logger.info(
        f"Final model saved at: {os.path.join(checkpoint_dir, 'final_model.pt')}"
    )

    # Print command to run evaluation
    print("\nTo evaluate the model, run:")
    print(
        f"python -u \"c:\\Users\\KIIT\\Desktop\\project\\RealTimeDungeonPathfindingOptimizer\\scripts\\enhanced_evaluation.py\" --checkpoint {os.path.join(checkpoint_dir, 'best_checkpoint.pt')} --num_test_samples 5"
    )


class EnhancedGNNTrainer(GNNTrainer):
    """Enhanced trainer with recall-focused loss and threshold adjustment"""

    def __init__(
        self, model, device=None, checkpoint_dir="checkpoints", recall_focused=True
    ):
        super().__init__(model, device, checkpoint_dir)
        self.recall_focused = recall_focused
        self.threshold = 0.5  # Default threshold

        # Initialize both loss functions
        self.recall_loss = RecallFocusedLoss(alpha=0.7, gamma=2.0)
        self.f1_loss = F1Loss()

    def configure_optimizer(self, lr=0.001, weight_decay=1e-5):
        # Use AdamW instead of Adam for better regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Use cosine annealing scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

    def train_epoch(self, train_loader, criterion=None):
        self.model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            path_scores, _ = self.model(data)

            # Calculate loss based on selected criterion
            if self.recall_focused:
                loss = self.recall_loss(path_scores, data.y)
            else:
                loss = self.f1_loss(path_scores, data.y)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * data.num_graphs

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader, criterion=None):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)

                # Forward pass
                path_scores, _ = self.model(data)

                # Calculate loss
                if self.recall_focused:
                    loss = self.recall_loss(path_scores, data.y)
                else:
                    loss = self.f1_loss(path_scores, data.y)

                total_loss += loss.item() * data.num_graphs

        return total_loss / len(val_loader.dataset)

    def train(
        self,
        train_loader,
        val_loader,
        criterion=None,
        num_epochs=100,
        patience=20,
        threshold_adjustment=0.0,
    ):
        """Train with threshold adjustment for improved recall"""
        if self.optimizer is None:
            self.configure_optimizer()

        # Adjust threshold for predictions
        self.threshold = 0.5 - threshold_adjustment

        no_improve_count = 0
        start_time = time.time()

        for epoch in range(self.epoch, self.epoch + num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Print progress
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{self.epoch + num_epochs}, "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.6f}, Time: {elapsed:.2f}s"
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
        self.training_time = time.time() - start_time

        # Load best model
        self.load_checkpoint(best=True)
        return self.train_losses, self.val_losses

    def evaluate(self, test_loader):
        """Evaluate model with detailed metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)

                # Forward pass
                path_scores, _ = self.model(data)
                path_probs = torch.sigmoid(path_scores)

                # Apply threshold
                predictions = (path_probs > self.threshold).float()

                # Collect predictions and targets
                all_preds.append(predictions.cpu())
                all_targets.append(data.y.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        # Calculate metrics
        tp = (all_preds * all_targets).sum()
        fp = (all_preds * (1 - all_targets)).sum()
        fn = ((1 - all_preds) * all_targets).sum()
        tn = ((1 - all_preds) * (1 - all_targets)).sum()

        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "threshold": self.threshold,
        }


if __name__ == "__main__":
    main()
