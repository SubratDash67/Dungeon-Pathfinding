import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_generator import DungeonPathfindingDataset
from models.gnn_model import PathfindingGNN
from training.trainer import GNNTrainer
from utils.logging_utils import Logger


def main():
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"full_training_{timestamp}"

    # Create directories
    checkpoint_dir = f"checkpoints/{run_name}"
    log_dir = f"logs/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = Logger(log_dir=log_dir)
    logger.logger.info(f"Starting full training run: {run_name}")

    # Create a dataset for full training
    logger.logger.info("Generating dataset...")
    dataset = DungeonPathfindingDataset(
        root="full_dataset",
        num_samples=200,  # Larger dataset
        width_range=(10, 20),
        height_range=(10, 20),
        force_reload=True,
    )

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size : train_size + val_size]
    test_dataset = dataset[train_size + val_size :]

    logger.logger.info(
        f"Dataset split: {train_size} train, {val_size} validation, {test_size} test"
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Create model
    model = PathfindingGNN(
        node_features=dataset.num_features, hidden_channels=64, num_layers=3
    )

    # Log model architecture
    logger.log_model_summary(model)

    # Create trainer
    trainer = GNNTrainer(model, checkpoint_dir=checkpoint_dir)
    trainer.configure_optimizer(lr=0.001, weight_decay=1e-5)

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()

    # Train for more epochs
    logger.logger.info("Training model for 50 epochs...")
    start_time = time.time()

    train_losses, val_losses = trainer.train(
        train_loader, val_loader, criterion, num_epochs=50, patience=15
    )

    training_time = time.time() - start_time
    logger.logger.info(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on test set
    logger.logger.info("Evaluating on test set...")
    test_loss = trainer.validate(test_loader, criterion)
    logger.logger.info(f"Test loss: {test_loss:.6f}")

    # Save final model
    final_checkpoint = {
        "epoch": trainer.epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_loss": test_loss,
    }

    torch.save(final_checkpoint, os.path.join(checkpoint_dir, "final_model.pt"))
    logger.logger.info(
        f"Final model saved at: {os.path.join(checkpoint_dir, 'final_model.pt')}"
    )

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.axhline(y=test_loss, color="r", linestyle="-", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "training_history.png"))

    logger.logger.info("Full training complete!")
    logger.logger.info(
        f"Best model saved at: {os.path.join(checkpoint_dir, 'best_checkpoint.pt')}"
    )
    logger.logger.info(
        f"Final model saved at: {os.path.join(checkpoint_dir, 'final_model.pt')}"
    )


if __name__ == "__main__":
    main()
