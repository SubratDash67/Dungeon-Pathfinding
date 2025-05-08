import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, DataLoader

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dungeon_graphs import DungeonGenerator
from models.gnn_model import PathfindingGNN, DungeonGraphToData
from training.trainer import GNNTrainer
from training.checkpoint import CheckpointManager


def generate_synthetic_data(num_samples=100, min_size=10, max_size=20):
    """Generate synthetic data for testing the training pipeline."""
    dataset = []
    converter = DungeonGraphToData()

    for i in range(num_samples):
        # Random dungeon size
        width = np.random.randint(min_size, max_size)
        height = np.random.randint(min_size, max_size)

        # Generate dungeon
        generator = DungeonGenerator(
            width=width,
            height=height,
            room_density=0.6,
            trap_probability=0.1,
            guard_count=3,
        )
        dungeon = generator.generate()

        # Convert to PyTorch Geometric Data
        data, node_map = converter.convert(dungeon)

        # Add synthetic path labels (for now, just random)
        # In a real scenario, these would be computed from optimal paths
        y = torch.rand(data.num_nodes, 1)
        data.y = y

        dataset.append(data)

    return dataset


def test_trainer():
    """Test the GNN trainer with synthetic data."""
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_data(num_samples=50)

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Create model
    model = PathfindingGNN(node_features=6, hidden_channels=32, num_layers=2)

    # Create trainer
    trainer = GNNTrainer(model, checkpoint_dir="test_checkpoints")
    trainer.configure_optimizer(lr=0.001)

    # Define loss function (MSE for regression task)
    criterion = nn.MSELoss()

    # Train for a few epochs
    print("Training model for 5 epochs...")
    trainer.train(train_loader, val_loader, criterion, num_epochs=5, patience=10)

    # Test checkpoint loading
    print("Testing checkpoint loading...")
    trainer.load_checkpoint(best=True)

    # Test prediction
    print("Testing prediction...")
    sample_data = val_dataset[0]
    path_scores, _ = trainer.predict(sample_data)

    print(f"Prediction shape: {path_scores.shape}")
    print("Training test passed!")


def test_checkpoint_manager():
    """Test the checkpoint manager."""
    # Create checkpoint manager
    checkpoint_dir = "test_checkpoints_manager"
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir, max_to_keep=3)

    # Create dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())

    # Save checkpoints
    for epoch in range(5):
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": 1.0 - epoch * 0.1,  # Dummy metric that improves
        }

        # Mark every other checkpoint as best
        is_best = epoch % 2 == 0

        metrics = {"loss": 1.0 - epoch * 0.1, "accuracy": 0.5 + epoch * 0.1}

        manager.save(state, is_best=is_best, metrics=metrics)

    # Load latest checkpoint
    latest = manager.load_latest()
    assert latest["epoch"] == 4, f"Expected epoch 4, got {latest['epoch']}"

    # Load best checkpoint
    best = manager.load_best()
    assert best["epoch"] == 4, f"Expected epoch 4, got {best['epoch']}"

    # Check checkpoint info
    info = manager.get_checkpoint_info()
    assert len(info) == 3, f"Expected 3 checkpoints, got {len(info)}"

    # Test resume or restart
    new_model = nn.Linear(10, 1)
    new_optimizer = torch.optim.Adam(new_model.parameters())

    start_epoch, best_metric = manager.resume_or_restart(new_model, new_optimizer)

    assert start_epoch == 5, f"Expected start_epoch 5, got {start_epoch}"

    print("Checkpoint manager test passed!")


if __name__ == "__main__":
    test_trainer()
    test_checkpoint_manager()
