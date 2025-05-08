import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_generator import DungeonPathfindingDataset
from models.gnn_model import PathfindingGNN
from training.trainer import GNNTrainer


def test_dataset_generation():
    """Test the generation of the dungeon pathfinding dataset."""
    # Create a small dataset for testing
    dataset = DungeonPathfindingDataset(
        root="test_dataset",
        num_samples=10,
        width_range=(10, 15),
        height_range=(10, 15),
        force_reload=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample data: {dataset[0]}")

    # Check that the dataset has the expected properties
    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
    assert dataset[0].x is not None, "Sample data has no node features"
    assert dataset[0].edge_index is not None, "Sample data has no edges"
    assert dataset[0].y is not None, "Sample data has no labels"

    # Visualize a sample
    visualize_sample(dataset[0])

    print("Dataset generation test passed!")


def test_training_pipeline():
    """Test the training pipeline with a small dataset."""
    # Create a small dataset for testing
    dataset = DungeonPathfindingDataset(
        root="test_dataset", num_samples=10, width_range=(10, 15), height_range=(10, 15)
    )

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Create model
    model = PathfindingGNN(
        node_features=dataset.num_features, hidden_channels=32, num_layers=2
    )

    # Create trainer
    trainer = GNNTrainer(model, checkpoint_dir="test_checkpoints")
    trainer.configure_optimizer(lr=0.01)

    # Define loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train for a few epochs
    print("Training model for 5 epochs...")
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, criterion, num_epochs=5, patience=10
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
    plt.savefig("training_history.png")
    plt.close()

    print("Training pipeline test passed!")


def visualize_sample(data):
    """Visualize a sample from the dataset."""
    # Create figure
    plt.figure(figsize=(10, 10))

    # Extract node positions and labels
    x_coords = data.x[:, 0].numpy()
    y_coords = data.x[:, 1].numpy()
    labels = data.y.numpy()

    # Scale coordinates
    x_coords = x_coords * 15
    y_coords = y_coords * 15

    # Draw edges
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        plt.plot(
            [x_coords[src], x_coords[dst]],
            [y_coords[src], y_coords[dst]],
            "k-",
            alpha=0.3,
        )

    # Draw nodes
    for i in range(data.num_nodes):
        if data.x[i, 3] == 1:  # Start
            plt.plot(x_coords[i], y_coords[i], "go", markersize=10)  # Green for start
        elif data.x[i, 4] == 1:  # Goal
            plt.plot(x_coords[i], y_coords[i], "bo", markersize=10)  # Blue for goal
        elif data.x[i, 2] == 1:  # Trap
            plt.plot(x_coords[i], y_coords[i], "ro", markersize=8)  # Red for trap
        elif data.x[i, 5] == 1:  # Guard
            plt.plot(x_coords[i], y_coords[i], "yo", markersize=8)  # Yellow for guard
        elif labels[i] > 0.5:  # On path
            plt.plot(x_coords[i], y_coords[i], "mo", markersize=8)  # Magenta for path
        else:
            plt.plot(
                x_coords[i], y_coords[i], "ko", markersize=6
            )  # Black for normal room

    plt.title("Sample Dungeon with Path")
    plt.axis("equal")
    plt.grid(True)

    # Save the figure
    plt.savefig("sample_dungeon.png")
    print("Sample dungeon visualization saved as 'sample_dungeon.png'")


if __name__ == "__main__":
    test_dataset_generation()
    test_training_pipeline()
