import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_generator import DungeonPathfindingDataset
from data.dungeon_graphs import DungeonGenerator
from models.gnn_model import PathfindingGNN, DungeonGraphToData
from training.trainer import GNNTrainer
from utils.graph_utils import GraphUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the dungeon pathfinding GNN")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/dataset",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=5,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Number of hidden channels in the GNN",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of GNN layers"
    )
    return parser.parse_args()


def evaluate_model(model, dataset, num_samples, output_dir):
    """Evaluate the model on test samples and visualize predictions."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Select random test samples
    indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    # Evaluate each sample
    for i, idx in enumerate(indices):
        data = dataset[idx]

        # Make prediction
        model.eval()
        with torch.no_grad():
            path_scores, _ = model(data)
            path_probs = torch.sigmoid(path_scores)

        # Visualize prediction
        visualize_prediction(data, path_probs, output_dir, i)

        # Calculate metrics
        true_path = data.y.squeeze().numpy()
        pred_path = (path_probs.squeeze().numpy() > 0.5).astype(float)

        accuracy = np.mean(true_path == pred_path)
        precision = np.sum(true_path * pred_path) / (np.sum(pred_path) + 1e-8)
        recall = np.sum(true_path * pred_path) / (np.sum(true_path) + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"Sample {i+1}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # Save metrics to file
        with open(os.path.join(output_dir, f"metrics_sample_{i+1}.txt"), "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")


def visualize_prediction(data, path_probs, output_dir, sample_idx):
    """Visualize the model's path prediction."""
    # Create figure
    plt.figure(figsize=(12, 6))

    # Extract node positions and features
    x_coords = data.x[:, 0].numpy() * 15  # Scale to actual coordinates
    y_coords = data.x[:, 1].numpy() * 15
    true_path = data.y.squeeze().numpy()
    pred_path = path_probs.squeeze().numpy()

    # Plot ground truth
    plt.subplot(1, 2, 1)

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
        elif true_path[i] > 0.5:  # On path
            plt.plot(x_coords[i], y_coords[i], "mo", markersize=8)  # Magenta for path
        else:
            plt.plot(
                x_coords[i], y_coords[i], "ko", markersize=6
            )  # Black for normal room

    plt.title("Ground Truth Path")
    plt.axis("equal")
    plt.grid(True)

    # Plot prediction
    plt.subplot(1, 2, 2)

    # Draw edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        plt.plot(
            [x_coords[src], x_coords[dst]],
            [y_coords[src], y_coords[dst]],
            "k-",
            alpha=0.3,
        )

    # Create colormap for path probabilities
    cmap = plt.cm.viridis

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
        else:
            # Color based on path probability
            color = cmap(pred_path[i])
            plt.plot(x_coords[i], y_coords[i], "o", color=color, markersize=6)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, label="Path Probability")

    plt.title("Predicted Path Probabilities")
    plt.axis("equal")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"prediction_sample_{sample_idx+1}.png"))
    plt.close()


def generate_new_dungeons(num_samples, output_dir):
    """Generate new dungeons for evaluation."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate dungeons
    dungeons = []
    converter = DungeonGraphToData()

    for i in tqdm(range(num_samples), desc="Generating new dungeons"):
        # Random dungeon size
        width = np.random.randint(10, 20)
        height = np.random.randint(10, 20)

        # Generate dungeon
        generator = DungeonGenerator(
            width=width,
            height=height,
            room_density=0.6,
            trap_probability=0.1,
            guard_count=np.random.randint(2, 5),
        )
        dungeon = generator.generate()

        # Find start and goal nodes
        start_node = None
        goal_node = None

        for node, features in dungeon.get_all_nodes_with_features().items():
            if features[3] == 1:  # is_start
                start_node = node
            elif features[4] == 1:  # is_goal
                goal_node = node

        # Find shortest path
        path = GraphUtils.find_shortest_path(dungeon, start_node, goal_node)

        # Convert to PyTorch Geometric Data
        data, node_map = converter.convert(dungeon)

        # Generate path labels
        path_labels = GraphUtils.generate_path_labels(dungeon, path)

        # Convert labels to tensor
        y = torch.zeros(data.num_nodes, 1)
        for node, label in path_labels.items():
            node_idx = node_map[node]
            y[node_idx] = label

        # Add labels to data
        data.y = y

        # Store data
        dungeons.append(data)

        # Visualize dungeon
        visualize_dungeon(dungeon, path, output_dir, i)

    return dungeons


def visualize_dungeon(dungeon, path, output_dir, sample_idx):
    """Visualize a dungeon with its optimal path."""
    # Create figure
    plt.figure(figsize=(10, 10))

    # Extract node positions and features
    node_positions = {}
    node_types = {}

    for node, features in dungeon.get_all_nodes_with_features().items():
        x, y = features[0], features[1]
        # Scale to actual coordinates
        x_coord = x * 15
        y_coord = y * 15
        node_positions[node] = (x_coord, y_coord)

        if features[3] == 1:  # Start
            node_types[node] = "start"
        elif features[4] == 1:  # Goal
            node_types[node] = "goal"
        elif features[2] == 1:  # Trap
            node_types[node] = "trap"
        elif features[5] == 1:  # Guard
            node_types[node] = "guard"
        else:
            node_types[node] = "room"

    # Draw edges
    for node in dungeon.nodes:
        x1, y1 = node_positions[node]
        for neighbor, weight in dungeon.get_neighbors(node):
            if node < neighbor:  # To avoid drawing edges twice
                x2, y2 = node_positions[neighbor]
                plt.plot([x1, x2], [y1, y2], "k-", alpha=0.3, linewidth=weight / 2)

    # Draw nodes
    for node, (x, y) in node_positions.items():
        if node in path:
            # Highlight path nodes
            plt.plot(x, y, "mo", markersize=8, alpha=0.7)  # Magenta for path
        elif node_types[node] == "start":
            plt.plot(x, y, "go", markersize=10)  # Green for start
        elif node_types[node] == "goal":
            plt.plot(x, y, "bo", markersize=10)  # Blue for goal
        elif node_types[node] == "trap":
            plt.plot(x, y, "ro", markersize=8)  # Red for trap
        elif node_types[node] == "guard":
            plt.plot(x, y, "yo", markersize=8)  # Yellow for guard
        else:
            plt.plot(x, y, "ko", markersize=6)  # Black for normal room

    # Draw path lines
    for i in range(len(path) - 1):
        node1, node2 = path[i], path[i + 1]
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        plt.plot(
            [x1, x2], [y1, y2], "m-", linewidth=2, alpha=0.7
        )  # Magenta line for path

    plt.title("Dungeon with Optimal Path")
    plt.axis("equal")
    plt.grid(True)

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"dungeon_sample_{sample_idx+1}.png"))
    plt.close()


def main():
    # Parse arguments
    args = parse_args()

    # Create model
    if os.path.exists(args.dataset_dir):
        # Load dataset to get number of features
        dataset = DungeonPathfindingDataset(root=args.dataset_dir)
        num_features = dataset.num_features
    else:
        # Default number of features
        num_features = 6

    model = PathfindingGNN(
        node_features=num_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model from checkpoint: {args.checkpoint}")
    print(f"Model was trained for {checkpoint.get('epoch', 0)} epochs")

    # Evaluate on existing dataset
    if os.path.exists(args.dataset_dir):
        print("Evaluating on existing dataset...")
        dataset = DungeonPathfindingDataset(root=args.dataset_dir)

        # Use test split
        test_size = min(len(dataset) // 5, 100)  # 20% or max 100 samples
        test_dataset = dataset[-test_size:]

        evaluate_model(
            model,
            test_dataset,
            args.num_test_samples,
            os.path.join(args.output_dir, "existing_dataset"),
        )

    # Generate and evaluate on new dungeons
    print("Generating and evaluating on new dungeons...")
    new_dungeons = generate_new_dungeons(
        args.num_test_samples, os.path.join(args.output_dir, "new_dungeons")
    )

    evaluate_model(
        model,
        new_dungeons,
        args.num_test_samples,
        os.path.join(args.output_dir, "new_dungeons_eval"),
    )

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
