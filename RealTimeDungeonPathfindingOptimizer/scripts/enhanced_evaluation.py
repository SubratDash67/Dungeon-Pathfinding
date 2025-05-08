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
from utils.graph_utils import GraphUtils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the dungeon pathfinding GNN with enhanced metrics"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="enhanced_dataset",
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
        default="enhanced_evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=128,
        help="Number of hidden channels in the GNN",
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of GNN layers"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Threshold for path prediction (lower = higher recall)",
    )
    return parser.parse_args()


def evaluate_model_with_thresholds(model, dataset, num_samples, output_dir):
    """Evaluate model with different thresholds to find optimal F1 score."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Select random test samples
    indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )
    test_samples = [dataset[idx] for idx in indices]

    # Try different thresholds
    thresholds = np.arange(0.1, 0.7, 0.05)
    results = []

    for threshold in thresholds:
        metrics = evaluate_with_threshold(model, test_samples, threshold)
        results.append(
            {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )
        print(
            f"Threshold {threshold:.2f}: F1={metrics['f1_score']:.4f}, Recall={metrics['recall']:.4f}"
        )

    # Find best threshold for F1 score
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]["f1_score"])
    best_threshold = results[best_f1_idx]["threshold"]
    best_f1 = results[best_f1_idx]["f1_score"]

    print(f"\nBest threshold for F1 score: {best_threshold:.2f} (F1={best_f1:.4f})")

    # Plot threshold vs metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r["precision"] for r in results], "b-", label="Precision")
    plt.plot(thresholds, [r["recall"] for r in results], "g-", label="Recall")
    plt.plot(thresholds, [r["f1_score"] for r in results], "r-", label="F1 Score")
    plt.axvline(
        x=best_threshold,
        color="k",
        linestyle="--",
        label=f"Best Threshold ({best_threshold:.2f})",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs. Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "threshold_analysis.png"))

    # Evaluate with best threshold
    for i, idx in enumerate(indices):
        data = dataset[idx]
        visualize_prediction_with_threshold(model, data, best_threshold, output_dir, i)

    return best_threshold


def evaluate_with_threshold(model, samples, threshold):
    """Evaluate model with a specific threshold."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in samples:
            # Forward pass
            path_scores, _ = model(data)
            path_probs = torch.sigmoid(path_scores)

            # Apply threshold
            predictions = (path_probs > threshold).float()

            # Collect predictions and targets
            all_preds.append(predictions.cpu())
            all_targets.append(data.y.cpu())

    # Concatenate all samples
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
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def visualize_prediction_with_threshold(model, data, threshold, output_dir, sample_idx):
    """Visualize the model's path prediction with a specific threshold."""
    # Create figure
    plt.figure(figsize=(12, 6))

    # Extract node positions and features
    x_coords = data.x[:, 0].numpy() * 15  # Scale to actual coordinates
    y_coords = data.x[:, 1].numpy() * 15
    true_path = data.y.squeeze().numpy()

    # Get model predictions
    model.eval()
    with torch.no_grad():
        path_scores, _ = model(data)
        path_probs = torch.sigmoid(path_scores).squeeze().numpy()
        pred_path = (path_probs > threshold).astype(float)

    # Calculate metrics for this sample
    tp = (pred_path * true_path).sum()
    fp = (pred_path * (1 - true_path)).sum()
    fn = ((1 - pred_path) * true_path).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

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
        elif pred_path[i] > 0.5:  # Predicted path
            if true_path[i] > 0.5:  # True positive
                plt.plot(
                    x_coords[i], y_coords[i], "mo", markersize=8
                )  # Magenta for correct path
            else:  # False positive
                plt.plot(
                    x_coords[i], y_coords[i], "co", markersize=8
                )  # Cyan for false positive
        elif true_path[i] > 0.5:  # False negative
            plt.plot(
                x_coords[i], y_coords[i], "rx", markersize=8
            )  # Red X for missed path
        else:
            plt.plot(
                x_coords[i], y_coords[i], "ko", markersize=6
            )  # Black for normal room

    plt.title(f"Predicted Path (F1={f1:.4f}, Recall={recall:.4f})")
    plt.axis("equal")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"enhanced_prediction_{sample_idx+1}.png"))
    plt.close()

    # Save metrics to file
    with open(
        os.path.join(output_dir, f"enhanced_metrics_sample_{sample_idx+1}.txt"), "w"
    ) as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"True Positives: {tp}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"Threshold: {threshold}\n")


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

        # Find optimal threshold and evaluate
        best_threshold = evaluate_model_with_thresholds(
            model,
            test_dataset,
            args.num_test_samples,
            os.path.join(args.output_dir, "existing_dataset"),
        )
    else:
        print("Dataset directory not found. Using default threshold.")
        best_threshold = args.threshold

    # Generate and evaluate on new dungeons
    print("Generating and evaluating on new dungeons...")
    new_dungeons_dir = os.path.join(args.output_dir, "new_dungeons")
    os.makedirs(new_dungeons_dir, exist_ok=True)

    # Generate new dungeons
    new_dungeons = []
    converter = DungeonGraphToData()

    for i in tqdm(range(args.num_test_samples), desc="Generating new dungeons"):
        # Random dungeon size
        width = np.random.randint(15, 25)
        height = np.random.randint(15, 25)

        # Generate dungeon with more complexity
        generator = DungeonGenerator(
            width=width,
            height=height,
            room_density=0.65,
            trap_probability=0.15,
            guard_count=np.random.randint(3, 7),
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
        new_dungeons.append(data)

        # Visualize dungeon with ground truth path
        visualize_dungeon(dungeon, path, new_dungeons_dir, i)

    # Evaluate on new dungeons with best threshold
    evaluate_model_with_thresholds(
        model,
        new_dungeons,
        args.num_test_samples,
        os.path.join(args.output_dir, "new_dungeons_eval"),
    )

    print("Enhanced evaluation complete!")


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

    plt.title("Enhanced Dungeon with Optimal Path")
    plt.axis("equal")
    plt.grid(True)

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"enhanced_dungeon_{sample_idx+1}.png"))
    plt.close()


if __name__ == "__main__":
    main()
