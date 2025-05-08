import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dungeon_graphs import DungeonGenerator
from models.gnn_model import PathfindingGNN, DungeonGraphToData


def test_gnn_model():
    # Create a dungeon generator
    generator = DungeonGenerator(
        width=15, height=15, room_density=0.6, trap_probability=0.1, guard_count=3
    )

    # Generate a dungeon
    dungeon = generator.generate()

    # Convert dungeon to PyTorch Geometric Data
    converter = DungeonGraphToData()
    data, node_map = converter.convert(dungeon)

    print(f"Converted dungeon to PyTorch Geometric Data:")
    print(f"  Number of nodes: {data.num_nodes}")
    print(f"  Number of edges: {data.num_edges}")
    print(f"  Node feature dimensions: {data.x.size(1)}")

    # Create GNN model
    model = PathfindingGNN(
        node_features=data.x.size(1), hidden_channels=64, num_layers=3
    )
    print(
        f"Created GNN model with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        path_scores, _ = model(data)

    print(f"Model output shape: {path_scores.shape}")

    # Visualize path scores on the dungeon
    visualize_path_scores(dungeon, path_scores, node_map)

    print("GNN model test passed!")


def visualize_path_scores(dungeon, path_scores, node_map):
    """Visualize path scores on the dungeon."""
    # Create figure
    plt.figure(figsize=(10, 10))

    # Extract node positions and scores
    node_positions = {}
    node_scores = {}
    node_types = {}

    for node, features in dungeon.get_all_nodes_with_features().items():
        x, y = features[0], features[1]
        # Scale to actual coordinates
        x_coord = x * 15
        y_coord = y * 15
        node_positions[node] = (x_coord, y_coord)

        # Get node score
        node_idx = node_map[node]
        score = path_scores[node_idx].item()
        node_scores[node] = score

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
                plt.plot([x1, x2], [y1, y2], "k-", alpha=0.5, linewidth=weight / 2)

    # Normalize scores for coloring
    scores = np.array(list(node_scores.values()))
    vmin, vmax = scores.min(), scores.max()

    # Draw nodes with color based on path score
    for node, (x, y) in node_positions.items():
        score = node_scores[node]
        color = plt.cm.viridis((score - vmin) / (vmax - vmin + 1e-8))

        if node_types[node] == "start":
            plt.plot(x, y, "go", markersize=10)  # Green for start
        elif node_types[node] == "goal":
            plt.plot(x, y, "bo", markersize=10)  # Blue for goal
        elif node_types[node] == "trap":
            plt.plot(x, y, "ro", markersize=8)  # Red for trap
        elif node_types[node] == "guard":
            plt.plot(x, y, "yo", markersize=8)  # Yellow for guard
        else:
            plt.plot(x, y, "o", color=color, markersize=6)

    plt.title("Path Scores from GNN Model (Untrained)")
    plt.axis("equal")
    plt.grid(True)

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    plt.colorbar(sm, label="Path Score")

    # Save the figure
    plt.savefig("gnn_path_scores.png")
    print("GNN path scores visualization saved as 'gnn_path_scores.png'")


if __name__ == "__main__":
    test_gnn_model()
