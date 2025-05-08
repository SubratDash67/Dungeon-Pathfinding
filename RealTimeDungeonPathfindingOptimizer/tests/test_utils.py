import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import copy

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dungeon_graphs import DungeonGenerator
from models.gnn_model import DungeonGraphToData
from utils.data_augmentation import DungeonDataAugmentation
from utils.graph_utils import GraphUtils


def test_data_augmentation():
    """Test data augmentation functions."""
    # Generate a dungeon
    generator = DungeonGenerator(
        width=15, height=15, room_density=0.6, trap_probability=0.1, guard_count=3
    )
    dungeon = generator.generate()

    # Convert to PyTorch Geometric Data
    converter = DungeonGraphToData()
    data, node_map = converter.convert(dungeon)

    print(f"Original data: {data.num_nodes} nodes, {data.num_edges} edges")

    # Test horizontal flip
    flipped_h = DungeonDataAugmentation.flip_horizontal(data, 1.0)
    print(
        f"Horizontally flipped: {flipped_h.num_nodes} nodes, {flipped_h.num_edges} edges"
    )

    # Test vertical flip
    flipped_v = DungeonDataAugmentation.flip_vertical(data, 1.0)
    print(
        f"Vertically flipped: {flipped_v.num_nodes} nodes, {flipped_v.num_edges} edges"
    )

    # Test rotation
    rotated = DungeonDataAugmentation.rotate_90(data)
    print(f"Rotated 90 degrees: {rotated.num_nodes} nodes, {rotated.num_edges} edges")

    # Test adding traps
    with_traps = DungeonDataAugmentation.add_random_traps(data)
    trap_count = sum(with_traps.x[:, 2]).item()
    original_trap_count = sum(data.x[:, 2]).item()
    print(f"Added traps: {original_trap_count} -> {trap_count}")

    # Test adding guards
    with_guards = DungeonDataAugmentation.add_random_guards(data)
    guard_count = sum(with_guards.x[:, 5]).item()
    original_guard_count = sum(data.x[:, 5]).item()
    print(f"Added guards: {original_guard_count} -> {guard_count}")

    # Test perturbing edge weights
    perturbed = DungeonDataAugmentation.perturb_edge_weights(data)
    print(f"Perturbed edge weights")

    # Test combined augmentation
    augmented = DungeonDataAugmentation.augment(data, 1.0, 1.0)
    print(
        f"Combined augmentation: {augmented.num_nodes} nodes, {augmented.num_edges} edges"
    )

    # Test generating augmented dataset
    dataset = [data]
    augmented_dataset = DungeonDataAugmentation.generate_augmented_dataset(
        dataset, num_augmentations_per_sample=3
    )
    print(f"Augmented dataset: {len(dataset)} -> {len(augmented_dataset)} samples")

    print("Data augmentation tests passed!")


def test_graph_utils():
    """Test graph utility functions."""
    # Generate a dungeon
    generator = DungeonGenerator(
        width=15, height=15, room_density=0.6, trap_probability=0.1, guard_count=3
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

    print(f"Start node: {start_node}, Goal node: {goal_node}")

    # Test converting to NetworkX
    G = GraphUtils.dungeon_to_networkx(dungeon)
    print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Test finding shortest path
    shortest_path = GraphUtils.find_shortest_path(dungeon, start_node, goal_node)
    print(f"Shortest path length: {len(shortest_path)}")

    # Test finding safest path
    safest_path = GraphUtils.find_safest_path(dungeon, start_node, goal_node)
    print(f"Safest path length: {len(safest_path)}")

    # Test generating path labels
    path_labels = GraphUtils.generate_path_labels(dungeon, shortest_path)
    on_path_count = sum(1 for v in path_labels.values() if v > 0.5)
    print(f"Path labels: {on_path_count} nodes on path out of {len(path_labels)}")

    # Test generating distance labels
    distance_labels = GraphUtils.generate_distance_labels(dungeon, shortest_path)
    print(f"Distance labels generated for {len(distance_labels)} nodes")

    # Test calculating centrality
    centrality = GraphUtils.calculate_centrality(dungeon)
    print(f"Centrality measures calculated for {len(centrality)} nodes")

    # Visualize the paths
    if shortest_path and safest_path:
        visualize_paths(dungeon, shortest_path, safest_path)

    print("Graph utils tests passed!")


def visualize_paths(dungeon, shortest_path, safest_path):
    """Visualize the shortest and safest paths."""
    # Create figure
    plt.figure(figsize=(12, 6))

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

    # Plot for shortest path
    plt.subplot(1, 2, 1)

    # Draw edges
    for node in dungeon.nodes:
        x1, y1 = node_positions[node]
        for neighbor, weight in dungeon.get_neighbors(node):
            if node < neighbor:  # To avoid drawing edges twice
                x2, y2 = node_positions[neighbor]
                plt.plot([x1, x2], [y1, y2], "k-", alpha=0.3, linewidth=weight / 2)

    # Draw nodes
    for node, (x, y) in node_positions.items():
        if node_types[node] == "start":
            plt.plot(x, y, "go", markersize=10)  # Green for start
        elif node_types[node] == "goal":
            plt.plot(x, y, "bo", markersize=10)  # Blue for goal
        elif node_types[node] == "trap":
            plt.plot(x, y, "ro", markersize=8)  # Red for trap
        elif node_types[node] == "guard":
            plt.plot(x, y, "yo", markersize=8)  # Yellow for guard
        else:
            plt.plot(x, y, "ko", markersize=6)  # Black for normal room

    # Draw shortest path
    for i in range(len(shortest_path) - 1):
        node1 = shortest_path[i]
        node2 = shortest_path[i + 1]
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        plt.plot([x1, x2], [y1, y2], "g-", linewidth=2)

    plt.title("Shortest Path")
    plt.axis("equal")
    plt.grid(True)

    # Plot for safest path
    plt.subplot(1, 2, 2)

    # Draw edges
    for node in dungeon.nodes:
        x1, y1 = node_positions[node]
        for neighbor, weight in dungeon.get_neighbors(node):
            if node < neighbor:  # To avoid drawing edges twice
                x2, y2 = node_positions[neighbor]
                plt.plot([x1, x2], [y1, y2], "k-", alpha=0.3, linewidth=weight / 2)

    # Draw nodes
    for node, (x, y) in node_positions.items():
        if node_types[node] == "start":
            plt.plot(x, y, "go", markersize=10)  # Green for start
        elif node_types[node] == "goal":
            plt.plot(x, y, "bo", markersize=10)  # Blue for goal
        elif node_types[node] == "trap":
            plt.plot(x, y, "ro", markersize=8)  # Red for trap
        elif node_types[node] == "guard":
            plt.plot(x, y, "yo", markersize=8)  # Yellow for guard
        else:
            plt.plot(x, y, "ko", markersize=6)  # Black for normal room

    # Draw safest path
    for i in range(len(safest_path) - 1):
        node1 = safest_path[i]
        node2 = safest_path[i + 1]
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]
        plt.plot([x1, x2], [y1, y2], "b-", linewidth=2)

    plt.title("Safest Path")
    plt.axis("equal")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("path_comparison.png")
    print("Path comparison visualization saved as 'path_comparison.png'")


if __name__ == "__main__":
    test_data_augmentation()
    test_graph_utils()
