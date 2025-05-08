import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dungeon_graphs import DungeonGenerator
from utils.graph_utils import GraphUtils


def test_dungeon_connectivity():
    """Test that the dungeon generator creates connected graphs."""
    # Create a dungeon generator
    generator = DungeonGenerator(
        width=15, height=15, room_density=0.6, trap_probability=0.1, guard_count=3
    )

    # Generate a dungeon
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

    # Check connectivity
    is_connected = GraphUtils.check_connectivity(dungeon, start_node, goal_node)
    print(f"Start and goal are connected: {is_connected}")

    # Get connected components
    components = GraphUtils.get_connected_components(dungeon)
    print(f"Number of connected components: {len(components)}")

    # Find shortest path
    shortest_path = GraphUtils.find_shortest_path(dungeon, start_node, goal_node)
    print(f"Shortest path length: {len(shortest_path)}")

    # Visualize the dungeon and path
    visualize_dungeon_path(dungeon, shortest_path)

    assert is_connected, "Start and goal nodes should be connected"
    assert len(shortest_path) > 0, "Shortest path should exist"

    print("Connectivity test passed!")


def visualize_dungeon_path(dungeon, path):
    """Visualize the dungeon and path."""
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

    plt.title("Dungeon with Shortest Path")
    plt.axis("equal")
    plt.grid(True)

    # Save the figure
    plt.savefig("dungeon_with_path.png")
    print("Dungeon visualization saved as 'dungeon_with_path.png'")


if __name__ == "__main__":
    test_dungeon_connectivity()
