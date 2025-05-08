import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import random

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dungeon_graphs import DungeonGraph, DungeonGenerator
from data.trap_guard_simulation import (
    TrapSimulation,
    GuardSimulation,
    DungeonSimulation,
)


def test_trap_simulation():
    # Create a dungeon generator
    generator = DungeonGenerator(
        width=15, height=15, room_density=0.6, trap_probability=0.2, guard_count=3
    )

    # Generate a dungeon
    dungeon = generator.generate()

    # Create trap simulation
    trap_sim = TrapSimulation(activation_probability=0.3)

    # Update trap states
    trap_sim.update(dungeon)

    # Count active traps
    active_traps = len(trap_sim.active_traps)

    # Count total traps
    total_traps = 0
    for node, features in dungeon.get_all_nodes_with_features().items():
        if features[2] == 1:  # is_trap
            total_traps += 1

    print(
        f"Trap simulation: {active_traps} active traps out of {total_traps} total traps"
    )

    # Test trap damage
    for node in trap_sim.active_traps:
        damage = trap_sim.get_trap_damage(node)
        assert 1 <= damage <= 5, f"Trap damage {damage} outside expected range"

    print("Trap simulation tests passed!")


def test_guard_simulation():
    # Create a dungeon generator
    generator = DungeonGenerator(
        width=15, height=15, room_density=0.6, trap_probability=0.1, guard_count=3
    )

    # Generate a dungeon
    dungeon = generator.generate()

    # Create guard simulation
    guard_sim = GuardSimulation(vision_range=2)

    # Initialize guard positions
    guard_sim.initialize(dungeon)

    # Get initial guard positions
    initial_positions = guard_sim.get_guard_positions()
    print(f"Initial guard positions: {initial_positions}")

    # Update guard positions a few times
    for i in range(5):
        guard_sim.update(dungeon)
        positions = guard_sim.get_guard_positions()
        print(f"Guard positions after update {i+1}: {positions}")

    print("Guard simulation tests passed!")


def test_dungeon_simulation():
    # Create a dungeon generator
    generator = DungeonGenerator(
        width=15, height=15, room_density=0.6, trap_probability=0.1, guard_count=3
    )

    # Generate a dungeon
    dungeon = generator.generate()

    # Create dungeon simulation
    sim = DungeonSimulation(dungeon)

    # Run simulation for a few steps with random player actions
    actions = ["up", "down", "left", "right"]

    for i in range(10):
        action = random.choice(actions)
        state = sim.step(action)
        print(f"Step {i+1}, Action: {action}")
        print(f"  Player position: {state['player_position']}")
        print(f"  Player health: {state['player_health']}")
        print(f"  Detected by guard: {state['detected']}")
        print(f"  Goal reached: {state['goal_reached']}")
        print(f"  Active traps: {len(state['active_traps'])}")
        print(f"  Guard positions: {state['guard_positions']}")

    print("Dungeon simulation tests passed!")


def visualize_simulation(dungeon, simulation, steps=20):
    """Visualize the dungeon simulation over time."""
    # Create figure
    plt.figure(figsize=(10, 10))

    # Run simulation for specified steps
    actions = ["up", "down", "left", "right"]
    states = []

    for i in range(steps):
        action = random.choice(actions)
        state = simulation.step(action)
        states.append(state)

        # Visualize every 5 steps
        if (i + 1) % 5 == 0 or i == steps - 1:
            plt.clf()  # Clear figure

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
                        plt.plot(
                            [x1, x2], [y1, y2], "k-", alpha=0.5, linewidth=weight / 2
                        )

            # Draw nodes
            for node, (x, y) in node_positions.items():
                if node in state["active_traps"]:
                    plt.plot(x, y, "ro", markersize=8)  # Red for active trap
                elif node_types[node] == "start":
                    plt.plot(x, y, "go", markersize=10)  # Green for start
                elif node_types[node] == "goal":
                    plt.plot(x, y, "bo", markersize=10)  # Blue for goal
                elif node_types[node] == "trap" and node not in state["active_traps"]:
                    plt.plot(x, y, "mo", markersize=8)  # Magenta for inactive trap
                elif node_types[node] == "guard":
                    plt.plot(
                        x, y, "ko", markersize=6
                    )  # Black for normal room (guard home)
                else:
                    plt.plot(x, y, "ko", markersize=6)  # Black for normal room

            # Draw guards at their current positions
            for guard_home, current_pos in state["guard_positions"].items():
                x, y = node_positions[current_pos]
                plt.plot(x, y, "yo", markersize=8)  # Yellow for guard

            # Draw player
            x, y = node_positions[state["player_position"]]
            plt.plot(x, y, "c*", markersize=12)  # Cyan star for player

            plt.title(f"Dungeon Simulation - Step {i+1}")
            plt.axis("equal")
            plt.grid(True)

            # Save the figure
            plt.savefig(f"simulation_step_{i+1}.png")
            print(f"Simulation visualization saved as 'simulation_step_{i+1}.png'")

            # Small delay to see the animation
            plt.pause(0.5)

    print("Simulation visualization complete!")


if __name__ == "__main__":
    test_trap_simulation()
    test_guard_simulation()
    test_dungeon_simulation()

    # Generate a dungeon for visualization
    generator = DungeonGenerator(
        width=15, height=15, room_density=0.6, trap_probability=0.1, guard_count=3
    )
    dungeon = generator.generate()

    # Create dungeon simulation
    sim = DungeonSimulation(dungeon)

    # Visualize simulation
    visualize_simulation(dungeon, sim, steps=20)
