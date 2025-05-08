import random
import numpy as np
import math


class TrapSimulation:
    def __init__(self, activation_probability=0.3, damage_range=(1, 5)):
        self.activation_probability = activation_probability
        self.damage_range = damage_range
        self.active_traps = set()

    def update(self, dungeon):
        """Update trap states in the dungeon."""
        # Reset active traps
        self.active_traps = set()

        # Randomly activate traps based on probability
        for node, features in dungeon.get_all_nodes_with_features().items():
            if features[2] == 1:  # If it's a trap
                if random.random() < self.activation_probability:
                    self.active_traps.add(node)

    def get_trap_damage(self, node):
        """Get damage for a trap if it's active."""
        if node in self.active_traps:
            return random.randint(self.damage_range[0], self.damage_range[1])
        return 0

    def is_trap_active(self, node):
        """Check if a trap is currently active."""
        return node in self.active_traps


class GuardSimulation:
    def __init__(self, vision_range=2, movement_staleness_weight=0.7):
        self.vision_range = vision_range
        self.movement_staleness_weight = movement_staleness_weight
        self.guard_positions = {}  # Maps guard node to current position
        self.staleness_map = {}  # Maps nodes to staleness values
        self.patrol_paths = {}  # Maps guard nodes to their current patrol path

    def initialize(self, dungeon):
        """Initialize guard positions and staleness map."""
        self.guard_positions = {}
        self.staleness_map = {
            node: 1.0 for node in dungeon.nodes
        }  # Initialize all nodes as stale
        self.patrol_paths = {}

        # Set initial guard positions
        for node, features in dungeon.get_all_nodes_with_features().items():
            if features[5] == 1:  # If it's a guard
                self.guard_positions[node] = node  # Guard starts at its assigned node

    def update(self, dungeon):
        """Update guard positions and staleness map."""
        # Update staleness for all nodes (increase staleness over time)
        for node in dungeon.nodes:
            self.staleness_map[node] = min(1.0, self.staleness_map[node] + 0.05)

        # Update guard positions and reduce staleness for visible nodes
        for guard_home, current_pos in self.guard_positions.items():
            # If guard has no patrol path or has reached the end of its path, generate a new one
            if guard_home not in self.patrol_paths or not self.patrol_paths[guard_home]:
                self._generate_patrol_path(dungeon, guard_home)

            # Move guard along patrol path
            if self.patrol_paths[guard_home]:
                next_pos = self.patrol_paths[guard_home].pop(0)
                self.guard_positions[guard_home] = next_pos

                # Reduce staleness for the current position and neighbors
                self._update_staleness_for_visible_nodes(dungeon, next_pos)

    def _generate_patrol_path(self, dungeon, guard_home):
        """Generate a patrol path for a guard based on staleness."""
        current_pos = self.guard_positions[guard_home]

        # Find the stalest node within reasonable distance
        target_node = self._find_stalest_node(dungeon, current_pos)

        # Generate path to target using A* or similar
        path = self._find_path(dungeon, current_pos, target_node)
        self.patrol_paths[guard_home] = path

    def _find_stalest_node(self, dungeon, start_node):
        """Find the stalest node, weighted by distance."""
        best_score = -float("inf")
        best_node = start_node

        for node in dungeon.nodes:
            # Skip the current node
            if node == start_node:
                continue

            # Calculate distance (using Manhattan distance as approximation)
            start_x, start_y = self._get_node_coords(start_node)
            node_x, node_y = self._get_node_coords(node)
            distance = abs(node_x - start_x) + abs(node_y - start_y)

            # Skip nodes that are too far away
            if distance > 10:  # Arbitrary limit to avoid guards wandering too far
                continue

            # Calculate score based on staleness and distance
            staleness = self.staleness_map[node]
            score = staleness - (self.movement_staleness_weight * distance / 10)

            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def _find_path(self, dungeon, start_node, end_node):
        """Find a path between two nodes using A* algorithm."""
        # A* implementation
        open_set = {start_node}
        closed_set = set()

        # For node n, g_score[n] is the cost of the cheapest path from start to n
        g_score = {node: float("inf") for node in dungeon.nodes}
        g_score[start_node] = 0

        # For node n, f_score[n] = g_score[n] + h(n)
        f_score = {node: float("inf") for node in dungeon.nodes}
        f_score[start_node] = self._heuristic(start_node, end_node)

        # For reconstructing the path
        came_from = {}

        while open_set:
            # Find node in open_set with lowest f_score
            current = min(open_set, key=lambda node: f_score[node])

            if current == end_node:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)
            closed_set.add(current)

            for neighbor, weight in dungeon.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Tentative g_score
                tentative_g_score = g_score[current] + weight

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue

                # This path is better, record it
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self._heuristic(
                    neighbor, end_node
                )

        # No path found
        return [end_node]  # Just return the end node as fallback

    def _heuristic(self, node1, node2):
        """Calculate heuristic distance between two nodes."""
        x1, y1 = self._get_node_coords(node1)
        x2, y2 = self._get_node_coords(node2)
        return abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance

    def _get_node_coords(self, node):
        """Extract x, y coordinates from node ID."""
        # Assuming node format is "room_x_y"
        parts = node.split("_")
        return int(parts[1]), int(parts[2])

    def _update_staleness_for_visible_nodes(self, dungeon, node):
        """Update staleness for nodes visible from the current position."""
        # Reduce staleness for current node
        self.staleness_map[node] = 0.0

        # Reduce staleness for neighboring nodes based on distance
        for neighbor, _ in dungeon.get_neighbors(node):
            self.staleness_map[neighbor] = max(0.0, self.staleness_map[neighbor] - 0.5)

            # Check second-level neighbors if within vision range
            if self.vision_range >= 2:
                for second_neighbor, _ in dungeon.get_neighbors(neighbor):
                    if (
                        second_neighbor != node
                    ):  # Avoid updating the original node again
                        self.staleness_map[second_neighbor] = max(
                            0.0, self.staleness_map[second_neighbor] - 0.25
                        )

    def get_guard_positions(self):
        """Get current positions of all guards."""
        return self.guard_positions

    def is_player_detected(self, player_node):
        """Check if player is detected by any guard."""
        for guard_pos in self.guard_positions.values():
            if player_node == guard_pos:
                return True

            # Check if player is in guard's vision range
            player_x, player_y = self._get_node_coords(player_node)
            guard_x, guard_y = self._get_node_coords(guard_pos)

            distance = abs(player_x - guard_x) + abs(player_y - guard_y)
            if distance <= self.vision_range:
                return True

        return False


class DungeonSimulation:
    def __init__(self, dungeon):
        self.dungeon = dungeon
        self.trap_simulation = TrapSimulation()
        self.guard_simulation = GuardSimulation()
        self.player_position = None
        self.player_health = 100
        self.steps = 0

        # Find start node
        for node, features in dungeon.get_all_nodes_with_features().items():
            if features[3] == 1:  # is_start
                self.player_position = node
                break

        # Initialize simulations
        self.trap_simulation.update(dungeon)
        self.guard_simulation.initialize(dungeon)

    def step(self, player_action):
        """Advance simulation by one step."""
        self.steps += 1

        # Update player position based on action
        if player_action in ["up", "down", "left", "right"]:
            self._move_player(player_action)

        # Update trap and guard simulations
        if self.steps % 3 == 0:  # Update traps less frequently
            self.trap_simulation.update(self.dungeon)

        self.guard_simulation.update(self.dungeon)

        # Check for trap damage
        if self.trap_simulation.is_trap_active(self.player_position):
            damage = self.trap_simulation.get_trap_damage(self.player_position)
            self.player_health -= damage

        # Check for guard detection
        detected = self.guard_simulation.is_player_detected(self.player_position)

        # Check for goal reached
        goal_reached = False
        node_features = self.dungeon.get_node_features(self.player_position)
        if node_features and node_features[4] == 1:  # is_goal
            goal_reached = True

        return {
            "player_position": self.player_position,
            "player_health": self.player_health,
            "detected": detected,
            "goal_reached": goal_reached,
            "active_traps": self.trap_simulation.active_traps,
            "guard_positions": self.guard_simulation.get_guard_positions(),
        }

    def _move_player(self, direction):
        """Move player in the specified direction if possible."""
        x, y = self._get_node_coords(self.player_position)

        if direction == "up":
            new_y = y + 1
            new_x = x
        elif direction == "down":
            new_y = y - 1
            new_x = x
        elif direction == "left":
            new_x = x - 1
            new_y = y
        elif direction == "right":
            new_x = x + 1
            new_y = y

        new_pos = f"room_{new_x}_{new_y}"

        # Check if the move is valid (node exists and is connected)
        if new_pos in self.dungeon.nodes:
            for neighbor, _ in self.dungeon.get_neighbors(self.player_position):
                if neighbor == new_pos:
                    self.player_position = new_pos
                    return

    def _get_node_coords(self, node):
        """Extract x, y coordinates from node ID."""
        # Assuming node format is "room_x_y"
        parts = node.split("_")
        return int(parts[1]), int(parts[2])
