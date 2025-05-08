import numpy as np
import networkx as nx
import torch
import heapq
from collections import deque
import copy


class GraphUtils:
    """Utility functions for working with graphs."""

    @staticmethod
    def dungeon_to_networkx(dungeon):
        """Convert a DungeonGraph to a NetworkX graph for analysis."""
        G = nx.Graph()

        # Add nodes with features
        for node, features in dungeon.get_all_nodes_with_features().items():
            G.add_node(node, features=features)

        # Add edges with weights
        for node in dungeon.nodes:
            for neighbor, weight in dungeon.get_neighbors(node):
                G.add_edge(node, neighbor, weight=weight)

        return G

    @staticmethod
    def find_shortest_path(dungeon, start_node, goal_node):
        """Find the shortest path using Dijkstra's algorithm."""
        # Check if start and goal nodes exist
        if start_node not in dungeon.nodes or goal_node not in dungeon.nodes:
            print(f"Start node {start_node} or goal node {goal_node} not in graph")
            return []

        # Initialize distances
        distances = {node: float("infinity") for node in dungeon.nodes}
        distances[start_node] = 0

        # Initialize previous nodes for path reconstruction
        previous = {node: None for node in dungeon.nodes}

        # Priority queue for Dijkstra's algorithm
        pq = [(0, start_node)]

        # Set of visited nodes
        visited = set()

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            # If we've reached the goal, we're done
            if current_node == goal_node:
                break

            # Skip if we've already processed this node
            if current_node in visited:
                continue

            visited.add(current_node)

            # Check all neighbors
            for neighbor, weight in dungeon.get_neighbors(current_node):
                if neighbor in visited:
                    continue

                # Calculate new distance
                new_distance = current_distance + weight

                # If we found a shorter path, update
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_distance, neighbor))

        # Reconstruct path
        if distances[goal_node] == float("infinity"):
            print(f"No path found from {start_node} to {goal_node}")
            return []

        path = []
        current = goal_node

        while current:
            path.append(current)
            current = previous[current]

        path.reverse()

        # Check if a path was found
        if not path or path[0] != start_node:
            print(f"Path reconstruction failed from {start_node} to {goal_node}")
            return []

        return path

    @staticmethod
    def find_safest_path(
        dungeon, start_node, goal_node, trap_weight=5.0, guard_weight=10.0
    ):
        """Find the safest path avoiding traps and guards."""
        # Check if start and goal nodes exist
        if start_node not in dungeon.nodes or goal_node not in dungeon.nodes:
            print(f"Start node {start_node} or goal node {goal_node} not in graph")
            return []

        # Create a copy of the dungeon with modified weights
        modified_dungeon = copy.deepcopy(dungeon)

        # Increase weights for edges connected to traps and guards
        for node, features in dungeon.get_all_nodes_with_features().items():
            is_trap = features[2] == 1
            is_guard = features[5] == 1

            if is_trap or is_guard:
                # Get all neighbors
                for neighbor, weight in list(dungeon.get_neighbors(node)):
                    # Modify the weight based on trap or guard
                    if is_trap:
                        new_weight = weight * trap_weight
                    else:  # is_guard
                        new_weight = weight * guard_weight

                    # Update edge in modified dungeon
                    # First, remove the existing edge
                    modified_dungeon.edges[node] = [
                        (n, w) for n, w in modified_dungeon.edges[node] if n != neighbor
                    ]
                    modified_dungeon.edges[neighbor] = [
                        (n, w) for n, w in modified_dungeon.edges[neighbor] if n != node
                    ]

                    # Then add the edge with modified weight
                    modified_dungeon.edges[node].append((neighbor, new_weight))
                    modified_dungeon.edges[neighbor].append((node, new_weight))

        # Find shortest path in the modified dungeon
        return GraphUtils.find_shortest_path(modified_dungeon, start_node, goal_node)

    @staticmethod
    def generate_path_labels(dungeon, path):
        """Generate binary labels for nodes on the path."""
        labels = {node: 0.0 for node in dungeon.nodes}

        # Set nodes on the path to 1.0
        for node in path:
            labels[node] = 1.0

        return labels

    @staticmethod
    def generate_distance_labels(dungeon, path):
        """Generate labels based on distance to the path."""
        labels = {node: 0.0 for node in dungeon.nodes}

        # Set path nodes to 1.0
        path_set = set(path)
        for node in path:
            labels[node] = 1.0

        # For non-path nodes, calculate distance to the path
        for node in dungeon.nodes:
            if node in path_set:
                continue

            # Find minimum distance to any node on the path
            min_distance = float("infinity")
            for path_node in path:
                # Use BFS to find distance
                distance = GraphUtils._bfs_distance(dungeon, node, path_node)
                min_distance = min(min_distance, distance)

            # Convert distance to a label value (closer = higher value)
            if min_distance == float("infinity"):
                labels[node] = 0.0
            else:
                labels[node] = 1.0 / (1.0 + min_distance)

        return labels

    @staticmethod
    def _bfs_distance(dungeon, start_node, end_node):
        """Calculate distance between two nodes using BFS."""
        visited = set([start_node])
        queue = deque([(start_node, 0)])  # (node, distance)

        while queue:
            node, distance = queue.popleft()

            if node == end_node:
                return distance

            for neighbor, _ in dungeon.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float("infinity")  # No path found

    @staticmethod
    def calculate_centrality(dungeon):
        """Calculate various centrality measures for nodes."""
        G = GraphUtils.dungeon_to_networkx(dungeon)

        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)

        # Combine into a dictionary of node features
        centrality_features = {}
        for node in dungeon.nodes:
            centrality_features[node] = {
                "degree": degree_centrality.get(node, 0.0),
                "betweenness": betweenness_centrality.get(node, 0.0),
                "closeness": closeness_centrality.get(node, 0.0),
            }

        return centrality_features

    @staticmethod
    def check_connectivity(dungeon, start_node, goal_node):
        """Check if there's a path between start and goal nodes."""
        visited = set()
        queue = deque([start_node])
        visited.add(start_node)

        while queue:
            node = queue.popleft()

            if node == goal_node:
                return True

            for neighbor, _ in dungeon.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    @staticmethod
    def get_connected_components(dungeon):
        """Get the connected components of the dungeon graph."""
        G = GraphUtils.dungeon_to_networkx(dungeon)
        return list(nx.connected_components(G))
