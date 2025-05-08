import random
import numpy as np
from collections import deque


class DungeonGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.node_features = {}  # Store features for each node

    def add_node(self, node, features=None):
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = []
        if features is not None:
            self.node_features[node] = features

    def add_edge(self, node1, node2, weight=1):
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("Both nodes must be in the graph")
        self.edges[node1].append((node2, weight))
        self.edges[node2].append((node1, weight))  # Assuming undirected graph

    def get_neighbors(self, node):
        return self.edges.get(node, [])

    def get_node_features(self, node):
        return self.node_features.get(node, None)

    def get_all_nodes_with_features(self):
        return self.node_features

    def __repr__(self):
        return f"DungeonGraph(nodes={list(self.nodes)}, edges={self.edges})"


class DungeonGenerator:
    def __init__(
        self, width=20, height=20, room_density=0.6, trap_probability=0.1, guard_count=3
    ):
        self.width = width
        self.height = height
        self.room_density = room_density
        self.trap_probability = trap_probability
        self.guard_count = guard_count

    def generate(self):
        """Generate a procedural dungeon as a graph."""
        dungeon = DungeonGraph()

        # Create a grid representation of the dungeon
        grid = np.zeros((self.height, self.width), dtype=int)

        # Start with a random room
        start_x, start_y = random.randint(0, self.width - 1), random.randint(
            0, self.height - 1
        )
        grid[start_y, start_x] = 1  # 1 represents a room

        # Use cellular automata to generate rooms
        self._generate_rooms(grid)

        # Create nodes for each room
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] == 1:
                    node_id = f"room_{x}_{y}"

                    # Generate random features for the room
                    # Features: [x_pos, y_pos, is_trap, is_start, is_goal]
                    is_trap = random.random() < self.trap_probability
                    is_start = x == start_x and y == start_y
                    is_goal = False  # Will set one room as goal later

                    features = [
                        x / self.width,
                        y / self.height,
                        int(is_trap),
                        int(is_start),
                        int(is_goal),
                        0,
                    ]  # Last 0 is for guard
                    dungeon.add_node(node_id, features)

        # Set a random room (not the start) as the goal
        rooms = list(dungeon.nodes)
        goal_room = random.choice(
            [r for r in rooms if not r.startswith(f"room_{start_x}_{start_y}")]
        )
        goal_x, goal_y = map(int, goal_room.split("_")[1:])
        dungeon.node_features[goal_room][4] = 1  # Set is_goal to True

        # Connect adjacent rooms with edges
        self._connect_adjacent_rooms(dungeon, grid)

        # Place guards
        self._place_guards(dungeon)

        # Ensure connectivity between start and goal
        self.ensure_start_goal_connectivity(dungeon)

        return dungeon

    def _generate_rooms(self, grid):
        """Use cellular automata to generate a natural-looking dungeon."""
        # First pass: randomly fill grid based on room_density
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.room_density:
                    grid[y, x] = 1

        # Apply cellular automata rules for a few iterations
        for _ in range(3):
            new_grid = np.copy(grid)
            for y in range(self.height):
                for x in range(self.width):
                    # Count neighbors
                    neighbors = 0
                    for ny in range(max(0, y - 1), min(self.height, y + 2)):
                        for nx in range(max(0, x - 1), min(self.width, x + 2)):
                            if nx == x and ny == y:
                                continue
                            if grid[ny, nx] == 1:
                                neighbors += 1

                    # Apply rules
                    if grid[y, x] == 1 and neighbors < 2:
                        new_grid[y, x] = 0  # Die from underpopulation
                    elif grid[y, x] == 0 and neighbors > 3:
                        new_grid[y, x] = 1  # Birth from reproduction

            grid = new_grid

        # Ensure connectivity using BFS
        self._ensure_connectivity(grid)

    def _ensure_connectivity(self, grid):
        """Ensure all rooms are connected using BFS."""
        # Find first room
        start_y, start_x = None, None
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] == 1:
                    start_y, start_x = y, x
                    break
            if start_y is not None:
                break

        if start_y is None:  # No rooms found
            # Create at least one room
            start_y, start_x = self.height // 2, self.width // 2
            grid[start_y, start_x] = 1
            return

        # BFS to find all connected rooms
        visited = np.zeros_like(grid)
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = 1

        while queue:
            y, x = queue.popleft()
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < self.height
                    and 0 <= nx < self.width
                    and grid[ny, nx] == 1
                    and visited[ny, nx] == 0
                ):
                    visited[ny, nx] = 1
                    queue.append((ny, nx))

        # Connect isolated rooms
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] == 1 and visited[y, x] == 0:
                    # Find closest visited room
                    min_dist = float("inf")
                    closest = None
                    for vy in range(self.height):
                        for vx in range(self.width):
                            if visited[vy, vx] == 1:
                                dist = abs(vy - y) + abs(vx - x)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest = (vy, vx)

                    if closest:
                        cy, cx = closest
                        # Create a path between (y,x) and (cy,cx)
                        while y != cy or x != cx:
                            if x < cx:
                                x += 1
                            elif x > cx:
                                x -= 1
                            elif y < cy:
                                y += 1
                            elif y > cy:
                                y -= 1
                            grid[y, x] = 1

                    # Mark this room and all connected to it as visited
                    queue = deque([(y, x)])
                    visited[y, x] = 1
                    while queue:
                        qy, qx = queue.popleft()
                        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            ny, nx = qy + dy, qx + dx
                            if (
                                0 <= ny < self.height
                                and 0 <= nx < self.width
                                and grid[ny, nx] == 1
                                and visited[ny, nx] == 0
                            ):
                                visited[ny, nx] = 1
                                queue.append((ny, nx))

    def _connect_adjacent_rooms(self, dungeon, grid):
        """Connect adjacent rooms with edges."""
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] == 1:
                    current = f"room_{x}_{y}"

                    # Check adjacent cells
                    for dx, dy in [(1, 0), (0, 1)]:  # Right and down
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < self.width
                            and 0 <= ny < self.height
                            and grid[ny, nx] == 1
                        ):
                            neighbor = f"room_{nx}_{ny}"

                            # Add edge with random weight (1-5)
                            weight = random.randint(1, 5)
                            dungeon.add_edge(current, neighbor, weight)

    def _place_guards(self, dungeon):
        """Place guards in the dungeon."""
        rooms = list(dungeon.nodes)

        # Don't place guards at start or goal
        valid_rooms = []
        for room in rooms:
            features = dungeon.get_node_features(room)
            if features[3] == 0 and features[4] == 0:  # Not start or goal
                valid_rooms.append(room)

        # Place guards
        guard_rooms = random.sample(
            valid_rooms, min(self.guard_count, len(valid_rooms))
        )

        for room in guard_rooms:
            features = dungeon.get_node_features(room)
            # Set guard feature (index 5)
            features[5] = 1  # 1 indicates guard presence

    def ensure_start_goal_connectivity(self, dungeon):
        """Ensure that start and goal nodes are connected."""
        # Find start and goal nodes
        start_node = None
        goal_node = None

        for node, features in dungeon.get_all_nodes_with_features().items():
            if features[3] == 1:  # is_start
                start_node = node
            elif features[4] == 1:  # is_goal
                goal_node = node

        if start_node is None or goal_node is None:
            print("Start or goal node not found")
            return

        # Check if there's a path from start to goal
        visited = set()
        queue = deque([start_node])
        visited.add(start_node)

        goal_reachable = False

        while queue:
            node = queue.popleft()

            if node == goal_node:
                goal_reachable = True
                break

            for neighbor, _ in dungeon.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # If goal is not reachable, create a path
        if not goal_reachable:
            print(
                f"Goal node {goal_node} not reachable from start node {start_node}. Creating path..."
            )

            # Extract coordinates
            start_x, start_y = map(int, start_node.split("_")[1:])
            goal_x, goal_y = map(int, goal_node.split("_")[1:])

            # Create a path between start and goal
            current_x, current_y = start_x, start_y
            prev_x, prev_y = start_x, start_y

            while current_x != goal_x or current_y != goal_y:
                # Save previous position
                prev_x, prev_y = current_x, current_y

                # Move towards goal
                if current_x < goal_x:
                    current_x += 1
                elif current_x > goal_x:
                    current_x -= 1
                elif current_y < goal_y:
                    current_y += 1
                elif current_y > goal_y:
                    current_y -= 1

                # Create node if it doesn't exist
                current_node = f"room_{current_x}_{current_y}"
                prev_node = f"room_{prev_x}_{prev_y}"

                if current_node not in dungeon.nodes:
                    # Create the node
                    is_trap = False
                    is_start = False
                    is_goal = current_x == goal_x and current_y == goal_y
                    is_guard = False

                    features = [
                        current_x / self.width,
                        current_y / self.height,
                        int(is_trap),
                        int(is_start),
                        int(is_goal),
                        int(is_guard),
                    ]
                    dungeon.add_node(current_node, features)

                # Connect to previous node
                if prev_node in dungeon.nodes and current_node in dungeon.nodes:
                    # Add edge with random weight (1-3)
                    weight = random.randint(1, 3)

                    # Check if edge already exists
                    edge_exists = False
                    for neighbor, _ in dungeon.get_neighbors(prev_node):
                        if neighbor == current_node:
                            edge_exists = True
                            break

                    if not edge_exists:
                        dungeon.add_edge(prev_node, current_node, weight)
