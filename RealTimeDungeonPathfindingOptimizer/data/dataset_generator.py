import os
import random
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import pickle
from tqdm import tqdm

from data.dungeon_graphs import DungeonGenerator
from models.gnn_model import DungeonGraphToData
from utils.graph_utils import GraphUtils


class DungeonPathfindingDataset(InMemoryDataset):
    """Dataset for training the dungeon pathfinding GNN."""

    def __init__(
        self,
        root,
        num_samples=1000,
        width_range=(10, 20),
        height_range=(10, 20),
        room_density=0.6,
        trap_probability=0.1,
        guard_count_range=(2, 5),
        transform=None,
        pre_transform=None,
        force_reload=False,
    ):
        """
        Initialize the dataset.

        Args:
            root: Root directory where the dataset should be saved
            num_samples: Number of dungeon samples to generate
            width_range: Range of dungeon widths (min, max)
            height_range: Range of dungeon heights (min, max)
            room_density: Density of rooms in the dungeon
            trap_probability: Probability of a room being a trap
            guard_count_range: Range of guard counts (min, max)
            transform: Transform to apply to the data
            pre_transform: Transform to apply to the data before saving
            force_reload: Whether to force regeneration of the dataset
        """
        self.num_samples = num_samples
        self.width_range = width_range
        self.height_range = height_range
        self.room_density = room_density
        self.trap_probability = trap_probability
        self.guard_count_range = guard_count_range
        self.force_reload = force_reload

        super(DungeonPathfindingDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dungeons.pkl"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Generate dungeons
        if not os.path.exists(self.raw_paths[0]) or self.force_reload:
            self._generate_dungeons()

    def process(self):
        # Load raw dungeons
        with open(self.raw_paths[0], "rb") as f:
            dungeons = pickle.load(f)

        # Convert dungeons to PyTorch Geometric Data objects
        data_list = []
        converter = DungeonGraphToData()

        for dungeon, path in tqdm(dungeons, desc="Processing dungeons"):
            # Convert dungeon to PyTorch Geometric Data
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

            # Apply pre-transform if available
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # Collate data into a single tensor
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _generate_dungeons(self):
        """Generate dungeons and save them to disk."""
        dungeons = []

        for _ in tqdm(range(self.num_samples), desc="Generating dungeons"):
            # Random dungeon size
            width = random.randint(*self.width_range)
            height = random.randint(*self.height_range)

            # Random guard count
            guard_count = random.randint(*self.guard_count_range)

            # Generate dungeon
            generator = DungeonGenerator(
                width=width,
                height=height,
                room_density=self.room_density,
                trap_probability=self.trap_probability,
                guard_count=guard_count,
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

            # Store dungeon and path
            dungeons.append((dungeon, path))

        # Create raw directory if it doesn't exist
        os.makedirs(os.path.dirname(self.raw_paths[0]), exist_ok=True)

        # Save dungeons to disk
        with open(self.raw_paths[0], "wb") as f:
            pickle.dump(dungeons, f)
