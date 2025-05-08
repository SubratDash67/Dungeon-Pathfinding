import random
import numpy as np
import torch
from torch_geometric.data import Data
import copy


class DungeonDataAugmentation:
    """Data augmentation techniques for dungeon graphs."""

    @staticmethod
    def flip_horizontal(data, width):
        """Flip the dungeon horizontally."""
        # Create a copy of the data
        new_data = copy.deepcopy(data)

        # Update node features (x, y coordinates)
        for i in range(new_data.x.size(0)):
            # Assuming x-coordinate is at index 0
            new_data.x[i, 0] = 1.0 - new_data.x[i, 0]

        return new_data

    @staticmethod
    def flip_vertical(data, height):
        """Flip the dungeon vertically."""
        # Create a copy of the data
        new_data = copy.deepcopy(data)

        # Update node features (x, y coordinates)
        for i in range(new_data.x.size(0)):
            # Assuming y-coordinate is at index 1
            new_data.x[i, 1] = 1.0 - new_data.x[i, 1]

        return new_data

    @staticmethod
    def rotate_90(data):
        """Rotate the dungeon by 90 degrees."""
        # Create a copy of the data
        new_data = copy.deepcopy(data)

        # Update node features (x, y coordinates)
        for i in range(new_data.x.size(0)):
            # Swap x and y, and flip y
            x, y = new_data.x[i, 0].item(), new_data.x[i, 1].item()
            new_data.x[i, 0] = y
            new_data.x[i, 1] = 1.0 - x

        return new_data

    @staticmethod
    def add_random_traps(data, trap_probability=0.05):
        """Add random traps to the dungeon."""
        # Create a copy of the data
        new_data = copy.deepcopy(data)

        # Add traps randomly
        for i in range(new_data.x.size(0)):
            # Skip start and goal nodes
            if new_data.x[i, 3] == 1 or new_data.x[i, 4] == 1:
                continue

            # Randomly add traps
            if random.random() < trap_probability:
                new_data.x[i, 2] = 1  # Set trap feature to 1

        return new_data

    @staticmethod
    def add_random_guards(data, guard_probability=0.03):
        """Add random guards to the dungeon."""
        # Create a copy of the data
        new_data = copy.deepcopy(data)

        # Add guards randomly
        for i in range(new_data.x.size(0)):
            # Skip start, goal, and trap nodes
            if new_data.x[i, 3] == 1 or new_data.x[i, 4] == 1 or new_data.x[i, 2] == 1:
                continue

            # Randomly add guards
            if random.random() < guard_probability:
                new_data.x[i, 5] = 1  # Set guard feature to 1

        return new_data

    @staticmethod
    def perturb_edge_weights(data, perturbation_range=0.2):
        """Perturb edge weights slightly."""
        # Create a copy of the data
        new_data = copy.deepcopy(data)

        # Perturb edge weights
        for i in range(new_data.edge_attr.size(0)):
            # Get current weight
            weight = new_data.edge_attr[i, 0].item()

            # Apply random perturbation
            perturbation = 1.0 + (random.random() * 2 - 1) * perturbation_range
            new_weight = max(0.1, weight * perturbation)

            new_data.edge_attr[i, 0] = new_weight

        return new_data

    @staticmethod
    def augment(data, width, height, augmentation_types=None):
        """Apply multiple augmentations to the data."""
        if augmentation_types is None:
            augmentation_types = [
                "flip_h",
                "flip_v",
                "rotate",
                "traps",
                "guards",
                "weights",
            ]

        # Choose a random subset of augmentations
        num_augmentations = random.randint(1, len(augmentation_types))
        selected_augmentations = random.sample(augmentation_types, num_augmentations)

        # Apply selected augmentations
        augmented_data = copy.deepcopy(data)

        for aug_type in selected_augmentations:
            if aug_type == "flip_h":
                augmented_data = DungeonDataAugmentation.flip_horizontal(
                    augmented_data, width
                )
            elif aug_type == "flip_v":
                augmented_data = DungeonDataAugmentation.flip_vertical(
                    augmented_data, height
                )
            elif aug_type == "rotate":
                augmented_data = DungeonDataAugmentation.rotate_90(augmented_data)
            elif aug_type == "traps":
                augmented_data = DungeonDataAugmentation.add_random_traps(
                    augmented_data
                )
            elif aug_type == "guards":
                augmented_data = DungeonDataAugmentation.add_random_guards(
                    augmented_data
                )
            elif aug_type == "weights":
                augmented_data = DungeonDataAugmentation.perturb_edge_weights(
                    augmented_data
                )

        return augmented_data

    @staticmethod
    def generate_augmented_dataset(dataset, num_augmentations_per_sample=3):
        """Generate an augmented dataset from the original dataset."""
        augmented_dataset = []

        for data in dataset:
            # Add the original data
            augmented_dataset.append(data)

            # Add augmented versions
            for _ in range(num_augmentations_per_sample):
                # Assuming width and height are 1.0 in normalized coordinates
                augmented_data = DungeonDataAugmentation.augment(data, 1.0, 1.0)
                augmented_dataset.append(augmented_data)

        return augmented_dataset
