import os
import json
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch


class Logger:
    """Utility class for logging training progress and results."""

    def __init__(self, log_dir="logs", experiment_name=None):
        """
        Initialize logger.

        Args:
            log_dir: Directory to store logs
            experiment_name: Name of the experiment (default: timestamp)
        """
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Set experiment name
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)

        # Create experiment directory
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # Configure logging
        self._configure_logging()

        # Initialize metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "epochs": [],
            "learning_rate": [],
            "best_epoch": None,
            "best_val_loss": float("inf"),
            "training_time": 0,
        }

        self.logger.info(f"Initialized experiment: {experiment_name}")

    def _configure_logging(self):
        """Configure logging to file and console."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create file handler
        log_file = os.path.join(self.experiment_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_hyperparameters(self, hparams):
        """Log hyperparameters to file."""
        self.logger.info(f"Hyperparameters: {hparams}")

        # Save to JSON
        hparams_file = os.path.join(self.experiment_dir, "hyperparameters.json")
        with open(hparams_file, "w") as f:
            json.dump(hparams, f, indent=2)

    def log_model_summary(self, model):
        """Log model summary."""
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Model summary:")
        self.logger.info(f"  Total parameters: {num_params}")
        self.logger.info(f"  Trainable parameters: {num_trainable}")

        # Save model architecture to file
        model_file = os.path.join(self.experiment_dir, "model_architecture.txt")
        with open(model_file, "w") as f:
            f.write(str(model))

    def log_epoch(self, epoch, train_loss, val_loss, learning_rate=None):
        """Log metrics for an epoch."""
        self.metrics["epochs"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)

        if learning_rate is not None:
            self.metrics["learning_rate"].append(learning_rate)

        # Check if this is the best epoch
        if val_loss < self.metrics["best_val_loss"]:
            self.metrics["best_val_loss"] = val_loss
            self.metrics["best_epoch"] = epoch
            is_best = True
        else:
            is_best = False

        # Log to file
        self.logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={learning_rate}"
        )

        # Save metrics to JSON
        self._save_metrics()

        # Update plots
        self._update_plots()

        return is_best

    def log_training_time(self, seconds):
        """Log total training time."""
        self.metrics["training_time"] = seconds
        self.logger.info(f"Total training time: {seconds:.2f} seconds")
        self._save_metrics()

    def log_prediction(self, dungeon, path_scores, node_map, filename="prediction.png"):
        """Log prediction visualization."""
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

        plt.title("Path Scores from GNN Model")
        plt.axis("equal")
        plt.grid(True)

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        cbar = plt.colorbar(sm, ax=plt.gca(), label="Path Score")

        # Save the figure
        plt.savefig(os.path.join(self.experiment_dir, filename))
        plt.close()

        self.logger.info(f"Saved prediction visualization to {filename}")

    def _save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = os.path.join(self.experiment_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def _update_plots(self):
        """Update and save plots of training metrics."""
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.metrics["epochs"], self.metrics["train_loss"], label="Training Loss"
        )
        plt.plot(
            self.metrics["epochs"], self.metrics["val_loss"], label="Validation Loss"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.experiment_dir, "loss_plot.png"))
        plt.close()

        # Learning rate plot
        if self.metrics["learning_rate"]:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics["epochs"], self.metrics["learning_rate"])
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True)
            plt.savefig(os.path.join(self.experiment_dir, "lr_plot.png"))
            plt.close()
