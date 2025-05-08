import os
import torch
import json
import numpy as np


class CheckpointManager:
    """Manages model checkpoints with additional metadata."""

    def __init__(self, checkpoint_dir="checkpoints", max_to_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.checkpoint_list = []

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Load checkpoint metadata if it exists
        self.metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
        self._load_metadata()

    def _load_metadata(self):
        """Load checkpoint metadata from file."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
                self.checkpoint_list = metadata.get("checkpoint_list", [])
        else:
            self.checkpoint_list = []

    def _save_metadata(self):
        """Save checkpoint metadata to file."""
        metadata = {"checkpoint_list": self.checkpoint_list}
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def save(self, state, is_best=False, metrics=None):
        """
        Save a checkpoint.

        Args:
            state: Dictionary containing model state and other info
            is_best: Whether this is the best model so far
            metrics: Dictionary of metrics to store with checkpoint
        """
        # Create checkpoint filename with epoch
        epoch = state.get("epoch", 0)
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        # Add metrics to state if provided
        if metrics is not None:
            state["metrics"] = metrics

        # Save the checkpoint
        torch.save(state, checkpoint_path)

        # Save as best if specified
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            torch.save(state, best_path)

        # Update checkpoint list
        checkpoint_info = {
            "filename": checkpoint_name,
            "epoch": epoch,
            "is_best": is_best,
            "metrics": metrics,
        }

        # Add to list and sort by epoch
        self.checkpoint_list.append(checkpoint_info)
        self.checkpoint_list.sort(key=lambda x: x["epoch"])

        # Remove old checkpoints if exceeding max_to_keep
        while len(self.checkpoint_list) > self.max_to_keep:
            oldest = self.checkpoint_list.pop(0)
            oldest_path = os.path.join(self.checkpoint_dir, oldest["filename"])
            if os.path.exists(oldest_path) and oldest["is_best"] == False:
                os.remove(oldest_path)

        # Save updated metadata
        self._save_metadata()

    def load_latest(self, map_location=None):
        """Load the latest checkpoint."""
        if not self.checkpoint_list:
            return None

        latest = self.checkpoint_list[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest["filename"])

        if os.path.exists(checkpoint_path):
            return torch.load(checkpoint_path, map_location=map_location)
        else:
            return None

    def load_best(self, map_location=None):
        """Load the best checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")

        if os.path.exists(best_path):
            return torch.load(best_path, map_location=map_location)
        else:
            return None

    def get_checkpoint_info(self):
        """Get information about available checkpoints."""
        return self.checkpoint_list

    def resume_or_restart(
        self, model, optimizer=None, scheduler=None, map_location=None
    ):
        """
        Resume from latest checkpoint or restart training.

        Returns:
            start_epoch: Epoch to start training from
            best_metric: Best metric value so far
        """
        checkpoint = self.load_latest(map_location=map_location)

        if checkpoint is not None:
            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state if provided
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state if provided
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint.get("epoch", 0) + 1
            best_metric = checkpoint.get("best_metric", float("inf"))

            print(f"Resumed from epoch {start_epoch-1}")
            return start_epoch, best_metric
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0, float("inf")
