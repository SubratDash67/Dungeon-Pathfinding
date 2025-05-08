import torch
import torch.nn as nn
import torch.nn.functional as F


class RecallFocusedLoss(nn.Module):
    """
    Custom loss function that dynamically weights examples based on recall performance.
    Based on the research in search result [4].
    """

    def __init__(self, alpha=0.9, gamma=2.0, beta=0.999):
        super(RecallFocusedLoss, self).__init__()
        self.alpha = alpha  # Focusing parameter
        self.gamma = gamma  # Modulating factor
        self.beta = beta  # EMA decay rate
        self.class_recalls = None

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Raw model outputs before sigmoid [batch_size, num_nodes, 1]
            targets: Binary targets [batch_size, num_nodes, 1]
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)

        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )

        # Calculate batch recall for positive class
        with torch.no_grad():
            true_positives = (probs > 0.5).float() * targets
            recall = true_positives.sum() / (targets.sum() + 1e-8)

            # Update EMA of recall
            if self.class_recalls is None:
                self.class_recalls = recall
            else:
                self.class_recalls = (
                    self.beta * self.class_recalls + (1 - self.beta) * recall
                )

            # Calculate weight based on recall performance
            # Lower recall means higher weight
            weight = torch.pow(1 - self.class_recalls, self.gamma)

        # Apply weight to positive examples
        weighted_loss = bce_loss * (targets * weight + (1 - targets) * self.alpha)

        return weighted_loss.mean()


class F1Loss(nn.Module):
    """
    Loss that directly optimizes F1 Score
    """

    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        # Apply sigmoid to get probabilities
        probas = torch.sigmoid(predictions)

        # Calculate true positives, false positives, false negatives
        tp = (probas * targets).sum(dim=1)
        fp = (probas * (1 - targets)).sum(dim=1)
        fn = ((1 - probas) * targets).sum(dim=1)

        # Calculate precision and recall
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)

        # Return 1 - F1 to minimize
        return 1 - f1.mean()
