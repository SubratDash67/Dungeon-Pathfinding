import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch


class DungeonGNNLayer(MessagePassing):
    """Custom GNN layer for dungeon pathfinding."""

    def __init__(self, in_channels, out_channels):
        super(DungeonGNNLayer, self).__init__(aggr="max")  # Max aggregation
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(1, out_channels)  # Edge features (weights)
        self.lin_update = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # x: Node features [N, in_channels]
        # edge_index: Graph connectivity [2, E]
        # edge_attr: Edge features [E, 1]

        # Transform node features
        node_feat = self.lin_node(x)

        # Start propagation
        return self.propagate(edge_index, x=x, node_feat=node_feat, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: Source node features [E, in_channels]
        # edge_attr: Edge features [E, 1]

        # Transform edge features
        edge_feat = self.lin_edge(edge_attr)

        # Combine node and edge features
        return x_j * edge_feat

    def update(self, aggr_out, x):
        # aggr_out: Aggregated messages [N, out_channels]
        # x: Original node features [N, in_channels]

        # Combine original and aggregated features
        combined = torch.cat([x, aggr_out], dim=1)
        return F.relu(self.lin_update(combined))


class PathfindingGNN(nn.Module):
    def __init__(self, node_features=6, hidden_channels=64, num_layers=3):
        super(PathfindingGNN, self).__init__()

        # Initial node embedding
        self.node_encoder = nn.Linear(node_features, hidden_channels)

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(DungeonGNNLayer(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(DungeonGNNLayer(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layers for different tasks
        self.path_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),  # Probability of being on optimal path
        )

        self.next_node_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),  # Score for next node selection
        )

    def forward(self, data):
        # data.x: Node features [N, node_features]
        # data.edge_index: Graph connectivity [2, E]
        # data.edge_attr: Edge weights [E, 1]

        # Initial node embedding
        x = self.node_encoder(data.x)

        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index, data.edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        # Predict path membership for each node
        path_scores = self.path_predictor(x)

        # For next node prediction, we need to compute scores for each edge
        edge_scores = None
        if hasattr(data, "current_node"):
            # Get the current node's embedding
            current_node_idx = data.current_node
            current_embedding = x[current_node_idx]

            # Get neighboring nodes
            neighbors = []
            for i in range(data.edge_index.size(1)):
                if data.edge_index[0, i] == current_node_idx:
                    neighbors.append(data.edge_index[1, i])

            if neighbors:
                # Compute scores for each neighbor
                neighbor_embeddings = x[neighbors]

                # Concatenate current node embedding with each neighbor
                combined = torch.cat(
                    [current_embedding.repeat(len(neighbors), 1), neighbor_embeddings],
                    dim=1,
                )

                # Predict scores
                edge_scores = self.next_node_predictor(combined)

        return path_scores, edge_scores


class DungeonGraphToData:
    """Convert DungeonGraph to PyTorch Geometric Data objects."""

    def __init__(self):
        self.node_map = {}  # Maps node IDs to indices

    def convert(self, dungeon):
        """Convert a DungeonGraph to a PyTorch Geometric Data object."""
        # Reset node map
        self.node_map = {}

        # Extract node features
        node_features = []
        for i, (node, features) in enumerate(
            dungeon.get_all_nodes_with_features().items()
        ):
            self.node_map[node] = i
            node_features.append(features)

        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)

        # Extract edges and edge weights
        edge_index = []
        edge_attr = []

        for node in dungeon.nodes:
            node_idx = self.node_map[node]
            for neighbor, weight in dungeon.get_neighbors(node):
                neighbor_idx = self.node_map[neighbor]
                edge_index.append([node_idx, neighbor_idx])
                edge_attr.append([weight])

        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data, self.node_map
