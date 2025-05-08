import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GraphAttentionLayer(MessagePassing):
    """Graph Attention Network (GAT) layer specifically designed for spatial reasoning."""

    def __init__(self, in_channels, out_channels, heads=1, dropout=0.1, edge_dim=1):
        super(GraphAttentionLayer, self).__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        # Linear transformations
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention mechanism
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

        if hasattr(self, "lin_edge"):
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)

        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        # Add self-loops to edge_index
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=0.0, num_nodes=x.size(0)
        )

        # Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)

        # Process edge attributes if present
        edge_embedding = None
        if self.edge_dim > 0 and edge_attr is not None:
            edge_embedding = self.lin_edge(edge_attr).view(
                -1, self.heads, self.out_channels
            )

        # Start propagation
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)

        # Reshape output
        out = out.view(-1, self.heads * self.out_channels)

        # Add bias
        out = out + self.bias

        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # Compute attention coefficients
        alpha_src = (x_i * self.att_src).sum(dim=-1)
        alpha_dst = (x_j * self.att_dst).sum(dim=-1)
        alpha = alpha_src + alpha_dst

        # Add edge feature attention if available
        if edge_attr is not None:
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        # Apply attention
        alpha = F.leaky_relu(alpha, 0.2)

        # Compute softmax over neighbors
        alpha = softmax(alpha, index, ptr, size_i)

        # Apply dropout to attention
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weight messages by attention
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        return aggr_out


class SpatialMessagePassingLayer(MessagePassing):
    """Custom message passing layer that emphasizes spatial relationships."""

    def __init__(self, in_channels, out_channels, edge_dim=1):
        super(SpatialMessagePassingLayer, self).__init__(aggr="mean")

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Node transformation
        self.lin_node = nn.Linear(in_channels, out_channels)

        # Edge transformation
        self.lin_edge = nn.Linear(edge_dim, out_channels)

        # Message transformation
        self.lin_message = nn.Linear(in_channels + out_channels, out_channels)

        # Update function
        self.lin_update = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Transform node features
        node_feat = self.lin_node(x)

        # Start propagation
        return self.propagate(edge_index, x=x, node_feat=node_feat, edge_attr=edge_attr)

    def message(self, x_j, edge_attr, node_feat_j):
        # Transform edge features
        edge_feat = self.lin_edge(edge_attr)

        # Combine node and edge features
        message_input = torch.cat([x_j, edge_feat], dim=1)
        return self.lin_message(message_input)

    def update(self, aggr_out, x):
        # Combine original and aggregated features
        update_input = torch.cat([x, aggr_out], dim=1)
        return F.relu(self.lin_update(update_input))


# Helper function for softmax with sparse inputs
def softmax(src, index, ptr=None, num_nodes=None):
    """Sparse softmax implementation for GAT."""
    if ptr is not None:
        out = src.exp()
        out_sum = torch.zeros(num_nodes, dtype=out.dtype, device=out.device)

        # Sum up contributions for each node
        for i in range(ptr.size(0) - 1):
            out_sum[i] = out[ptr[i] : ptr[i + 1]].sum()

        # Normalize
        for i in range(ptr.size(0) - 1):
            out[ptr[i] : ptr[i + 1]] = out[ptr[i] : ptr[i + 1]] / (out_sum[i] + 1e-16)

        return out
    else:
        out = src.exp()

        # Compute sum for each target node
        out_sum = torch.zeros(num_nodes, dtype=out.dtype, device=out.device)
        out_sum.scatter_add_(0, index, out)

        # Normalize
        out = out / (out_sum[index] + 1e-16)

        return out
