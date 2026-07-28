"""A compact edge-aware GAT for G-Safeguard-style malicious-agent detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv


class DialogueEmbeddingPool(nn.Module):
    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"mean", "last"}:
            raise ValueError(f"Unsupported dialogue pooling mode: {mode}")
        self.mode = mode

    def forward(self, edge_attr: Tensor) -> Tensor:
        if edge_attr.dim() != 3:
            raise ValueError(f"edge_attr must be [edges, turns, dim], got {tuple(edge_attr.shape)}")
        if self.mode == "last":
            return edge_attr[:, -1, :]
        return edge_attr.mean(dim=1)


class GSafeguardGAT(nn.Module):
    """Communication-only detector with the same I/O contract as G-Safeguard.

    Input node features and temporal edge utterance embeddings come from agent
    replies. The model predicts one malicious-agent logit per node.
    """

    def __init__(
        self,
        in_channels: int = 384,
        hidden_channels: int = 1024,
        out_channels: int = 1,
        heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        edge_dim: int = 384,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        self.dropout = float(dropout)
        self.pool = DialogueEmbeddingPool(pool)
        head_dim = hidden_channels // heads
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_channels, head_dim, heads=heads, concat=True, edge_dim=edge_dim, add_self_loops=False))
        for _ in range(max(0, num_layers - 1)):
            self.layers.append(GATConv(hidden_channels, head_dim, heads=heads, concat=True, edge_dim=edge_dim, add_self_loops=False))
        self.out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        pooled_edge_attr = self.pool(edge_attr)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=pooled_edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)

