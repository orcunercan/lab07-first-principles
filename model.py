"""The network: a small feedforward regressor for bike rentals."""

import torch
from torch import nn


class BikeNet(nn.Module):
    """Hidden layers with ReLU, ending in one value per row with no activation."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        # first hidden layer: from the input columns to `hidden` units
        blocks: list[nn.Module] = [nn.Linear(n_features, hidden), nn.ReLU()]
        # each extra hidden layer keeps the same width
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.ReLU()]
        # output: one number per row, no activation because this is regression
        blocks.append(nn.Linear(hidden, 1))
        # Sequential registers the layers, so the optimizer can see their weights
        self.layers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze(1) turns shape (N, 1) into (N,), the same shape as the targets
        return self.layers(x).squeeze(1)