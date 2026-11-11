"""Tests for the network."""

import pytest
import torch
from torch import nn

from model import BikeNet


@pytest.mark.parametrize("hidden", [8, 64, 256])
def test_network_returns_one_value_per_row(hidden):
    out = BikeNet(n_features=8, hidden=hidden)(torch.zeros(5, 8))
    # (N,) like the targets, not (N, 1), which would broadcast silently in the loss
    assert out.shape == (5,)


def test_network_has_two_hidden_layers_with_relu():
    model = BikeNet(n_features=8)
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    relu = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    assert len(linear) == 3 and len(relu) == 2
    assert isinstance(list(model.layers)[-1], nn.Linear)


def test_one_optimizer_step_changes_every_layer():
    model = BikeNet(n_features=8)
    before = [p.clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.MSELoss()(model(torch.randn(16, 8)), torch.full((16,), 10.0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for old, new in zip(before, model.parameters(), strict=True):
        assert not torch.equal(old, new)
