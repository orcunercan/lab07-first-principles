"""Tests for the epoch function and the training function."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import BikeNet
from train import fit, run_epoch


def test_network_can_memorise_one_small_batch(small_batch):
    x, y = small_batch
    loader = DataLoader(TensorDataset(x, y), batch_size=len(y))
    model = BikeNet(n_features=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    start, _ = run_epoch(model, loader, nn.MSELoss())
    for _ in range(300):
        run_epoch(model, loader, nn.MSELoss(), optimizer)
    end, _ = run_epoch(model, loader, nn.MSELoss())
    assert end < 0.05 * start


def test_gradients_are_cleared_between_batches(small_batch):
    # With lr=0 the weights never move, so every batch gives the same gradient.
    # If optimizer.zero_grad() were missing, two batches would leave twice as much.
    x, y = small_batch
    model = BikeNet(n_features=x.shape[1])
    nn.MSELoss()(model(x), y).backward()
    one_batch = [p.grad.clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loader = DataLoader(TensorDataset(torch.cat([x, x]), torch.cat([y, y])), 16)
    run_epoch(model, loader, nn.MSELoss(), optimizer)
    for expected, p in zip(one_batch, model.parameters(), strict=True):
        assert torch.allclose(p.grad, expected, atol=1e-5)


def test_measuring_an_epoch_leaves_the_weights_unchanged(small_batch):
    x, y = small_batch
    model = BikeNet(n_features=x.shape[1])
    before = [p.clone() for p in model.parameters()]
    run_epoch(model, DataLoader(TensorDataset(x, y), batch_size=4), nn.MSELoss())
    for old, new in zip(before, model.parameters(), strict=True):
        assert torch.equal(old, new)


def test_fit_logs_every_epoch_and_restores_the_best_weights(small_batch):
    x, y = small_batch
    train_loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(x, y + 5.0), batch_size=16)
    model = BikeNet(n_features=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    history = fit(model, train_loader, val_loader, nn.MSELoss(), optimizer, 6)
    assert [row["epoch"] for row in history] == [1, 2, 3, 4, 5, 6]
    assert set(history[0]) == {
        "epoch",
        "train_loss",
        "train_mae",
        "val_loss",
        "val_mae",
    }
    best = min(row["val_loss"] for row in history)
    restored, _ = run_epoch(model, val_loader, nn.MSELoss())
    assert abs(restored - best) < 1e-4
