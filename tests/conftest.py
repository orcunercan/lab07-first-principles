"""Shared fixtures: the prepared splits and one small batch of real rows."""

import pytest
import torch

from prepare import make_splits


@pytest.fixture(scope="session")
def splits():
    """The real 60/20/20 split, prepared once for the whole test session."""
    return make_splits()


@pytest.fixture
def small_batch(splits):
    """Sixteen training rows: features and rentals."""
    return splits.train.x[:16], splits.train.y[:16]


@pytest.fixture(autouse=True)
def seeded():
    """Every test starts from the same random numbers."""
    torch.manual_seed(0)
