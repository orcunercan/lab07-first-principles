"""Tests for the split, the preparation, the Dataset and the loaders."""

import pandas as pd
import torch

from bikes_data import make_rentals
from prepare import NUMERIC, make_loaders, split_frame


def test_dataset_length_and_float32_shapes(splits):
    assert len(splits.train) == 3600
    x, y = splits.train[0]
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.shape == (splits.n_features,)
    assert y.shape == ()


def test_splits_have_expected_sizes_and_do_not_overlap():
    frame = make_rentals()
    train_df, val_df, test_df = split_frame(frame)
    assert (len(train_df), len(val_df), len(test_df)) == (3600, 1200, 1200)
    combined = pd.concat([train_df, val_df, test_df])
    assert combined.index.is_unique
    assert len(combined) == len(frame)


def test_scaler_was_fitted_on_training_rows_only(splits):
    n = len(NUMERIC)
    train_means = splits.train.x[:, :n].mean(dim=0)
    val_means = splits.val.x[:, :n].mean(dim=0)
    assert torch.allclose(train_means, torch.zeros(n), atol=1e-5)
    # validation rows were scaled with the training mean, so they are not exactly 0
    assert not torch.allclose(val_means, torch.zeros(n), atol=1e-5)
    scaler = splits.preprocessor.named_transformers_["num"]
    assert scaler.n_samples_seen_ == len(splits.train)


def test_training_loader_shuffles_but_repeats_with_a_seed(splits):
    first = next(iter(make_loaders(splits, seed=1)[0]))[1]
    again = next(iter(make_loaders(splits, seed=1)[0]))[1]
    in_order = splits.train.y[: len(first)]
    assert torch.equal(first, again)
    assert not torch.equal(first, in_order)
