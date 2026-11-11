"""Split the table, prepare the features, and wrap them in a Dataset and loaders."""

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]
SEED = 42


class RentalsDataset(Dataset):
    """Holds features and targets as float32 tensors and hands out one row."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class Splits:
    """The three prepared datasets and the preprocessor fitted on training rows."""

    def __init__(
        self,
        train: RentalsDataset,
        val: RentalsDataset,
        test: RentalsDataset,
        preprocessor: ColumnTransformer,
    ) -> None:
        self.train = train
        self.val = val
        self.test = test
        self.preprocessor = preprocessor
        # number of columns after preparing: what the network's first layer needs
        self.n_features = train.x.shape[1]


def make_preprocessor() -> ColumnTransformer:
    """Scale the numbers and one-hot encode the station type."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        sparse_threshold=0,
    )


def split_frame(
    frame: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the rows 60/20/20 into training, validation and test, before preparing."""
    train_df, rest = train_test_split(frame, test_size=0.4, random_state=seed)
    val_df, test_df = train_test_split(rest, test_size=0.5, random_state=seed)
    return train_df, val_df, test_df


def build_dataset(df: pd.DataFrame, preprocessor: ColumnTransformer) -> RentalsDataset:
    """Transform one split with the already fitted preprocessor and wrap it."""
    features = preprocessor.transform(df[NUMERIC + CATEGORICAL])
    return RentalsDataset(features, df[TARGET].to_numpy())


def make_splits(seed: int = SEED) -> Splits:
    """Split first, fit the preprocessor on the training rows, transform all three."""
    train_df, val_df, test_df = split_frame(make_rentals(), seed)
    preprocessor = make_preprocessor()
    preprocessor.fit(train_df[NUMERIC + CATEGORICAL])  # training rows only
    return Splits(
        build_dataset(train_df, preprocessor),
        build_dataset(val_df, preprocessor),
        build_dataset(test_df, preprocessor),
        preprocessor,
    )


def make_loaders(
    splits: Splits, batch_size: int = 64, seed: int = SEED
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Shuffle the training batches with a seeded generator; keep the rest in order."""
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(
        splits.train, batch_size=batch_size, shuffle=True, generator=generator
    )
    val = DataLoader(splits.val, batch_size=256)
    test = DataLoader(splits.test, batch_size=256)
    return train, val, test
