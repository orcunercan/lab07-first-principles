"""Train the bike network with a hand-written loop, compare it, run experiments.

Run it with: python train.py
Everything in reports/ is recreated from nothing.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn

from bikes_data import make_rentals
from model import BikeNet
from prepare import SEED, make_loaders, make_splits, split_frame

REPORTS = Path("reports")

# The main run: the settings I believe in. Each experiment changes one of them.
MAIN = {"lr": 3e-3, "hidden": 64, "layers": 2, "batch_size": 64, "epochs": 150}
EXPERIMENTS = {
    "main": {},
    "lr_too_small": {"lr": 1e-5},
    "lr_too_large": {"lr": 0.1},
    "much_wider_deeper": {"hidden": 512, "layers": 4},
}


def run_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over a loader. Trains if given an optimizer, otherwise only measures.

    Returns the mean loss and the mean absolute error in rentals.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_error, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:  # 1. a batch
            prediction = model(x)  # 2. forward
            loss = loss_fn(prediction, y)  # 3. loss
            if training:
                optimizer.zero_grad()  # clear the gradients of the last batch
                loss.backward()  # 4. backward
                optimizer.step()  # 5. step
            # weight by batch size, because the last batch is smaller
            total_loss += loss.item() * len(y)
            total_error += (prediction - y).abs().sum().item()
            count += len(y)
    return total_loss / count, total_error / count


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    """Train for many epochs, log every epoch, and put back the best weights."""
    history = []
    best_loss, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history


def train_run(splits, settings):
    """Train one network with the given settings, seeded so it repeats exactly."""
    torch.manual_seed(SEED)
    train_loader, val_loader, _ = make_loaders(
        splits, batch_size=settings["batch_size"], seed=SEED
    )
    model = BikeNet(splits.n_features, settings["hidden"], settings["layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    history = fit(
        model, train_loader, val_loader, nn.MSELoss(), optimizer, settings["epochs"]
    )
    return model, pd.DataFrame(history)


def plot_curves(frame, title, path):
    """Training and validation loss per epoch, with the best epoch marked."""
    best = int(frame.loc[frame["val_loss"].idxmin(), "epoch"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame["epoch"], frame["train_loss"], label="training loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="validation loss")
    ax.axvline(best, color="grey", linestyle="--", label=f"best epoch ({best})")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error (log scale)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rentals_by_hour(path):
    """Mean rentals per hour, on the training rows, by weekday and weekend."""
    train_df, _, _ = split_frame(make_rentals(), SEED)
    means = train_df.groupby(["hour", "is_weekend"])["rentals"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(means.index, means[0], marker="o", label="weekday")
    ax.plot(means.index, means[1], marker="o", label="weekend")
    ax.set_xlabel("hour of the day")
    ax.set_ylabel("mean bikes rented")
    ax.set_title("Rentals by hour (training rows)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def baselines(splits):
    """Two simple models on the same rows: the training mean and a straight line."""
    x_train, y_train = splits.train.x.numpy(), splits.train.y.numpy()
    mean = float(y_train.mean())
    linear = LinearRegression().fit(x_train, y_train)
    results = {}
    for name, part in [("val", splits.val), ("test", splits.test)]:
        x, y = part.x.numpy(), part.y.numpy()
        results[f"training_mean_{name}_mae"] = mean_absolute_error(y, [mean] * len(y))
        results[f"linear_regression_{name}_mae"] = mean_absolute_error(
            y, linear.predict(x)
        )
    return {k: round(float(v), 3) for k, v in results.items()}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    plot_rentals_by_hour(REPORTS / "rentals_by_hour.png")
    splits = make_splits(SEED)

    rows, main_model = [], None
    for name, change in EXPERIMENTS.items():
        settings = {**MAIN, **change}
        model, frame = train_run(splits, settings)
        best = frame.loc[frame["val_loss"].idxmin()]
        last = frame.iloc[-1]
        rows.append(
            {
                "run": name,
                "changed": ", ".join(f"{k}={v}" for k, v in change.items()) or "-",
                "best_epoch": int(best["epoch"]),
                "best_val_mae": round(float(best["val_mae"]), 3),
                "final_train_loss": round(float(last["train_loss"]), 3),
                "final_val_loss": round(float(last["val_loss"]), 3),
            }
        )
        plot_curves(frame, f"Run: {name}", REPORTS / f"curves_{name}.png")
        print(rows[-1])
        if name == "main":
            main_model = model
            frame.round(4).to_csv(REPORTS / "history.csv", index=False)
            plot_curves(frame, "Main run", REPORTS / "learning_curves.png")
    pd.DataFrame(rows).to_csv(REPORTS / "experiments.csv", index=False)

    # The final network is the main run, chosen on validation rows only.
    # The test rows are used once, here, at the end.
    _, val_loader, test_loader = make_loaders(splits, seed=SEED)
    _, network_val_mae = run_epoch(main_model, val_loader, nn.MSELoss())
    _, network_test_mae = run_epoch(main_model, test_loader, nn.MSELoss())

    metrics = baselines(splits)
    metrics["network_val_mae"] = round(network_val_mae, 3)
    metrics["network_test_mae"] = round(network_test_mae, 3)
    metrics["main_settings"] = MAIN
    metrics["seed"] = SEED
    metrics["torch"] = torch.__version__
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
