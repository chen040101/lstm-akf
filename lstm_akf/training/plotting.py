from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence


def _prepare_matplotlib(mpl_config_dir: Path):
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.set_loglevel("error")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt

    return plt


def _to_series(history: Sequence[dict], key: str) -> list[float | None]:
    values: list[float | None] = []
    for row in history:
        value = row.get(key)
        if value in ("", None):
            values.append(None)
        else:
            values.append(float(value))
    return values


def plot_training_curves(
    history: Sequence[dict],
    output_path: str | Path,
    title: str = "Coordinate Prediction Model Training Curves",
) -> Path | None:
    if not history:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(output_path.parent / ".mplconfig")

    epochs = [int(row["epoch"]) for row in history]
    train_loss = _to_series(history, "train_loss")
    val_loss = _to_series(history, "val_loss")
    learning_rate = _to_series(history, "lr")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="blue", linewidth=2, label="Training Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    valid_val_points = [(epoch, value) for epoch, value in zip(epochs, val_loss) if value is not None]
    if valid_val_points:
        ax.plot(
            [epoch for epoch, _ in valid_val_points],
            [value for _, value in valid_val_points],
            color="red",
            linewidth=2,
            label="Validation Loss",
        )
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Validation data not available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Validation Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, learning_rate, color="green", linewidth=2, label="Learning Rate")
    ax.set_title("Learning Rate Change", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, train_loss, color="blue", linewidth=2, label="Training Loss")
    if valid_val_points:
        ax.plot(
            [epoch for epoch, _ in valid_val_points],
            [value for _, value in valid_val_points],
            color="red",
            linewidth=2,
            label="Validation Loss",
        )
    ax.set_title("Training vs Validation Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_comparison(
    results: Sequence[dict],
    output_path: str | Path,
    title: str = "Baseline Model Comparison",
) -> Path | None:
    metric_specs = [
        ("loss", "Validation Loss"),
        ("ade", "Validation ADE"),
        ("fde", "Validation FDE"),
    ]
    return plot_metric_grid(results, metric_specs=metric_specs, output_path=output_path, title=title)


def plot_metric_grid(
    results: Sequence[dict],
    metric_specs: Sequence[tuple[str, str]],
    output_path: str | Path,
    title: str,
) -> Path | None:
    if not results:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(output_path.parent / ".mplconfig")

    model_names = [str(row["model_name"]) for row in results]
    colors = [
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#76B7B2",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
    ]
    num_metrics = len(metric_specs)
    cols = 3
    rows = max(1, (num_metrics + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, (key, label) in zip(axes_list, metric_specs):
        values = [float(row[key]) for row in results]
        bars = axis.bar(model_names, values, color=colors[: len(values)])
        axis.set_title(label, fontweight="bold")
        axis.set_ylabel(label)
        axis.grid(True, axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.6f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for axis in axes_list[num_metrics:]:
        axis.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = ["plot_metric_grid", "plot_model_comparison", "plot_training_curves"]
