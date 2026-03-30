from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_benchmark_history import (
    CONFIG_MARKERS,
    CONFIG_ORDER,
    MODEL_COLORS,
    MODEL_ORDER,
    load_data,
)

# Hard-coded switches: 1 means plot, 0 means skip.
PLOT_FEEDFORWARD_FP32 = 0
PLOT_FORWARD_BACKWARD_FP32 = 0
PLOT_FEEDFORWARD_MIXED_PRECISION = 1
PLOT_FORWARD_BACKWARD_MIXED_PRECISION = 1

CONFIG_ENABLED = {
    "feedforward + fp32": PLOT_FEEDFORWARD_FP32,
    "forward+backward + fp32": PLOT_FORWARD_BACKWARD_FP32,
    "feedforward + mixed precision": PLOT_FEEDFORWARD_MIXED_PRECISION,
    "forward+backward + mixed precision": PLOT_FORWARD_BACKWARD_MIXED_PRECISION,
}


def plot_benchmark_history_linear(csv_path: Path, output_path: Path) -> None:
    df = load_data(csv_path)

    fig, ax = plt.subplots(figsize=(10, 15))

    for config_label in CONFIG_ORDER:
        if CONFIG_ENABLED.get(config_label, 0) != 1:
            continue

        for model_size in MODEL_ORDER:
            group = df[
                (df["config_label"] == config_label) & (df["model_size"] == model_size)
            ].sort_values("context_length")

            if group.empty:
                continue

            color = MODEL_COLORS[model_size]
            marker = CONFIG_MARKERS[config_label]

            ax.plot(
                group["context_length"],
                group["mean_step_time"],
                color=color,
                linewidth=1.5,
                alpha=0.85,
            )
            ax.scatter(
                group["context_length"],
                group["mean_step_time"],
                color=color,
                marker=marker,
                s=30,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

    ax.set_xlabel("context_length")
    ax.set_ylabel("mean_step_time (s)")
    ax.set_title("Benchmark History by Model Size and Configuration (Linear Y Axis)")
    ax.set_xticks(sorted(df["context_length"].unique()))
    ax.grid(True, linestyle="--", alpha=0.3)

    color_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MODEL_COLORS[model_size],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            label=model_size,
        )
        for model_size in MODEL_ORDER
    ]
    marker_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=CONFIG_MARKERS[config_label],
            color="black",
            linestyle="none",
            markersize=8,
            label=config_label,
        )
        for config_label in CONFIG_ORDER
        if CONFIG_ENABLED.get(config_label, 0) == 1
    ]

    legend1 = ax.legend(handles=color_handles, title="model_size", loc="upper left")
    ax.add_artist(legend1)
    if marker_handles:
        ax.legend(handles=marker_handles, title="configuration", loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=script_dir / "benchmark_history.csv",
        help="Path to benchmark_history.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "benchmark_history_plot_linear.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args()

    plot_benchmark_history_linear(args.csv, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
