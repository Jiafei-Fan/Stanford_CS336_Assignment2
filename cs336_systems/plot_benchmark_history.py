from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = ["small", "medium", "large", "xl", "2_7B"]
MODEL_COLORS = {
    "small": "#4C78A8",
    "medium": "#F58518",
    "large": "#54A24B",
    "xl": "#E45756",
    "2_7B": "#B279A2",
}

CONFIG_ORDER = [
    "feedforward + fp32",
    "forward+backward + fp32",
    "feedforward + mixed precision",
    "forward+backward + mixed precision",
]
CONFIG_MARKERS = {
    "feedforward + fp32": "o",
    "forward+backward + fp32": "s",
    "feedforward + mixed precision": "^",
    "forward+backward + mixed precision": "D",
}


def build_config_label(row: pd.Series) -> str:
    run_type = (
        "forward+backward"
        if int(row["both_feedfoward_backward"]) == 1
        else "feedforward"
    )
    precision = "mixed precision" if int(row["mixed_precision"]) == 1 else "fp32"
    return f"{run_type} + {precision}"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "success"].copy()
    df["config_label"] = df.apply(build_config_label, axis=1)

    # If the CSV later contains repeated runs for the same setup, keep the latest one.
    df = df.sort_values("timestamp_utc").drop_duplicates(
        subset=["model_size", "context_length", "config_label"],
        keep="last",
    )

    df["model_size"] = pd.Categorical(df["model_size"], categories=MODEL_ORDER, ordered=True)
    df["config_label"] = pd.Categorical(df["config_label"], categories=CONFIG_ORDER, ordered=True)
    return df.sort_values(["config_label", "model_size", "context_length"])


def plot_benchmark_history(csv_path: Path, output_path: Path) -> None:
    df = load_data(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    for config_label in CONFIG_ORDER:
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
                s=40,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

    ax.set_xlabel("context_length")
    ax.set_ylabel("mean_step_time (s)")
    ax.set_title("Benchmark History by Model Size and Configuration")
    ax.set_xticks(sorted(df["context_length"].unique()))
    ax.set_yscale("log")
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
    ]

    legend1 = ax.legend(handles=color_handles, title="model_size", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, title="configuration", loc="lower right")

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
        default=script_dir / "benchmark_history_plot.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args()

    plot_benchmark_history(args.csv, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
