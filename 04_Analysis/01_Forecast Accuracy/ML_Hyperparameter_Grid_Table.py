"""
Publication-style table of machine-learning hyperparameter grids.

This reports the candidate grid / search space used during tuning, not the
selected parameters.
"""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "99_Thesis Graphs" / "03_Results" / "01_Forecast Accuracy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "T16c_ml_hyperparameter_grids.png"
OUT_CSV = OUTPUT_DIR / "csv_ml_hyperparameter_grids.csv"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


ROWS = [
    {
        "Model": "XGBoost",
        "Search": "HalvingGridSearchCV",
        "Grid / search space": (
            "n_estimators: [5, 50, 80, 100, 150, 200]\n"
            "max_depth: [2, 3, 4, 5, 6]\n"
            "learning_rate: [0.09, 0.10, 0.14, 0.18, 0.20, 0.21]"
        ),
    },
    {
        "Model": "KNN",
        "Search": "GridSearchCV",
        "Grid / search space": (
            "n_neighbors: [10, 15, 50, 100, 500]\n"
            "weights: [uniform, distance]\n"
            "algorithm: [auto, ball_tree, kd_tree]\n"
            "leaf_size: [1, 5, 30, 100, 200]"
        ),
    },
    {
        "Model": "SVR",
        "Search": "GridSearchCV",
        "Grid / search space": (
            "kernel: [rbf, linear, poly, sigmoid]\n"
            "degree: [1, 2, 3]\n"
            "gamma: [scale, auto]\n"
            "tol: [0.0001, 0.0005]\n"
            "epsilon: [0.1, 0.2, 0.7, 1.0, 10.0]\n"
            "C: [1.0]"
        ),
    },
    {
        "Model": "Random Forest",
        "Search": "GridSearchCV",
        "Grid / search space": (
            "n_estimators: [100, 200, 400, 500, 700]\n"
            "criterion: [squared_error, absolute_error]\n"
            "max_depth: [2, 3, 4, 15, 20]\n"
            "min_samples_split: [2, 5, 10, 50, 100]\n"
            "max_features: [None, sqrt, log2, 0.3]"
        ),
    },
    {
        "Model": "Feed Forward NN",
        "Search": "HalvingGridSearchCV",
        "Grid / search space": (
            "hidden_layer_sizes: [(50,), (25,), (12,), (25, 10), (25, 10, 5)]\n"
            "activation: [relu, tanh]\n"
            "learning_rate: [constant, invscaling, adaptive]\n"
            "learning_rate_init: [0.001, 0.005]\n"
            "max_iter: [2000]\n"
            "tol: [0.0001, 0.0005]\n"
            "early_stopping: [True]\n"
            "validation_fraction: [0.1, 0.2]"
        ),
    },
    {
        "Model": "LSTM",
        "Search": "Optuna, 3 trials",
        "Grid / search space": (
            "days_in: integer [2, 3]\n"
            "input_chunk_length: 365 x days_in\n"
            "output_chunk_length: 1\n"
            "hidden_dim: [8, 16, 32]\n"
            "n_rnn_layers: integer [1, 1]\n"
            "dropout: continuous [0.0, 0.2]\n"
            "activation: [ReLU, tanh]\n"
            "learning_rate: log-uniform [1e-4, 8e-4]\n"
            "batch_size: 128; trial_epochs: 8; final_epochs: 25"
        ),
    },
]


def wrap_cell(text: str, width: int = 115) -> str:
    compact = "; ".join(line.strip() for line in text.splitlines() if line.strip())
    return textwrap.fill(compact, width=width, break_long_words=False)


def main() -> None:
    df = pd.DataFrame(ROWS)
    df.to_csv(OUT_CSV, index=False)

    table_df = df.copy()
    table_df["Grid / search space"] = table_df["Grid / search space"].map(
        lambda s: wrap_cell(s, width=118)
    )

    headers = ["Model", "Search method", "Candidate grid / search space"]
    col_x = [0.02, 0.20, 0.42]
    col_right = [0.18, 0.39, 0.98]
    row_units = [
        max(1.05, row["Grid / search space"].count("\n") * 0.72 + 1.0)
        for _, row in table_df.iterrows()
    ]
    header_units = 0.62
    total_units = header_units + sum(row_units) + 0.04

    fig, ax = plt.subplots(figsize=(13.8, 0.42 * total_units))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_units)

    y = total_units - 0.06
    ax.hlines(y, 0.01, 0.99, colors="black", linewidth=1.0)
    y -= header_units * 0.55
    for label, x in zip(headers, col_x):
        ax.text(x, y, label, ha="left", va="center", fontsize=9.2,
                fontfamily="Times New Roman")
    y -= header_units * 0.42
    ax.hlines(y, 0.01, 0.99, colors="black", linewidth=0.8)

    for i, (_, row) in enumerate(table_df.iterrows()):
        row_h = row_units[i]
        top = y
        bottom = y - row_h
        mid = (top + bottom) / 2

        ax.text(col_x[0], mid, row["Model"], ha="left", va="center",
                fontsize=7.8, fontfamily="Times New Roman")
        ax.text(col_x[1], mid, row["Search"], ha="left", va="center",
                fontsize=7.8, fontfamily="Times New Roman")
        ax.text(col_x[2], top - 0.17, row["Grid / search space"],
                ha="left", va="top", fontsize=7.4,
                fontfamily="Times New Roman", linespacing=1.0)

        if i < len(table_df) - 1:
            ax.hlines(bottom, col_x[2] - 0.01, col_right[2],
                      colors="0.78", linewidth=0.45)
        y = bottom

    ax.hlines(y, 0.01, 0.99, colors="black", linewidth=1.0)

    fig.tight_layout(pad=0.05)
    fig.savefig(OUT_PNG, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
