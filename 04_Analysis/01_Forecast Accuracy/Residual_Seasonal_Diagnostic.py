"""
Visual diagnostic for residual seasonality.

For each model, plots the mean residual ε̄(d) = mean ε over all years × regions
where day-of-year(t) = d.  A model that captured seasonality gives a flat
scatter around zero.  A smooth wave reveals systematic calendar bias.

Outputs:
  T08_residual_by_dayofyear.png   3×3 grid, one panel per model
  T09_residual_by_month.png       3×3 grid, monthly aggregation (cleaner read)
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parents[2]
FORECAST_CSV = Path(__file__).resolve().parent / "bk_forecasts.csv"
OUTPUT_DIR   = ROOT / "99_Thesis Graphs" / "03_Results" / "01_Forecast Accuracy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "HBA", "Alaton", "Benth", "ARMA",
    "XGB", "LSTM", "FeedForwardNN", "KNN", "SVM", "RF",
]
LABELS = {
    "HBA": "Naïve", "Alaton": "Alaton", "Benth": "Benth", "ARMA": "ARMA",
    "XGB": "XGBoost", "LSTM": "LSTM", "FeedForwardNN": "Feed Forward NN",
    "KNN": "KNN", "SVM": "SVR", "RF": "Random Forest",
}
REGION_NAMES = {
    11: "Île-de-France",           24: "Centre-Val de Loire",
    27: "Bourgogne-Franche-Comté", 28: "Normandie",
    32: "Hauts-de-France",         44: "Grand Est",
    52: "Pays de la Loire",        53: "Bretagne",
}

DPI = 220
SMOOTH_WINDOW = 15  # days, for the DOY plot

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def load() -> pd.DataFrame:
    df = pd.read_csv(FORECAST_CSV, parse_dates=["date"])
    df = df.dropna(subset=["actual", "forecast"])
    df["resid"] = df["actual"] - df["forecast"]
    df["doy"]   = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df = df[df["doy"] != 366]                       # drop leap-day extra
    return df


def doy_panel(ax, model: str, df: pd.DataFrame) -> None:
    sub = df[df["model"] == model]
    if sub.empty:
        ax.set_visible(False)
        return

    # Faint per-region lines + heavy pooled mean
    for region, grp in sub.groupby("region"):
        m = grp.groupby("doy")["resid"].mean()
        m_smooth = m.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
        ax.plot(m_smooth.index, m_smooth.values,
                color="0.75", lw=0.6, alpha=0.85)

    pooled = sub.groupby("doy")["resid"].mean()
    pooled_smooth = pooled.rolling(SMOOTH_WINDOW, center=True,
                                    min_periods=1).mean()
    ax.plot(pooled_smooth.index, pooled_smooth.values,
            color="black", lw=1.5)

    ax.axhline(0, color="black", lw=0.5, linestyle="--", alpha=0.6)
    ax.set_xlim(1, 365)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks([1, 91, 182, 274, 365])
    ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"], fontsize=8)
    ax.set_title(LABELS.get(model, model), fontsize=10, loc="left",
                 fontweight="bold")
    ax.tick_params(labelsize=7)
    rmse_doy = float(np.sqrt(np.mean(pooled.values ** 2)))
    ax.text(0.97, 0.95,
            f"RMS of ε̄(d) = {rmse_doy:.2f} °C",
            transform=ax.transAxes, fontsize=7.5,
            ha="right", va="top", color="0.3")


def month_panel(ax, model: str, df: pd.DataFrame) -> None:
    sub = df[df["model"] == model]
    if sub.empty:
        ax.set_visible(False)
        return

    # Per-region monthly means (faint), pooled (heavy)
    region_means = (sub.groupby(["region", "month"])["resid"]
                    .mean().unstack("month"))
    months = np.arange(1, 13)
    for _, row in region_means.iterrows():
        ax.plot(months, row.values, color="0.75", lw=0.6, alpha=0.85,
                marker="o", markersize=2)

    pooled = sub.groupby("month")["resid"].mean()
    ax.plot(months, pooled.values, color="black", lw=1.5,
            marker="o", markersize=4)

    ax.axhline(0, color="black", lw=0.5, linestyle="--", alpha=0.6)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks(months)
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                       fontsize=8)
    ax.set_title(LABELS.get(model, model), fontsize=10, loc="left",
                 fontweight="bold")
    ax.tick_params(labelsize=7)


def time_panel(ax, model: str, df: pd.DataFrame) -> None:
    sub = df[df["model"] == model]
    if sub.empty:
        ax.set_visible(False)
        return

    sub = sub.copy()
    sub["ym"] = sub["date"].dt.to_period("M").dt.to_timestamp()

    pooled = sub.groupby("ym")["resid"].mean().sort_index()
    ax.plot(pooled.index, pooled.values, color="black", lw=1.1)
    ax.axhline(0, color="black", lw=0.5, linestyle=":", alpha=0.6)

    ax.set_ylim(-5, 5)
    ax.set_title(LABELS.get(model, model), fontsize=10, loc="left",
                 fontweight="bold")
    ax.tick_params(labelsize=7)


def make_grid(df: pd.DataFrame, panel_fn, title: str, out_name: str,
              ylabel: str) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(12, 11.2),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.38, "wspace": 0.16})
    axes_flat = axes.flatten()

    present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    for i, model in enumerate(present):
        panel_fn(axes_flat[i], model, df)

    for j in range(len(present), len(axes_flat)):
        axes_flat[j].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=9)

    fig.suptitle(title, fontsize=11, y=0.995)
    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.name}")


def main():
    df = load()
    print(f"Loaded {len(df):,} rows  |  models: {sorted(df['model'].unique())}")

    make_grid(
        df, doy_panel,
        title=("Mean residual by day-of-year — light grey lines: per-region; "
               "black line: pooled across regions  (smoothed, "
               f"{SMOOTH_WINDOW}-day window)"),
        out_name="T08_residual_by_dayofyear.png",
        ylabel="ε̄(d)  [°C]",
    )

    make_grid(
        df, month_panel,
        title=("Mean residual by calendar month — light grey: per-region; "
               "black: pooled"),
        out_name="T09_residual_by_month.png",
        ylabel="ε̄(month)  [°C]",
    )

    make_grid(
        df, time_panel,
        title="Residual over the full forecasting range — black: pooled monthly residual across regions",
        out_name="T10_residual_over_time.png",
        ylabel="ε̄  [°C]",
    )


if __name__ == "__main__":
    main()
