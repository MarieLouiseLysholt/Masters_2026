"""
Build pooled simulated-index distribution figure for pricing accuracy.

The figure pools bootstrap path index values across all configured mainland
regions and OOS years for each model, with panel-level average strike and
average realised index overlays.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


ROOT = Path(r"/Users/marielouiselysholt/Desktop/copy repo/Masters_2026")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIONS = [11, 24, 27, 28, 32, 44, 52, 53]
OOS_YEARS = list(range(2015, 2025))
MODELS = ["HBA", "Alaton", "Benth", "ARMA", "XGB", "LSTM", "FeedForwardNN", "KNN", "SVM"]

LABELS = {
    "HBA": "Naive",
    "Alaton": "Alaton",
    "Benth": "Benth",
    "ARMA": "ARMA",
    "XGB": "XGBoost",
    "LSTM": "LSTM",
    "FeedForwardNN": "Feed Forward NN",
    "KNN": "KNN",
    "SVM": "SVR",
}

MODEL_COLORS = {
    "HBA": "#000000",
    "Alaton": "#0072B2",
    "Benth": "#D55E00",
    "ARMA": "#009E73",
    "XGB": "#CC79A7",
    "LSTM": "#E69F00",
    "FeedForwardNN": "#56B4E9",
    "KNN": "#8C8C8C",
    "SVM": "#6A3D9A",
}

T_REF = 10.0
HORIZONS = {
    "H1": (1, 90),
    "H2": (91, 181),
}
HORIZON_LABELS = {
    "H1": "H1: Jan-Mar",
    "H2": "H2: Apr-Jun",
}
INDEX_TYPES = ["HDD", "CDD", "CAT"]
STRIKE_HISTORY_START = 1970
N_BINS = 180

PATHS_DIR = ROOT / "04_Analysis" / "02_Bootstrap Paths" / "bootstrap_paths"
TEMP_CSV = ROOT / "02_Data" / "01_Temprature" / "region_avg.csv"
OUTPUT_DIR = ROOT / "99_Thesis Graphs" / "03_Results" / "03_Pricing Accuracy"
OUTPUT_PNG = OUTPUT_DIR / "simulated_index_distributions.png"


def _doy_noleap(ts: pd.Timestamp) -> int:
    doy = ts.timetuple().tm_yday
    if ts.is_leap_year and ts.month > 2:
        doy -= 1
    return doy


def load_temp(temp_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(temp_csv, parse_dates=["date"])
    df.columns = df.columns.str.lower().str.strip()
    reg_col = next(c for c in ["region_code", "region", "reg"] if c in df.columns)
    temp_col = next(c for c in ["daily_avg_temperature", "temp", "temperature"] if c in df.columns)
    df = df.rename(columns={reg_col: "region", temp_col: "temp"})
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))].copy()
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].map(lambda d: _doy_noleap(pd.Timestamp(d)))
    return df[["date", "region", "year", "doy", "temp"]].sort_values(["region", "date"])


def load_paths(region: int, year: int, model: str):
    stem = PATHS_DIR / f"paths_region{region}_{year}_{model}"
    npy_path = stem.with_suffix(".npy")
    date_path = PATHS_DIR / f"{stem.name}_dates.csv"
    if not npy_path.exists() or not date_path.exists():
        return None
    paths = np.load(npy_path, mmap_mode="r")
    path_dates = pd.DatetimeIndex(pd.read_csv(date_path, parse_dates=["date"])["date"])
    if paths.shape[1] != len(path_dates):
        print(f"[skip] shape mismatch: {model} R{region} {year}")
        return None
    return paths, path_dates


def daily_index(temp: np.ndarray, index_type: str) -> np.ndarray:
    if index_type == "CDD":
        return np.maximum(temp - T_REF, 0.0)
    if index_type == "HDD":
        return np.maximum(T_REF - temp, 0.0)
    if index_type == "CAT":
        return temp.astype(float, copy=True)
    raise ValueError(f"Unknown index: {index_type}")


def path_index(paths, path_dates: pd.DatetimeIndex, index_type: str, horizon_window: tuple[int, int]) -> np.ndarray:
    start_doy, end_doy = horizon_window
    doys = np.array([_doy_noleap(pd.Timestamp(d)) for d in path_dates])
    mask = (doys >= start_doy) & (doys <= end_doy)
    if not mask.any():
        raise ValueError(f"No path days in DOY window {horizon_window}")
    sliced = paths[:, mask]
    if index_type == "CDD":
        return np.maximum(sliced - T_REF, 0.0).sum(axis=1)
    if index_type == "HDD":
        return np.maximum(T_REF - sliced, 0.0).sum(axis=1)
    if index_type == "CAT":
        return sliced.sum(axis=1)
    raise ValueError(f"Unknown index: {index_type}")


def historical_window_index(
    temp_df: pd.DataFrame,
    region: int,
    horizon_window: tuple[int, int],
    index_type: str,
    year_start: int,
    year_end: int,
) -> pd.Series:
    start_doy, end_doy = horizon_window
    expected = end_doy - start_doy + 1
    df_r = temp_df[
        (temp_df["region"] == region)
        & (temp_df["year"].between(year_start, year_end))
        & (temp_df["doy"].between(start_doy, end_doy))
    ].copy()
    df_r["idx"] = daily_index(df_r["temp"].to_numpy(), index_type)
    return pd.Series(
        {yr: float(g["idx"].sum()) for yr, g in df_r.groupby("year") if len(g) == expected}
    )


def compute_strike(temp_df: pd.DataFrame, region: int, horizon_window: tuple[int, int], index_type: str, year: int) -> float:
    hist = historical_window_index(
        temp_df, region, horizon_window, index_type, STRIKE_HISTORY_START, year - 1
    )
    if hist.empty:
        return float("nan")
    return float(hist.mean())


def realised_index(temp_df: pd.DataFrame, region: int, horizon_window: tuple[int, int], index_type: str, year: int) -> float:
    start_doy, end_doy = horizon_window
    expected = end_doy - start_doy + 1
    df_r = temp_df[
        (temp_df["region"] == region)
        & (temp_df["year"] == year)
        & (temp_df["doy"].between(start_doy, end_doy))
    ].sort_values("date")
    if len(df_r) != expected:
        return float("nan")
    return float(daily_index(df_r["temp"].to_numpy(), index_type).sum())


def panel_reference_values(temp_df: pd.DataFrame, horizon_window: tuple[int, int], index_type: str) -> tuple[float, float]:
    strikes = []
    realised = []
    for region in REGIONS:
        for year in OOS_YEARS:
            strikes.append(compute_strike(temp_df, region, horizon_window, index_type, year))
            realised.append(realised_index(temp_df, region, horizon_window, index_type, year))
    return float(np.nanmean(strikes)), float(np.nanmean(realised))


def pooled_indices(model: str, horizon_window: tuple[int, int], index_type: str) -> np.ndarray:
    chunks = []
    missing = 0
    for region in REGIONS:
        for year in OOS_YEARS:
            loaded = load_paths(region, year, model)
            if loaded is None:
                missing += 1
                continue
            paths, path_dates = loaded
            try:
                idx = path_index(paths, path_dates, index_type, horizon_window)
            except ValueError as exc:
                print(f"[skip] {model} R{region} {year} {index_type}: {exc}")
                missing += 1
                continue
            chunks.append(np.asarray(idx, dtype=np.float32))
    if missing:
        print(f"  {model} {index_type}: skipped {missing} region-year path files")
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def plot_distribution(ax, values: np.ndarray, bin_edges: np.ndarray, model: str) -> None:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return
    density, edges = np.histogram(values, bins=bin_edges, density=True)
    density = gaussian_filter1d(density, sigma=1.2)
    x = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(
        x,
        density,
        color=MODEL_COLORS.get(model, "0.35"),
        linewidth=1.35,
        label=LABELS.get(model, model),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_df = load_temp(TEMP_CSV)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

    for row, (horizon_name, horizon_window) in enumerate(HORIZONS.items()):
        for col, index_type in enumerate(INDEX_TYPES):
            ax = axes[row, col]
            print(f"Panel: {horizon_name} {index_type}", flush=True)

            by_model = {}
            finite_pool = []
            for model in MODELS:
                vals = pooled_indices(model, horizon_window, index_type)
                by_model[model] = vals
                if vals.size:
                    finite_pool.append(vals[np.isfinite(vals)])
                print(f"  {model}: {vals.size:,} simulated indices", flush=True)

            avg_strike, avg_realised = panel_reference_values(temp_df, horizon_window, index_type)
            if finite_pool:
                pooled = np.concatenate(finite_pool)
                lo, hi = np.nanpercentile(pooled, [0.5, 99.5])
                lo = min(lo, avg_strike, avg_realised)
                hi = max(hi, avg_strike, avg_realised)
                pad = max((hi - lo) * 0.08, 1.0)
                bin_edges = np.linspace(lo - pad, hi + pad, N_BINS + 1)
                for model in MODELS:
                    plot_distribution(ax, by_model[model], bin_edges, model)

            ax.axvline(avg_strike, color="black", linestyle="--", linewidth=1.0, label="Avg strike")
            ax.axvline(avg_realised, color="black", linestyle=":", linewidth=1.15, label="Avg realised")
            ax.set_title(f"{HORIZON_LABELS[horizon_name]} - {index_type}", loc="left", pad=4)
            ax.set_xlabel("Index value")
            ax.set_ylabel("Density")
            ax.grid(axis="y", visible=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Pooled Simulated Index Distributions by Model", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    print(f"saved -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
