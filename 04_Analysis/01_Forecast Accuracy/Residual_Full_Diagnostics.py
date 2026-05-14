"""
Full residual diagnostics for all forecast models.

Complements the existing residual plots:
  T08/T09  residual seasonality by day/month
  T10      residual drift over time
  T11      residual autocorrelation / white-noise check

This script adds:
  T12_residual_distribution_all_models.png
  T13_residual_qq_all_models.png
  T14_residual_abs_by_month_all_models.png
  T15_residual_squared_acf_all_models.png
  csv_residual_distribution_variance_diagnostics.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kurtosis, norm, probplot, skew
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
FORECAST_CSV = Path(__file__).resolve().parent / "bk_forecasts.csv"
OUTPUT_DIR = ROOT / "99_Thesis Graphs" / "03_Results" / "01_Forecast Accuracy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "HBA", "Alaton", "Benth", "ARMA",
    "XGB", "LSTM", "FeedForwardNN", "KNN", "SVM", "RF",
]
LABELS = {
    "HBA": "Naïve",
    "Alaton": "Alaton",
    "Benth": "Benth",
    "ARMA": "ARMA",
    "XGB": "XGBoost",
    "LSTM": "LSTM",
    "FeedForwardNN": "Feed Forward NN",
    "KNN": "KNN",
    "SVM": "SVR",
    "RF": "Random Forest",
}

DPI = 240
MAX_LAG = 60
ARCH_LAG = 30
X_TICKS = [1, 10, 20, 30, 40, 50, 60]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_residuals() -> pd.DataFrame:
    df = pd.read_csv(FORECAST_CSV, parse_dates=["date"])
    df = df[df["model"].isin(MODEL_ORDER)].dropna(subset=["actual", "forecast"])
    df["resid"] = df["actual"] - df["forecast"]
    df["abs_resid"] = df["resid"].abs()
    df["sq_resid"] = df["resid"] ** 2
    df["month"] = df["date"].dt.month
    df = df[np.isfinite(df["resid"])]
    return df.sort_values(["model", "region", "date"])


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.array([])
    s = np.std(x, ddof=1)
    if not np.isfinite(s) or s <= 0:
        return np.array([])
    return (x - np.mean(x)) / s


def clipped_standardized_resid(df: pd.DataFrame, model: str) -> np.ndarray:
    z = standardize(df.loc[df["model"] == model, "resid"].to_numpy(dtype=float))
    return z[np.abs(z) <= 6]


def squared_acf(vals: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= max_lag + 1:
        return np.full(max_lag + 1, np.nan)
    centered = vals - np.mean(vals)
    sq = centered ** 2
    if np.std(sq) <= 0:
        return np.full(max_lag + 1, np.nan)
    return acf(sq, nlags=max_lag, fft=True, missing="drop")


def diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        model_rows = []
        for region, grp in df[df["model"] == model].groupby("region"):
            resid = grp["resid"].to_numpy(dtype=float)
            resid = resid[np.isfinite(resid)]
            if resid.size < ARCH_LAG + 5:
                continue
            arch_stat, arch_p, _, _ = het_arch(resid - np.mean(resid), nlags=ARCH_LAG)
            model_rows.append({
                "model": model,
                "region": int(region),
                "n": int(resid.size),
                "mean": float(np.mean(resid)),
                "std": float(np.std(resid, ddof=1)),
                "skew": float(skew(resid, bias=False)),
                "excess_kurtosis": float(kurtosis(resid, fisher=True, bias=False)),
                "p01": float(np.quantile(resid, 0.01)),
                "p99": float(np.quantile(resid, 0.99)),
                "ARCH30_stat": float(arch_stat),
                "ARCH30_p": float(arch_p),
            })

        if not model_rows:
            continue
        detail = pd.DataFrame(model_rows)
        w = detail["n"].to_numpy(dtype=float)
        rows.append({
            "model": model,
            "label": LABELS.get(model, model),
            "regions": int(detail["region"].nunique()),
            "n": int(detail["n"].sum()),
            "mean": float(np.average(detail["mean"], weights=w)),
            "std": float(np.average(detail["std"], weights=w)),
            "skew_median": float(np.median(detail["skew"])),
            "excess_kurtosis_median": float(np.median(detail["excess_kurtosis"])),
            "p01_median": float(np.median(detail["p01"])),
            "p99_median": float(np.median(detail["p99"])),
            "ARCH30_p_median": float(np.median(detail["ARCH30_p"])),
            "ARCH30_reject_share_5pct": float(np.mean(detail["ARCH30_p"] < 0.05)),
        })
    return pd.DataFrame(rows)


def make_grid(panel_fn, title: str, out_name: str, ylabel: str = "") -> None:
    fig, axes = plt.subplots(
        4, 3, figsize=(12, 11.2), sharex=False, sharey=False,
        gridspec_kw={"hspace": 0.38, "wspace": 0.16},
    )
    axes_flat = axes.flatten()
    for ax, model in zip(axes_flat, MODEL_ORDER):
        panel_fn(ax, model)
    for ax in axes_flat[len(MODEL_ORDER):]:
        ax.set_visible(False)
    if ylabel:
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=11, y=0.995)
    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.name}")


def draw_distribution(df: pd.DataFrame) -> None:
    def panel(ax, model):
        z = clipped_standardized_resid(df, model)
        if z.size < 20:
            ax.set_visible(False)
            return
        xs = np.linspace(-5, 5, 300)
        ax.hist(z, bins=55, density=True, color="0.86", edgecolor="white")
        ax.plot(xs, gaussian_kde(z)(xs), color="black", lw=1.35)
        ax.plot(xs, norm.pdf(xs), color="crimson", lw=1.0, linestyle="--")
        ax.axvline(0, color="black", lw=0.5, linestyle=":", alpha=0.7)
        ax.set_xlim(-5, 5)
        ax.set_title(LABELS.get(model, model), loc="left", fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.text(
            0.98, 0.93,
            f"skew={skew(z):+.2f}\nkurt={kurtosis(z, fisher=True):+.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        )

    make_grid(
        panel,
        "Standardized residual distribution — grey: empirical; black: KDE; red dashed: normal",
        "T12_residual_distribution_all_models.png",
        ylabel="Density",
    )


def draw_qq(df: pd.DataFrame) -> None:
    def panel(ax, model):
        z = clipped_standardized_resid(df, model)
        if z.size < 20:
            ax.set_visible(False)
            return
        osm, osr = probplot(z, dist="norm", fit=False)
        ax.scatter(osm, osr, s=4, color="0.35", alpha=0.28, linewidths=0)
        lim = max(abs(np.nanmin(osm)), abs(np.nanmax(osm)), abs(np.nanmin(osr)), abs(np.nanmax(osr)))
        lim = min(max(lim, 3), 6)
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(LABELS.get(model, model), loc="left", fontweight="bold")
        ax.tick_params(labelsize=8)

    make_grid(
        panel,
        "Normal QQ plot of standardized residuals",
        "T13_residual_qq_all_models.png",
        ylabel="Empirical quantile",
    )


def draw_abs_by_month(df: pd.DataFrame) -> None:
    months = np.arange(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    def panel(ax, model):
        sub = df[df["model"] == model]
        if sub.empty:
            ax.set_visible(False)
            return
        region_means = sub.groupby(["region", "month"])["abs_resid"].mean().unstack("month")
        for _, row in region_means.iterrows():
            ax.plot(months, row.reindex(months).values, color="0.82", lw=0.55, alpha=0.85)
        pooled = sub.groupby("month")["abs_resid"].mean().reindex(months)
        ax.plot(months, pooled.values, color="black", lw=1.35, marker="o", markersize=3)
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels)
        ax.set_ylim(0, 4.8)
        ax.set_title(LABELS.get(model, model), loc="left", fontweight="bold")
        ax.tick_params(labelsize=8)

    make_grid(
        panel,
        "Mean absolute residual by calendar month — light grey: per-region; black: pooled",
        "T14_residual_abs_by_month_all_models.png",
        ylabel="Mean |residual| [deg C]",
    )


def draw_squared_acf(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    lags = np.arange(1, MAX_LAG + 1)

    def panel(ax, model):
        sub = df[df["model"] == model]
        acfs = []
        ns = []
        for _, grp in sub.groupby("region"):
            vals = squared_acf(grp["resid"].to_numpy(dtype=float), MAX_LAG)
            if np.isfinite(vals).all():
                acfs.append(vals[1:])
                ns.append(len(grp))
        if not acfs:
            ax.set_visible(False)
            return
        acfs_arr = np.vstack(acfs)
        mean_acf = np.nanmean(acfs_arr, axis=0)
        band = 1.96 / np.sqrt(float(np.median(ns)))
        ax.axhspan(-band, band, color="0.88", zorder=0)
        for row_acf in acfs_arr:
            ax.plot(lags, row_acf, color="0.82", lw=0.55, alpha=0.85)
        ax.plot(lags, mean_acf, color="black", lw=1.35)
        ax.axhline(0, color="black", lw=0.5, linestyle="--", alpha=0.6)
        row = summary[summary["model"] == model]
        p = row["ARCH30_p_median"].iloc[0] if not row.empty else np.nan
        p_text = "<0.0001" if np.isfinite(p) and p < 1e-4 else f"{p:.4f}"
        ax.text(
            0.98, 0.93, f"median ARCH(30) p = {p_text}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        )
        ax.set_xlim(1, MAX_LAG)
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels([str(x) for x in X_TICKS])
        ax.set_ylim(-0.08, 0.45)
        ax.set_title(LABELS.get(model, model), loc="left", fontweight="bold")
        ax.tick_params(labelsize=8)

    make_grid(
        panel,
        "Squared-residual autocorrelation by lag — light grey: per-region; black: pooled mean",
        "T15_residual_squared_acf_all_models.png",
        ylabel="ACF of squared residuals",
    )


def main() -> None:
    df = load_residuals()
    print(f"Loaded {len(df):,} forecast rows | models: {sorted(df['model'].unique())}")
    summary = diagnostics_table(df)
    csv_path = OUTPUT_DIR / "csv_residual_distribution_variance_diagnostics.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  -> {csv_path.name}")
    draw_distribution(df)
    draw_qq(df)
    draw_abs_by_month(df)
    draw_squared_acf(df, summary)


if __name__ == "__main__":
    main()
