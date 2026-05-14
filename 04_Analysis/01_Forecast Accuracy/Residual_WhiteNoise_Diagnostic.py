"""
Residual white-noise diagnostics for all forecast models.

Reads bk_forecasts.csv, constructs residuals as:

    epsilon(t) = actual(t) - forecast(t)

and produces diagnostics for whether post-forecast residuals still
contain serial dependence.

Outputs:
  T11_residual_acf_all_models.png
  csv_residual_white_noise_all_models.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
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

MAX_LAG = 60
LB_LAGS = [30, 60, 365]
X_TICKS = [1, 10, 20, 30, 40, 50, 60]
DPI = 240

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
    df = df[np.isfinite(df["resid"])]
    return df.sort_values(["model", "region", "date"])


def region_acf(resid: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size <= max_lag + 1 or np.nanstd(resid) <= 0:
        return np.full(max_lag + 1, np.nan)
    return acf(resid, nlags=max_lag, fft=True, missing="drop")


def diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        model_rows = []
        for region, grp in df[df["model"] == model].groupby("region"):
            resid = grp["resid"].to_numpy(dtype=float)
            resid = resid[np.isfinite(resid)]
            if resid.size < max(LB_LAGS) + 5:
                continue

            lb = acorr_ljungbox(resid, lags=LB_LAGS, return_df=True)
            row = {
                "model": model,
                "region": int(region),
                "n": int(resid.size),
                "mean_resid": float(np.mean(resid)),
                "std_resid": float(np.std(resid, ddof=1)),
                "rmse": float(np.sqrt(np.mean(resid ** 2))),
                "mae": float(np.mean(np.abs(resid))),
            }
            for lag in LB_LAGS:
                row[f"LB{lag}_stat"] = float(lb.loc[lag, "lb_stat"])
                row[f"LB{lag}_p"] = float(lb.loc[lag, "lb_pvalue"])
            model_rows.append(row)

        if not model_rows:
            continue

        detail = pd.DataFrame(model_rows)
        w = detail["n"].to_numpy(dtype=float)
        summary = {
            "model": model,
            "label": LABELS.get(model, model),
            "regions": int(detail["region"].nunique()),
            "n": int(detail["n"].sum()),
            "mean_resid": float(np.average(detail["mean_resid"], weights=w)),
            "std_resid_avg": float(np.average(detail["std_resid"], weights=w)),
            "rmse": float(np.sqrt(np.average(detail["rmse"] ** 2, weights=w))),
            "mae": float(np.average(detail["mae"], weights=w)),
        }
        for lag in LB_LAGS:
            p = detail[f"LB{lag}_p"].to_numpy(dtype=float)
            summary[f"LB{lag}_p_median"] = float(np.median(p))
            summary[f"LB{lag}_reject_share_5pct"] = float(np.mean(p < 0.05))
        rows.append(summary)
    return pd.DataFrame(rows)


def draw_acf_grid(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        4, 3, figsize=(12, 11.2), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.38, "wspace": 0.16},
    )
    axes_flat = axes.flatten()
    lags = np.arange(1, MAX_LAG + 1)

    for ax, model in zip(axes_flat, MODEL_ORDER):
        sub = df[df["model"] == model]
        acfs = []
        ns = []
        for _, grp in sub.groupby("region"):
            vals = region_acf(grp["resid"].to_numpy(dtype=float), MAX_LAG)
            if np.isfinite(vals).all():
                acfs.append(vals[1:])
                ns.append(len(grp))

        if not acfs:
            ax.set_visible(False)
            continue

        acfs = np.vstack(acfs)
        mean_acf = np.nanmean(acfs, axis=0)
        n_eff = float(np.median(ns))
        band = 1.96 / np.sqrt(n_eff)

        ax.axhspan(-band, band, color="0.88", zorder=0)
        for row_acf in acfs:
            ax.plot(lags, row_acf, color="0.82", lw=0.55, alpha=0.85)
        ax.plot(lags, mean_acf, color="black", lw=1.35)
        ax.axhline(0, color="black", lw=0.5, linestyle="--", alpha=0.6)

        row = summary[summary["model"] == model]
        p60 = row["LB60_p_median"].iloc[0] if not row.empty else np.nan
        p_text = "<0.0001" if np.isfinite(p60) and p60 < 1e-4 else f"{p60:.4f}"
        ax.text(
            0.98, 0.93, f"median LB(60) p = {p_text}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
        )
        ax.set_title(LABELS.get(model, model), loc="left", fontweight="bold")
        ax.set_xlim(1, MAX_LAG)
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels([str(x) for x in X_TICKS])
        ax.set_ylim(-0.18, 0.85)
        ax.tick_params(labelsize=8)

    for ax in axes[:, 0]:
        ax.set_ylabel("Average residual ACF")
    for ax in axes[-1, :]:
        ax.set_xlabel("Lag in days")
    for ax in axes_flat[len(MODEL_ORDER):]:
        ax.set_visible(False)

    fig.suptitle(
        "Residual autocorrelation by lag — light grey: per-region; black: pooled mean",
        fontsize=11, y=0.995,
    )
    fig.text(
        0.5, 0.01,
        "Residuals are computed as actual minus forecast. Grey bands show "
        "approximate 95% white-noise bounds "
        "using the median per-region sample size.",
        ha="center", fontsize=8,
    )
    out = OUTPUT_DIR / "T11_residual_acf_all_models.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.name}")


def main() -> None:
    df = load_residuals()
    print(f"Loaded {len(df):,} forecast rows | models: {sorted(df['model'].unique())}")
    summary = diagnostics_table(df)
    csv_path = OUTPUT_DIR / "csv_residual_white_noise_all_models.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  -> {csv_path.name}")
    draw_acf_grid(df, summary)


if __name__ == "__main__":
    main()
