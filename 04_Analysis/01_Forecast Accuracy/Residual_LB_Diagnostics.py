"""
Daily residual diagnostics — Murat-style accuracy table.
Reads bk_forecasts.csv and produces a per-model table reporting:

  RMSE  MAE  MASE  LB(365) stat  p-value (lag = 365)

on the daily residuals  ε(t) = actual(t) − forecast(t).

Two outputs:
  T06_residual_lb_AvgT.png    one row per model, aggregated across regions
  T06_residual_lb_by_region.csv  one row per (model, region)  — full detail
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import chi2 as scipy_chi2

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parents[2]
FORECAST_CSV = Path(__file__).resolve().parent / "bk_forecasts.csv"
OUTPUT_DIR   = ROOT / "99_Thesis Graphs" / "03_Results" / "01_Forecast Accuracy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LB_LAG = 365
SEASONAL_M = 365
DPI = 260

MODEL_ORDER = [
    "HBA", "Alaton", "Benth", "ARMA",
    "XGB", "LSTM", "FeedForwardNN", "KNN", "SVM",
]

LABELS = {
    "HBA": "HBA", "Alaton": "Alaton", "Benth": "Benth", "ARMA": "ARMA",
    "XGB": "XGBoost", "LSTM": "LSTM", "FeedForwardNN": "Feed Forward NN",
    "KNN": "KNN", "SVM": "SVM",
}

REGION_NAMES = {
    11: "Île-de-France",           24: "Centre-Val de Loire",
    27: "Bourgogne-Franche-Comté", 28: "Normandie",
    32: "Hauts-de-France",         44: "Grand Est",
    52: "Pays de la Loire",        53: "Bretagne",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def stars(p: float) -> str:
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def mase_from_residuals(actual: np.ndarray, residuals: np.ndarray,
                         m: int = SEASONAL_M) -> float:
    """Hyndman-Koehler MASE with seasonal-naive denominator."""
    if len(actual) <= m:
        return np.nan
    naive = np.abs(actual[m:] - actual[:-m])
    denom = float(np.mean(naive)) if naive.size else np.nan
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.mean(np.abs(residuals)) / denom)


def single_lag_test(resid: np.ndarray, lag: int) -> tuple:
    """
    Single-lag autocorrelation test (non-cumulative).
    Under H₀ of no autocorrelation at this lag, n·r_k² ~ χ²(1).
    Returns (statistic, two-sided p-value).
    """
    n = len(resid)
    if n <= lag + 1:
        return np.nan, np.nan
    e = resid - resid.mean()
    v = float(np.mean(e * e))
    if v <= 0:
        return np.nan, np.nan
    rk = float(np.mean(e[lag:] * e[:-lag]) / v)
    stat = n * rk * rk
    p    = float(1.0 - scipy_chi2.cdf(stat, df=1))
    return stat, p


def monthly_lb_test(dates: pd.Series, resid: np.ndarray,
                     lag: int = 12) -> tuple:
    """
    Aggregate daily residuals to monthly means, then Ljung-Box at lag months.
    Daily weather persistence (~5-15 days) collapses, so what remains
    reflects genuine seasonal / annual mis-fit.
    Returns (statistic, p-value, n_months).
    """
    s = (pd.DataFrame({"date": dates, "e": resid})
         .dropna()
         .assign(ym=lambda d: d["date"].dt.to_period("M"))
         .groupby("ym")["e"].mean()
         .sort_index())
    if len(s) <= lag + 1:
        return np.nan, np.nan, int(len(s))
    out = acorr_ljungbox(s.values, lags=[lag], return_df=True)
    return float(out["lb_stat"].iloc[0]), float(out["lb_pvalue"].iloc[0]), int(len(s))


def per_region_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE/MAE/MASE + three Ljung-Box variants per (model, region):
       LB(1..365)   cumulative on daily residuals — sensitive to weather persistence
       LB@365       single-lag at 365 days only — direct annual-seasonality test
       LB(12)/mo    cumulative on monthly-mean residuals — long-range structure
    """
    rows = []
    for model in MODEL_ORDER:
        for region in sorted(df["region"].unique()):
            sub = (df[(df["model"] == model) & (df["region"] == region)]
                   .dropna(subset=["actual", "forecast"])
                   .sort_values("date"))
            if sub.empty:
                continue
            actual = sub["actual"].values.astype(float)
            resid  = (sub["actual"] - sub["forecast"]).values.astype(float)
            mask   = np.isfinite(resid) & np.isfinite(actual)
            if mask.sum() < LB_LAG + 5:
                continue
            dates  = sub["date"].values
            actual = actual[mask]
            resid  = resid[mask]
            dates  = pd.Series(dates[mask])

            rmse = float(np.sqrt(np.mean(resid ** 2)))
            mae  = float(np.mean(np.abs(resid)))
            mase = mase_from_residuals(actual, resid, m=SEASONAL_M)

            lb = acorr_ljungbox(resid, lags=[LB_LAG], return_df=True)
            lb_stat = float(lb["lb_stat"].iloc[0])
            lb_p    = float(lb["lb_pvalue"].iloc[0])

            sl_stat, sl_p           = single_lag_test(resid, LB_LAG)
            mo_stat, mo_p, n_mo     = monthly_lb_test(dates, resid, lag=12)

            rows.append({
                "model":      model,
                "region":     region,
                "n":          int(mask.sum()),
                "RMSE":       rmse,
                "MAE":        mae,
                "MASE":       mase,
                "LB_stat":    lb_stat,
                "LB_p":       lb_p,
                "SL365_stat": sl_stat,
                "SL365_p":    sl_p,
                "MOLB_stat":  mo_stat,
                "MOLB_p":     mo_p,
                "n_months":   n_mo,
            })
    return pd.DataFrame(rows)


def per_model_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detail rows to one row per model.

    RMSE/MAE: sample-size-weighted root-mean-square / mean across regions.
    MASE:     sample-size-weighted mean across regions.
    LB stats: median across regions.
    LB p:     median across regions (use min for the strictest reading).
    """
    out = []
    for model in MODEL_ORDER:
        sub = detail[detail["model"] == model]
        if sub.empty:
            continue
        w = sub["n"].values.astype(float)
        rmse = float(np.sqrt(np.sum(w * sub["RMSE"].values ** 2) / w.sum()))
        mae  = float(np.sum(w * sub["MAE"].values) / w.sum())
        mase_vals = sub["MASE"].values.astype(float)
        valid     = np.isfinite(mase_vals)
        mase = (float(np.sum(w[valid] * mase_vals[valid]) / w[valid].sum())
                if valid.any() else np.nan)
        lb_stat_med = float(np.median(sub["LB_stat"].values))
        lb_p_med    = float(np.median(sub["LB_p"].values))
        lb_p_min    = float(np.min(sub["LB_p"].values))
        sl_stat_med = float(np.median(sub["SL365_stat"].values))
        sl_p_med    = float(np.median(sub["SL365_p"].values))
        mo_stat_med = float(np.median(sub["MOLB_stat"].values))
        mo_p_med    = float(np.median(sub["MOLB_p"].values))
        out.append({
            "model":      model,
            "RMSE":       rmse,
            "MAE":        mae,
            "MASE":       mase,
            "LB_stat":    lb_stat_med,
            "LB_p_med":   lb_p_med,
            "LB_p_min":   lb_p_min,
            "SL365_stat": sl_stat_med,
            "SL365_p":    sl_p_med,
            "MOLB_stat":  mo_stat_med,
            "MOLB_p":     mo_p_med,
        })
    return pd.DataFrame(out)


def _p_str(p):
    if not np.isfinite(p): return "—"
    return ("< 0.0001" if p < 1e-4 else f"{p:.4f}") + stars(p)


def draw_table(summary: pd.DataFrame, out_path: Path) -> None:
    headers = ["Model", "RMSE", "MAE", "MASE",
               f"LB({LB_LAG})", "p", f"r²·n @365", "p", "LB(12) mo.", "p"]
    cell_text = []
    for _, r in summary.iterrows():
        cell_text.append([
            LABELS.get(r["model"], r["model"]),
            f"{r['RMSE']:.3f}",
            f"{r['MAE']:.3f}",
            f"{r['MASE']:.3f}" if np.isfinite(r["MASE"]) else "—",
            f"{r['LB_stat']:,.0f}",
            _p_str(r["LB_p_med"]),
            f"{r['SL365_stat']:.2f}" if np.isfinite(r["SL365_stat"]) else "—",
            _p_str(r["SL365_p"]),
            f"{r['MOLB_stat']:,.1f}" if np.isfinite(r["MOLB_stat"]) else "—",
            _p_str(r["MOLB_p"]),
        ])

    n_rows = len(cell_text)
    fig, ax = plt.subplots(figsize=(13.5, 0.55 + 0.34 * n_rows))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        cell.visible_edges = ""
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_text_props(ha="center", fontfamily="Times New Roman")
        else:
            if row == n_rows:
                cell.visible_edges = "B"
            ha = "left" if col == 0 else "right"
            cell.set_text_props(ha=ha, fontfamily="Times New Roman")

    fig.text(0.5, 0.01,
             f"Daily residual diagnostics ε(t) = actual(t) − forecast(t).  "
             f"LB({LB_LAG}) cumulative on daily residuals.  "
             "r²·n @365 single-lag χ²(1) test at lag 365 (annual seasonality only).  "
             "LB(12) on monthly-mean residuals.  Stats are median across 8 regions.  "
             "* p<0.05  ** p<0.01  *** p<0.001",
             ha="center", fontsize=7.5)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.name}")


def draw_per_region_table(detail: pd.DataFrame, out_path: Path) -> None:
    """Murat-style per-region × model table.

    Region label appears once per group (centered), models listed below it.
    Mirrors Murat 2018 Tables 3-4 layout.
    """
    from matplotlib.lines import Line2D

    regions = sorted(detail["region"].unique())

    blocks = []
    for region in regions:
        sub = detail[detail["region"] == region]
        sub = sub.set_index("model").reindex(MODEL_ORDER).dropna(how="all")
        rows = []
        for model, r in sub.iterrows():
            rows.append({
                "model":   LABELS.get(model, model),
                "rmse":    f"{r['RMSE']:.3f}",
                "mae":     f"{r['MAE']:.3f}",
                "mase":    f"{r['MASE']:.3f}" if np.isfinite(r["MASE"]) else "—",
                "lb_stat": f"{r['LB_stat']:,.0f}",
                "lb_p":    _p_str(r["LB_p"]),
                "sl_stat": f"{r['SL365_stat']:.2f}" if np.isfinite(r["SL365_stat"]) else "—",
                "sl_p":    _p_str(r["SL365_p"]),
                "mo_stat": f"{r['MOLB_stat']:,.1f}" if np.isfinite(r["MOLB_stat"]) else "—",
                "mo_p":    _p_str(r["MOLB_p"]),
            })
        blocks.append((REGION_NAMES.get(region, str(region)), rows))

    headers = ["Site", "Model", "RMSE", "MAE", "MASE",
               f"LB({LB_LAG})", "p", "r²·n @365", "p", "LB(12) mo.", "p"]
    col_keys = ["site", "model", "rmse", "mae", "mase",
                "lb_stat", "lb_p", "sl_stat", "sl_p", "mo_stat", "mo_p"]
    col_x    = [0.00, 0.16, 0.30, 0.38, 0.46, 0.54, 0.62, 0.71, 0.79, 0.87, 0.95]

    n_data = sum(len(rows) for _, rows in blocks)
    n_blocks = len(blocks)

    row_h  = 0.26
    hdr_h  = 0.32
    foot_h = 0.32
    # extra small gap between region blocks
    gap_h  = 0.06
    fig_w  = 14.5
    fig_h  = (hdr_h + row_h * n_data + gap_h * max(0, n_blocks - 1)
              + foot_h + 0.05)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    BLACK = "#000000"

    def hline(y, lw):
        ax.add_line(Line2D([0, 1], [y, y], lw=lw, color=BLACK,
                           solid_capstyle="butt", transform=ax.transAxes))

    def hline_partial(x0, x1, y, lw):
        ax.add_line(Line2D([x0, x1], [y, y], lw=lw, color=BLACK,
                           solid_capstyle="butt", transform=ax.transAxes))

    def txt(fx, y, text, bold=False, fontsize=8.5, ha="left"):
        offset = 0.005 if ha == "left" else 0.0
        ax.text(fx + offset, y, text, fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                ha=ha, va="center", color=BLACK,
                transform=ax.transAxes)

    def ay(y_data):
        return y_data / fig_h

    hline(ay(fig_h), 1.2)

    y_hdr = fig_h - hdr_h
    for label, fx in zip(headers, col_x):
        txt(fx, ay(y_hdr + hdr_h * 0.5), label, bold=True)
    hline(ay(y_hdr), 0.7)

    # Render blocks
    cursor_y = y_hdr
    for bi, (site, rows) in enumerate(blocks):
        block_top = cursor_y
        # Site label centered vertically across the block
        block_n   = len(rows)
        block_h   = row_h * block_n
        site_y    = block_top - block_h * 0.5
        txt(col_x[0], ay(site_y), site, fontsize=9, bold=False)

        for i, row in enumerate(rows):
            y_row = block_top - row_h * (i + 1)
            y_mid = ay(y_row + row_h * 0.5)
            for fx, key in zip(col_x[1:], col_keys[1:]):
                txt(fx, y_mid, row[key])

        cursor_y = block_top - block_h
        # Thin separator between blocks (not after the last)
        if bi < n_blocks - 1:
            hline_partial(col_x[1] - 0.005, 1.0, ay(cursor_y - gap_h * 0.5), 0.3)
            cursor_y -= gap_h

    # Footer
    y_foot = row_h * 0.15
    hline(ay(y_foot + foot_h * 0.92), 0.5)
    txt(0.0, ay(y_foot + foot_h * 0.45),
        "Daily residual diagnostics  ε(t) = actual(t) − forecast(t).  "
        f"LB({LB_LAG}) cumulative on daily residuals (sensitive to weather "
        "persistence).  r²·n @365 single-lag χ²(1) at lag 365 (annual seasonality).  "
        "LB(12) on monthly-mean residuals (long-range structure).  "
        "* p<0.05  ** p<0.01  *** p<0.001",
        fontsize=7.5)
    hline(ay(y_foot - 0.01), 1.2)

    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.name}")


def main():
    df = pd.read_csv(FORECAST_CSV, parse_dates=["date"])
    df = df.dropna(subset=["actual", "forecast"])
    print(f"Loaded {len(df):,} rows  |  models: {sorted(df['model'].unique())}")

    detail  = per_region_diagnostics(df)
    summary = per_model_summary(detail)

    detail_csv = OUTPUT_DIR / "csv_residual_lb_by_region.csv"
    detail.to_csv(detail_csv, index=False)
    print(f"  -> {detail_csv.name}")

    draw_table(summary, OUTPUT_DIR / "T06_residual_lb_AvgT.png")
    draw_per_region_table(detail,
                          OUTPUT_DIR / "T07_residual_lb_by_region.png")


if __name__ == "__main__":
    main()
