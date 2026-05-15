"""
Daily residual diagnostics — Murat-style accuracy table.
Reads bk_forecasts.csv and produces residual autocorrelation diagnostics:

  LB(1..7), LB(1..30), LB(1..90), and LB(1..365) p-values

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

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parents[2]
FORECAST_CSV = Path(__file__).resolve().parent / "bk_forecasts.csv"
OUTPUT_DIR   = ROOT / "99_Thesis Graphs" / "03_Results" / "01_Forecast Accuracy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LB_LAGS = [7, 30, 90, 365]
DPI = 260

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


def per_region_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative Ljung-Box tests per (model, region)."""
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
            if mask.sum() < max(LB_LAGS) + 5:
                continue
            actual = actual[mask]
            resid  = resid[mask]

            lb = acorr_ljungbox(resid, lags=LB_LAGS, return_df=True)
            lb_by_lag = {
                lag: (
                    float(lb.loc[lag, "lb_stat"]),
                    float(lb.loc[lag, "lb_pvalue"]),
                )
                for lag in LB_LAGS
            }

            row = {
                "model":      model,
                "region":     region,
                "n":          int(mask.sum()),
            }
            for lag, (stat, p_value) in lb_by_lag.items():
                row[f"LB{lag}_stat"] = stat
                row[f"LB{lag}_p"] = p_value
            rows.append(row)
    return pd.DataFrame(rows)


def per_model_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detail rows to one row per model.

    Test statistics and p-values are medians across regions.
    """
    out = []
    for model in MODEL_ORDER:
        sub = detail[detail["model"] == model]
        if sub.empty:
            continue
        row = {
            "model":      model,
            "n_regions":  int(len(sub)),
        }
        for lag in LB_LAGS:
            row[f"LB{lag}_stat"] = float(np.median(sub[f"LB{lag}_stat"].values))
            row[f"LB{lag}_p"] = float(np.median(sub[f"LB{lag}_p"].values))
        out.append(row)
    return pd.DataFrame(out)


def _p_str(p):
    if not np.isfinite(p): return "—"
    return ("< 0.0001" if p < 1e-4 else f"{p:.4f}") + stars(p)


def _decision(*p_values: float) -> str:
    finite = [p for p in p_values if np.isfinite(p)]
    if not finite:
        return "n/a"
    return "Reject" if min(finite) < 0.05 else "No reject"


def draw_table(summary: pd.DataFrame, out_path: Path) -> None:
    headers = ["Model", "LB(7) p", "LB(30) p", "LB(90) p", "LB(365) p"]
    cell_text = []
    for _, r in summary.iterrows():
        cell_text.append([
            LABELS.get(r["model"], r["model"]),
            _p_str(r["LB7_p"]),
            _p_str(r["LB30_p"]),
            _p_str(r["LB90_p"]),
            _p_str(r["LB365_p"]),
        ])

    n_rows = len(cell_text)
    fig, ax = plt.subplots(figsize=(7.4, 0.55 + 0.34 * n_rows))
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
                "lb7_p":   _p_str(r["LB7_p"]),
                "lb30_p":  _p_str(r["LB30_p"]),
                "lb90_p":  _p_str(r["LB90_p"]),
                "lb365_p": _p_str(r["LB365_p"]),
                "decision": _decision(r["LB7_p"], r["LB30_p"], r["LB90_p"],
                                      r["LB365_p"]),
            })
        blocks.append((REGION_NAMES.get(region, str(region)), rows))

    headers = ["Site", "Model", "LB(7) p", "LB(30) p",
               "LB(90) p", "LB(365) p", "Decision"]
    col_keys = ["site", "model", "lb7_p", "lb30_p",
                "lb90_p", "lb365_p", "decision"]
    col_x    = [0.00, 0.20, 0.39, 0.52, 0.65, 0.78, 0.93]

    n_data = sum(len(rows) for _, rows in blocks)
    n_blocks = len(blocks)

    row_h  = 0.26
    hdr_h  = 0.32
    foot_h = 0.32
    # extra small gap between region blocks
    gap_h  = 0.06
    fig_w  = 11.0
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

    y_foot = row_h * 0.15
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
    detail_out = detail.copy()
    detail_out["model"] = detail_out["model"].map(lambda m: LABELS.get(m, m))
    detail_out.to_csv(detail_csv, index=False)
    print(f"  -> {detail_csv.name}")

    draw_table(summary, OUTPUT_DIR / "T06_residual_lb_AvgT.png")
    draw_per_region_table(detail,
                          OUTPUT_DIR / "T07_residual_lb_by_region.png")


if __name__ == "__main__":
    main()
