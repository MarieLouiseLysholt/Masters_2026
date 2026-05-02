"""
bk_forecast_accuracy.py  [v4]
==============================
Reads bk_forecasts.csv and produces four publication-ready tables
(black-and-white, Times New Roman).

Outputs (all written to OUTPUT_DIR)
--------------------------------------
  T01_accuracy_metrics.png     ME / MAE / RMSE / MAPE — per model
  T02_dm_test.png              Diebold-Mariano pairwise test matrix
  T03_metrics_by_region.png    ME / MAE / RMSE / MAPE — per NUTS-2 region
  T04_metrics_by_year.png      ME / MAE / RMSE / MAPE — per OOS year

Each output is produced for all four index types (AvgT, HDD, CDD, CAT)
as a 2×2 panel figure.

Methodology notes
-----------------
Metrics
  ME   = mean(forecast − actual)            [signed bias; +ve = over-forecast]
  MAE  = mean(|forecast − actual|)
  RMSE = sqrt(mean((forecast − actual)²))
  MAPE = mean(|error| / |actual|) × 100
         HDD/CDD: months with actual < MIN_INDEX excluded from MAPE.
         AvgT/CAT: months with |actual| < 0.5 excluded from MAPE.

Metrics are computed on monthly aggregated index values.

Diebold-Mariano test
  Loss:      squared error  L(e) = e²
  d_{i}:     L(e_model_i) − L(e_model_j) per matched (region, year, month)
  Variance:  Newey-West HAC (h = 1 lag) with Harvey-Leybourne-Newbold
             small-sample correction factor √((n+1−2h+h(h−1)/n)/n)
  H0:        equal predictive accuracy (two-sided)
  Positive DM stat => row model has higher loss (column model is more accurate)
  Multiple testing: Benjamini-Hochberg FDR within each index type

T03 / T04
  Metrics aggregated across ALL models.  Purpose: reveal how forecast
  difficulty varies by region and by OOS year, not model ranking.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm as scipy_norm

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT         = Path(__file__).resolve().parents[2]
FORECAST_CSV = Path(__file__).resolve().parent / "bk_forecasts.csv"
OUTPUT_DIR   = ROOT / "99_Thesis Graphs" / "03_Results" / "01_Forecast Accuracy"

T_REF     = 10.0   # degree-day threshold (°C)
MIN_INDEX = 1.0    # min actual HDD / CDD for MAPE denominator
MAPE_AVGT_MIN = 0.5  # min |actual AvgT| and |actual CAT/n| for MAPE denom
DPI       = 260

# Full candidate model list — script filters to those present in the CSV
MODEL_ORDER = [
    "HBA", "Alaton", "Benth", "ARMA",
    "XGB", "LSTM", "FeedForwardNN",
    "KNN", "SVM",
]

BASELINE = "HBA"

LABELS = {
    "HBA":           "HBA",
    "Alaton":        "Alaton",
    "Benth":         "Benth",
    "ARMA":          "ARMA",
    "XGB":           "XGBoost",
    "LSTM":          "LSTM",
    "FeedForwardNN": "Feed Forward NN",
    "KNN":           "KNN",
    "SVM":           "SVM",
}

REGION_NAMES = {
    11: "Île-de-France",
    24: "Centre-Val de Loire",
    27: "Bourgogne-Franche-Comté",
    28: "Normandie",
    32: "Hauts-de-France",
    44: "Grand Est",
    52: "Pays de la Loire",
    53: "Bretagne",
}

INDEX_TYPES = ["AvgT", "HDD", "CDD", "CAT"]
INDEX_LABELS = {
    "AvgT": "Average Temperature (°C / month)",
    "HDD":  r"HDD  (T$_{ref}$ = 10 °C)",
    "CDD":  r"CDD  (T$_{ref}$ = 10 °C)",
    "CAT":  "CAT  (°C · days)",
}

# ── global rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

def save_fig(fig, stem: str) -> None:
    p = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {p.name}")


def _fmt(v, decimals=3):
    """Format float; return em-dash for NaN / Inf."""
    try:
        if pd.isna(v) or not np.isfinite(float(v)):
            return "—"
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _stars(p: float) -> str:
    """Significance stars from p-value."""
    if pd.isna(p) or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _bh_correct(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.  NaN pass through."""
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    p_fin  = p[finite]
    m = len(p_fin)
    if m == 0:
        return p.copy()
    order    = np.argsort(p_fin)
    sorted_p = p_fin[order]
    adj      = sorted_p * m / np.arange(1, m + 1)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    result        = np.empty(m)
    result[order] = adj
    out           = p.copy()
    out[finite]   = result
    return out


def _draw_econ_table(ax, cell_text, col_labels, row_labels,
                     fontsize=9,
                     col_width_scale=1.0, row_height_scale=1.5):
    """
    Plain econometric-literature B&W table: top rule, mid rule under header,
    bottom rule. No shading, no bold, no titles, no footnotes.
    """
    ax.axis("off")
    n_rows = len(cell_text)

    tbl = ax.table(
        cellText  = cell_text,
        colLabels = col_labels,
        rowLabels = row_labels,
        loc       = "center",
        cellLoc   = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(col_width_scale, row_height_scale)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        cell.visible_edges = ""

        if row == 0:                       # header row
            cell.visible_edges = "TB"      # top rule + mid rule
            cell.set_text_props(fontweight="normal",
                                ha="left" if col == -1 else "center",
                                fontfamily="Times New Roman")
        else:
            if row == n_rows:              # last data row → bottom rule
                cell.visible_edges = "B"
            ha = "left" if col == -1 else "right"
            cell.set_text_props(ha=ha, fontweight="normal",
                                fontfamily="Times New Roman")
    return tbl


# ============================================================================
# DATA LOADING
# ============================================================================

def load_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["actual", "forecast"])
    # Filter to models that are actually present
    present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    df = df[df["model"].isin(present)].copy()
    df["month"] = df["date"].dt.month
    return df


def active_models(df: pd.DataFrame) -> list:
    """Return MODEL_ORDER subset actually present in the data."""
    present = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in present]


# ============================================================================
# MONTHLY INDEX COMPUTATION
# ============================================================================

def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily temps to monthly HDD, CDD, CAT, AvgT per model."""

    def _agg(grp):
        a = grp["actual"].values.astype(float)
        f = grp["forecast"].values.astype(float)
        return pd.Series({
            "HDD_actual":    np.sum(np.maximum(0.0, T_REF - a)),
            "HDD_forecast":  np.sum(np.maximum(0.0, T_REF - f)),
            "CDD_actual":    np.sum(np.maximum(0.0, a - T_REF)),
            "CDD_forecast":  np.sum(np.maximum(0.0, f - T_REF)),
            "CAT_actual":    np.sum(a),
            "CAT_forecast":  np.sum(f),
            "AvgT_actual":   np.mean(a),
            "AvgT_forecast": np.mean(f),
            "n_days":        len(a),
        })

    mon = (
        df.groupby(["region", "year", "month", "model"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    for idx in INDEX_TYPES:
        e = mon[f"{idx}_forecast"] - mon[f"{idx}_actual"]
        mon[f"{idx}_error"]    = e
        mon[f"{idx}_sq_error"] = e ** 2
        mon[f"{idx}_abs_error"] = e.abs()

    return mon


# ============================================================================
# METRICS  (ME, MAE, RMSE, MAPE)
# ============================================================================

def _mape_mask(actual: np.ndarray, idx: str) -> np.ndarray:
    """Boolean mask selecting rows safe to include in MAPE."""
    if idx in ("HDD", "CDD"):
        return np.abs(actual) >= MIN_INDEX
    else:  # AvgT, CAT
        return np.abs(actual) >= MAPE_AVGT_MIN


def _metrics_from_arrays(errors: np.ndarray, actual: np.ndarray,
                          idx: str) -> dict:
    """Compute ME, MAE, RMSE, MAPE from aligned arrays."""
    if len(errors) == 0:
        return dict(ME=np.nan, MAE=np.nan, RMSE=np.nan, MAPE=np.nan)
    me   = float(np.mean(errors))
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mask = _mape_mask(actual, idx)
    mape = (float(np.mean(np.abs(errors[mask]) / np.abs(actual[mask])) * 100)
            if mask.sum() > 0 else np.nan)
    return dict(ME=me, MAE=mae, RMSE=rmse, MAPE=mape)


def compute_metrics_by_model(df_m: pd.DataFrame, idx: str,
                              models: list) -> pd.DataFrame:
    """ME, MAE, RMSE, MAPE per model (rows) for a given index type."""
    rows = []
    for model in models:
        sub = df_m[df_m["model"] == model]
        if len(sub) == 0:
            rows.append(dict(model=model, ME=np.nan, MAE=np.nan,
                             RMSE=np.nan, MAPE=np.nan, N=0))
            continue
        m = _metrics_from_arrays(sub[f"{idx}_error"].values,
                                  sub[f"{idx}_actual"].values, idx)
        m["model"] = model
        m["N"]     = len(sub)
        rows.append(m)
    return pd.DataFrame(rows).set_index("model")


def compute_metrics_by_group(df_m: pd.DataFrame, idx: str,
                              group_col: str) -> pd.DataFrame:
    """
    ME, MAE, RMSE, MAPE per region or per year, aggregating across ALL models.
    """
    groups = sorted(df_m[group_col].unique())
    rows = []
    for g in groups:
        sub = df_m[df_m[group_col] == g]
        m = _metrics_from_arrays(sub[f"{idx}_error"].values,
                                  sub[f"{idx}_actual"].values, idx)
        m[group_col] = g
        rows.append(m)
    return pd.DataFrame(rows).set_index(group_col)


# ============================================================================
# DIEBOLD-MARIANO TEST
# ============================================================================

def _dm_stat(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple:
    """
    Diebold-Mariano (1995) test, squared-error loss.
    Harvey, Leybourne & Newbold (1997) small-sample correction applied.

    Parameters
    ----------
    e1, e2  : forecast error arrays for model 1 and model 2 (matched pairs)
    h       : forecast horizon (set to 1 for monthly indices)

    Returns
    -------
    (dm_stat, two-sided p-value)
    Positive dm_stat => model 1 has higher squared loss (model 2 is more accurate).
    """
    d = e1 ** 2 - e2 ** 2          # loss differential
    n = len(d)
    if n < 10:
        return np.nan, np.nan

    d_bar = d.mean()

    # Newey-West HAC variance with h−1 autocovariance lags
    gamma0 = float(np.var(d, ddof=0))
    gamma_sum = 0.0
    for lag in range(1, h):
        if n > lag:
            gamma_sum += float(
                np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
            )
    V_d = (gamma0 + 2.0 * gamma_sum) / n
    if V_d <= 0:
        return np.nan, np.nan

    # HLN small-sample correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm = (d_bar / np.sqrt(V_d)) * correction

    p = float(2.0 * (1.0 - scipy_norm.cdf(abs(dm))))
    return float(dm), p


def compute_dm_matrices(df_m: pd.DataFrame, idx: str,
                        models: list) -> tuple:
    """
    Compute pairwise DM test for all model pairs.

    Returns
    -------
    dm_stat_df : DataFrame of DM statistics (row vs col)
    adj_p_df   : DataFrame of BH-adjusted p-values
    """
    # Pivot to matched pairs on (region, year, month)
    pivot = df_m.pivot_table(
        index=["region", "year", "month"],
        columns="model",
        values=f"{idx}_error",
        aggfunc="first",
    )

    dm_mat  = pd.DataFrame(np.nan, index=models, columns=models)
    p_raw   = pd.DataFrame(np.nan, index=models, columns=models)

    pairs, plist = [], []

    for mi in models:
        for mj in models:
            if mi == mj:
                continue
            if mi not in pivot.columns or mj not in pivot.columns:
                continue
            matched = pivot[[mi, mj]].dropna()
            if len(matched) < 10:
                continue
            dm, p = _dm_stat(matched[mi].values, matched[mj].values, h=1)
            dm_mat.loc[mi, mj] = dm
            p_raw.loc[mi, mj]  = p
            pairs.append((mi, mj))
            plist.append(p)

    # BH correction across all pairs for this index type
    adj_p = pd.DataFrame(np.nan, index=models, columns=models)
    if plist:
        corr = _bh_correct(np.array(plist))
        for (mi, mj), pc in zip(pairs, corr):
            adj_p.loc[mi, mj] = float(pc)

    return dm_mat, adj_p


# ============================================================================
# FIGURE / TABLE RENDERING
# ============================================================================

# ── T01 — Accuracy metrics per model (one PNG per index) ───────────────────

def fig_metrics_by_model_for_index(monthly: pd.DataFrame, models: list,
                                   idx: str) -> None:
    met = compute_metrics_by_model(monthly, idx, models)

    row_labels = [LABELS[m] for m in models]
    col_labels = ["ME", "MAE", "RMSE", "MAPE (%)"]
    cell_text  = []

    for model in models:
        r = met.loc[model]
        cell_text.append([
            _fmt(r["ME"],   3),
            _fmt(r["MAE"],  3),
            _fmt(r["RMSE"], 3),
            _fmt(r["MAPE"], 2),
        ])

    fig, ax = plt.subplots(figsize=(7, 0.45 + 0.32 * len(models)))
    _draw_econ_table(ax, cell_text, col_labels, row_labels, fontsize=9)
    save_fig(fig, f"T01_metrics_{idx}")


# ── T02 — Diebold-Mariano pairwise test ─────────────────────────────────────

def fig_dm_for_index(monthly: pd.DataFrame, models: list, idx: str) -> None:
    n_mod = len(models)
    dm_mat, adj_p = compute_dm_matrices(monthly, idx, models)

    row_labels = [LABELS[m] for m in models]
    col_labels = [LABELS[m] for m in models]
    cell_text  = []

    for mi in models:
        row = []
        for mj in models:
            if mi == mj:
                row.append("—"); continue
            dm = dm_mat.loc[mi, mj]
            pa = adj_p.loc[mi, mj]
            if pd.isna(dm) or not np.isfinite(dm):
                row.append("n/a")
            else:
                row.append(f"{dm:+.2f}{_stars(pa)}")
        cell_text.append(row)

    fs = max(5, 8 - max(0, n_mod - 6))
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n_mod),
                                     0.4 + 0.22 * n_mod))
    _draw_econ_table(ax, cell_text, col_labels, row_labels, fontsize=fs,
                     col_width_scale=0.85, row_height_scale=1.05)
    save_fig(fig, f"T02_dm_{idx}")


# ── T03 — Accuracy by region ─────────────────────────────────────────────────

def fig_region_for_index(monthly: pd.DataFrame, idx: str) -> None:
    met     = compute_metrics_by_group(monthly, idx, "region")
    regions = sorted(monthly["region"].unique())

    row_labels = [REGION_NAMES.get(int(r), str(r)) for r in regions]
    col_labels = ["ME", "MAE", "RMSE", "MAPE (%)"]
    cell_text  = []

    for reg in regions:
        if reg not in met.index:
            cell_text.append(["—"] * 4)
            continue
        r = met.loc[reg]
        cell_text.append([
            _fmt(r["ME"],   3),
            _fmt(r["MAE"],  3),
            _fmt(r["RMSE"], 3),
            _fmt(r["MAPE"], 2),
        ])

    fig, ax = plt.subplots(figsize=(7, 0.45 + 0.32 * len(regions)))
    _draw_econ_table(ax, cell_text, col_labels, row_labels, fontsize=9)
    save_fig(fig, f"T03_region_{idx}")


# ── T04 — Accuracy by year ───────────────────────────────────────────────────

def fig_year_for_index(monthly: pd.DataFrame, idx: str) -> None:
    years = sorted(monthly["year"].unique())
    met   = compute_metrics_by_group(monthly, idx, "year")

    row_labels = [str(y) for y in years]
    col_labels = ["ME", "MAE", "RMSE", "MAPE (%)"]
    cell_text  = []

    for yr in years:
        if yr not in met.index:
            cell_text.append(["—"] * 4)
            continue
        r = met.loc[yr]
        cell_text.append([
            _fmt(r["ME"],   3),
            _fmt(r["MAE"],  3),
            _fmt(r["RMSE"], 3),
            _fmt(r["MAPE"], 2),
        ])

    fig, ax = plt.subplots(figsize=(7, 0.45 + 0.30 * len(years)))
    _draw_econ_table(ax, cell_text, col_labels, row_labels, fontsize=9)
    save_fig(fig, f"T04_year_{idx}")


# ============================================================================
# CSV EXPORTS (optional diagnostics)
# ============================================================================

def export_csvs(monthly: pd.DataFrame, models: list) -> None:
    """Write per-model metrics and DM p-values to CSV for further inspection."""
    for idx in INDEX_TYPES:
        # Per-model metrics
        met = compute_metrics_by_model(monthly, idx, models)
        met.to_csv(OUTPUT_DIR / f"csv_{idx}_model_metrics.csv")

        # DM adjusted p-values
        _, adj_p = compute_dm_matrices(monthly, idx, models)
        adj_p.index   = [LABELS[m] for m in models if m in adj_p.index]
        adj_p.columns = [LABELS[m] for m in models if m in adj_p.columns]
        adj_p.to_csv(OUTPUT_DIR / f"csv_{idx}_dm_adj_pvalues.csv")

    print("  CSV exports written.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading forecasts …")
    df = load_forecasts(FORECAST_CSV)
    models = active_models(df)

    print(f"  {len(df):,} daily rows")
    print(f"  Models present : {models}")
    print(f"  Regions        : {sorted(df['region'].unique())}")
    print(f"  Years          : {sorted(df['year'].unique())}")

    print("Building monthly indices …")
    monthly = build_monthly(df)
    print(f"  {len(monthly):,} model × region × year × month records")

    for idx in INDEX_TYPES:
        print(f"Rendering tables for {idx} …")
        fig_metrics_by_model_for_index(monthly, models, idx)
        fig_dm_for_index(monthly, models, idx)
        fig_region_for_index(monthly, idx)
        fig_year_for_index(monthly, idx)

    print("Exporting CSVs …")
    export_csvs(monthly, models)

    print("\nAll outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()