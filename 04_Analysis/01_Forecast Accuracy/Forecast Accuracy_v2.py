""""
bk_forecast_accuracy.py  [v3]
==============================
Reads bk_forecasts.csv.  Produces five publication-ready outputs
(black-and-white, Times New Roman) and a full set of result CSVs.

Outputs (all written to OUTPUT_DIR)
-------------------------------------
  bk_forecasts.csv              raw daily forecasts (region/year/date/actual/model/forecast)
  T01_accuracy_metrics.png      RMSE / MSE / UAPE per model × index type
  T02_murphy_decomposition.png  Murphy (1988) MSE components as % of MSE + RMSE
  T03_model_comparison.png      Theil U vs baseline + paired Wilcoxon p (BH-adj)
  F01_uape_by_month.png         UAPE by calendar month, B&W line styles
  T04_rmse_region_year.png      RMSE broken down by region and by OOS year

Methodology notes
-----------------
- RMSE / MSE: full monthly sample; no MIN_INDEX filter.
- UAPE: MIN_INDEX filter applied for HDD and CDD only (avoids 0/0).
- Significance: paired Wilcoxon signed-rank on d_i = |err_A| - |err_B|
  per matched (region, year, month).  Not Mann-Whitney (data are paired).
- Multiple-testing: Benjamini-Hochberg FDR within each index type.
- Murphy decomposition uses population sigma (ddof=0) so components
  sum exactly to MSE: Bias² + Conditional-Bias² + Unpredictable Variance.
- HBA is labelled as the burn-analysis climatological baseline.
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
import matplotlib.lines as mlines
from pathlib import Path
from scipy.stats import wilcoxon as scipy_wilcoxon

# ============================================================================
# CONFIGURATION
# ============================================================================

FORECAST_CSV = Path(
    r"C:\Users\U434016\Downloads\Masters_2026"
    r"\04_Analysis\01_Forecast Accuracy\bk_forecasts.csv"
)
OUTPUT_DIR = Path(
    r"C:\Users\U434016\Downloads\Masters_2026"
    r"\99_Thesis Graphs\03_Results\01_Forecast Accuracy"
)

T_REF     = 10.0   # degree-day threshold (°C)
MIN_INDEX = 1.0    # min monthly HDD / CDD for UAPE only
DPI       = 260

MODEL_ORDER = ["HBA", "Alaton", "Benth", "ARMA", "XGB", "WaveletFNN", "LSTM", "FeedForwardNN", "KNN", "SVM"]
BASELINE    = "HBA"

LABELS = {
    "HBA":           "Burn Analysis",
    "Alaton":        "Alaton (2002)",
    "Benth":         "Benth (2007)",
    "ARMA":          "ARMA",
    "XGB":           "XGBoost",
    "WaveletFNN":    "Wavelet FNN",
    "LSTM":          "LSTM",
    "FeedForwardNN": "Feed Forward NN",
    "KNN":           "KNN",
    "SVM":           "SVM (SVR)",
}

INDEX_TYPES  = ["HDD", "CDD", "CAT", "AvgT"]
INDEX_LABELS = {
    "HDD":  r"HDD  (T$_{ref}$ = 10 °C)",
    "CDD":  r"CDD  (T$_{ref}$ = 10 °C)",
    "CAT":  "CAT  (°C·days)",
    "AvgT": "Avg. Temperature (°C)",
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# B&W line styles
PLOT_STYLES = {
    "HBA":           dict(ls="-",   marker="o", color="black", lw=1.5, ms=4),
    "Alaton":        dict(ls="--",  marker="s", color="black", lw=1.5, ms=4),
    "Benth":         dict(ls="-.", marker="^", color="black", lw=1.5, ms=4),
    "ARMA":          dict(ls="-",   marker="h", color="0.20",  lw=1.4, ms=4),
    "XGB":           dict(ls=":",   marker="D", color="black", lw=2.0, ms=4),
    "WaveletFNN":    dict(ls="-",   marker="v", color="0.50",  lw=1.2, ms=4),
    "LSTM":          dict(ls="--",  marker="P", color="0.50",  lw=1.2, ms=4),
    "FeedForwardNN": dict(ls="-.", marker="X", color="0.50",  lw=1.2, ms=4),
    "KNN":           dict(ls=":",   marker="*", color="0.50",  lw=1.2, ms=5),
    "SVM":           dict(ls="-",   marker="p", color="0.35",  lw=1.2, ms=4),
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

_HDR   = "#c8c8c8"   # header cell background
_ALT   = "#f0f0f0"   # alternating data row background
_ROWLB = "#e4e4e4"   # row-label column background


# ============================================================================
# UTILITY
# ============================================================================

def save_fig(fig, stem: str) -> None:
    p = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {p.name}")


def _fmt(v, decimals=3):
    return f"{v:.{decimals}f}" if pd.notna(v) and np.isfinite(float(v)) else "—"


def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _bh_correct(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    NaN inputs pass through as NaN.  Finite p-values corrected jointly.
    """
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    p_fin  = p[finite]
    m      = len(p_fin)
    if m == 0:
        return p.copy()
    order     = np.argsort(p_fin)           # ascending sort
    sorted_p  = p_fin[order]
    adj       = sorted_p * m / np.arange(1, m + 1)
    # enforce monotonicity (cumulative min from right)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    result         = np.empty(m)
    result[order]  = adj                    # put back in original order
    out            = p.copy()
    out[finite]    = result
    return out


def _draw_bw_table(ax, cell_text, col_labels, row_labels,
                   bold_rows=None, fontsize=8, footnote=None):
    """
    Render a clean black-and-white academic table on *ax*.

    bold_rows : set of 0-based data-row indices whose cells are rendered bold.
    """
    ax.axis("off")
    bold_rows = bold_rows or set()

    tbl = ax.table(
        cellText  = cell_text,
        colLabels = col_labels,
        rowLabels = row_labels,
        loc       = "center",
        cellLoc   = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.05, 1.80)

    n_data = len(row_labels)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0)           # remove all individual borders

        if row == 0:                    # header
            cell.set_facecolor(_HDR)
            cell.set_text_props(fontweight="bold", ha="center",
                               fontfamily="Times New Roman")
        elif col == -1:                 # row-label column
            cell.set_facecolor(_ROWLB)
            cell.set_text_props(ha="left", fontfamily="Times New Roman")
        else:                           # data cell
            cell.set_facecolor(_ALT if row % 2 == 0 else "white")
            fw = "bold" if (row - 1) in bold_rows else "normal"
            cell.set_text_props(ha="right", fontweight=fw,
                               fontfamily="Times New Roman")

    if footnote:
        ax.text(0.0, -0.01, footnote, transform=ax.transAxes,
               fontsize=6.5, style="italic",
               fontfamily="Times New Roman", va="top", ha="left")
    return tbl


def _panel_label(ax, letter, x=-0.06, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom",
            fontfamily="Times New Roman")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["actual", "forecast"])
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["month"] = df["date"].dt.month
    return df


# ============================================================================
# MONTHLY INDEX COMPUTATION
# ============================================================================

def build_monthly(df: pd.DataFrame) -> pd.DataFrame:

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
        .apply(_agg)
        .reset_index()
    )

    for idx in INDEX_TYPES:
        e = mon[f"{idx}_forecast"] - mon[f"{idx}_actual"]
        mon[f"{idx}_error"]     = e
        mon[f"{idx}_abs_error"] = e.abs()
        mon[f"{idx}_sq_error"]  = e ** 2
        denom = (mon[f"{idx}_actual"].abs() + mon[f"{idx}_forecast"].abs()) / 2
        mon[f"{idx}_uape"] = np.where(denom > 1e-8,
                                       mon[f"{idx}_abs_error"] / denom, np.nan)
    return mon


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(df_m: pd.DataFrame, idx: str) -> pd.DataFrame:
    """
    RMSE / MSE  : full monthly sample — no MIN_INDEX filter.
    UAPE        : MIN_INDEX filter for HDD / CDD only.
    """
    rows = []
    for model in MODEL_ORDER:
        m = df_m[df_m["model"] == model]
        if len(m) == 0:
            rows.append(dict(model=model, RMSE=np.nan, MSE=np.nan, UAPE=np.nan, N=0))
            continue
        mse = float(m[f"{idx}_sq_error"].mean())
        if idx in ("HDD", "CDD"):
            m_u = m[m[f"{idx}_actual"] >= MIN_INDEX]
        else:
            m_u = m
        uape = float(m_u[f"{idx}_uape"].mean(skipna=True)) if len(m_u) else np.nan
        rows.append(dict(model=model, RMSE=float(np.sqrt(mse)), MSE=mse, UAPE=uape,
                         N=len(m)))
    return pd.DataFrame(rows).set_index("model")


# ============================================================================
# MURPHY (1988) DECOMPOSITION
# ============================================================================

def _murphy_cell(a: np.ndarray, f: np.ndarray):
    """Return (bias_sq, cond_bias_sq, unpred_var) or (nan, nan, nan)."""
    if len(a) < 3:
        return np.nan, np.nan, np.nan
    sa  = a.std(ddof=0);  sf = f.std(ddof=0)
    rho = 0.0 if (sa < 1e-10 or sf < 1e-10) else float(np.corrcoef(a, f)[0, 1])
    return ((f.mean() - a.mean()) ** 2,
            (sf - rho * sa) ** 2,
            (1 - rho ** 2) * sa ** 2)


def murphy_decompose(df_m: pd.DataFrame, idx: str) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        sub = df_m[df_m["model"] == model]
        bs, cb, uv = _murphy_cell(sub[f"{idx}_actual"].values,
                                   sub[f"{idx}_forecast"].values)
        total = float(sub[f"{idx}_sq_error"].mean()) if len(sub) else np.nan
        rows.append(dict(model=model, bias_sq=bs, cond_bias_sq=cb,
                         unpred_var=uv, total_mse=total))
    return pd.DataFrame(rows).set_index("model")


# ============================================================================
# THEIL U  &  PAIRED WILCOXON + BH
# ============================================================================

def theil_u_vs_baseline(df_m: pd.DataFrame, idx: str) -> pd.Series:
    """Theil U_i = RMSE_i / RMSE_baseline."""
    rmses = {m: float(np.sqrt(df_m[df_m["model"] == m][f"{idx}_sq_error"].mean()))
             for m in MODEL_ORDER}
    rb = rmses[BASELINE]
    return pd.Series({m: rmses[m] / rb if np.isfinite(rb) and rb > 0 else np.nan
                      for m in MODEL_ORDER})


def pairwise_wilcoxon_bh(df_m: pd.DataFrame, idx: str):
    """
    d_i = |err_A_i| − |err_B_i| per matched (region, year, month).
    Wilcoxon signed-rank (two-sided).  Returns (raw_p_df, adj_p_df).
    """
    pivot = df_m.pivot_table(
        index=["region", "year", "month"],
        columns="model",
        values=f"{idx}_abs_error",
        aggfunc="first",
    )

    raw = pd.DataFrame(np.nan, index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    pairs, plist = [], []

    for mi in MODEL_ORDER:
        for mj in MODEL_ORDER:
            if mi == mj or mi not in pivot.columns or mj not in pivot.columns:
                continue
            d = (pivot[mi] - pivot[mj]).dropna().values
            if len(d) < 5 or np.all(d == 0):
                continue
            try:
                _, p = scipy_wilcoxon(d, alternative="two-sided")
                raw.loc[mi, mj] = float(p)
                pairs.append((mi, mj));  plist.append(float(p))
            except Exception:
                pass

    adj = pd.DataFrame(np.nan, index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    if plist:
        corr = _bh_correct(np.array(plist))
        for (mi, mj), pc in zip(pairs, corr):
            adj.loc[mi, mj] = float(pc)

    return raw, adj


# ============================================================================
# FIGURE / TABLE RENDERING FUNCTIONS
# ============================================================================

# ── T01 — Accuracy metrics ───────────────────────────────────────────────────

def fig_accuracy_metrics(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()

    for ax, idx in zip(axes, INDEX_TYPES):
        met = compute_metrics(monthly, idx)
        row_labels = [LABELS[m] for m in MODEL_ORDER]
        col_labels = ["RMSE", "MSE", "UAPE\u2020"]
        cell_text  = []
        best_rmse  = met["RMSE"].dropna().idxmin() if not met["RMSE"].dropna().empty else None

        bold_rows = set()
        for i, model in enumerate(MODEL_ORDER):
            r = met.loc[model]
            cell_text.append([_fmt(r["RMSE"]), _fmt(r["MSE"]),
                               _fmt(r["UAPE"], 4)])
            if model == best_rmse:
                bold_rows.add(i)

        fn = (f"\u2020UAPE for {idx}: months with actual < {MIN_INDEX} \u00b0C\u00b7day excluded"
              if idx in ("HDD", "CDD") else None)
        _draw_bw_table(ax, cell_text, col_labels, row_labels,
                       bold_rows=bold_rows, footnote=fn)
        ax.set_title(INDEX_LABELS[idx], fontsize=10, fontweight="bold",
                     fontfamily="Times New Roman", pad=6)

    fig.suptitle("Table 1.  Forecast Accuracy — All Regions \u00d7 All OOS Years\n"
                 "Bold = lowest RMSE.  RMSE and MSE computed on full monthly sample.",
                 fontsize=10, fontweight="bold", fontfamily="Times New Roman")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "T01_accuracy_metrics")


# ── T02 — Murphy decomposition ───────────────────────────────────────────────

def fig_murphy(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()

    for ax, idx in zip(axes, INDEX_TYPES):
        dc = murphy_decompose(monthly, idx)
        row_labels = [LABELS[m] for m in MODEL_ORDER]
        col_labels = ["Bias\u00b2 (%)", "Cond.Bias\u00b2 (%)", "Unpred. (%)", "RMSE"]
        cell_text  = []

        for model in MODEL_ORDER:
            r   = dc.loc[model]
            tot = r["total_mse"]
            def pct(v):
                return f"{v / tot * 100:.1f}" if (pd.notna(v) and pd.notna(tot)
                                                   and tot > 0) else "—"
            cell_text.append([pct(r["bias_sq"]), pct(r["cond_bias_sq"]),
                               pct(r["unpred_var"]),
                               _fmt(np.sqrt(tot) if pd.notna(tot) else np.nan)])

        _draw_bw_table(ax, cell_text, col_labels, row_labels, fontsize=8)
        ax.set_title(INDEX_LABELS[idx], fontsize=10, fontweight="bold",
                     fontfamily="Times New Roman", pad=6)

    fig.suptitle("Table 2.  Murphy (1988) MSE Decomposition\n"
                 r"MSE = Bias$^2$ + Conditional-Bias$^2$ + Unpredictable Variance"
                 "  (population \u03c3; components sum to MSE).",
                 fontsize=10, fontweight="bold", fontfamily="Times New Roman")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "T02_murphy_decomposition")


# ── T03 — Theil U + Wilcoxon vs baseline ─────────────────────────────────────

def fig_model_comparison(monthly: pd.DataFrame) -> None:
    """
    For each index type: one table with columns
        RMSE | Theil U | p_raw | p_adj | Sig.
    Only the comparison against the burn-analysis baseline is shown here.
    Full pairwise matrices are exported to CSV.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()

    for ax, idx in zip(axes, INDEX_TYPES):
        raw_p, adj_p = pairwise_wilcoxon_bh(monthly, idx)
        theil        = theil_u_vs_baseline(monthly, idx)
        met          = compute_metrics(monthly, idx)

        row_labels = [LABELS[m] for m in MODEL_ORDER]
        col_labels = ["RMSE", "Theil U", "p (raw)", "p (BH-adj)", "Sig."]
        cell_text  = []
        bold_rows  = set()

        for i, model in enumerate(MODEL_ORDER):
            rmse = met.loc[model, "RMSE"]
            u    = theil.loc[model]
            # comparison: model (row) vs baseline (col)
            pr   = raw_p.loc[model, BASELINE]
            pa   = adj_p.loc[model, BASELINE]

            if model == BASELINE:
                cell_text.append([_fmt(rmse), "1.000", "—", "—", "ref."])
            else:
                row = [_fmt(rmse), _fmt(u), _fmt(pr, 4), _fmt(pa, 4),
                       _stars(pa if pd.notna(pa) else np.nan)]
                cell_text.append(row)
                if pd.notna(u) and u < 1.0:
                    bold_rows.add(i)

        fn = ("Theil U = RMSE_model / RMSE_baseline.  "
              "Wilcoxon: d_i = |err_model| \u2212 |err_baseline| per "
              "(region \u00d7 year \u00d7 month).  p BH-adjusted within index type.")
        _draw_bw_table(ax, cell_text, col_labels, row_labels,
                       bold_rows=bold_rows, fontsize=7.5, footnote=fn)
        ax.set_title(INDEX_LABELS[idx], fontsize=10, fontweight="bold",
                     fontfamily="Times New Roman", pad=6)

    fig.suptitle("Table 3.  Model Comparison vs Burn Analysis Baseline\n"
                 "Bold = Theil U < 1 (outperforms baseline).  "
                 "* p\u2080 < 0.05  ** p\u2080 < 0.01  *** p\u2080 < 0.001.",
                 fontsize=10, fontweight="bold", fontfamily="Times New Roman")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "T03_model_comparison")


# ── F01 — UAPE by month (B&W line plot) ──────────────────────────────────────

def fig_uape_by_month(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, idx in zip(axes, INDEX_TYPES):
        for model in MODEL_ORDER:
            uapes = []
            for m in range(1, 13):
                sub = monthly[(monthly["model"] == model) & (monthly["month"] == m)]
                if idx in ("HDD", "CDD"):
                    sub = sub[sub[f"{idx}_actual"] >= MIN_INDEX]
                uapes.append(float(sub[f"{idx}_uape"].mean(skipna=True))
                             if len(sub) else np.nan)
            st = PLOT_STYLES[model]
            ax.plot(range(1, 13), uapes,
                    ls=st["ls"], marker=st["marker"],
                    color=st["color"], lw=st["lw"], ms=st["ms"],
                    label=LABELS[model])

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES, fontsize=8, fontfamily="Times New Roman")
        ax.set_ylabel("UAPE", fontsize=9, fontfamily="Times New Roman")
        ax.set_title(INDEX_LABELS[idx], fontsize=10, fontweight="bold",
                     fontfamily="Times New Roman")
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx in ("HDD", "CDD"):
            ax.set_xlabel(f"Months with actual < {MIN_INDEX} °C·day excluded",
                          fontsize=7, style="italic", fontfamily="Times New Roman")

    # shared legend below all panels
    handles = [
        mlines.Line2D([], [],
                      ls=PLOT_STYLES[m]["ls"],
                      marker=PLOT_STYLES[m]["marker"],
                      color=PLOT_STYLES[m]["color"],
                      lw=PLOT_STYLES[m]["lw"],
                      ms=PLOT_STYLES[m]["ms"],
                      label=LABELS[m])
        for m in MODEL_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=True, edgecolor="black",
               prop={"family": "Times New Roman", "size": 8},
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Figure 1.  Scale-Adjusted Forecast Error (UAPE) by Calendar Month\n"
                 "All regions × all OOS years.  UAPE = |error| / mean(|actual|, |forecast|).",
                 fontsize=10, fontweight="bold", fontfamily="Times New Roman")
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    save_fig(fig, "F01_uape_by_month")


# ── T04 — RMSE by region and year ────────────────────────────────────────────

def fig_rmse_breakdown(monthly: pd.DataFrame) -> None:
    """
    Two-panel figure for each of the four index types:
      left  — RMSE by NUTS-2 region  (regions as rows, models as cols)
      right — RMSE by OOS year       (years as rows, models as cols)
    Laid out as a 4-row × 2-col grid (one row per index type).
    """
    regions = sorted(monthly["region"].unique())
    years   = sorted(monthly["year"].unique())
    n_idx   = len(INDEX_TYPES)

    fig, axes = plt.subplots(n_idx, 2, figsize=(16, 4.5 * n_idx))

    for i, idx in enumerate(INDEX_TYPES):
        # ── region table ────────────────────────────────────────────────────
        ax_r = axes[i, 0]
        r_rows = [f"R{r}" for r in regions]
        r_cols = [LABELS[m] for m in MODEL_ORDER]
        r_data = []
        for reg in regions:
            row = []
            for mdl in MODEL_ORDER:
                sub = monthly[(monthly["model"] == mdl) & (monthly["region"] == reg)]
                row.append(_fmt(np.sqrt(sub[f"{idx}_sq_error"].mean())
                                if len(sub) else np.nan, 2))
            r_data.append(row)

        _draw_bw_table(ax_r, r_data, r_cols, r_rows, fontsize=7)
        ax_r.set_title(f"{INDEX_LABELS[idx]} — by Region",
                       fontsize=9, fontweight="bold",
                       fontfamily="Times New Roman", pad=5)

        # ── year table ──────────────────────────────────────────────────────
        ax_y = axes[i, 1]
        y_rows = [str(y) for y in years]
        y_cols = [LABELS[m] for m in MODEL_ORDER]
        y_data = []
        for yr in years:
            row = []
            for mdl in MODEL_ORDER:
                sub = monthly[(monthly["model"] == mdl) & (monthly["year"] == yr)]
                row.append(_fmt(np.sqrt(sub[f"{idx}_sq_error"].mean())
                                if len(sub) else np.nan, 2))
            y_data.append(row)

        _draw_bw_table(ax_y, y_data, y_cols, y_rows, fontsize=7)
        ax_y.set_title(f"{INDEX_LABELS[idx]} — by Year",
                       fontsize=9, fontweight="bold",
                       fontfamily="Times New Roman", pad=5)

    fig.suptitle("Table 4.  RMSE by Region and by OOS Year  (full monthly sample)",
                 fontsize=10, fontweight="bold", fontfamily="Times New Roman")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, "T04_rmse_region_year")




# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading bk_forecasts.csv ...")
    df = load_forecasts(FORECAST_CSV)
    print(f"  {len(df):,} rows | "
          f"regions: {sorted(df['region'].unique())} | "
          f"years: {sorted(df['year'].unique())}")

    # Mirror the replication script: write bk_forecasts.csv to OUTPUT_DIR
    FORECAST_COLS = ["region", "year", "date", "actual", "model", "forecast"]
    df[FORECAST_COLS].to_csv(OUTPUT_DIR / "bk_forecasts.csv", index=False)
    print(f"  -> bk_forecasts.csv  ({len(df):,} rows)")

    print("Computing monthly indices ...")
    monthly = build_monthly(df)
    print(f"  Monthly table: {len(monthly):,} rows")

    print("\nRendering figures ...")
    fig_accuracy_metrics(monthly)
    fig_murphy(monthly)
    fig_model_comparison(monthly)
    fig_uape_by_month(monthly)
    fig_rmse_breakdown(monthly)

    print(f"\n{'='*60}")
    print(f"  All outputs -> {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()