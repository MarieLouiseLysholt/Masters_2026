"""
Futures Hedging Effectiveness — model strike vs historical-mean strike.

Setup:
    Virtual farmer observes I_hat (forecast index) before the season and
    enters a futures position with strike K = I_hat.  The hedge ratio
        N = price · dY/dI |_{I = I_bar}
    is derived from the frozen yield model slope at the calibration mean
    index — revenue change per unit of index movement.

    Total wealth at harvest:   W = R_act + N · (I_act − K)
    Hedging effectiveness:     HE = 1 − Var(W) / Var(R_act)

Benchmark:
    Naive farmer sets K = I_bar (calibration mean of the observed index).
    Same N, no model needed.

Theoretical claim being tested:
    For a futures contract (linear payoff) the strike is an additive
    constant in expectation, so the naive K = I_bar achieves the structural
    R² ceiling unconditionally.  A model-forecast strike adds variance
    proportional to Var(I_hat), so it can only beat the naive when the
    forecast genuinely reduces uncertainty.  A noisy forecast strictly
    underperforms.

Output:  PFUT_strike_HE_test.png   (one figure, no extras)
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration — must match Forecast Yield Predictability
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
TEMP_CSV     = ROOT / "02_Data" / "01_Temprature" / "region_avg.csv"
YIELD_CSV    = ROOT / "02_Data" / "02_Wheat" / "ble_tendre_hiver_yield_2000_2024_regions_mainland.csv"
FORECAST_CSV = ROOT / "04_Analysis" / "01_Forecast Accuracy" / "bk_forecasts.csv"
OUT_DIR      = ROOT / "99_Thesis Graphs" / "03_Results" / "04_Hedging Effectiveness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS         = ["HBA", "Alaton", "Benth", "ARMA",
                  "XGB", "LSTM", "FeedForwardNN", "KNN", "SVM"]
REGIONS        = [11, 24, 27, 28, 32, 44, 52, 53]
HOLDOUT_YEARS  = list(range(2016, 2025))
TREND_END_YEAR = 2015
BASE_TEMP      = 10.0
YIELD_T0       = 2000
WHEAT_PRICE    = 160.0
MIN_CALIB_N    = 6

INDEX_TYPES   = ["CDD", "HDD", "CAT"]
MONTH_WINDOWS = {"Mar": (60, 90),  "Apr": (91, 120), "May": (121, 151),
                 "Jun": (152, 181), "Jul": (182, 212)}
MONTH_ORDER   = ["Mar", "Apr", "May", "Jun", "Jul"]

REGION_NAMES = {
    11: "Île-de-France", 24: "Centre-Val de Loire",
    27: "Bourgogne-Franche-Comté", 28: "Normandie",
    32: "Hauts-de-France", 44: "Grand Est",
    52: "Pays de la Loire", 53: "Bretagne",
}
LABELS = {
    "HBA": "HBA", "Alaton": "Alaton", "Benth": "Benth", "ARMA": "ARMA",
    "XGB": "XGBoost", "LSTM": "LSTM",
    "FeedForwardNN": "Feed Forward NN", "KNN": "KNN", "SVM": "SVM",
}

FORMS = ["linear", "quad_only", "lin_quad"]


# ---------------------------------------------------------------------------
# Helpers — copied from Forecast Yield Predictability
# ---------------------------------------------------------------------------
def _doy_noleap(ts):
    doy = ts.timetuple().tm_yday
    if ts.is_leap_year and ts.month > 2:
        doy -= 1
    return doy


def load_yield(region):
    raw = pd.read_csv(YIELD_CSV)
    raw.columns = [c.strip().lower() for c in raw.columns]
    rcol = next(c for c in ["reg", "region_code", "region"] if c in raw.columns)
    ycol = next(c for c in ["yield", "yield_t_ha", "rendement"] if c in raw.columns)
    tcol = next(c for c in ["year", "annee"] if c in raw.columns)
    df = raw[raw[rcol] == region][[tcol, ycol]].copy()
    df.columns = ["year", "yield_t_ha"]
    df["year"]       = df["year"].astype(int)
    df["yield_t_ha"] = df["yield_t_ha"].astype(float)
    if df["yield_t_ha"].median() > 20:
        df["yield_t_ha"] /= 10.0
    return df.sort_values("year").reset_index(drop=True)


def load_temperature(region):
    raw = pd.read_csv(TEMP_CSV, parse_dates=["date"])
    raw.columns = raw.columns.str.lower().str.strip()
    rcol = next(c for c in ["region_code", "region", "reg"] if c in raw.columns)
    tcol = next(c for c in ["daily_avg_temperature", "temp", "temperature"]
                if c in raw.columns)
    df = raw[raw[rcol] == region][["date", tcol]].copy().rename(columns={tcol: "T"})
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))].copy()
    df["doy"]  = df["date"].apply(lambda d: _doy_noleap(pd.Timestamp(d)))
    df["year"] = df["date"].dt.year
    return df.sort_values("date").reset_index(drop=True)


def load_forecasts():
    df = pd.read_csv(FORECAST_CSV, parse_dates=["date"])
    df = df[df["model"].isin(MODELS)].copy()
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))].copy()
    df["doy"]    = df["date"].apply(lambda d: _doy_noleap(pd.Timestamp(d)))
    df["year"]   = df["date"].dt.year.astype(int)
    df["region"] = df["region"].astype(int)
    return df


def index_obs_by_window(temp_df, s, e, idx_type):
    win = temp_df[(temp_df["doy"] >= s) & (temp_df["doy"] <= e)].copy()
    T = win["T"].values
    if   idx_type == "CDD": win["v"] = np.maximum(T - BASE_TEMP, 0.0)
    elif idx_type == "HDD": win["v"] = np.maximum(BASE_TEMP - T, 0.0)
    else:                    win["v"] = T
    return win.groupby("year")["v"].sum().rename("idx").reset_index()


def index_from_forecast(fc_slice, s, e, idx_type):
    win = fc_slice[(fc_slice["doy"] >= s) & (fc_slice["doy"] <= e)]
    if win.empty:
        return None
    T = win["forecast"].values.astype(float)
    if   idx_type == "CDD": return float(np.maximum(T - BASE_TEMP, 0.0).sum())
    elif idx_type == "HDD": return float(np.maximum(BASE_TEMP - T, 0.0).sum())
    else:                    return float(T.sum())


def detrend_calib(yield_df, end_year):
    df = yield_df[(yield_df["yield_t_ha"] > 0) &
                  (yield_df["year"] <= end_year)].dropna(subset=["yield_t_ha"])
    if len(df) < 6:
        return None
    y = df["yield_t_ha"].values
    t = (df["year"].values - YIELD_T0).astype(float)
    X = np.column_stack([np.ones(len(y)), t])
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_bar = float(np.mean(y))
    out = df[["year"]].copy()
    out["yield_detrended"] = y - X @ coefs + y_bar
    return out.reset_index(drop=True), coefs, y_bar


def apply_trend(y_actual, year, coefs, y_bar):
    t = float(year - YIELD_T0)
    return float(y_actual - (coefs[0] + coefs[1] * t) + y_bar)


def _design(x, form):
    x = np.asarray(x, dtype=float)
    if form == "linear":    return np.column_stack([np.ones(len(x)), x])
    if form == "quad_only": return np.column_stack([np.ones(len(x)), x ** 2])
    if form == "lin_quad":  return np.column_stack([np.ones(len(x)), x, x ** 2])
    raise ValueError(form)


def predict_form(coefs, x, form):
    return _design(x, form) @ coefs


def slope_at(coefs, x_eval, form):
    """dY/dI at x = x_eval — the hedge ratio per unit of price."""
    if form == "linear":    return float(coefs[1])
    if form == "quad_only": return float(2.0 * coefs[1] * x_eval)
    if form == "lin_quad":  return float(coefs[1] + 2.0 * coefs[2] * x_eval)
    raise ValueError(form)


def fit_form(y, x, form):
    X = _design(x, form)
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat  = X @ coefs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    n, k   = int(len(y)), X.shape[1]
    nll2   = n * np.log(max(ss_res, 1e-12) / n)
    bic    = nll2 + k * np.log(n)
    return {"coefs": coefs, "r2": r2, "bic": float(bic), "k": int(k)}


def select_form_bic(y, x, slack=2.0):
    results = {f: fit_form(y, x, f) for f in FORMS}
    best_form, best = None, None
    for form in FORMS:
        bic, k = results[form]["bic"], results[form]["k"]
        if best is None:
            best_form, best = form, (bic, k); continue
        if bic < best[0] - slack:
            best_form, best = form, (bic, k)
        elif abs(bic - best[0]) <= slack and k < best[1]:
            best_form, best = form, (bic, k)
    return best_form, results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

print("=" * 72)
print("  Futures hedging effectiveness -- model strike vs naive mean strike")
print("=" * 72)

print("\n  Loading data ...", end=" ", flush=True)
temp_cache  = {r: load_temperature(r) for r in REGIONS}
yield_cache = {r: load_yield(r) for r in REGIONS}
fc_all      = load_forecasts()
fc_grouped  = {k: g for k, g in fc_all.groupby(["model", "region", "year"],
                                                sort=False)}
print(f"done  ({len(fc_all):,} forecast rows)")

# Build window grid (same as parent script)
WINDOWS = []
for i, ms in enumerate(MONTH_ORDER):
    for me in MONTH_ORDER[i:]:
        s, e = MONTH_WINDOWS[ms][0], MONTH_WINDOWS[me][1]
        lab  = f"{ms}-{me}" if ms != me else ms
        WINDOWS.append((s, e, lab))

# Stage 1 — select frozen (itype, window, form) per region
print("\n  Selecting (itype, window, form) per region ...")
frozen = {}
for region in REGIONS:
    yield_df = yield_cache[region]
    temp_df  = temp_cache[region]

    det_calib = detrend_calib(yield_df, TREND_END_YEAR)
    if det_calib is None:
        continue
    det_calib_df, trend_coefs, y_bar = det_calib

    best = None
    for itype in INDEX_TYPES:
        for s, e, lab in WINDOWS:
            idx_obs = index_obs_by_window(temp_df, s, e, itype)
            idx_obs = idx_obs[idx_obs["year"] <= TREND_END_YEAR]
            merged  = pd.merge(det_calib_df, idx_obs, on="year",
                                how="inner").dropna()
            if len(merged) < MIN_CALIB_N:
                continue
            xs, ys = merged["idx"].values, merged["yield_detrended"].values
            if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
                continue
            rho, _ = pearsonr(ys, xs)
            if np.isnan(rho):
                continue
            if best is None or abs(rho) > best[0]:
                best = (abs(rho), float(rho), itype, lab, s, e)

    if best is None:
        continue
    _, rho_sel, itype_sel, lab_sel, s_sel, e_sel = best
    obs_idx_calib = index_obs_by_window(temp_df, s_sel, e_sel, itype_sel)
    obs_idx_calib = obs_idx_calib[obs_idx_calib["year"] <= TREND_END_YEAR]
    fit_df = pd.merge(det_calib_df, obs_idx_calib, on="year",
                       how="inner").dropna()
    y_fit, x_fit = fit_df["yield_detrended"].values, fit_df["idx"].values
    form_sel, form_res = select_form_bic(y_fit, x_fit)
    fr = form_res[form_sel]

    I_bar_calib = float(np.mean(x_fit))
    dYdI_at_Ibar = slope_at(fr["coefs"], I_bar_calib, form_sel)
    N_region     = WHEAT_PRICE * dYdI_at_Ibar       # EUR/ha per unit of index

    frozen[region] = {
        "trend_coefs": trend_coefs, "y_bar": y_bar,
        "itype": itype_sel, "s": s_sel, "e": e_sel, "label": lab_sel,
        "form": form_sel, "coefs": fr["coefs"],
        "r2": fr["r2"],
        "I_bar": I_bar_calib,
        "N":     N_region,
        "rho_select": rho_sel,
    }
    print(f"    R{region}: {itype_sel:<3s} {lab_sel:<7s}  "
          f"R²={fr['r2']:.3f}  I_bar={I_bar_calib:8.2f}  "
          f"N={N_region:+8.2f} EUR/ha per unit")


# Stage 2 — assemble per-(model, region, year) hedge cash flows
print("\n  Computing hedging variants over holdout 2016-2024 ...")
records = []
for region in REGIONS:
    if region not in frozen:
        continue
    fz       = frozen[region]
    yield_df = yield_cache[region]
    temp_df  = temp_cache[region]

    # Pre-compute observed I_act per holdout year for this region
    idx_obs_all = index_obs_by_window(temp_df, fz["s"], fz["e"], fz["itype"])
    idx_obs_all = idx_obs_all.set_index("year")["idx"]

    for model in MODELS:
        for hy in HOLDOUT_YEARS:
            fc = fc_grouped.get((model, region, hy))
            if fc is None or fc.empty:
                continue
            I_hat = index_from_forecast(fc, fz["s"], fz["e"], fz["itype"])
            if I_hat is None or hy not in idx_obs_all.index:
                continue
            I_act = float(idx_obs_all.loc[hy])

            act_row = yield_df[yield_df["year"] == hy]
            if act_row.empty:
                continue
            Y_act = apply_trend(float(act_row["yield_t_ha"].values[0]),
                                hy, fz["trend_coefs"], fz["y_bar"])
            R_act = Y_act * WHEAT_PRICE

            N     = fz["N"]
            I_bar = fz["I_bar"]

            # Total wealth under each strike convention.  The farmer's
            # natural exposure has slope dR/dI = N, so a hedge offsets it
            # via the position whose payoff is  -N (I_act - K).
            W_unhedged = R_act
            W_naive    = R_act - N * (I_act - I_bar)
            W_model    = R_act - N * (I_act - I_hat)

            records.append({
                "model": model, "region": region, "year": hy,
                "I_act": I_act, "I_hat": I_hat, "I_bar": I_bar,
                "R_act": R_act,
                "N": N,
                "W_unhedged": W_unhedged,
                "W_naive":    W_naive,
                "W_model":    W_model,
            })

cells = pd.DataFrame(records)
print(f"    {len(cells):,} (model, region, year) cells")


# Stage 3 — HE per (model, region) and aggregate per model
def _safe_var(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.var(v, ddof=1)) if len(v) >= 2 else np.nan


he_rows = []
for region in REGIONS:
    if region not in frozen:
        continue
    sub_reg = cells[cells["region"] == region]
    if sub_reg.empty:
        continue

    # Naive HE — same series for every model since K = I_bar (constant), so
    # we compute it once per region from any model's rows (W_naive doesn't
    # depend on the model).
    naive_subset = sub_reg.drop_duplicates(subset=["year"])
    var_R   = _safe_var(naive_subset["R_act"].values)
    var_Wn  = _safe_var(naive_subset["W_naive"].values)
    HE_naive = (1.0 - var_Wn / var_R) if var_R and np.isfinite(var_R) else np.nan

    for model in MODELS:
        sub = sub_reg[sub_reg["model"] == model]
        if len(sub) < 3:
            continue
        var_R_m  = _safe_var(sub["R_act"].values)
        var_Wm   = _safe_var(sub["W_model"].values)
        HE_model = (1.0 - var_Wm / var_R_m) \
                   if var_R_m and np.isfinite(var_R_m) else np.nan
        # forecast-error variance for context
        fe_var = _safe_var((sub["I_act"] - sub["I_hat"]).values)
        I_var  = _safe_var(sub["I_act"].values)
        he_rows.append({
            "region": region, "model": model,
            "HE_naive": HE_naive, "HE_model": HE_model,
            "FEvar":    fe_var,   "Ivar":     I_var,
            "R2_struct": frozen[region]["r2"],
        })

he = pd.DataFrame(he_rows)


# Per-model average across regions
agg = (he.groupby("model")
         .agg(HE_model_mean=("HE_model", "mean"),
              HE_naive_mean=("HE_naive", "mean"),
              FEvar_med=("FEvar", "median"),
              Ivar_med=("Ivar",  "median"),
              n_regions=("region", "nunique"))
         .reindex(MODELS).dropna())
agg["FE_to_I_ratio"] = agg["FEvar_med"] / agg["Ivar_med"]
agg = agg.sort_values("HE_model_mean", ascending=True)

# Single naive line — average across regions of HE_naive (same per region
# regardless of model)
HE_naive_overall = float(he.drop_duplicates("region")["HE_naive"].mean())
print(f"\n  Naive HE  (mean across regions): {HE_naive_overall * 100:6.2f}%")
print("  Model HE  (mean across regions, sorted):")
for m, row in agg.iterrows():
    flag = "+" if row["HE_model_mean"] >= HE_naive_overall else "-"
    print(f"    {LABELS.get(m, m):<16}  {row['HE_model_mean'] * 100:6.2f}%  "
          f"{flag}   FE/I var = {row['FE_to_I_ratio']:.3f}")


# ---------------------------------------------------------------------------
# Render — single PNG
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Build per-model deltas vs naive (region-level then averaged).
he["delta_pp"] = (he["HE_model"] - he["HE_naive"]) * 100.0
delta_agg = (he.groupby("model")
               .agg(delta_mean=("delta_pp", "mean"),
                    fe_med=("FEvar", "median"),
                    iv_med=("Ivar",  "median"))
               .reindex(MODELS).dropna())
delta_agg["fe_to_i"] = delta_agg["fe_med"] / delta_agg["iv_med"]
delta_agg = delta_agg.sort_values("delta_mean")

fig, ax = plt.subplots(figsize=(9.5, 5.0))

models_sorted = list(delta_agg.index)
y_pos    = np.arange(len(models_sorted))
delta_v  = delta_agg["delta_mean"].values
fe_to_i  = delta_agg["fe_to_i"].values

colors = ["#2c7a3e" if v >= 0 else "#a0392c" for v in delta_v]

bars = ax.barh(y_pos, delta_v, color=colors, edgecolor="black",
               linewidth=0.6, height=0.6)

ax.axvline(0, color="black", linestyle="--", linewidth=1.0,
           label="Naive K = mean(I) benchmark")

struct_r2 = float(np.mean([fz["r2"] for fz in frozen.values()]) * 100.0)

ax.set_yticks(y_pos)
ax.set_yticklabels([LABELS.get(m, m) for m in models_sorted])
ax.set_xlabel("delta HE  =  HE(model strike)  -  HE(naive mean strike)   "
              "[percentage points, mean across 8 regions]")
ax.set_title(
    "Futures hedge -- does the model strike beat the historical mean?\n"
    f"Holdout {HOLDOUT_YEARS[0]}-{HOLDOUT_YEARS[-1]}.  Same hedge ratio "
    "N = price * dY/dI evaluated at calibration mean.",
    fontsize=10.5, loc="left")

xpad = max(1.0, (delta_v.max() - delta_v.min()) * 0.025)
for bar, ratio in zip(bars, fe_to_i):
    v = bar.get_width()
    ax.text(v + (xpad if v >= 0 else -xpad),
            bar.get_y() + bar.get_height() / 2,
            f"{v:+5.2f} pp   Var(I-Ihat)/Var(I) = {ratio:.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8.5, color="black")

ax.set_xlim(delta_v.min() - 14, delta_v.max() + 14)
ax.legend(loc="upper right", frameon=True, fontsize=9, edgecolor="black")

sub_caption = (
    f"Naive HE level (mean across regions): {HE_naive_overall * 100:+.1f}%\n"
    f"Calibration R-squared ceiling (mean): {struct_r2:+.1f}%\n"
    "Absolute HE is below zero because the calibrated hedge ratio does\n"
    "not extrapolate cleanly to the 9-yr holdout sample; this test\n"
    "isolates the relative contribution of strike choice."
)
ax.text(0.02, 0.02, sub_caption, transform=ax.transAxes, fontsize=7.5,
        ha="left", va="bottom", color="0.30", family="serif",
        bbox=dict(facecolor="white", edgecolor="0.6", linewidth=0.4,
                  boxstyle="round,pad=0.3"))

caption = (
    "For a futures contract (linear payoff) the strike K enters total\n"
    "wealth additively, so Var(W) is invariant to K in expectation --\n"
    "naive K = mean(I) attains the R-squared ceiling unconditionally.\n"
    "A model strike K = Ihat only beats it if the forecast tracks I,\n"
    "i.e. Var(I - Ihat) is small relative to Var(I).  Forecasts with\n"
    "ratio > 1 (noisier than the index) under-perform the naive."
)
fig.text(0.99, 0.02, caption, fontsize=7.5, ha="right", va="bottom",
         color="0.25", family="serif", style="italic")

fig.tight_layout()
out = OUT_DIR / "PFUT_strike_HE_test.png"
fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\n  -> {out.name}")
print(f"\n  Naive HE: {HE_naive_overall * 100:+.2f}%")
for m in models_sorted:
    d = float(delta_agg.loc[m, "delta_mean"])
    r = float(delta_agg.loc[m, "fe_to_i"])
    print(f"    {LABELS.get(m, m):<16}  delta HE = {d:+5.2f} pp   FE/I = {r:.2f}")
print("\nDone.")
