"""
ARIMAF Rolling PKL Exporter — Murat et al. (2018) exact
=========================================================
Implements ARIMAF(p, 0, q)[K] following Murat et al. (2018, eq. 14):

    y_t = c + Σ_{l=1}^{K} [α_l sin(2πlt/365) + β_l cos(2πlt/365)] + U_t

where U_t ~ ARMA(p, q), d = 0 throughout.

Design
───────
Phase 1 — Specification selection  (once per region, pre-OOS)
    Selection training : 1980-01-01 – 2008-12-31
    Validation         : 2009-01-01 – 2009-12-31
    Joint grid         : p,q ∈ 0..3, K ∈ 1..10 → 150 SARIMAX fits
                         (p=0, q=0 excluded)
    Criterion          : validation RMSE  [Murat et al. Table 3 primary]
    K is selected jointly with p and q — not separately.

Phase 2 — Rolling parameter refit  (one PKL per region × calib_year)
    Specification fixed from Phase 1.
    Parameters re-estimated by MLE on expanding calibration window.
    One PKL per region × calib_year, consistent with all other models.

Reference: Murat, Malinowska, Gos & Krzyszczak (2018, Int. Agrophys. 32)
"""

import warnings
warnings.simplefilter("ignore")

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
TEMP_DIR  = ROOT / "02_Data" / "01_Temprature"
MODEL_DIR = Path(__file__).resolve().parent
PKL_DIR   = MODEL_DIR / "01_PKL Files" / "13_ARIMAF"
PKL_DIR.mkdir(parents=True, exist_ok=True)

# ── Phase 1: specification selection ──────────────────────────────────────────
SEL_TRAIN_END = pd.Timestamp("2008-12-31")
VAL_START     = pd.Timestamp("2009-01-01")
VAL_END       = pd.Timestamp("2009-12-31")
TRAINING_START = pd.Timestamp("1980-01-01")
TEMP_COL = "TAVG_imptd"

# ── Phase 2: rolling OOS ──────────────────────────────────────────────────────
CALIB_YEARS      = range(2010, 2024)
FORECAST_HORIZON = 365

REGIONS = [11, 24, 27, 28, 32, 44, 52, 53]

# Murat et al. §Results exact grid
MAX_P = 3    # p ∈ 0..3
MAX_Q = 3    # q ∈ 0..3
MAX_K = 6  # K ∈ 1..10  (reduce to 6 if runtime is prohibitive)

N_JOBS = -1

tdy = datetime.datetime.today().strftime('%Y-%m-%d')


# ============================================================================
# FOURIER REGRESSORS  (Murat et al. eq. 14, period m = 365)
# ============================================================================

def make_fourier(doy_array: np.ndarray, K: int,
                 period: float = 365.0) -> np.ndarray:
    cols = []
    for l in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * l * doy_array / period))
        cols.append(np.cos(2 * np.pi * l * doy_array / period))
    return np.column_stack(cols)


# ============================================================================
# METRICS  (Murat et al. eqs 9-11)
# ============================================================================

def _rmse(a, f):
    return float(np.sqrt(np.mean((a - f) ** 2)))

def _mae(a, f):
    return float(np.mean(np.abs(a - f)))

def _mase(a, f, train):
    Q = float(np.mean(np.abs(np.diff(train))))
    return _mae(a, f) / Q if Q > 1e-10 else np.nan


# ============================================================================
# SINGLE CANDIDATE FIT
# ============================================================================

def _fit_candidate(train_y, train_doy, val_y, val_doy,
                   p, q, K) -> tuple:
    """
    Fit ARIMAF(p,0,q)[K] on training data, forecast validation period.
    Returns (rmse, mae, mase) or (inf, inf, inf) on failure.
    """
    X_tr = make_fourier(train_doy, K)
    X_va = make_fourier(val_doy,   K)

    try:
        mod = SARIMAX(
            train_y, exog=X_tr, order=(p, 0, q), trend="c",
            enforce_stationarity=False, enforce_invertibility=False,
        )
        res = mod.fit(disp=False, maxiter=150,
                      method="lbfgs", warn_convergence=False)

        if not np.isfinite(res.llf):
            return np.inf, np.inf, np.inf

        fcast = np.asarray(
            res.forecast(steps=len(val_y), exog=X_va), dtype=float
        ).reshape(-1)

        if not np.all(np.isfinite(fcast)):
            return np.inf, np.inf, np.inf

        return (_rmse(val_y, fcast),
                _mae(val_y, fcast),
                _mase(val_y, fcast, train_y))

    except Exception:
        return np.inf, np.inf, np.inf


# ============================================================================
# PHASE 1 — JOINT GRID SEARCH  (Murat et al. §Results exact)
# ============================================================================

def select_spec(region: int, subset: pd.DataFrame) -> dict | None:
    """
    Joint grid over p,q ∈ 0..3, K ∈ 1..10.
    150 SARIMAX fits per region, all evaluated on 2009 validation RMSE.
    K is not selected separately — it is part of the joint search.
    """
    df_sel_train = subset.loc[:SEL_TRAIN_END].copy()
    df_val       = subset.loc[VAL_START:VAL_END].copy()

    if len(df_sel_train) < 365 or len(df_val) < 30:
        print(f"  [SELECT|{region}] insufficient data")
        return None

    train_y   = df_sel_train[TEMP_COL].values.astype(float)
    train_doy = df_sel_train.index.dayofyear.values.astype(float)
    val_y     = df_val[TEMP_COL].values.astype(float)
    val_doy   = df_val.index.dayofyear.values.astype(float)

    best_rmse = np.inf
    best_mae  = np.inf
    best_mase = np.inf
    best_p = best_q = best_K = None
    n_fits = 0

    for K in range(1, MAX_K + 1):
        for p in range(0, MAX_P + 1):
            for q in range(0, MAX_Q + 1):
                if p == 0 and q == 0:
                    continue
                rmse, mae, mase = _fit_candidate(
                    train_y, train_doy, val_y, val_doy, p, q, K
                )
                n_fits += 1
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_mae  = mae
                    best_mase = mase
                    best_p, best_q, best_K = p, q, K

    if best_p is None:
        print(f"  [SELECT|{region}] no converged model in {n_fits} fits")
        return None

    print(
        f"  [SELECT|{region}] ARIMAF({best_p},0,{best_q})[K={best_K}]  "
        f"valRMSE={best_rmse:.3f}  valMAE={best_mae:.3f}  "
        f"valMASE={best_mase:.3f}  ({n_fits} fits)"
    )

    return {
        "region"   : region,
        "p"        : best_p,
        "q"        : best_q,
        "K"        : best_K,
        "sel_rmse" : round(best_rmse, 4),
        "sel_mae"  : round(best_mae,  4),
        "sel_mase" : round(best_mase, 4),
        "n_fits"   : n_fits,
    }


# ============================================================================
# PHASE 2 — ROLLING PARAMETER REFIT
# ============================================================================

def refit_and_forecast(region: int, calib_year: int,
                       subset: pd.DataFrame,
                       p: int, q: int, K: int) -> dict | None:
    """
    Refit ARIMAF(p,0,q)[K] on the full calibration window up to
    calib_year-12-31 and produce a 365-day forecast.
    Specification (p,q,K) is fixed from Phase 1.
    """
    calib_end = pd.Timestamp(f"{calib_year}-12-31")
    df_calib  = subset.loc[:calib_end].copy()

    if len(df_calib) < 365 or df_calib[TEMP_COL].isna().any():
        return None

    t0        = datetime.datetime.now()
    calib_y   = df_calib[TEMP_COL].values.astype(float)
    calib_doy = df_calib.index.dayofyear.values.astype(float)
    X_calib   = make_fourier(calib_doy, K)

    try:
        mod = SARIMAX(
            calib_y, exog=X_calib, order=(p, 0, q), trend="c",
            enforce_stationarity=False, enforce_invertibility=False,
        )
        res = mod.fit(disp=False, maxiter=150,
                      method="lbfgs", warn_convergence=False)
    except Exception as e:
        print(f"  [REFIT|{region}|{calib_year}] fit failed: {e}")
        return None

    # Forecast dates
    fcast_dates = []
    d = calib_end
    while len(fcast_dates) < FORECAST_HORIZON:
        d += pd.Timedelta(days=1)
        fcast_dates.append(d)
    fcast_dates = pd.DatetimeIndex(fcast_dates)
    X_fcast     = make_fourier(
        fcast_dates.dayofyear.values.astype(float), K
    )

    try:
        forecast_temp = np.asarray(
            res.forecast(steps=FORECAST_HORIZON, exog=X_fcast),
            dtype=float
        ).reshape(-1)
    except Exception as e:
        print(f"  [REFIT|{region}|{calib_year}] forecast failed: {e}")
        return None

    if not np.all(np.isfinite(forecast_temp)):
        print(f"  [REFIT|{region}|{calib_year}] non-finite forecast")
        return None

    pkl_path = PKL_DIR / f"arimaf_model_{region}_calib_{calib_year}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "fitted_model"          : res,
            "order"                 : (p, 0, q),
            "K"                     : K,
            "forecast_temp_365"     : forecast_temp,
            "forecast_dates"        : fcast_dates,
            "n_calib_days"          : len(df_calib),
            "df_calib"              : df_calib.copy(),
            "calib_end"             : f"{calib_year}-12-31",
            "region_code"           : region,
            "model_name"            : "ARIMAF",
            "spec_selected_pre_oos" : True,
            "sel_train_end"         : str(SEL_TRAIN_END.date()),
            "val_period"            : f"{VAL_START.date()} – "
                                      f"{VAL_END.date()}",
        }, f)

    elapsed = (datetime.datetime.now() - t0).total_seconds()
    print(
        f"  [REFIT|{region}|{calib_year}] "
        f"ARIMAF({p},0,{q})[K={K}]  "
        f"N={len(df_calib):,}  t={elapsed:.1f}s"
    )

    return {
        "region"       : region,
        "calib_year"   : calib_year,
        "p"            : p,
        "d"            : 0,
        "q"            : q,
        "K"            : K,
        "n_calib_days" : len(df_calib),
        "time_s"       : round(elapsed, 1),
        "pkl_path"     : str(pkl_path),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    overall_start = datetime.datetime.now()

    raw = pd.read_csv(TEMP_DIR / "region_temp_extended.csv",
                      parse_dates=["date"])

    region_data = {}
    for region in REGIONS:
        sub = (
            raw[raw["region_code"] == region]
            .set_index("date")[[TEMP_COL]]
            .sort_index()
        )
        if sub.empty:
            continue

        sub = sub.loc[TRAINING_START:].copy()
        if sub.empty:
            print(f"  [DATA|{region}] no data from "
                  f"{TRAINING_START.date()} forward")
            continue

        full_idx = pd.date_range(
            sub.index.min(), sub.index.max(), freq="D"
        )
        sub = sub.reindex(full_idx)
        sub.index.name = "date"
        sub[TEMP_COL] = sub[TEMP_COL].bfill()
        region_data[region] = sub

    n_candidates = MAX_K * ((MAX_P + 1) * (MAX_Q + 1) - 1)

    print("=" * 65)
    print("ARIMAF — Murat et al. (2018) exact joint grid")
    print("=" * 65)
    print(f"Phase 1 sel. training : "
          f"{TRAINING_START.date()} – {SEL_TRAIN_END.date()}")
    print(f"Phase 1 validation    : "
          f"{VAL_START.date()} – {VAL_END.date()}")
    print(f"Joint grid            : p,q ∈ 0..{MAX_P}, K ∈ 1..{MAX_K}  "
          f"→ {n_candidates} SARIMAX fits per region")
    print(f"Phase 2 calib years   : "
          f"{min(CALIB_YEARS)} – {max(CALIB_YEARS)}")
    print(f"Phase 2 OOS years     : "
          f"{min(CALIB_YEARS)+1} – {max(CALIB_YEARS)+1}")
    print(f"Workers               : {N_JOBS}")
    print(f"\nPhase 1 total fits    : "
          f"{len(REGIONS)} regions × {n_candidates} = "
          f"{len(REGIONS) * n_candidates}\n")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print("PHASE 1 — Joint grid search  [Murat et al. §Results exact]")
    print("-" * 55)

    sel_results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(select_spec)(region, region_data[region])
        for region in REGIONS if region in region_data
    )

    specs = {r["region"]: r for r in sel_results if r is not None}

    sel_df = pd.DataFrame(list(specs.values()))
    if not sel_df.empty:
        sel_csv = MODEL_DIR / f"arimaf_selection_summary_{tdy}.csv"
        sel_df.to_csv(sel_csv, index=False)
        print(f"\nSelection results → {sel_csv}")
        print(f"\n{'─'*60}")
        print(f"{'Region':<8} {'p':>3} {'q':>3} {'K':>3} "
              f"{'valRMSE':>8} {'valMAE':>7} {'valMASE':>8} {'fits':>5}")
        print(f"{'─'*60}")
        for _, r in sel_df.sort_values("region").iterrows():
            print(
                f"  {int(r.region):<6} {int(r.p):>3} {int(r.q):>3} "
                f"{int(r.K):>3} {r.sel_rmse:>8.3f} {r.sel_mae:>7.3f} "
                f"{r.sel_mase:>8.3f} {int(r.n_fits):>5}"
            )

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print("\nPHASE 2 — Rolling parameter refit")
    print("-" * 55)

    refit_tasks = [
        (region, year, region_data[region],
         specs[region]["p"],
         specs[region]["q"],
         specs[region]["K"])
        for region in REGIONS
        if region in specs and region in region_data
        for year in CALIB_YEARS
    ]

    print(f"Total refit tasks: {len(refit_tasks)}\n")

    refit_results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(refit_and_forecast)(region, year, subset, p, q, K)
        for region, year, subset, p, q, K in refit_tasks
    )

    rows = [r for r in refit_results if r is not None]
    if rows:
        out_df  = pd.DataFrame(rows)
        out_csv = MODEL_DIR / f"arimaf_rolling_summary_{tdy}.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\nRolling summary → {out_csv}")
        print(f"\n{'Region':<8} {'Calib':<6} {'p':>3} {'q':>3} "
              f"{'K':>3} {'N_calib':>8} {'t(s)':>6}")
        print("─" * 42)
        for _, r in out_df.sort_values(
                ["region", "calib_year"]).iterrows():
            print(
                f"  {int(r.region):<6} {int(r.calib_year):<6} "
                f"{int(r.p):>3} {int(r.q):>3} {int(r.K):>3} "
                f"{int(r.n_calib_days):>8} {r.time_s:>6.1f}"
            )

    print(f"\nTotal time : {datetime.datetime.now() - overall_start}")
    print(f"PKL folder : {PKL_DIR}  "
          f"({len(list(PKL_DIR.glob('*.pkl')))} PKLs)")


if __name__ == "__main__":
    main()
