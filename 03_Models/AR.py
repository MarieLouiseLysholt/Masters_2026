"""
AR Models rolling PKL exporter — ARMA, AutoARIMA, ARIMA.

Fits all three AR benchmarks from Barnor, Kampouridis & Kanellopoulos (2026)
Section 3.2.1 on the RAW daily average temperature series (after missing
value imputation). Per Section 3.1.1 of the paper, February 29th is retained
for AR models since sktime's ARIMA-based models explicitly encode the temporal
dimension and require a complete daily frequency index without gaps.

Models
------
ARMA      : ARIMA(p, 0, q) — AutoARIMA(d=0) selects (p,q), then refit clean
AutoARIMA : AutoARIMA(d=0) fitted directly, order stored from pmdarima object
ARIMA     : ARIMA(p, 1, q) — AutoARIMA(d=1) selects (p,q), then refit clean

PKL folders created
-------------------
01_PKL Files/10_ARMA/        arma_model_{region}_calib_{year}.pkl
01_PKL Files/11_AutoARIMA/   autoarima_model_{region}_calib_{year}.pkl
01_PKL Files/12_ARIMA/       arima_model_{region}_calib_{year}.pkl

PKL layout (consistent across all three models):
    {
        'fitted_model'       : fitted sktime model object,
        'order'              : (p, d, q) tuple,
        'forecast_temp_365'  : np.ndarray, shape (365,), raw temperature
                               forecasts for the 365 days following calib_end,
        'forecast_dates'     : pd.DatetimeIndex of those 365 calendar days
                               (includes Feb 29 where it falls),
        'n_calib_days'       : int, number of calibration observations,
        'df'                 : raw region DataFrame (date-indexed),
        'df_calib'           : calibration slice of df,
        'calib_end'          : str 'YYYY-12-31',
        'region_code'        : int,
        'model_name'         : str, one of 'ARMA' / 'AutoARIMA' / 'ARIMA',
    }

Notes
-----
- Missing values (including any Feb 29 gaps from upstream data) are filled
  with backfill imputation before fitting, matching the reference script.
- (p, q) selection uses AutoARIMA BIC internally, consistent with the paper's
  "Bayesian information criteria was used to select the optimal lag length".
- The fitted object is saved directly for AutoARIMA; for ARMA and ARIMA a
  pure ARIMA(p, d, q) is refit after order discovery so the stored object is
  unambiguous.
- Forecasts are in absolute temperature units (degrees Celsius) because the
  models are fitted on raw temperatures, not residuals.
"""

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import os
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["NUMEXPR_NUM_THREADS"] = "5"

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import datetime

from sktime.forecasting.arima import AutoARIMA, ARIMA
from sktime.transformations.series.impute import Imputer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
TEMP_DIR    = ROOT / "02_Data" / "01_Temprature"
MODEL_DIR   = Path(__file__).resolve().parent
PKL_ROOT    = MODEL_DIR / "01_PKL Files"

PKL_FOLDERS = {
    'ARMA'      : PKL_ROOT / "10_ARMA",
    'AutoARIMA' : PKL_ROOT / "11_AutoARIMA",
    'ARIMA'     : PKL_ROOT / "12_ARIMA",
}
for folder in PKL_FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

PKL_PREFIXES = {
    'ARMA'      : 'arma_model',
    'AutoARIMA' : 'autoarima_model',
    'ARIMA'     : 'arima_model',
}

tdy   = datetime.datetime.today().strftime('%Y-%m-%d')
start = datetime.datetime.now()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGIONS          = [11, 24, 27, 28, 32, 44, 52, 53]
CALIB_YEARS      = range(2010, 2024)
MIN_CALIB_OBS    = 1000
FORECAST_HORIZON = 365
MAX_P            = 2
MAX_Q            = 2
TRAINING_START   = pd.Timestamp('1980-01-01')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
bftransformer = Imputer(method='backfill')


def _safe_order(auto_model):
    """
    Extract the selected (p, d, q) order from a fitted sktime AutoARIMA.
    Accesses the underlying pmdarima object directly to avoid dependency on
    get_fitted_params() key names which vary across sktime versions.
    """
    try:
        order = auto_model._fitted_forecaster.order
        return (int(order[0]), int(order[1]), int(order[2]))
    except AttributeError:
        # fallback: if internal structure changes, return max order
        return (MAX_P, 0, MAX_Q)


def _build_forecast_dates(calib_end, horizon):
    """
    Build a DatetimeIndex of `horizon` calendar days after calib_end,
    including Feb 29 where it falls (AR models retain leap days).
    """
    dates = []
    d = calib_end
    while len(dates) < horizon:
        d = d + pd.Timedelta(days=1)
        dates.append(d)
    return pd.DatetimeIndex(dates)


def _fit_arma(sub_train):
    """
    Two-step ARMA fit: AutoARIMA(d=0) discovers (p, q),
    then a pure ARIMA(p, 0, q) is refit for a clean object.
    Returns (fitted_model, order_tuple) or raises on failure.
    """
    auto = AutoARIMA(d=0, max_p=MAX_P, max_q=MAX_Q,
                     suppress_warnings=True, error_action='ignore')
    auto.fit(sub_train)
    p, _, q = _safe_order(auto)
    order = (p, 0, q)
    mod = ARIMA(order=order, suppress_warnings=True)
    mod.fit(sub_train)
    return mod, order


def _fit_autoarima(sub_train):
    """
    AutoARIMA fitted directly (d=0). Order extracted from pmdarima object.
    Returns (fitted_model, order_tuple) or raises on failure.
    """
    mod = AutoARIMA(d=0, max_p=MAX_P, max_q=MAX_Q,
                    suppress_warnings=True, error_action='ignore')
    mod.fit(sub_train)
    p, _, q = _safe_order(mod)
    order = (p, 0, q)
    return mod, order


def _fit_arima(sub_train):
    """
    Two-step ARIMA fit: AutoARIMA(d=1) discovers (p, q),
    then a pure ARIMA(p, 1, q) is refit for a clean object.
    Returns (fitted_model, order_tuple) or raises on failure.
    """
    auto = AutoARIMA(d=1, max_p=MAX_P, max_q=MAX_Q,
                     suppress_warnings=True, error_action='ignore')
    auto.fit(sub_train)
    p, _, q = _safe_order(auto)
    order = (p, 1, q)
    mod = ARIMA(order=order, suppress_warnings=True)
    mod.fit(sub_train)
    return mod, order


FITTERS = {
    'ARMA'      : _fit_arma,
    'AutoARIMA' : _fit_autoarima,
    'ARIMA'     : _fit_arima,
}

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
raw = pd.read_csv(TEMP_DIR / "region_temp_extended.csv", parse_dates=['date'])

# summary containers — one per model type
summary_rows = {name: [] for name in FITTERS}
all_param_dfs = {name: pd.DataFrame() for name in FITTERS}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for region in REGIONS:
    print(f"\n{'='*50}")
    print(f"Region {region}")
    print(f"{'='*50}")

    subset_data = (raw[raw['region_code'] == region]
                   .set_index('date')
                   [['TAVG_imptd']]
                   .dropna()
                   .sort_index())

    if subset_data.empty:
        print(f'  No data for region {region}, skipping.')
        continue

    subset_data = subset_data.loc[TRAINING_START:].copy()

    if subset_data.empty:
        print(f'  No data from {TRAINING_START.date()} forward for region {region}, skipping.')
        continue

    # Feb 29 RETAINED — per Barnor et al. Section 3.1.1 for AR models.
    # Reindex to full daily calendar so sktime gets a gapless DatetimeIndex.
    full_idx = pd.date_range(
        start=subset_data.index.min(),
        end=subset_data.index.max(),
        freq='D'
    )
    subset_data = subset_data.reindex(full_idx)
    subset_data.index.name = 'date'
    subset_data['dayofyear'] = subset_data.index.dayofyear
    subset_data['month']     = subset_data.index.month
    subset_data['name']      = region

    for calib_year in CALIB_YEARS:
        calib_end = pd.Timestamp(f'{calib_year}-12-31')
        calib_df  = subset_data.loc[:calib_end].copy()

        if len(calib_df) < MIN_CALIB_OBS:
            print(f'  {calib_year}: insufficient observations ({len(calib_df)}), skipping.')
            continue

        # Build daily-frequency series for sktime; backfill any missing values
        # (handles sparse gaps and any upstream missing data)
        raw_series = calib_df['TAVG_imptd'].asfreq('D')
        raw_series = bftransformer.fit_transform(raw_series)

        if raw_series.isna().any():
            print(f'  {calib_year}: NaNs remain after imputation, skipping.')
            continue

        # Forecast dates: full calendar days after calib_end (Feb 29 included)
        forecast_dates = _build_forecast_dates(calib_end, FORECAST_HORIZON)

        # ----------------------------------------------------------------
        # Fit each model
        # ----------------------------------------------------------------
        for model_name, fitter in FITTERS.items():
            run_start = datetime.datetime.now()

            try:
                mod, order = fitter(raw_series)
            except Exception as e:
                print(f'  {calib_year} [{model_name}]: fit failed '
                      f'({type(e).__name__}: {e}), skipping.')
                continue

            fh = np.arange(1, FORECAST_HORIZON + 1)
            try:
                preds        = mod.predict(fh=fh)
                forecast_temp = np.asarray(preds, dtype=float).reshape(-1)
            except Exception as e:
                print(f'  {calib_year} [{model_name}]: predict failed '
                      f'({type(e).__name__}: {e}), skipping.')
                continue

            export_pkl = {
                'fitted_model'      : mod,
                'order'             : order,
                'forecast_temp_365' : forecast_temp,
                'forecast_dates'    : forecast_dates,
                'n_calib_days'      : int(raw_series.notna().sum()),
                'df'                : subset_data.copy(),
                'df_calib'          : calib_df.copy(),
                'calib_end'         : f'{calib_year}-12-31',
                'region_code'       : region,
                'model_name'        : model_name,
            }

            pkl_path = str(
                PKL_FOLDERS[model_name] /
                f'{PKL_PREFIXES[model_name]}_{region}_calib_{calib_year}.pkl'
            )
            with open(pkl_path, 'wb') as f:
                pickle.dump(export_pkl, f)

            elapsed = (datetime.datetime.now() - run_start).total_seconds()

            param_row = pd.DataFrame(
                [[region, calib_year, order[0], order[1], order[2]]],
                columns=['region', 'calib_year', 'p', 'd', 'q']
            )
            all_param_dfs[model_name] = pd.concat(
                [all_param_dfs[model_name], param_row], axis=0
            )

            summary_rows[model_name].append({
                'region'       : region,
                'calib_year'   : calib_year,
                'p'            : order[0],
                'd'            : order[1],
                'q'            : order[2],
                'n_calib_days' : export_pkl['n_calib_days'],
                'time_taken_s' : elapsed,
                'pkl_path'     : pkl_path,
            })

            print(
                f'  {calib_year} [{model_name}]: '
                f'N={export_pkl["n_calib_days"]:,}  '
                f'order={order}  '
                f't={elapsed:.1f}s  '
                f'-> {pkl_path}'
            )

# ---------------------------------------------------------------------------
# Save aggregate summary files — one per model type
# ---------------------------------------------------------------------------
for model_name in FITTERS:
    if not all_param_dfs[model_name].empty:
        all_param_dfs[model_name].to_csv(
            MODEL_DIR / f'{model_name.lower()}_oos_param_df_{tdy}.csv',
            index=False
        )
    summary_df = pd.DataFrame(summary_rows[model_name])
    if not summary_df.empty:
        summary_df.to_csv(
            MODEL_DIR / f'{model_name.lower()}_model_summary.csv',
            index=False
        )

print(f'\nTotal time: {datetime.datetime.now() - start}')
print(f'PKL folders:')
for name, folder in PKL_FOLDERS.items():
    n_pkls = len(list(folder.glob('*.pkl')))
    print(f'  [{name}] {folder}  ({n_pkls} PKLs)')
