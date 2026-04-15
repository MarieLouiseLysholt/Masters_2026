"""
Literal-style Benth rolling PKL exporter.

Goal:
- stay as close as possible to the original example `Benth.py`
- while exporting one PKL per region per calibration year:
    benth_models/ou_levy_model_{region}_calib_{year}.pkl

Expected input file:
    EDA/region_avg.csv
with columns:
    date, region_code, daily_avg_temperature

Notes on closeness to the original example:
- keeps the original helper names:
    benth_seas, ss_linreg, benth_reg
- uses the same seasonal fit:
    a0 + a1 * cos((2π/365) * (t - t0))
- uses the same deseasonalise -> regress lag1 -> residual variance ->
  smoothed log-volatility -> generalised hyperbolic fit workflow
- exports one PKL per region x calibration year rather than directly writing
  simulation CSVs

Compatibility notes:
- the original script stores seasonal parameters as a0, a1, t0 and the AR(1)
  regression as slope/intercept.
- for downstream code that expects a Benth OU-style interface, this exporter also
  stores aliases:
      a   -> a0
      b   -> 0.0
      b1  -> a1
      g1  -> t0
      alpha -> -log(slope)   if 0 < slope < 1 else NaN
  Those aliases are convenience fields only; the literal model remains the
  original Benth.py structure.
"""

# imports
import warnings
warnings.simplefilter("ignore")

# import libraries
import pandas as pd
import numpy as np
import os
import pickle
import datetime

# scipy
from scipy.optimize import curve_fit
import scipy.stats as ss
from scipy.stats import genhyperbolic

# optional plotting backend (safe on headless machines)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# set paths
input_data_loc = 'EDA'
pkl_folder = 'Outputs/benth_models'
os.makedirs(pkl_folder, exist_ok=True)

tdy = datetime.datetime.today().strftime('%Y-%m-%d')

# begin code
start = datetime.datetime.now()

# configuration
REGIONS = [11, 24, 27, 28, 32, 44, 52, 53]
CALIB_YEARS = range(2014, 2024)
MIN_CALIB_OBS = 1000
MAX_LAG = 60


# define functions

# function to extract seasonality from daily average temperature
def benth_seas(t, a0, a1, t0):
    return a0 + (a1 * np.cos(((2 * np.pi) / 365) * (t - t0)))

# benth regression model regressing deseasonalised temp on lagged deseasonalised temp
def ss_linreg(x):
    return (slope * x) + intercept

# benth cyclic & regression analysis
def benth_reg(a, x):
    return a * x

# helper for diagnostics
def compute_acf(x, max_lag=60):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.full(max_lag + 1, np.nan)
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom == 0:
        out = np.zeros(max_lag + 1)
        out[0] = 1.0
        return out
    n = len(xc)
    acf = [1.0]
    for k in range(1, max_lag + 1):
        if k >= n:
            acf.append(np.nan)
        else:
            acf.append(np.dot(xc[:n-k], xc[k:]) / denom)
    return np.array(acf)

# remove feb 29 to preserve a 365-day cycle
def remove_feb29(df):
    return df[~((df.index.month == 2) & (df.index.day == 29))].copy()


# import raw data
raw = pd.read_csv(f'{input_data_loc}/region_avg.csv', parse_dates=['date'])

summary_rows = []
all_param_df = pd.DataFrame()
diag_store = []
diag_year = min(CALIB_YEARS)

# globals used by ss_linreg to stay close to the original source style
slope = np.nan
intercept = np.nan

# begin main loop
for region in REGIONS:
    print(region)

    subset_data = (raw[raw['region_code'] == region]
                   .set_index('date')
                   .rename(columns={'daily_avg_temperature': 'TAVG_imptd'})
                   [['TAVG_imptd']]
                   .dropna()
                   .sort_index())

    if subset_data.empty:
        print(f'No data for region {region}, skipping.')
        continue

    subset_data = remove_feb29(subset_data)
    subset_data['dayofyear'] = subset_data.index.dayofyear
    subset_data['month'] = subset_data.index.month
    subset_data['name'] = region

    for calib_year in CALIB_YEARS:
        calib_end = pd.Timestamp(f'{calib_year}-12-31')
        calib_df = subset_data.loc[:calib_end].copy()

        if len(calib_df) < MIN_CALIB_OBS:
            print(f'  {calib_year}: insufficient data, skipping.')
            continue

        # fit benth seasonality function to data and extract parameters
        x = calib_df['dayofyear'].values.astype(float)
        y = calib_df['TAVG_imptd'].values.astype(float)
        popt, _ = curve_fit(f=benth_seas, xdata=x, ydata=y, maxfev=10000)
        a0, a1, t0 = popt

        # calculate seasonality and subtract from average temperature
        subset1yr = calib_df.copy()
        subset1yr.loc[:, 'seas'] = benth_seas(subset1yr.dayofyear, a0, a1, t0)
        subset1yr.loc[:, 'de_seas_temp'] = subset1yr.TAVG_imptd - subset1yr.seas
        subset1yr.loc[:, 'de_seas_temp_lag1'] = subset1yr['de_seas_temp'].shift(1)

        # regress deseasonalised temp against lagged deseasonalised temp
        reg_df = subset1yr.dropna(subset=['de_seas_temp_lag1', 'de_seas_temp']).copy()
        reg_x = reg_df['de_seas_temp_lag1'].values
        reg_y = reg_df['de_seas_temp'].values
        slope, intercept, r, p, std_err = ss.linregress(reg_x, reg_y)

        # remove linear reg components from seasonal residuals
        subset1yr['lag_seas_pred'] = subset1yr['de_seas_temp_lag1'].apply(ss_linreg)
        subset1yr['lag_seas_pred_resid'] = subset1yr['lag_seas_pred'] - subset1yr['de_seas_temp']
        subset1yr['lag_seas_pred_resid_sqd'] = subset1yr['lag_seas_pred_resid'] ** 2

        # calculate expected daily volatility of residuals by benth methodology
        subset1yr['expected_daily_vol'] = subset1yr.groupby(['dayofyear'])['lag_seas_pred_resid_sqd'].transform('mean')
        subset1yr['expected_daily_vol_log'] = np.log(subset1yr['expected_daily_vol'])
        subset1yr['expected_daily_vol_log_mvnavge'] = subset1yr['expected_daily_vol_log'].rolling(window=3).mean()
        subset1yr['expected_daily_vol_smthd'] = np.exp(subset1yr['expected_daily_vol_log_mvnavge'])

        # calculate residuals after removing volatility
        subset1yr['vol_resid'] = subset1yr['lag_seas_pred_resid'] / np.sqrt(subset1yr['expected_daily_vol_smthd'])

        # fit residuals to hyperbolic distribution using scipy implementation
        vol_resid_fit = subset1yr['vol_resid'][~np.isnan(subset1yr['vol_resid'])]
        if len(vol_resid_fit) < 20:
            print(f'  {calib_year}: insufficient residual data, skipping.')
            continue

        gh_p, gh_a, gh_b, gh_loc, gh_scale = genhyperbolic.fit(vol_resid_fit)

        # diagnostics
        e = vol_resid_fit.values
        if len(e) >= 8 and np.isfinite(np.std(e)) and np.std(e) > 0:
            _, ks_p = ss.kstest(e, lambda x: ss.norm.cdf(x, np.mean(e), np.std(e)))
        else:
            ks_p = np.nan

        subset1yr['X'] = subset1yr['de_seas_temp']
        acf_X = compute_acf(subset1yr['X'].dropna().values, MAX_LAG)
        acf_e = compute_acf(e, MAX_LAG)
        ci_bound = 1.96 / np.sqrt(len(subset1yr)) if len(subset1yr) > 0 else np.nan

        sigma_dayofyear = (subset1yr[['dayofyear', 'expected_daily_vol_smthd']]
                           .dropna()
                           .groupby('dayofyear')['expected_daily_vol_smthd']
                           .last()
                           .to_dict())

        # convenience OU-style aliases for downstream comparison scripts
        if np.isfinite(slope) and 0 < slope < 1:
            alpha_alias = -np.log(slope)
        else:
            alpha_alias = np.nan

        df_calib = subset1yr.copy()

        export_model = {
            # literal-style Benth.py parameters
            'a0': float(a0),
            'a1': float(a1),
            't0': float(t0),
            'slope': float(slope),
            'intercept': float(intercept),
            'gh_p': float(gh_p),
            'gh_a': float(gh_a),
            'gh_b': float(gh_b),
            'gh_loc': float(gh_loc),
            'gh_scale': float(gh_scale),
            'sigma_dayofyear': sigma_dayofyear,

            # compatibility aliases
            'a': float(a0),
            'b': 0.0,
            'b1': float(a1),
            'g1': float(t0),
            'alpha': float(alpha_alias),

            # stored data / diagnostics
            'df': subset_data.copy(),
            'df_calib': df_calib.copy(),
            'model_name': 'Benth.py literal-style export',
            'calib_end': f'{calib_year}-12-31',
            'region_code': region,
            'e': e,
            'ks_p': ks_p,
            'acf_X': acf_X,
            'acf_e': acf_e,
            'ci_bound': ci_bound,
        }

        pkl_path = os.path.join(
            pkl_folder,
            f'ou_levy_model_{region}_calib_{calib_year}.pkl'
        )
        with open(pkl_path, 'wb') as f:
            pickle.dump(export_model, f)

        param_df = pd.DataFrame([[
            region, calib_year, a0, a1, t0, slope, intercept,
            gh_p, gh_a, gh_b, gh_loc, gh_scale, alpha_alias
        ]], columns=[
            'name', 'calib_year', 'a0', 'a1', 't0', 'slope', 'intercept',
            'p', 'a', 'b', 'loc', 'scale', 'alpha_alias'
        ])
        all_param_df = pd.concat([all_param_df, param_df], axis=0)

        vol_vals = np.array(list(sigma_dayofyear.values())) if sigma_dayofyear else np.array([])
        summary_rows.append({
            'region': region,
            'calib_year': calib_year,
            'a0': float(a0),
            'a1': float(a1),
            't0': float(t0),
            'slope': float(slope),
            'intercept': float(intercept),
            'alpha_alias': float(alpha_alias) if np.isfinite(alpha_alias) else np.nan,
            'gh_p': float(gh_p),
            'gh_a': float(gh_a),
            'gh_b': float(gh_b),
            'gh_loc': float(gh_loc),
            'gh_scale': float(gh_scale),
            'vol_min': vol_vals.min() if len(vol_vals) else np.nan,
            'vol_max': vol_vals.max() if len(vol_vals) else np.nan,
            'ks_p': ks_p,
            'pkl_path': pkl_path,
        })

        print(
            f'  {calib_year}: '
            f'N_calib={len(df_calib):,}  '
            f'slope={slope:.4f}  '
            f'a0={a0:.3f}  a1={a1:.3f}  t0={t0:.3f}  '
            f'-> {pkl_path}'
        )

        if calib_year == diag_year:
            diag_store.append((
                region,
                df_calib[['dayofyear', 'expected_daily_vol_smthd']].copy(),
                e.copy(),
                ks_p,
                acf_X.copy(),
                acf_e.copy(),
                ci_bound
            ))

# save results to file
all_param_df.to_csv(f'benth_oos_param_df_{tdy}.csv', index=False)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('benth_model_summary.csv', index=False)

# lightweight diagnostics figure (optional)
if len(diag_store) > 0:
    n = len(diag_store)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 10))
    if n == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(f'Benth literal-style calibration overview ({diag_year})',
                 fontsize=12, fontweight='bold', y=1.005)

    lags = np.arange(MAX_LAG + 1)

    for col, (region, vol_df, e, ks_p, acf_X, acf_e, ci_bound) in enumerate(diag_store):
        ax = axes[0, col]
        vol_plot = (vol_df.dropna()
                    .groupby('dayofyear')['expected_daily_vol_smthd']
                    .mean()
                    .reindex(np.arange(1, 366)))
        ax.plot(np.arange(1, 366), vol_plot.values)
        ax.set_title(f'R{region}', fontsize=9, fontweight='bold')
        if col == 0:
            ax.set_ylabel('smoothed vol', fontsize=8)

        ax = axes[1, col]
        if len(e) > 0:
            ax.hist(e, bins=60, density=True, alpha=0.4)
            x_grid = np.linspace(np.percentile(e, 0.5), np.percentile(e, 99.5), 300)
            if np.isfinite(np.std(e)) and np.std(e) > 0:
                ax.plot(x_grid, ss.norm.pdf(x_grid, np.mean(e), np.std(e)), lw=1.5)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 2)
        ax.text(0.97, 0.97, f'KS p={ks_p:.3f}' if pd.notna(ks_p) else 'KS p=nan',
                transform=ax.transAxes, ha='right', va='top', fontsize=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        if col == 0:
            ax.set_ylabel('Density (log)', fontsize=8)

        ax = axes[2, col]
        ax.bar(lags[1:], acf_e[1:], width=0.8)
        if pd.notna(ci_bound):
            ax.axhline(ci_bound, ls='--', lw=0.8)
            ax.axhline(-ci_bound, ls='--', lw=0.8)
        ax.axhline(0, lw=0.4, color='k')
        ax.set_xlabel('Lag', fontsize=7)
        if col == 0:
            ax.set_ylabel('ACF(e)', fontsize=8)

    fig.tight_layout()
    fig.savefig('benth_model_diagnostics.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

print('Time taken ', datetime.datetime.now() - start)
print(f'Wrote PKLs to: {pkl_folder}')
