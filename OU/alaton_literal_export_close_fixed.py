"""
Alaton-style rolling PKL exporter.

Goal:
- stay as close as possible to the original example `Alaton.py`
- while exporting one PKL per region per calibration year:
    alaton_models/alaton_model_{region}_calib_{year}.pkl

Expected input file:
    EDA/region_avg.csv
with columns:
    date, region_code, daily_avg_temperature

Notes on closeness to the original example:
- Keeps the original-style helper function names and formulas where possible.
- Uses `curve_fit(alatonTmfunc, ...)` exactly like the example for the deterministic
  temperature fit.
- Keeps the example's monthly sigma-1, mean-reversion, and sigma-2 workflow.
- The original example calibrated from prebuilt train/test feature tables and then
  simulated out-of-sample paths. This version instead calibrates on an expanding
  window and writes PKLs, but leaves the core calibration logic deliberately close
  to the original.

Important corrections vs prior draft:
- Uses the original phase formula exactly:
      D = (np.arctan(popt[3] / popt[2])) - np.pi
  rather than arctan2.
- Uses a 1-based sequential t index consistently, matching the original fit.
- Stores df_calib['t'] so downstream forecasting can continue the same time index
  instead of reconstructing t from calendar dates.
"""

# imports
import warnings
warnings.simplefilter("ignore")

# import libraries
import pandas as pd
import numpy as np
import os
import pickle
from scipy.optimize import curve_fit
import datetime
from scipy import stats

# optional plotting backend (safe on headless machines)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# set paths
input_data_loc = 'EDA'
pkl_folder = 'Outputs/alaton_models'
os.makedirs(pkl_folder, exist_ok=True)

tdy = datetime.datetime.today().strftime('%Y-%m-%d')

# begin script
start = datetime.datetime.now()

# configuration
REGIONS = [11, 24, 27, 28, 32, 44, 52, 53]
CALIB_YEARS = range(2014, 2024)
MIN_CALIB_OBS = 1000
MAX_LAG = 60

# global fit params to preserve the original-style alatmeanT(t) helper behavior
popt = None


# define functions

# define alaton function (3.9) to compute parameters for the mean of temperature
# comprising linear trend and seasonal cycle
def alatonTmfunc(t, a1, a2, a3, a4):
    w = (2 * np.pi) / 365
    return a1 + (a2 * t) + (a3 * (np.sin(w * t))) + (a4 * (np.cos(w * t)))


# function for mean/deterministic part of temperature per alaton paper
# kept as close to the original example as possible: uses current global `popt`
def alatmeanT(t):
    A = popt[0]
    B = popt[1]
    C = np.sqrt((popt[2] ** 2) + (popt[3] ** 2))
    # literal match to the original example
    D = (np.arctan(popt[3] / popt[2])) - np.pi
    osc = (2 * np.pi) / 365
    return A + (B * t) + (C * np.sin((osc * t) + D))


# function to calculate second estimate of sigma - part after summation
# kept in the same style as the example
def sigma2(row):
    return ((row['TAVG_imptd']
             - (row['meanrev'] * row['Tm-1'])
             - ((1 - row['meanrev']) * row['tavgcel-1'])) ** 2).sum()


# helper for acf diagnostics
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


# remove feb 29 to preserve a 365-day seasonal cycle
def remove_feb29(df):
    return df[~((df.index.month == 2) & (df.index.day == 29))].copy()


# import raw data
raw = pd.read_csv(f'{input_data_loc}/region_avg.csv', parse_dates=['date'])

summary_rows = []
all_popt_df = pd.DataFrame()

diag_year = min(CALIB_YEARS)
diag_store = []

# main loop, kept close in spirit to the original `for city in ...` block
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
    subset_data['month'] = subset_data.index.month
    subset_data['dayofyear'] = subset_data.index.dayofyear
    subset_data['name'] = region

    # full-series t for reference / debugging only
    subset_data['t'] = np.arange(1, len(subset_data) + 1)

    for calib_year in CALIB_YEARS:
        calib_end = pd.Timestamp(f'{calib_year}-12-31')
        calib_df = subset_data.loc[:calib_end].copy()

        if len(calib_df) < MIN_CALIB_OBS:
            print(f'  {calib_year}: insufficient data, skipping.')
            continue

        # call data prep func (very close to original example)
        x = np.arange(1, len(calib_df['TAVG_imptd'].dropna(how='any')) + 1)
        y = calib_df['TAVG_imptd'].dropna(how='any').reset_index().values[:, -1]

        # fit actual data to alaton Tm parameter estimation function
        popt, _ = curve_fit(alatonTmfunc, x, y, maxfev=10000)

        # summarize parameter values
        a1, a2, a3, a4 = popt
        A = popt[0]
        B = popt[1]
        C = np.sqrt((popt[2] ** 2) + (popt[3] ** 2))
        phi = (np.arctan(popt[3] / popt[2])) - np.pi

        # create calibration copy with a sequential day index matching the example's fit
        subset1yr = calib_df.copy()
        subset1yr['t'] = np.arange(1, len(subset1yr) + 1)

        # compute sigma estimator 1 exactly in example style
        subset1yr['1daydiff'] = subset1yr.groupby(['month'])['TAVG_imptd'].diff(1)
        subset1yr['1daydiffsqrd'] = subset1yr['1daydiff'] ** 2

        sigma1sqd = (subset1yr.groupby(['month'])['1daydiffsqrd'].sum()
                     / subset1yr.groupby(['month'])['1daydiffsqrd'].count())
        sigma1 = np.sqrt(sigma1sqd)

        # add back sigma to df in example style
        subset1yr = subset1yr.reset_index().merge(
            sigma1, how='left', on=['month', 'month']
        ).rename(columns={'1daydiffsqrd_x': '1daydiffsqrd',
                          '1daydiffsqrd_y': 'sigma1'}).merge(
            sigma1sqd, how='left', on=['month', 'month']
        ).rename(columns={'1daydiffsqrd_x': '1daydiffsqrd',
                          '1daydiffsqrd_y': 'sigma1sqd'})

        # second sigma estimate inputs
        subset1yr['Tm'] = subset1yr['t'].apply(alatmeanT)
        subset1yr['Tm-1'] = subset1yr['Tm'].shift(1)
        subset1yr['tavgcel-1'] = subset1yr['TAVG_imptd'].shift(1)

        subset1yr['sigma1sqd-1'] = subset1yr.groupby(['month'])['sigma1sqd'].shift(1)
        subset1yr['Yi-1'] = ((subset1yr['Tm-1'] - subset1yr['tavgcel-1'])
                             / subset1yr['sigma1sqd-1'])

        # equation 3.26 calculating the mean reversion per month using estimator one variance
        def _month_meanrev(x):
            num = ((x['Yi-1']) * (x['TAVG_imptd'] - x['Tm'])).sum()
            den = ((x['Yi-1']) * (x['tavgcel-1'] - x['Tm-1'])).sum()
            if pd.isna(num) or pd.isna(den) or den == 0:
                return np.nan
            ratio = num / den
            if not np.isfinite(ratio) or ratio <= 0:
                return np.nan
            return -np.log(ratio)

        alameanr = subset1yr.groupby(['month']).apply(_month_meanrev)

        # add back monthly mean reversion value to data frame
        subset1yr = subset1yr.merge(alameanr.rename('meanrev'), how='left', on=['month', 'month'])

        # create sigma estimate table for region
        alatsigma2 = np.sqrt(
            subset1yr.groupby(['month']).apply(sigma2)
            / subset1yr.groupby(['month'])['1daydiffsqrd'].count()
        )

        sigma_df = pd.concat([
            sigma1.rename('Estimation1'),
            alatsigma2.rename('Estimation2')
        ], axis=1)

        sigma_df['avg_sigma'] = (sigma_df.Estimation1 + sigma_df.Estimation2) / 2

        subset1yr = subset1yr.merge(sigma_df.reset_index(), how='left', on=['month', 'month'])

        # collapse month-wise mean reversion to a single model value for export
        a_est = subset1yr['meanrev'].dropna().mean()
        sigma_monthly = sigma_df['avg_sigma'].dropna().to_dict()

        # diagnostics in the style of the user's rolling script / sample PKL
        df_calib = subset1yr.set_index('date').copy()
        df_calib['X'] = df_calib['TAVG_imptd'] - df_calib['Tm']
        df_calib['sigma'] = df_calib.index.month.map(sigma_monthly)

        T_v = df_calib['TAVG_imptd'].values
        Tm_v = df_calib['Tm'].values

        cond_mean = np.full(len(T_v), np.nan)
        if pd.notna(a_est) and a_est > 0:
            cond_mean[1:] = ((T_v[:-1] - Tm_v[:-1]) * np.exp(-a_est)) + Tm_v[1:]
            cstd = df_calib['sigma'].values * np.sqrt((1 - np.exp(-2 * a_est)) / (2 * a_est))
            with np.errstate(divide='ignore', invalid='ignore'):
                df_calib['resid'] = (T_v - cond_mean) / cstd
        else:
            df_calib['resid'] = np.nan

        e = df_calib['resid'].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(e) >= 8 and np.isfinite(e.std()) and e.std() > 0:
            _, ks_p = stats.kstest(e, lambda x: stats.norm.cdf(x, e.mean(), e.std()))
        else:
            ks_p = np.nan

        ci_bound = 1.96 / np.sqrt(len(T_v)) if len(T_v) > 0 else np.nan
        acf_X = compute_acf(df_calib['X'].dropna().values, MAX_LAG)
        acf_e = compute_acf(e, MAX_LAG) if len(e) > 0 else np.full(MAX_LAG + 1, np.nan)

        # build export dict to match the target PKL style
        export_model = {
            'A': A,
            'B': B,
            'C': C,
            'phi': phi,
            'a': a_est,
            'sigma_monthly': sigma_monthly,
            'df': subset_data.copy(),
            'df_calib': df_calib.copy(),
            'sig1': sigma1.to_dict(),
            'sig2': alatsigma2.dropna().to_dict(),
            'model_name': 'Alaton (2002)',
            'calib_end': f'{calib_year}-12-31',
            'e': e,
            'ks_p': ks_p,
            'acf_X': acf_X,
            'acf_e': acf_e,
            'ci_bound': ci_bound,
            'region_code': region,
        }

        pkl_path = os.path.join(
            pkl_folder,
            f'alaton_model_{region}_calib_{calib_year}.pkl'
        )
        with open(pkl_path, 'wb') as f:
            pickle.dump(export_model, f)

        # create data frame of Tm params and add to main params df (same spirit as example)
        popt_df = pd.DataFrame([[a1, a2, a3, a4]])
        popt_df['name'] = region
        popt_df['calib_year'] = calib_year
        popt_df['A'] = A
        popt_df['B'] = B
        popt_df['C'] = C
        popt_df['phi'] = phi
        popt_df['a'] = a_est
        all_popt_df = pd.concat([all_popt_df, popt_df], axis=0)

        sigma_arr = np.array(list(sigma_monthly.values())) if sigma_monthly else np.array([])
        summary_rows.append({
            'region': region,
            'calib_year': calib_year,
            'A': A,
            'B_C_per_decade': B * 365 * 10,
            'C': C,
            'phi': phi,
            'a': a_est,
            'half_life_days': (np.log(2) / a_est) if pd.notna(a_est) and a_est > 0 else np.nan,
            'sigma_min': sigma_arr.min() if len(sigma_arr) else np.nan,
            'sigma_max': sigma_arr.max() if len(sigma_arr) else np.nan,
            'ks_p': ks_p,
            'normal_rejected_5pct': (ks_p < 0.05) if pd.notna(ks_p) else np.nan,
            'pkl_path': pkl_path,
        })

        a_print = a_est if pd.notna(a_est) else float('nan')
        sigma_min = sigma_arr.min() if len(sigma_arr) else np.nan
        sigma_max = sigma_arr.max() if len(sigma_arr) else np.nan

        print(
            f'  {calib_year}: '
            f'N_calib={len(df_calib):,}  '
            f'a={a_print:.4f}  '
            f'trend={B * 365 * 10:.3f} C/dec  '
            f'sigma=[{sigma_min:.3f},{sigma_max:.3f}]  '
            f'-> {pkl_path}'
        )

        if calib_year == diag_year:
            diag_store.append((region, df_calib.copy(), sigma1.to_dict(), sigma_monthly, e, ks_p, acf_X, acf_e, ci_bound))

# save results to file, preserving example-like outputs plus summary
all_popt_df.to_csv(f'alaton_oos_popt_df_{tdy}.csv', index=False)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('alaton_model_summary.csv', index=False)

# lightweight diagnostics figure (optional)
if len(diag_store) > 0:
    n = len(diag_store)
    fig, axes = plt.subplots(4, n, figsize=(3.0 * n, 13))
    if n == 1:
        axes = axes.reshape(4, 1)
    fig.suptitle(f'Alaton literal-style calibration overview ({diag_year})', fontsize=12, fontweight='bold', y=1.005)
    lags = np.arange(MAX_LAG + 1)
    months = np.arange(1, 13)
    month_names = ['J','F','M','A','M','J','J','A','S','O','N','D']

    for col, (region, df_calib, sig1_dict, sigma_monthly, e, ks_p, acf_X, acf_e, ci_bound) in enumerate(diag_store):
        ax = axes[0, col]
        ax.bar(np.arange(12) - 0.16, [sig1_dict.get(m, 0) for m in months], width=0.32, alpha=0.6, label='Est1')
        ax.bar(np.arange(12) + 0.16, [sigma_monthly.get(m, 0) for m in months], width=0.32, alpha=0.8, label='Mean')
        ax.set_title(f'R{region}', fontsize=9, fontweight='bold')
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(month_names, fontsize=6)
        if col == 0:
            ax.set_ylabel('sigma', fontsize=8)
            ax.legend(fontsize=5, frameon=False)

        ax = axes[1, col]
        if len(e) > 0:
            ax.hist(e, bins=60, density=True, alpha=0.4)
            x_grid = np.linspace(np.percentile(e, 0.5), np.percentile(e, 99.5), 300)
            if np.isfinite(np.std(e)) and np.std(e) > 0:
                ax.plot(x_grid, stats.norm.pdf(x_grid, np.mean(e), np.std(e)), lw=1.5)
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 2)
        ax.text(0.97, 0.97, f'KS p={ks_p:.3f}' if pd.notna(ks_p) else 'KS p=nan',
                transform=ax.transAxes, ha='right', va='top', fontsize=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        if col == 0:
            ax.set_ylabel('Density (log)', fontsize=8)

        ax = axes[2, col]
        ax.bar(lags[1:], acf_X[1:], width=0.8)
        if pd.notna(ci_bound):
            ax.axhline(ci_bound, ls='--', lw=0.8)
            ax.axhline(-ci_bound, ls='--', lw=0.8)
        ax.axhline(0, lw=0.4, color='k')
        if col == 0:
            ax.set_ylabel('ACF(X)', fontsize=8)

        ax = axes[3, col]
        ax.bar(lags[1:], acf_e[1:], width=0.8)
        if pd.notna(ci_bound):
            ax.axhline(ci_bound, ls='--', lw=0.8)
            ax.axhline(-ci_bound, ls='--', lw=0.8)
        ax.axhline(0, lw=0.4, color='k')
        ax.set_xlabel('Lag', fontsize=7)
        if col == 0:
            ax.set_ylabel('ACF(e)', fontsize=8)

    fig.tight_layout()
    fig.savefig('alaton_model_diagnostics.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

print('Time taken ', datetime.datetime.now() - start)
print(f'Wrote PKLs to: {pkl_folder}')
