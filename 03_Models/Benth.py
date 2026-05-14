"""
Benth model — Schiller, Seidler & Wimmer (2012) specification with AIC-selected
truncation orders.

Implements the OU process from Benth & Saltyte-Benth (2007) as restated in SSW:

    dT_t = (a(theta_t - T_t) + d theta_t/dt) dt + sigma_t dW_t      (SSW eq. 3)

with deterministic mean

    theta_t = b + c t + sum_{i=1..I1} A_i sin(2 pi i t / 365)
                      + sum_{j=1..J1} B_j cos(2 pi j t / 365)        (SSW eq. 7)

and seasonal variance

    sigma^2_t = d + sum_{i=1..I2} c_i sin(2 pi i t / 365)
                  + sum_{j=1..J2} d_j cos(2 pi j t / 365)             (SSW eq. 8)

Innovations are Gaussian (Brownian-driven), the SSW choice; this replaces the
generalised hyperbolic / Levy distribution used in the prior implementation.

Mean-reversion is estimated by SSW eq. (6), the Bibby-Sorensen martingale
estimating function used by both Alaton and Benth in SSW for fair comparison.

Truncation orders are selected per region by minimising in-sample AIC,
addressing the SSW conclusion that the Stockholm-tuned values do not generalise
across stations.

Phase 1 (specification selection, once per region):
    Window   : 1980-01-01 -> 2010-12-31
    Grid     : (I1, J1) in {0..4}^2 \\ {(0,0)}, (I2, J2) in {1..4}^2
    Criterion: in-sample AIC

Phase 2 (rolling parameter refit, one PKL per region x calib_year):
    Window : 1980-01-01 -> calib_year-12-31
    Spec   : (I1*, J1*, I2*, J2*) frozen from Phase 1
    Input  : 02_Data/01_Temprature/region_temp_extended.csv, TAVG_imptd
    Output : 01_PKL Files/03_Benth/ou_brownian_model_{region}_calib_{year}.pkl
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

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
import scipy.stats as ss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
TEMP_DIR  = ROOT / "02_Data" / "01_Temprature"
MODEL_DIR = Path(__file__).resolve().parent
PKL_ROOT  = MODEL_DIR / "01_PKL Files"
pkl_folder = PKL_ROOT / "03_Benth"
pkl_folder.mkdir(parents=True, exist_ok=True)

tdy = datetime.datetime.today().strftime('%Y-%m-%d')
start_time = datetime.datetime.now()

# ── configuration ──────────────────────────────────────────────────────────────
REGIONS = [11, 24, 27, 28, 32, 44, 52, 53]

# Phase 1: specification selection window
SEL_TRAIN_START = pd.Timestamp("1980-01-01")
SEL_TRAIN_END   = pd.Timestamp("2010-12-31")

# Phase 2: rolling OOS calibrations
CALIB_YEARS    = range(2010, 2024)
MIN_CALIB_OBS  = 1000

# AIC grid (SSW four-tuple)
I1_RANGE = range(0, 5)   # 0..4
J1_RANGE = range(0, 5)   # 0..4
I2_RANGE = range(1, 5)   # 1..4 (variance must have at least one harmonic)
J2_RANGE = range(1, 5)   # 1..4

MAX_LAG       = 60
SIGMA2_FLOOR  = 1e-3        # numerical safety for OLS-fitted sigma^2(t)
N_REFINE_ITER = 2           # iterations of (alpha, sigma^2) joint refinement

OMEGA = 2.0 * np.pi / 365.0


# ── helpers ────────────────────────────────────────────────────────────────────

def remove_feb29(df):
    return df[~((df.index.month == 2) & (df.index.day == 29))].copy()


def fourier_basis(t, K_sin, K_cos):
    """[sin(omega t), .., sin(K_sin omega t), cos(omega t), .., cos(K_cos omega t)]."""
    cols = []
    t = np.asarray(t, dtype=float)
    for k in range(1, K_sin + 1):
        cols.append(np.sin(k * OMEGA * t))
    for k in range(1, K_cos + 1):
        cols.append(np.cos(k * OMEGA * t))
    if not cols:
        return np.zeros((len(t), 0))
    return np.column_stack(cols)


def design_theta(t, I1, J1):
    """Design matrix for theta_t = b + c t + Fourier(I1, J1)."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([np.ones(len(t)), t, fourier_basis(t, I1, J1)])


def design_sigma2(t, I2, J2):
    """Design matrix for sigma^2_t = d + Fourier(I2, J2)."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([np.ones(len(t)), fourier_basis(t, I2, J2)])


def fit_theta(t, y, I1, J1):
    X = design_theta(t, I1, J1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, X @ beta


def eval_theta(t, theta_coefs, I1, J1):
    X = design_theta(t, I1, J1)
    return X @ theta_coefs


def fit_sigma2(doy_unique, v_d, I2, J2):
    X = design_sigma2(doy_unique, I2, J2)
    beta, *_ = np.linalg.lstsq(X, v_d, rcond=None)
    return beta


def eval_sigma2(t_or_doy, sigma2_coefs, I2, J2):
    X = design_sigma2(t_or_doy, I2, J2)
    s2 = X @ sigma2_coefs
    return np.maximum(s2, SIGMA2_FLOOR)


def alpha_ssw(eps, sigma2_t):
    """SSW eq. (6) Bibby-Sorensen martingale mean-reversion estimator.

    eps[0..N-1]      : residuals (T_t - theta_t)
    sigma2_t[0..N-1] : variance evaluated at the same times

    Returns continuous-time alpha = -log(weighted lag-1 slope).
    """
    e_prev = eps[:-1]
    e_curr = eps[1:]
    s2_prev = sigma2_t[:-1]
    valid = (np.isfinite(e_prev) & np.isfinite(e_curr) &
             np.isfinite(s2_prev) & (s2_prev > 0))
    if valid.sum() < 10:
        return np.nan
    num = np.sum(e_prev[valid] * e_curr[valid] / s2_prev[valid])
    den = np.sum(e_prev[valid] ** 2 / s2_prev[valid])
    if den <= 0:
        return np.nan
    ratio = num / den
    if ratio <= 0 or not np.isfinite(ratio):
        return np.nan
    return -float(np.log(ratio))


def fit_full_model(t, y, doy, I1, J1, I2, J2):
    """Joint fit of theta + sigma^2 + alpha for one (I1, J1, I2, J2) candidate.

    Returns a dict with fitted parameters and AIC, or None on failure.
    """
    if len(t) < 50:
        return None

    # 1) theta fit
    theta_coefs, theta = fit_theta(t, y, I1, J1)
    eps = y - theta

    # 2) initial sigma^2 (constant = sample variance)
    sigma2_t = np.full(len(t), float(np.var(eps)))
    sigma2_coefs = None
    alpha = np.nan

    # 3) iterative refinement of alpha and sigma^2(t)
    for it in range(N_REFINE_ITER):
        a_new = alpha_ssw(eps, sigma2_t)
        if not np.isfinite(a_new):
            return None
        alpha = a_new
        rho = float(np.exp(-alpha))

        # innovations r_t = eps_t - rho * eps_{t-1}
        r = np.full(len(eps), np.nan)
        r[1:] = eps[1:] - rho * eps[:-1]

        # per-DOY mean of r_t^2
        df_v = pd.DataFrame({'doy': doy, 'r2': r ** 2}).dropna()
        if len(df_v) < 100:
            return None
        v_d = df_v.groupby('doy')['r2'].mean().reset_index()
        if len(v_d) < 50:
            return None

        sigma2_coefs_new = fit_sigma2(v_d['doy'].values.astype(float),
                                      v_d['r2'].values, I2, J2)
        sigma2_t_new = eval_sigma2(np.asarray(doy, dtype=float),
                                   sigma2_coefs_new, I2, J2)

        # convergence check
        if it > 0:
            mean_s2 = float(np.mean(sigma2_t)) + 1e-9
            change = float(np.mean(np.abs(sigma2_t_new - sigma2_t))) / mean_s2
            if change < 1e-3:
                sigma2_t = sigma2_t_new
                sigma2_coefs = sigma2_coefs_new
                break
        sigma2_t = sigma2_t_new
        sigma2_coefs = sigma2_coefs_new

    # final innovations and Gaussian log-likelihood
    rho = float(np.exp(-alpha))
    r = np.full(len(eps), np.nan)
    r[1:] = eps[1:] - rho * eps[:-1]

    valid = np.isfinite(r) & np.isfinite(sigma2_t) & (sigma2_t > 0)
    if valid.sum() < 10:
        return None

    s2 = sigma2_t[valid]
    rr = r[valid]
    log_lik = float(-0.5 * np.sum(np.log(2 * np.pi * s2) + (rr ** 2) / s2))

    # parameter count: theta has 2 + I1 + J1, sigma^2 has 1 + I2 + J2, plus alpha
    k_params = (2 + I1 + J1) + (1 + I2 + J2) + 1
    aic = float(2 * k_params - 2 * log_lik)

    return {
        'I1': int(I1), 'J1': int(J1),
        'I2': int(I2), 'J2': int(J2),
        'theta_coefs': np.asarray(theta_coefs, dtype=float),
        'sigma2_coefs': np.asarray(sigma2_coefs, dtype=float),
        'alpha': float(alpha),
        'rho': rho,
        'log_lik': log_lik,
        'aic': aic,
        'k_params': int(k_params),
        'eps': eps,
        'theta': theta,
        'sigma2_t': sigma2_t,
        'r': r,
    }


def select_orders(t, y, doy):
    """Joint AIC search over (I1, J1, I2, J2)."""
    best = None
    best_aic = np.inf
    n_eval = 0
    for I1 in I1_RANGE:
        for J1 in J1_RANGE:
            if I1 == 0 and J1 == 0:
                continue
            for I2 in I2_RANGE:
                for J2 in J2_RANGE:
                    res = fit_full_model(t, y, doy, I1, J1, I2, J2)
                    if res is None:
                        continue
                    n_eval += 1
                    if res['aic'] < best_aic:
                        best_aic = res['aic']
                        best = res
    return best, n_eval


def compute_acf(x, max_lag=MAX_LAG):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.full(max_lag + 1, np.nan)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        out = np.zeros(max_lag + 1); out[0] = 1.0
        return out
    n = len(xc)
    out = [1.0]
    for k in range(1, max_lag + 1):
        if k >= n:
            out.append(np.nan)
        else:
            out.append(float(np.dot(xc[:n - k], xc[k:]) / denom))
    return np.array(out)


# ── main ───────────────────────────────────────────────────────────────────────

raw = pd.read_csv(TEMP_DIR / "region_temp_extended.csv", parse_dates=['date'])
required_cols = {'date', 'region_code', 'TAVG_imptd'}
missing_cols = required_cols.difference(raw.columns)
if missing_cols:
    raise ValueError(
        "Missing required columns in region_temp_extended.csv: "
        + ", ".join(sorted(missing_cols))
    )

selection_summary = []
rolling_summary = []
diag_store = []
diag_year = min(CALIB_YEARS)

print("=" * 72)
print("Benth — SSW (2012) spec, AIC-selected (I1, J1, I2, J2), Brownian noise")
print("=" * 72)

for region in REGIONS:
    print(f"\n[Region {region}]")

    sub = (raw[raw['region_code'] == region]
             .set_index('date')
             .rename(columns={'TAVG_imptd': 'TAVG'})
             [['TAVG']]
             .dropna()
             .sort_index())

    if sub.empty:
        print("  no data, skipping"); continue

    sub = remove_feb29(sub)
    sub = sub.loc[SEL_TRAIN_START:].copy()

    if sub.empty:
        print(f"  no data from {SEL_TRAIN_START.date()} forward, skipping"); continue

    t_origin = sub.index.min()
    sub['t'] = ((sub.index - t_origin).days + 1).astype(float)
    sub['doy'] = sub.index.dayofyear.values.astype(float)

    # ── Phase 1: AIC selection on selection window ──────────────────────────
    sel_df = sub.loc[SEL_TRAIN_START:SEL_TRAIN_END].copy()
    if len(sel_df) < MIN_CALIB_OBS:
        print(f"  Phase 1 skipped: only {len(sel_df)} obs in selection window"); continue

    t0 = datetime.datetime.now()
    print(f"  Phase 1: AIC search on {sel_df.index.min().date()} -> "
          f"{sel_df.index.max().date()}  (N={len(sel_df):,})")
    best, n_eval = select_orders(sel_df['t'].values,
                                 sel_df['TAVG'].values.astype(float),
                                 sel_df['doy'].values)
    elapsed = (datetime.datetime.now() - t0).total_seconds()

    if best is None:
        print(f"  Phase 1 failed for region {region}, skipping"); continue

    I1, J1, I2, J2 = best['I1'], best['J1'], best['I2'], best['J2']
    print(f"  -> selected (I1={I1}, J1={J1}, I2={I2}, J2={J2})  "
          f"AIC={best['aic']:.1f}  alpha={best['alpha']:.4f}  "
          f"({n_eval} candidates, {elapsed:.1f}s)")

    selection_summary.append({
        'region': region,
        'I1': I1, 'J1': J1, 'I2': I2, 'J2': J2,
        'sel_aic': float(best['aic']),
        'sel_log_lik': float(best['log_lik']),
        'sel_alpha': float(best['alpha']),
        'k_params': int(best['k_params']),
        'n_candidates': int(n_eval),
    })

    # ── Phase 2: rolling refit per calib_year ──────────────────────────────
    for calib_year in CALIB_YEARS:
        calib_end = pd.Timestamp(f'{calib_year}-12-31')
        calib_df = sub.loc[SEL_TRAIN_START:calib_end].copy()

        if len(calib_df) < MIN_CALIB_OBS:
            print(f"  calib={calib_year}: insufficient data, skipping"); continue

        res = fit_full_model(calib_df['t'].values,
                             calib_df['TAVG'].values.astype(float),
                             calib_df['doy'].values,
                             I1, J1, I2, J2)
        if res is None:
            print(f"  calib={calib_year}: refit failed, skipping"); continue

        eps = res['eps']
        r = res['r']
        sigma2_t = res['sigma2_t']

        valid = np.isfinite(r) & np.isfinite(sigma2_t) & (sigma2_t > 0)
        std_resid = np.full(len(r), np.nan)
        std_resid[valid] = r[valid] / np.sqrt(sigma2_t[valid])
        e_clean = std_resid[np.isfinite(std_resid)]

        if len(e_clean) >= 8 and np.isfinite(e_clean.std()) and e_clean.std() > 0:
            _, ks_p = ss.kstest(e_clean, lambda x: ss.norm.cdf(x, e_clean.mean(), e_clean.std()))
        else:
            ks_p = np.nan

        acf_X = compute_acf(eps, MAX_LAG)
        acf_e = compute_acf(e_clean, MAX_LAG)
        ci_bound = float(1.96 / np.sqrt(len(calib_df))) if len(calib_df) > 0 else np.nan

        # per-DOY sigma values for compatibility with downstream tools
        all_doy = np.arange(1, 366).astype(float)
        sigma2_by_doy = eval_sigma2(all_doy, res['sigma2_coefs'], I2, J2)
        sigma_dayofyear = {int(d): float(np.sqrt(s2))
                           for d, s2 in zip(all_doy, sigma2_by_doy)}

        export_model = {
            # specification
            'model_name': 'Benth Brownian (SSW 2012, AIC-selected)',
            'model_version': 'ssw_brownian_v1',
            'model_id': 'Benth Brownian',
            'I1': I1, 'J1': J1, 'I2': I2, 'J2': J2,

            # fitted parameters
            'theta_coefs':  res['theta_coefs'].astype(float),   # [b, c, A_1..A_I1, B_1..B_J1]
            'sigma2_coefs': res['sigma2_coefs'].astype(float),  # [d, c_1..c_I2, d_1..d_J2]
            'alpha': float(res['alpha']),
            'rho':   float(res['rho']),                        # exp(-alpha), discrete AR(1) coef

            # convenience for downstream simulators
            'sigma_dayofyear': sigma_dayofyear,                # {doy: sigma_t}
            'omega':           float(OMEGA),
            'temperature_source': 'region_temp_extended.csv:TAVG_imptd',
            't_origin':        t_origin,
            't_start':         float(calib_df['t'].iloc[0]),
            't_end':           float(calib_df['t'].iloc[-1]),
            't_start_date':    calib_df.index[0],
            't_end_date':      calib_df.index[-1],

            # data and diagnostics
            'df':         sub.copy(),
            'df_calib':   calib_df.copy(),
            'calib_end':  f'{calib_year}-12-31',
            'region_code': region,
            'e':          e_clean,
            'ks_p':       float(ks_p) if np.isfinite(ks_p) else np.nan,
            'acf_X':      acf_X,
            'acf_e':      acf_e,
            'ci_bound':   ci_bound,
            'log_lik':    float(res['log_lik']),
            'aic':        float(res['aic']),
            'k_params':   int(res['k_params']),
        }

        pkl_path = os.path.join(
            pkl_folder,
            f'ou_brownian_model_{region}_calib_{calib_year}.pkl'
        )
        with open(pkl_path, 'wb') as f:
            pickle.dump(export_model, f)

        rolling_summary.append({
            'region': region, 'calib_year': calib_year,
            'I1': I1, 'J1': J1, 'I2': I2, 'J2': J2,
            'alpha':   float(res['alpha']),
            'rho':     float(res['rho']),
            'log_lik': float(res['log_lik']),
            'aic':     float(res['aic']),
            'ks_p':    float(ks_p) if np.isfinite(ks_p) else np.nan,
            'pkl_path': pkl_path,
        })

        print(f"  calib={calib_year}: alpha={res['alpha']:.4f}  "
              f"AIC={res['aic']:.1f}  N={len(calib_df):,}  "
              f"-> {os.path.basename(pkl_path)}")

        if calib_year == diag_year:
            diag_store.append((region, sigma_dayofyear, e_clean,
                               ks_p, acf_X, acf_e, ci_bound))

# ── write summaries ────────────────────────────────────────────────────────────
sel_df_out  = pd.DataFrame(selection_summary)
roll_df_out = pd.DataFrame(rolling_summary)

sel_csv  = MODEL_DIR / f'benth_selection_summary_{tdy}.csv'
roll_csv = MODEL_DIR / f'benth_rolling_summary_{tdy}.csv'
sel_df_out.to_csv(sel_csv, index=False)
roll_df_out.to_csv(roll_csv, index=False)

print(f"\n[summary] selection -> {sel_csv}")
print(f"[summary] rolling   -> {roll_csv}")
print(f"[summary] PKLs in   -> {pkl_folder}")
print(f"[summary] elapsed   -> {datetime.datetime.now() - start_time}")

# ── lightweight diagnostics figure ─────────────────────────────────────────────
if len(diag_store) > 0:
    n = len(diag_store)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 10))
    if n == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(f'Benth (SSW 2012, AIC-selected) calibration overview ({diag_year})',
                 fontsize=12, fontweight='bold', y=1.005)
    lags = np.arange(MAX_LAG + 1)

    for col, (region, sigma_dayofyear, e_clean,
              ks_p, acf_X, acf_e, ci_bound) in enumerate(diag_store):
        ax = axes[0, col]
        doys = np.array(sorted(sigma_dayofyear.keys()))
        sig_vals = np.array([sigma_dayofyear[d] for d in doys])
        ax.plot(doys, sig_vals)
        ax.set_title(f'R{region}', fontsize=9, fontweight='bold')
        if col == 0:
            ax.set_ylabel('sigma(t)', fontsize=8)

        ax = axes[1, col]
        if len(e_clean) > 0:
            ax.hist(e_clean, bins=60, density=True, alpha=0.4)
            x_grid = np.linspace(np.percentile(e_clean, 0.5),
                                 np.percentile(e_clean, 99.5), 300)
            if np.isfinite(np.std(e_clean)) and np.std(e_clean) > 0:
                ax.plot(x_grid, ss.norm.pdf(x_grid, np.mean(e_clean),
                                            np.std(e_clean)), lw=1.5)
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
            ax.axhline( ci_bound, ls='--', lw=0.8)
            ax.axhline(-ci_bound, ls='--', lw=0.8)
        ax.axhline(0, lw=0.4, color='k')
        ax.set_xlabel('Lag', fontsize=7)
        if col == 0:
            ax.set_ylabel('ACF(std resid)', fontsize=8)

    fig.tight_layout()
    fig.savefig(MODEL_DIR / 'benth_model_diagnostics.png',
                bbox_inches='tight', dpi=150)
    plt.close(fig)
