"""
Burn Analysis (HBA) rolling PKL exporter.

Based on:
    Schiller, Seidler & Wimmer (2012)
    "Temperature Models for Pricing Weather Derivatives"

Updated behavior:
- exports one PKL per region per calibration year, matching the rolling-export
  pattern used for the Alaton and Benth models
- output files:
    01_PKL Files/01_HBA/hba_model_{region}_calib_{year}.pkl

Expected input file:
    EDA/region_avg.csv
with columns:
    date, region_code, daily_avg_temperature
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = ROOT / "02_Data" / "01_Temprature"
MODEL_DIR = Path(__file__).resolve().parent
PKL_ROOT = MODEL_DIR / "01_PKL Files"

REGIONS      = [11, 24, 27, 28, 32, 44, 52, 53]
CALIB_YEARS  = range(2010, 2024)
DIAG_YEAR    = 2010
PKL_FOLDER   = PKL_ROOT / "01_HBA"
PKL_FOLDER.mkdir(parents=True, exist_ok=True)

T_REF = 18.0

CONTRACTS = [
    ("Summer",    "CDD",  5,  1,  9, 30),
    ("May",       "CDD",  5,  1,  5, 31),
    ("June",      "CDD",  6,  1,  6, 30),
    ("July",      "CDD",  7,  1,  7, 31),
    ("August",    "CDD",  8,  1,  8, 31),
    ("September", "CDD",  9,  1,  9, 30),
    ("Winter",    "HDD", 11,  1,  3, 31),
    ("November",  "HDD", 11,  1, 11, 30),
    ("December",  "HDD", 12,  1, 12, 31),
    ("January",   "HDD",  1,  1,  1, 31),
    ("February",  "HDD",  2,  1,  2, 28),
    ("March",     "HDD",  3,  1,  3, 31),
]

BLUE, RED, GRAY = "#2563EB", "#DC2626", "#6B7280"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

raw = pd.read_csv(TEMP_DIR / "region_avg.csv", parse_dates=["date"])

def compute_index(temps, index_type):
    if index_type == "HDD":
        return float(np.maximum(T_REF - temps, 0).sum())
    return float(np.maximum(temps - T_REF, 0).sum())

def get_period_temps(df, contract, year):
    _, idx_type, sm, sd, em, ed = contract
    start = pd.Timestamp(year=year, month=sm, day=sd)
    end_year = year + 1 if sm > em else year
    end = pd.Timestamp(year=end_year, month=em, day=ed)
    return df.loc[(df.index >= start) & (df.index <= end), "T"]

def fit_hba(Y):
    n = len(Y)
    i_vec = np.arange(1, n + 1, dtype=float)
    Y_bar = Y.mean()
    Y_pred = Y_bar
    resid = Y - Y_bar
    s2 = np.sum(resid**2) / (n - 2) if n > 2 else np.nan

    if n >= 5 and not np.isnan(s2):
        pred_var = (n + 2) * (n + 1) * (n - 2) / (n * (n - 1) * (n - 4)) * s2
    else:
        pred_var = np.nan

    pred_std = np.sqrt(pred_var) if not np.isnan(pred_var) else np.nan
    pred_lo95 = Y_pred - 1.96 * pred_std if not np.isnan(pred_std) else np.nan
    pred_hi95 = Y_pred + 1.96 * pred_std if not np.isnan(pred_std) else np.nan

    return {
        "n": n,
        "Y_bar": Y_bar,
        "beta0": Y_bar,
        "beta1": 0.0,
        "Y_pred": Y_pred,
        "Y_hat": np.full(n, Y_bar),
        "i_vec": i_vec,
        "s2": s2,
        "s": np.sqrt(s2) if not np.isnan(s2) else np.nan,
        "pred_var": pred_var,
        "pred_std": pred_std,
        "pred_lo95": pred_lo95,
        "pred_hi95": pred_hi95,
        "Y_hist": Y,
        "resid": resid,
    }

def predict_index(df, contract, valuation_year, n_history=20):
    _, idx_type, sm, sd, em, ed = contract
    years_used, index_values = [], []
    for yr in range(valuation_year - n_history - 5, valuation_year):
        series = get_period_temps(df, contract, yr)
        if len(series) == 0:
            continue
        start = pd.Timestamp(year=yr, month=sm, day=sd)
        end_yr = yr + 1 if sm > em else yr
        end = pd.Timestamp(year=end_yr, month=em, day=ed)
        if len(series) < 0.85 * ((end - start).days + 1):
            continue
        years_used.append(yr)
        index_values.append(compute_index(series, idx_type))
    if len(years_used) < 5:
        return None
    years_used = years_used[-n_history:]
    index_values = index_values[-n_history:]
    fit = fit_hba(np.array(index_values))
    fit["years"] = np.array(years_used)
    return fit

summary_rows = []
diag_data = []

for region in REGIONS:
    print(f"\n{'='*60}\n  Region {region}\n{'='*60}")

    df_raw = (raw[raw["region_code"] == region]
                .set_index("date")
                .rename(columns={"daily_avg_temperature": "T"})
                [["T"]]
                .dropna()
                .sort_index())

    print(f"Full: N={len(df_raw):,}")

    if len(df_raw) < 1000:
        print("  WARNING: insufficient full-series data, skipping region.")
        continue

    for calib_year in CALIB_YEARS:
        calib_end = f"{calib_year}-12-31"
        df_calib = df_raw.loc[:calib_end].copy()

        if len(df_calib) < 1000:
            print(f"  {calib_year}: insufficient data, skipping.")
            continue

        all_years = np.arange(df_calib.index.year.min(), df_calib.index.year.max() + 1)
        contract_results = {}

        for contract in CONTRACTS:
            name, idx_type, sm, sd, em, ed = contract
            years_used, index_values = [], []

            for yr in all_years:
                series = get_period_temps(df_calib, contract, yr)
                if len(series) == 0:
                    continue
                start = pd.Timestamp(year=yr, month=sm, day=sd)
                end_yr = yr + 1 if sm > em else yr
                end = pd.Timestamp(year=end_yr, month=em, day=ed)
                if len(series) < 0.85 * ((end - start).days + 1):
                    continue
                years_used.append(yr)
                index_values.append(compute_index(series, idx_type))

            if len(years_used) < 5:
                continue

            fit = fit_hba(np.array(index_values))
            fit["years"] = np.array(years_used)
            contract_results[name] = {"contract": contract, "fit": fit}

        model = {
            "model_name": "HBA (Schiller 2012)",
            "region_code": region,
            "T_ref": T_REF,
            "df": df_raw,
            "df_calib": df_calib,
            "contracts": CONTRACTS,
            "contract_results": contract_results,
            "calib_end": calib_end,
            "calib_year": calib_year,
        }

        pkl_path = os.path.join(PKL_FOLDER, f"hba_model_{region}_calib_{calib_year}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)

        winter = contract_results.get("Winter", {}).get("fit", {})
        summer = contract_results.get("Summer", {}).get("fit", {})

        summary_rows.append({
            "region": region,
            "calib_year": calib_year,
            "winter_Y_bar": round(winter.get("Y_bar", np.nan), 1),
            "winter_s": round(winter.get("s", np.nan), 2),
            "summer_Y_bar": round(summer.get("Y_bar", np.nan), 1),
            "summer_s": round(summer.get("s", np.nan), 2),
            "n_contracts": len(contract_results),
            "pkl_path": pkl_path,
        })

        print(
            f"  {calib_year}: n_contracts={len(contract_results)}  "
            f"winter_mu={winter.get('Y_bar', np.nan):.1f}  "
            f"summer_mu={summer.get('Y_bar', np.nan):.1f}  "
            f"-> {pkl_path}"
        )

        if calib_year == DIAG_YEAR:
            diag_data.append({
                "region": region,
                "contract_results": contract_results,
            })

summary = pd.DataFrame(summary_rows).set_index(["region", "calib_year"])
summary.to_csv(MODEL_DIR / "hba_model_summary.csv")
print(f"\nSaved -> {MODEL_DIR / 'hba_model_summary.csv'}")
print(summary.to_string())

N = len(diag_data)
if N > 0:
    fig, axes = plt.subplots(2, N, figsize=(3.0 * N, 8))
    if N == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"Burn Analysis (HBA) — Historical Index & Mean Forecast Across All Regions ({DIAG_YEAR})",
        fontsize=12, fontweight="bold", y=1.005
    )

    ROW_CONTRACTS = ["Winter", "Summer"]

    for col, d in enumerate(diag_data):
        region = d["region"]
        for row, cname in enumerate(ROW_CONTRACTS):
            ax = axes[row, col]
            res = d["contract_results"].get(cname, {}).get("fit")
            if res is None:
                ax.set_visible(False)
                continue
            yr = res["years"]
            Y = res["Y_hist"]
            ax.bar(yr, Y, color=GRAY, alpha=0.5, width=0.7)
            ax.axhline(res["Y_bar"], color=RED, lw=1.5, ls="--")
            ax.axhline(res["pred_lo95"], color=BLUE, lw=0.8, ls=":")
            ax.axhline(res["pred_hi95"], color=BLUE, lw=0.8, ls=":")
            ax.set_title(f"R{region}", fontweight="bold", fontsize=9)
            ax.tick_params(labelsize=6)
            ax.text(
                0.97, 0.97,
                f"μ={res['Y_bar']:.0f}\ns={res['s']:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
            )
            if col == 0:
                itype = d["contract_results"][cname]["contract"][1]
                ax.set_ylabel(f"{cname}\n({itype} °C·days)", fontsize=8)

    fig.tight_layout()
    fig.savefig(MODEL_DIR / "hba_model_diagnostics.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved -> {MODEL_DIR / 'hba_model_diagnostics.png'}")

print(f"\nDone. Wrote rolling HBA PKLs to: {PKL_FOLDER}")
