"""
Replication of Barnor, Kampouridis & Kanellopoulos (2026)
"Machine learning for temperature forecasting in weather derivatives"

Updated to load and use rolling PKL exports for:
  - HBA
  - Alaton
  - Benth
  - XGB
  - WaveletFNN
  - LSTM
  - FeedForwardNN

Important note on HBA:
- The exported HBA PKLs are contract-index models, not native daily-temperature
  forecasters.
- To keep this script comparable to the existing daily-temperature evaluation,
  the HBA forecast here is generated from the HBA PKL's stored calibration data
  (`df_calib`) as a day-of-year climatology baseline.
"""

import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from darts import TimeSeries
from scipy.stats import friedmanchisquare, wilcoxon as scipy_wilcoxon
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import pickle, warnings, os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ===========================================================================
# CONFIGURATION
# ===========================================================================

REGIONS      = [11, 24, 27, 28, 32, 44, 52, 53]
T_REF        = 18.0
OMEGA        = 2 * np.pi / 365
RAW_CSV      = "EDA/region_avg.csv"
EXTENDED_CSV = "EDA/region_temp_extended.csv"
HBA_DIR      = "hba_models"
ALATON_DIR   = "alaton_models"
BENTH_DIR    = "benth_models"
PAPER_XGB_DIR    = Path(r"/Users/marielouiselysholt/Desktop/Master/Masters_2026/Outputs/xgBoost_rolling_exports")
WAVELET_FNN_DIR  = Path(r"/Users/marielouiselysholt/Desktop/Master/Masters_2026/Outputs/WaveletFNN_rolling_exports")
LSTM_DIR         = Path(r"/Users/marielouiselysholt/Desktop/Master/Masters_2026/Outputs/lstm_rolling_exports")
FEEDFORWARD_DIR  = Path(r"/Users/marielouiselysholt/Desktop/Master/Masters_2026/Outputs/FeedForwardNN_rolling_exports")

OOS_START    = 2015
OOS_END      = 2024
TEST_YEARS   = list(range(OOS_START, OOS_END + 1))
MIN_INDEX    = 15.0
N_BOOTSTRAP  = 1_000
N_HARMONICS  = 3

MODEL_ORDER  = ["HBA", "Alaton", "Benth", "XGB", "WaveletFNN", "LSTM", "FeedForwardNN"]
LABELS       = {
    "HBA":           "Burn Analysis (HBA)",
    "Alaton":        "Alaton (2002)",
    "Benth":         "Benth (2007)",
    "XGB":           "XGBoost (Barnor et al. 2026)",
    "WaveletFNN":    "Wavelet FNN",
    "LSTM":          "LSTM",
    "FeedForwardNN": "Feed Forward NN",
}

HORIZONS = {
    "30":(0,30), "60":(0,60), "90":(0,90), "120":(0,120), "180":(0,180),
    "270":(0,270), "365":(0,365), "ISL":(90,180), "IISL":(180,270),
    "mid90":(135,225), "-270":(95,365), "-180":(185,365), "-120":(245,365),
    "-90":(275,365), "-60":(305,365), "-30":(335,365),
}
HORIZON_NAMES = list(HORIZONS.keys())

FEATURE_COLS = [
    "dayofyear", "pdtn_doy", "de_trend_seas",
    "dts_doyavge", "dts_doyvar",
    "yday", "IIdays_ago", "IIIdays_ago", "IVdays_ago",
    "Vdays_ago", "VIdays_ago", "VIIdays_ago",
    "last_year", "last_2year", "last_3year", "last_4year", "last_5year",
    "h1_last_year", "h1_last_2years", "h1_last_3years",
    "h1_last_4years", "h1_last_5years",
    "h1_ly_2days", "h1_ly_3day", "h1_ly_4day",
    "h1_ly_5day", "h1_ly_6day", "h1_ly_7day",
    "h1_ly_next_1day", "h1_ly_next_2days", "h1_ly_next_3days",
    "h1_ly_next_4days", "h1_ly_next_5days",
    "h1_ly_next_6days", "h1_ly_next_7days",
    "diff_1year", "diff_2year", "diff_3year", "diff_4year", "diff_5year",
    "diff_yday", "diff_2days", "diff_3days", "diff_4days",
    "diff_5days", "diff_6days", "diff_7days",
    "last_7_1day_deltas_mean", "last_7_1day_deltas_min",
    "last_7_1day_deltas_max",
]

# ===========================================================================
# WAVELET FEATURE TRANSFORMER
# ===========================================================================

class WaveletFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_wavelets=3, wavelet="mexican_hat", include_original=True):
        self.n_wavelets = n_wavelets
        self.wavelet = wavelet
        self.include_original = include_original

    def fit(self, X, y=None):
        X = self._to_numpy(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._to_numpy(X)
        n_samples, n_features = X.shape
        t = np.linspace(-1.0, 1.0, n_samples)
        features = []
        if self.include_original:
            features.append(X)
        for j in range(n_features):
            x = X[:, j]
            for i in range(self.n_wavelets):
                if self.n_wavelets == 1:
                    shift = 0.0
                else:
                    shift = -0.75 + i * (1.5 / (self.n_wavelets - 1))
                scale = 0.25 + 0.20 * i
                tau = (t - shift) / scale
                if self.wavelet == "mexican_hat":
                    psi = (1.0 - tau**2) * np.exp(-0.5 * tau**2)
                elif self.wavelet == "morlet":
                    psi = np.cos(5.0 * tau) * np.exp(-0.5 * tau**2)
                else:
                    raise ValueError(f"Unknown wavelet type: {self.wavelet}")
                features.append((x * psi).reshape(-1, 1))
        X_out = np.hstack(features).astype(float)
        if not np.isfinite(X_out).all():
            raise ValueError("WaveletFeatureTransformer produced NaN/inf.")
        return X_out

    @staticmethod
    def _to_numpy(X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.asarray(X, dtype=float)

# ===========================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ===========================================================================

def _doy_noleap(date):
    doy = date.timetuple().tm_yday
    if (date.year % 4 == 0 and (date.year % 100 != 0 or date.year % 400 == 0)):
        if date.month > 2:
            doy -= 1
    return max(1, min(365, doy))

def load_region(region):
    raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
    raw = raw.rename(columns={"daily_avg_temperature": "T"})
    df = (raw[raw["region_code"] == region][["date", "T"]]
          .dropna(subset=["date"]).sort_values("date").reset_index(drop=True))
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))].copy()
    df = df.reset_index(drop=True)
    df["doy"] = df["date"].apply(_doy_noleap)
    df["t_idx"] = np.arange(len(df), dtype=float)
    if df["T"].isna().any():
        doy_means = df.dropna(subset=["T"]).groupby("doy")["T"].mean().to_dict()
        for pos in df[df["T"].isna()].index.tolist():
            doy_val = int(df.at[pos, "doy"])
            T_avy = doy_means.get(doy_val, float(df["T"].mean()))
            nbrs = []
            for off in list(range(-7, 0)) + list(range(1, 8)):
                p = pos + off
                if 0 <= p < len(df) and not pd.isna(df.at[p, "T"]):
                    nbrs.append(df.at[p, "T"])
            T_avd = float(np.mean(nbrs)) if nbrs else T_avy
            df.at[pos, "T"] = (T_avy + T_avd) / 2.0
    return df

def get_train_test_for_year(df, test_year):
    df_train = df[df["date"].dt.year < test_year].copy().reset_index(drop=True)
    df_test = df[df["date"].dt.year == test_year].copy().reset_index(drop=True)
    return df_train, df_test

# ===========================================================================
# SECTION 2: DETRENDING / DESEASONALIZING
# ===========================================================================

def _detrend_linear(df_train):
    t = df_train["t_idx"].values
    T = df_train["T"].values
    mask = ~np.isnan(T)
    A = np.column_stack([np.ones(mask.sum()), t[mask]])
    coefs, _, _, _ = np.linalg.lstsq(A, T[mask], rcond=None)
    intercept, slope = coefs[0], coefs[1]
    detrended = T - (intercept + slope * t)
    return intercept, slope, detrended

def _deseasonalize_fourier(doy, detrended, n_harmonics=N_HARMONICS):
    mask = ~np.isnan(detrended)
    cols = [np.ones(mask.sum())]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * doy[mask] / 365.0))
        cols.append(np.cos(2 * np.pi * k * doy[mask] / 365.0))
    X = np.column_stack(cols)
    coefs, _, _, _ = np.linalg.lstsq(X, detrended[mask], rcond=None)
    cols_all = [np.ones(len(doy))]
    for k in range(1, n_harmonics + 1):
        cols_all.append(np.sin(2 * np.pi * k * doy / 365.0))
        cols_all.append(np.cos(2 * np.pi * k * doy / 365.0))
    seasonal = np.column_stack(cols_all) @ coefs
    return coefs, detrended - seasonal

def _build_feature_df(df):
    df = df.copy().sort_values("date").set_index("date")
    dts = df["de_trend_seas"]
    df["pdtn_doy"] = df["dayofyear"].shift(-1)
    last_doy = df["dayofyear"].iloc[-1]
    df.iloc[-1, df.columns.get_loc("pdtn_doy")] = 1 if last_doy == 365 else last_doy + 1
    df["pdtn_doy"] = df["pdtn_doy"].astype(int)
    doy_stats = df.groupby("dayofyear")["de_trend_seas"].agg(["mean", "var"])
    df["dts_doyavge"] = df["dayofyear"].map(doy_stats["mean"])
    df["dts_doyvar"] = df["dayofyear"].map(doy_stats["var"])
    df["yday"] = dts.shift(1)
    for name, lag in [("IIdays_ago",2),("IIIdays_ago",3),("IVdays_ago",4),
                      ("Vdays_ago",5),("VIdays_ago",6),("VIIdays_ago",7)]:
        df[name] = dts.shift(lag)
    for name, lag in [("last_year",365),("last_2year",730),("last_3year",1095),
                      ("last_4year",1460),("last_5year",1825)]:
        df[name] = dts.shift(lag)
    for name, lag in [("h1_last_year",364),("h1_last_2years",729),("h1_last_3years",1094),
                      ("h1_last_4years",1459),("h1_last_5years",1824)]:
        df[name] = dts.shift(lag)
    for name, lag in [("h1_ly_2days",366),("h1_ly_3day",367),("h1_ly_4day",368),
                      ("h1_ly_5day",369),("h1_ly_6day",370),("h1_ly_7day",371)]:
        df[name] = dts.shift(lag)
    for name, lag in [("h1_ly_next_1day",363),("h1_ly_next_2days",362),("h1_ly_next_3days",361),
                      ("h1_ly_next_4days",360),("h1_ly_next_5days",359),
                      ("h1_ly_next_6days",358),("h1_ly_next_7days",357)]:
        df[name] = dts.shift(lag)
    for name, lag in [("diff_1year",365),("diff_2year",730),("diff_3year",1095),
                      ("diff_4year",1460),("diff_5year",1825)]:
        df[name] = dts - dts.shift(lag)
    for name, lag in [("diff_yday",1),("diff_2days",2),("diff_3days",3),("diff_4days",4),
                      ("diff_5days",5),("diff_6days",6),("diff_7days",7)]:
        df[name] = dts - dts.shift(lag)
    diff_1d = dts - dts.shift(1)
    df["last_7_1day_deltas_mean"] = diff_1d.rolling(7).mean()
    df["last_7_1day_deltas_min"] = diff_1d.rolling(7).min()
    df["last_7_1day_deltas_max"] = diff_1d.rolling(7).max()
    return df

def compute_calib_context(df_train):
    df = df_train.copy()
    if "dayofyear" not in df.columns and "doy" in df.columns:
        df["dayofyear"] = df["doy"]
    trend_int, trend_slope, detrended = _detrend_linear(df)
    fourier_coefs, dts_residual = _deseasonalize_fourier(df["dayofyear"].values, detrended, N_HARMONICS)
    df["de_trend_seas"] = dts_residual
    feat_df = _build_feature_df(df)
    working_df = feat_df.sort_index(ascending=False).head(365 * 5)
    return {
        "trend_intercept": float(trend_int),
        "trend_slope": float(trend_slope),
        "fourier_coefs": fourier_coefs,
        "n_harmonics": N_HARMONICS,
        "n_calib_days": len(df),
        "working_df": working_df,
    }

def compute_wavelet_calib_context(region, calib_year, df_train_raw):
    ext = pd.read_csv(EXTENDED_CSV)
    ext["date"] = pd.to_datetime(ext["date"])
    region_col = None
    for c in ["region_code", "name"]:
        if c in ext.columns:
            region_col = c
            break
    if region_col is None:
        raise ValueError("Cannot find region column in region_temp_extended.csv")
    ext = ext[ext[region_col] == region].copy()
    ext = ext[~((ext["date"].dt.month == 2) & (ext["date"].dt.day == 29))].copy()
    ext = ext.sort_values("date").reset_index(drop=True)
    calib_end = pd.Timestamp(f"{calib_year}-12-31")
    ext_calib = ext[ext["date"] <= calib_end].copy()
    if len(ext_calib) < 365 * 6:
        return None
    if "dayofyear" not in ext_calib.columns:
        ext_calib["dayofyear"] = ext_calib["date"].apply(_doy_noleap)
    ext_calib = ext_calib.set_index("date")
    working_df = ext_calib.sort_index(ascending=False).head(365 * 5)
    if "diff_yday" not in working_df.columns:
        dts_sorted = ext_calib.sort_index()["de_trend_seas"]
        ext_calib["diff_yday"] = dts_sorted - dts_sorted.shift(1)
        working_df = ext_calib.sort_index(ascending=False).head(365 * 5)

    raw_train = df_train_raw.copy()
    raw_train = raw_train[~((raw_train["date"].dt.month == 2) & (raw_train["date"].dt.day == 29))].copy()
    raw_train = raw_train.sort_values("date").reset_index(drop=True)

    ext_flat = ext[ext["date"] <= calib_end][["date", "de_trend_seas"]].copy()
    ext_flat = ext_flat.sort_values("date").reset_index(drop=True)
    merged = pd.merge(raw_train[["date", "T"]], ext_flat, on="date", how="inner")
    if len(merged) < 365:
        return None

    merged["t_idx"] = np.arange(len(merged), dtype=float)
    merged["doy"] = merged["date"].apply(_doy_noleap)
    merged["trend_seas"] = merged["T"] - merged["de_trend_seas"]

    t = merged["t_idx"].values
    doy = merged["doy"].values
    ts = merged["trend_seas"].values
    mask = ~np.isnan(ts)

    A_lin = np.column_stack([np.ones(mask.sum()), t[mask]])
    lin_coefs, _, _, _ = np.linalg.lstsq(A_lin, ts[mask], rcond=None)
    trend_int, trend_slope = lin_coefs[0], lin_coefs[1]
    detrended_ts = ts - (trend_int + trend_slope * t)

    cols = [np.ones(mask.sum())]
    for k in range(1, N_HARMONICS + 1):
        cols.append(np.sin(2 * np.pi * k * doy[mask] / 365.0))
        cols.append(np.cos(2 * np.pi * k * doy[mask] / 365.0))
    X_four = np.column_stack(cols)
    fourier_coefs, _, _, _ = np.linalg.lstsq(X_four, detrended_ts[mask], rcond=None)

    return {
        "trend_intercept": float(trend_int),
        "trend_slope": float(trend_slope),
        "fourier_coefs": fourier_coefs,
        "n_harmonics": N_HARMONICS,
        "n_calib_days": len(merged),
        "working_df": working_df,
    }

# ===========================================================================
# SECTION 3: PKL MODEL LOADERS AND FORECASTERS
# ===========================================================================

def load_hba(region, calib_year):
    with open(os.path.join(HBA_DIR, f"hba_model_{region}_calib_{calib_year}.pkl"), "rb") as f:
        return pickle.load(f)

def load_alaton(region, calib_year):
    with open(os.path.join(ALATON_DIR, f"alaton_model_{region}_calib_{calib_year}.pkl"), "rb") as f:
        return pickle.load(f)

def load_benth(region, calib_year):
    with open(os.path.join(BENTH_DIR, f"ou_levy_model_{region}_calib_{calib_year}.pkl"), "rb") as f:
        return pickle.load(f)

def load_paper_xgb(region, calib_year):
    pkl_path = PAPER_XGB_DIR / f"paper_xgb_region_{region}_calib_{calib_year}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_wavelet_fnn(region, calib_year):
    prediction_year = calib_year + 1
    pkl_path = WAVELET_FNN_DIR / f"wavelet_fnn_region_{region}_calib_{calib_year}_predict_{prediction_year}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_lstm(region, calib_year):
    candidates = [
        LSTM_DIR / f"lstm_region_{region}_calib_{calib_year}.pkl",
        LSTM_DIR / f"lstm_model_region_{region}_calib_{calib_year}.pkl",
        LSTM_DIR / f"lstm_region_{region}_calib_{calib_year}_predict_{calib_year+1}.pkl",
    ]

    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                payload = pickle.load(f)
            return payload

    raise FileNotFoundError(f"No LSTM PKL found for region={region}, calib_year={calib_year}")

def load_feedforward_nn(region, calib_year):
    pred_year = calib_year + 1
    candidates = [
        FEEDFORWARD_DIR / f"paper_ffnn_region_{region}_calib_{calib_year}_predict_{pred_year}.pkl",
        FEEDFORWARD_DIR / f"feedforward_nn_region_{region}_calib_{calib_year}.pkl",
        FEEDFORWARD_DIR / f"FeedForwardNN_region_{region}_calib_{calib_year}.pkl",
        FEEDFORWARD_DIR / f"feedforward_region_{region}_calib_{calib_year}.pkl",
        FEEDFORWARD_DIR / f"ffnn_region_{region}_calib_{calib_year}.pkl",
        FEEDFORWARD_DIR / f"feedforward_nn_region_{region}_calib_{calib_year}_predict_{pred_year}.pkl",
        FEEDFORWARD_DIR / f"ffnn_region_{region}_calib_{calib_year}_predict_{pred_year}.pkl",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                payload = pickle.load(f)
            return payload
    raise FileNotFoundError(
        f"No FeedForwardNN PKL found for region={region}, calib_year={calib_year}"
    )

def _extract_model_from_payload(payload):
    for key in ["fitted_model", "model", "pipeline", "estimator", "net"]:
        if isinstance(payload, dict) and key in payload:
            return payload[key]
    return payload

def _extract_feature_cols(payload, default_cols=None):
    if isinstance(payload, dict):
        for key in ["feature_cols", "features", "input_features", "covariate_cols", "past_covariate_cols"]:
            if key in payload:
                return list(payload[key])
    return default_cols

def _extract_working_df(payload, calib_ctx):
    if isinstance(payload, dict) and "working_df" in payload:
        return payload["working_df"]
    return calib_ctx["working_df"]

def _extract_trend_ctx(payload, calib_ctx):
    if isinstance(payload, dict):
        return (
            payload.get("trend_intercept", calib_ctx["trend_intercept"]),
            payload.get("trend_slope", calib_ctx["trend_slope"]),
            payload.get("fourier_coefs", calib_ctx["fourier_coefs"]),
            payload.get("n_harmonics", calib_ctx["n_harmonics"]),
            payload.get("n_calib_days", calib_ctx["n_calib_days"]),
        )
    return (
        calib_ctx["trend_intercept"],
        calib_ctx["trend_slope"],
        calib_ctx["fourier_coefs"],
        calib_ctx["n_harmonics"],
        calib_ctx["n_calib_days"],
    )

def _predict_from_model_obj(model_obj, X):
    if hasattr(model_obj, "predict"):
        out = model_obj.predict(X)
        arr = np.asarray(out)
        return float(arr.reshape(-1)[0])
    raise AttributeError("Loaded model object does not implement predict().")

def _t_idx_for_dates(dates, t_origin):
    t_vals = []
    for d in dates:
        dr = pd.date_range(t_origin, d, freq="D")
        mask = ~((dr.month == 2) & (dr.day == 29))
        t_vals.append(int(mask.sum()) - 1)
    return np.array(t_vals, dtype=float)

def alaton_seasonal_mean(t_idx, A, B, C, phi):
    return A + B * t_idx + C * np.sin(OMEGA * t_idx + phi)

def benth_seasonal_mean(doy, a0, a1, t0):
    return a0 + a1 * np.cos((2 * np.pi / 365.0) * (doy - t0))

def forecast_hba(hba_params, test_dates):
    df_calib = hba_params.get("df_calib", hba_params.get("df"))
    if df_calib is None:
        return np.full(len(test_dates), np.nan)

    dfc = df_calib.copy()
    if isinstance(dfc.index, pd.DatetimeIndex):
        idx_dates = pd.DatetimeIndex(dfc.index)
    elif "date" in dfc.columns:
        idx_dates = pd.to_datetime(dfc["date"])
    else:
        return np.full(len(test_dates), np.nan)

    temp_col = None
    for c in ["T", "TAVG_imptd", "daily_avg_temperature"]:
        if c in dfc.columns:
            temp_col = c
            break
    if temp_col is None:
        return np.full(len(test_dates), np.nan)

    doy = np.array([_doy_noleap(d) for d in idx_dates])
    vals = pd.Series(dfc[temp_col].values, index=doy).groupby(level=0).mean().to_dict()
    global_mean = float(np.nanmean(dfc[temp_col].values))
    return np.array([vals.get(_doy_noleap(d), global_mean) for d in test_dates], dtype=float)

def forecast_alaton(alaton_params, df_train, test_dates):
    A, B, C, phi = [float(alaton_params[k]) for k in ("A", "B", "C", "phi")]
    a = float(alaton_params["a"])
    exp_a = np.exp(-a)
    df_calib = alaton_params["df_calib"].copy()
    T_prev = float(df_train.iloc[-1]["T"])

    if "t" in df_calib.columns:
        t_last = float(df_calib["t"].iloc[-1])
    elif "t_idx" in df_calib.columns:
        t_last = float(df_calib["t_idx"].iloc[-1])
    else:
        raise KeyError("Alaton PKL must contain df_calib['t'] or df_calib['t_idx'].")

    forecasts = np.empty(len(test_dates), dtype=float)
    for i in range(len(test_dates)):
        t_prev_step = t_last + i
        t_curr_step = t_last + i + 1
        T_m_prev = alaton_seasonal_mean(np.array([t_prev_step]), A, B, C, phi)[0]
        T_m_curr = alaton_seasonal_mean(np.array([t_curr_step]), A, B, C, phi)[0]
        T_curr = T_m_curr + exp_a * (T_prev - T_m_prev)
        forecasts[i] = T_curr
        T_prev = T_curr
    return forecasts
def forecast_benth(benth_params, df_train, test_dates):
    if all(k in benth_params for k in ["a0", "a1", "t0", "slope", "intercept"]):
        a0 = float(benth_params["a0"])
        a1 = float(benth_params["a1"])
        t0 = float(benth_params["t0"])
        slope = float(benth_params["slope"])
        intercept = float(benth_params["intercept"])
    else:
        a0 = float(benth_params["a"])
        a1 = float(benth_params["b1"])
        t0 = float(benth_params["g1"])
        alpha = float(benth_params["alpha"])
        slope = np.exp(-alpha) if np.isfinite(alpha) else np.nan
        intercept = 0.0

    T_prev = float(df_train.iloc[-1]["T"])
    doy_prev = int(df_train.iloc[-1]["doy"])
    seas_prev = benth_seasonal_mean(doy_prev, a0, a1, t0)
    de_prev = T_prev - seas_prev

    forecasts = np.empty(len(test_dates), dtype=float)
    for i, d in enumerate(test_dates):
        doy_curr = _doy_noleap(d)
        seas_curr = benth_seasonal_mean(doy_curr, a0, a1, t0)
        de_curr = slope * de_prev + intercept
        T_curr = seas_curr + de_curr
        forecasts[i] = T_curr
        de_prev = de_curr
    return forecasts

# ===========================================================================
# SECTION 4: ML MODEL SHARED HELPERS
# ===========================================================================

_EXTENDED_CACHE = None

def load_extended_data():
    global _EXTENDED_CACHE
    if _EXTENDED_CACHE is not None:
        return _EXTENDED_CACHE

    # notebook/export uses index_col=0, parse_dates=True
    ext = pd.read_csv(EXTENDED_CSV, index_col=0, parse_dates=True)
    ext.index = pd.to_datetime(ext.index)
    ext = ext.sort_index().copy()
    _EXTENDED_CACHE = ext
    return ext


def get_extended_region_frame(region):
    ext = load_extended_data()

    region_col = None
    for c in ["name", "region_code"]:
        if c in ext.columns:
            region_col = c
            break
    if region_col is None:
        raise ValueError("Could not find region column in region_temp_extended.csv")

    df = ext[ext[region_col] == region].copy()
    df = df.sort_index()
    return df

def _next_date_skip_feb29(d):
    nxt = d + datetime.timedelta(days=1)
    if nxt.month == 2 and nxt.day == 29:
        nxt = nxt + datetime.timedelta(days=1)
    return nxt

def _compute_trend_seasonal_single(t_idx, doy, trend_int, trend_slope, fourier_coefs, n_harmonics):
    trend = trend_int + trend_slope * t_idx
    cols = [1.0]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * doy / 365.0))
        cols.append(np.cos(2 * np.pi * k * doy / 365.0))
    return trend + np.dot(cols, fourier_coefs)

def _create_feat(df, pdtn):
    df = df.sort_index(ascending=False)
    today = df.index.max()
    today_row = df.loc[[today]]

    dayofyear = int(today_row["pdtn_doy"].iloc[0])
    pdtn_doy = 1 if dayofyear == 365 else dayofyear + 1
    de_trend_seas = pdtn
    dts_doyavge = df[df["dayofyear"] == dayofyear]["dts_doyavge"].iloc[0]
    dts_doyvar = df[df["dayofyear"] == dayofyear]["dts_doyvar"].iloc[0]

    yday = today_row["de_trend_seas"].iloc[0]
    IIdays_ago = df["de_trend_seas"].iloc[1]
    IIIdays_ago = df["de_trend_seas"].iloc[2]
    IVdays_ago = df["de_trend_seas"].iloc[3]
    Vdays_ago = df["de_trend_seas"].iloc[4]
    VIdays_ago = df["de_trend_seas"].iloc[5]
    VIIdays_ago = df["de_trend_seas"].iloc[6]

    last_year = df["de_trend_seas"].iloc[364]
    last_2year = df["de_trend_seas"].iloc[729]
    last_3year = df["de_trend_seas"].iloc[1094]
    last_4year = df["de_trend_seas"].iloc[1459]
    last_5year = df["de_trend_seas"].iloc[1824]

    h1_last_year = df["de_trend_seas"].iloc[363]
    h1_last_2years = df["de_trend_seas"].iloc[728]
    h1_last_3years = df["de_trend_seas"].iloc[1093]
    h1_last_4years = df["de_trend_seas"].iloc[1458]
    h1_last_5years = df["de_trend_seas"].iloc[1823]

    h1_ly_2days = df["de_trend_seas"].iloc[365]
    h1_ly_3day = df["de_trend_seas"].iloc[366]
    h1_ly_4day = df["de_trend_seas"].iloc[367]
    h1_ly_5day = df["de_trend_seas"].iloc[368]
    h1_ly_6day = df["de_trend_seas"].iloc[369]
    h1_ly_7day = df["de_trend_seas"].iloc[370]

    h1_ly_next_1day = df["de_trend_seas"].iloc[362]
    h1_ly_next_2days = df["de_trend_seas"].iloc[361]
    h1_ly_next_3days = df["de_trend_seas"].iloc[360]
    h1_ly_next_4days = df["de_trend_seas"].iloc[359]
    h1_ly_next_5days = df["de_trend_seas"].iloc[358]
    h1_ly_next_6days = df["de_trend_seas"].iloc[357]
    h1_ly_next_7days = df["de_trend_seas"].iloc[356]

    diff_1year = pdtn - last_year
    diff_2year = pdtn - last_2year
    diff_3year = pdtn - last_3year
    diff_4year = pdtn - last_4year
    diff_5year = pdtn - last_5year
    diff_yday = pdtn - yday
    diff_2days = pdtn - IIdays_ago
    diff_3days = pdtn - IIIdays_ago
    diff_4days = pdtn - IVdays_ago
    diff_5days = pdtn - Vdays_ago
    diff_6days = pdtn - VIdays_ago
    diff_7days = pdtn - VIIdays_ago

    last_7_1day_deltas_mean = df["diff_yday"].iloc[0:7].mean()
    last_7_1day_deltas_min = df["diff_yday"].iloc[0:7].min()
    last_7_1day_deltas_max = df["diff_yday"].iloc[0:7].max()

    return [[
        dayofyear, pdtn_doy, de_trend_seas, dts_doyavge, dts_doyvar,
        yday, IIdays_ago, IIIdays_ago, IVdays_ago, Vdays_ago, VIdays_ago, VIIdays_ago,
        last_year, last_2year, last_3year, last_4year, last_5year,
        h1_last_year, h1_last_2years, h1_last_3years, h1_last_4years, h1_last_5years,
        h1_ly_2days, h1_ly_3day, h1_ly_4day, h1_ly_5day, h1_ly_6day, h1_ly_7day,
        h1_ly_next_1day, h1_ly_next_2days, h1_ly_next_3days,
        h1_ly_next_4days, h1_ly_next_5days, h1_ly_next_6days, h1_ly_next_7days,
        diff_1year, diff_2year, diff_3year, diff_4year, diff_5year,
        diff_yday, diff_2days, diff_3days, diff_4days, diff_5days, diff_6days, diff_7days,
        last_7_1day_deltas_mean, last_7_1day_deltas_min, last_7_1day_deltas_max,
    ]]

def _recursive_forecast(predict_fn, working_df, test_dates, trend_int, trend_slope, fourier_coefs,
                         n_harmonics, n_calib_days, feature_cols):
    oodf = working_df.copy()
    if "diff_yday" not in oodf.columns:
        dts = oodf.sort_index()["de_trend_seas"]
        oodf["diff_yday"] = dts - dts.shift(1)

    n_steps = len(test_dates)
    forecasts_actual = np.full(n_steps, np.nan)

    for i in range(n_steps):
        latest_day = oodf.index.max()
        next_day = _next_date_skip_feb29(latest_day)
        latest_row = oodf.loc[[latest_day]]
        dts_pred = predict_fn(latest_row[feature_cols].values.reshape(1, -1))
        try:
            new_row_data = _create_feat(df=oodf, pdtn=dts_pred)
        except (IndexError, KeyError) as e:
            print(f"    WARNING: _create_feat failed at step {i}: {e}")
            break
        new_row = pd.DataFrame(data=new_row_data, index=[next_day], columns=feature_cols)
        oodf = pd.concat([oodf, new_row], axis=0)
        t_idx = float(n_calib_days + i)
        doy = _doy_noleap(next_day)
        trend_seas = _compute_trend_seasonal_single(t_idx, doy, trend_int, trend_slope, fourier_coefs, n_harmonics)
        forecasts_actual[i] = dts_pred + trend_seas

    return forecasts_actual

def forecast_paper_xgb(payload, test_dates, calib_ctx):
    model = payload["fitted_model"]
    scaler = payload["scaler"]
    trend_int, trend_slope, fourier_coefs, n_harmonics, n_calib_days = _extract_trend_ctx(payload, calib_ctx)
    working_df = _extract_working_df(payload, calib_ctx)
    feature_cols = _extract_feature_cols(payload, FEATURE_COLS)

    def predict_fn(X):
        return float(model.predict(scaler.transform(X))[0])

    return _recursive_forecast(predict_fn, working_df, test_dates, trend_int, trend_slope, fourier_coefs,
                               n_harmonics, n_calib_days, feature_cols)

def forecast_wavelet_fnn(payload, test_dates, calib_ctx):
    pipeline = payload["fitted_model"]
    trend_int = calib_ctx["trend_intercept"]
    trend_slope = calib_ctx["trend_slope"]
    fourier_coefs = calib_ctx["fourier_coefs"]
    n_harmonics = calib_ctx["n_harmonics"]
    n_calib_days = calib_ctx["n_calib_days"]
    working_df = calib_ctx["working_df"]

    def predict_fn(X):
        return float(pipeline.predict(X)[0])

    return _recursive_forecast(predict_fn, working_df, test_dates, trend_int, trend_slope, fourier_coefs,
                               n_harmonics, n_calib_days, FEATURE_COLS)

def forecast_lstm(payload, test_dates, calib_ctx=None):
    n = len(test_dates)

    if not isinstance(payload, dict):
        return np.full(n, np.nan)

    pred_df = payload.get("prediction_year_df")
    if pred_df is None or len(pred_df) == 0:
        raise ValueError("LSTM rolling export does not contain 'prediction_year_df'.")

    df = pred_df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # In your export, pred_df comes from pred_unscaled.to_dataframe()
    # where target_value_col == "de_trend_seas"
    value_col = None
    if "de_trend_seas" in df.columns:
        value_col = "de_trend_seas"
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("No numeric forecast column found in LSTM prediction_year_df.")
        value_col = num_cols[0]

    target_idx = pd.DatetimeIndex(test_dates)
    dts_forecast = df.reindex(target_idx)[value_col].to_numpy(dtype=float)

    if len(dts_forecast) < n:
        dts_forecast = np.pad(dts_forecast, (0, n - len(dts_forecast)), constant_values=np.nan)

    dts_forecast = dts_forecast[:n]

    # Reconstruct actual temperature from the SAME extended-data decomposition
    region = payload.get("region", payload.get("region_code"))
    if region is None:
        raise ValueError("LSTM payload does not contain region information.")

    ext_region = get_extended_region_frame(region)

    needed = ext_region.reindex(target_idx)[["trend", "four_seas"]].copy()
    trend_part = needed["trend"].to_numpy(dtype=float)
    seas_part = needed["four_seas"].to_numpy(dtype=float)

    forecasts_actual = dts_forecast + trend_part + seas_part
    return forecasts_actual

def forecast_feedforward_nn(payload, test_dates, ffnn_ctx):
    n = len(test_dates)

    if payload is None:
        return np.full(n, np.nan)

    if ffnn_ctx is None or "working_df" not in ffnn_ctx or "full_df" not in ffnn_ctx:
        return np.full(n, np.nan)

    if not isinstance(payload, dict):
        raise ValueError("FFNN payload must be a dict.")

    model = payload.get("fitted_model", None)
    if model is None or not hasattr(model, "predict"):
        raise ValueError("FFNN payload does not contain a usable fitted_model pipeline.")

    feature_cols = payload.get("feature_cols", FEATURE_COLS)

    # IMPORTANT:
    # Notebook labels target_col='next_day', but actually builds:
    # next_day = de_trend_seas.shift(-1)
    # So the model predicts NEXT-DAY de_trend_seas.
    oodf = ffnn_ctx["working_df"].copy().sort_index()
    full_df = ffnn_ctx["full_df"]

    missing = [c for c in feature_cols if c not in oodf.columns]
    if missing:
        raise KeyError(f"FFNN missing feature columns in working_df: {missing}")

    forecasts_actual = np.full(n, np.nan)

    for i in range(n):
        latest_day = oodf.index.max()
        next_day = _next_date_skip_feb29(latest_day)

        if next_day not in full_df.index:
            print(f"    WARNING: FFNN next_day {next_day.date()} missing in full_df")
            break

        X = oodf.loc[[latest_day], feature_cols]

        # pipeline already includes imputer + scaler + mlp
        y_pred_dts = float(model.predict(X)[0])

        trend_val = float(full_df.loc[next_day, "trend"])
        seas_val = float(full_df.loc[next_day, "four_seas"])

        y_pred_actual = y_pred_dts + trend_val + seas_val
        forecasts_actual[i] = y_pred_actual

        try:
            new_row_data = _create_feat(df=oodf.sort_index(ascending=False), pdtn=y_pred_dts)
        except (IndexError, KeyError) as e:
            print(f"    WARNING: FFNN _create_feat failed at step {i}: {e}")
            break

        new_row = pd.DataFrame(new_row_data, index=[next_day], columns=FEATURE_COLS)

        # preserve auxiliary columns needed for future steps / debugging
        if "trend" in full_df.columns:
            new_row["trend"] = trend_val
        if "four_seas" in full_df.columns:
            new_row["four_seas"] = seas_val
        new_row["de_trend_seas"] = y_pred_dts

        if "month" in oodf.columns:
            new_row["month"] = next_day.month
        if "next_day" in oodf.columns:
            new_row["next_day"] = np.nan

        oodf = pd.concat([oodf, new_row], axis=0).sort_index()

    return forecasts_actual

def compute_ffnn_calib_context(region, calib_year):
    ext = pd.read_csv(EXTENDED_CSV)
    ext["date"] = pd.to_datetime(ext["date"])

    region_col = None
    for c in ["region_code", "name"]:
        if c in ext.columns:
            region_col = c
            break
    if region_col is None:
        raise ValueError("Cannot find region column in region_temp_extended.csv")

    ext = ext[ext[region_col] == region].copy()
    ext = ext[~((ext["date"].dt.month == 2) & (ext["date"].dt.day == 29))].copy()
    ext = ext.sort_values("date").reset_index(drop=True)

    calib_end = pd.Timestamp(f"{calib_year}-12-31")
    ext_calib = ext[ext["date"] <= calib_end].copy()

    if len(ext_calib) < 365 * 6:
        return None

    ext_calib = ext_calib.set_index("date").sort_index()
    full_df = ext.set_index("date").sort_index()

    # make sure the recursive feature base has diff_yday
    if "diff_yday" not in ext_calib.columns and "de_trend_seas" in ext_calib.columns:
        ext_calib["diff_yday"] = ext_calib["de_trend_seas"] - ext_calib["de_trend_seas"].shift(1)

    working_df = ext_calib.tail(365 * 5).copy()

    required = set(FEATURE_COLS) | {"trend", "four_seas", "de_trend_seas"}
    missing = [c for c in required if c not in full_df.columns]
    if missing:
        raise KeyError(f"FFNN extended data missing required columns: {missing}")

    return {
        "working_df": working_df,
        "full_df": full_df,
    }
# ===========================================================================
# SECTION 5: METRICS
# ===========================================================================

def cumulative_index(T_arr, metric):
    if metric == "CAT":
        return float(np.sum(T_arr))
    elif metric == "HDD":
        return float(np.sum(np.maximum(0.0, T_REF - T_arr)))
    else:
        return float(np.sum(np.maximum(0.0, T_arr - T_REF)))

def compute_ape(T_actual, T_forecast, metric):
    d = {}
    for h, (s, e) in HORIZONS.items():
        ya = cumulative_index(T_actual[s:e], metric)
        yf = cumulative_index(T_forecast[s:e], metric)
        if abs(ya) < 1e-8:
            d[h] = np.nan
        elif metric in ("HDD", "CDD") and ya < MIN_INDEX:
            d[h] = np.nan
        else:
            d[h] = abs(ya - yf) / abs(ya)
    return d

def compute_mape(T_actual, T_forecast):
    r = {}
    for h, (s, e) in HORIZONS.items():
        ya = T_actual[s:e]
        yf = T_forecast[s:e]
        m = np.abs(ya) > 1e-8
        r[h] = float(np.mean(np.abs(ya[m] - yf[m]) / np.abs(ya[m]))) if m.any() else np.nan
    return r

def compute_rmse(T_actual, T_forecast):
    return {h: float(np.sqrt(np.mean((T_actual[s:e] - T_forecast[s:e]) ** 2))) for h, (s, e) in HORIZONS.items()}

def compute_mae(T_actual, T_forecast):
    return {h: float(np.mean(np.abs(T_actual[s:e] - T_forecast[s:e]))) for h, (s, e) in HORIZONS.items()}

def compute_wmape(T_actual, T_forecast):
    r = {}
    for h, (s, e) in HORIZONS.items():
        ya = T_actual[s:e]
        yf = T_forecast[s:e]
        d = np.sum(np.abs(ya))
        r[h] = float(np.sum(np.abs(ya - yf)) / d) if d > 1e-8 else np.nan
    return r

# ===========================================================================
# SECTION 6: STATISTICAL TESTS
# ===========================================================================

def rank_models(ape_matrix, metadata):
    n_obs = len(metadata)
    rd = {}
    for h in HORIZON_NAMES:
        rd[h] = {}
        for oi in range(n_obs):
            vals = {m: ape_matrix[m][oi][h] for m in MODEL_ORDER}
            valid = {m: v for m, v in vals.items() if not np.isnan(v)}
            if not valid:
                rd[h][oi] = {m: np.nan for m in MODEL_ORDER}
                continue
            sm = sorted(valid, key=lambda m: valid[m])
            ranks = {m: np.nan for m in MODEL_ORDER}
            for r, m in enumerate(sm, 1):
                ranks[m] = float(r)
            rd[h][oi] = ranks
    return rd

def rank_count_table(rd):
    rows = {m: {} for m in MODEL_ORDER}
    for h in HORIZON_NAMES:
        counts = {m: 0 for m in MODEL_ORDER}
        for _, rdict in rd[h].items():
            best = min((rdict[m] for m in MODEL_ORDER if not np.isnan(rdict[m])), default=np.nan)
            if np.isnan(best):
                continue
            for m in MODEL_ORDER:
                if rdict[m] == best:
                    counts[m] += 1
        for m in MODEL_ORDER:
            rows[m][h] = counts[m]
    return pd.DataFrame(rows, index=HORIZON_NAMES).T

def avg_rank_table(rd):
    rows = {m: {} for m in MODEL_ORDER}
    for h in HORIZON_NAMES:
        for m in MODEL_ORDER:
            vals = [rd[h][ri][m] for ri in rd[h] if not np.isnan(rd[h][ri][m])]
            rows[m][h] = float(np.mean(vals)) if vals else np.nan
    return pd.DataFrame(rows, index=HORIZON_NAMES).T

def run_friedman_nemenyi(rd):
    results = []
    for h in HORIZON_NAMES:
        rr = []
        for ri in rd[h]:
            row = [rd[h][ri][m] for m in MODEL_ORDER]
            if not any(np.isnan(v) for v in row):
                rr.append(row)
        if len(rr) < 3:
            results.append({"horizon": h, "friedman_p": np.nan, **{m: np.nan for m in MODEL_ORDER}})
            continue
        mat = np.array(rr)
        try:
            _, p = friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
        except Exception:
            p = np.nan
        ar = {m: float(mat[:, j].mean()) for j, m in enumerate(MODEL_ORDER)}
        results.append({"horizon": h, "friedman_p": round(p, 6), **ar})
    return pd.DataFrame(results).set_index("horizon")

def cluster_regions(rm, n_clusters=None):
    M = np.array(rm)
    k = n_clusters or max(2, min(4, len(M) // 2))
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(M)

def blocked_bootstrap_wilcoxon(ape_matrix, metadata, cluster_labels, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(42)
    nm = len(MODEL_ORDER)
    no = len(metadata)
    r2c = {r: int(cluster_labels[i]) for i, r in enumerate(REGIONS)}
    oc = np.array([r2c[m["region"]] for m in metadata])
    cl = np.unique(oc)
    results = []
    for h in HORIZON_NAMES:
        aa = np.array([[ape_matrix[m][oi][h] for oi in range(no)] for m in MODEL_ORDER])
        br = np.zeros((n_bootstrap, nm))
        for b in range(n_bootstrap):
            s = []
            for c in cl:
                ci = np.where(oc == c)[0]
                s.extend(rng.choice(ci, size=len(ci), replace=True).tolist())
            s = np.array(s)
            sub = aa[:, s]
            por = np.full_like(sub, np.nan)
            for col in range(sub.shape[1]):
                cv = sub[:, col]
                v = ~np.isnan(cv)
                if v.sum() < 2:
                    continue
                rk = np.full(nm, np.nan)
                vi = np.where(v)[0]
                sv = vi[np.argsort(cv[v])]
                for r, vi2 in enumerate(sv, 1):
                    rk[vi2] = r
                por[:, col] = rk
            br[b] = np.nanmean(por, axis=1)
        abr = br.mean(axis=0)
        ci2 = int(np.argmin(abr))
        row = {"horizon": h, "n_obs": no, "control_model": MODEL_ORDER[ci2]}
        for j, m in enumerate(MODEL_ORDER):
            row[f"{m}_avg_rank"] = round(float(abr[j]), 3)
        for j, m in enumerate(MODEL_ORDER):
            if j == ci2:
                row[f"{m}_wilcoxon_p"] = np.nan
                continue
            diff = br[:, j] - br[:, ci2]
            if np.all(diff == 0) or len(diff) < 10:
                row[f"{m}_wilcoxon_p"] = np.nan
                continue
            try:
                _, p = scipy_wilcoxon(diff, alternative="two-sided")
            except Exception:
                p = np.nan
            row[f"{m}_wilcoxon_p"] = round(float(p), 6)
        results.append(row)
    return pd.DataFrame(results).set_index("horizon")

def metric_table(md, meta):
    rows = []
    for oi, mr in enumerate(meta):
        for m in MODEL_ORDER:
            row = {"region": mr["region"], "year": mr["year"], "model": m}
            row.update(md[m][oi])
            rows.append(row)
    return pd.DataFrame(rows).set_index(["region", "year", "model"])

def region_temperature_moments(df):
    T = df["T"].values
    return (float(np.mean(T)), float(np.std(T)), float(sp_skew(T)), float(sp_kurt(T)))

# ===========================================================================
# SECTION 7: MAIN EVALUATION LOOP
# ===========================================================================

print("=" * 70)
print("  Barnor et al. (2026) Replication — French NUTS-2 Regions")
print(f"  Models: {', '.join(MODEL_ORDER)}")
print(f"  OOS: {OOS_START}–{OOS_END} ({len(TEST_YEARS)} yrs × {len(REGIONS)} regions = {len(TEST_YEARS)*len(REGIONS)} obs)")
print("=" * 70)

ape_results = {"CAT": {m: [] for m in MODEL_ORDER}, "HDD": {m: [] for m in MODEL_ORDER}, "CDD": {m: [] for m in MODEL_ORDER}}
mape_res = {m: [] for m in MODEL_ORDER}
rmse_res = {m: [] for m in MODEL_ORDER}
mae_res = {m: [] for m in MODEL_ORDER}
wmape_res = {m: [] for m in MODEL_ORDER}
obs_metadata = []
region_moments_list = []
all_forecasts = []

region_data_cache = {}
for region in REGIONS:
    print(f"  Loading region {region} ...", end=" ", flush=True)
    region_data_cache[region] = load_region(region)
    df_pre = region_data_cache[region][region_data_cache[region]["date"].dt.year < OOS_START]
    region_moments_list.append(region_temperature_moments(df_pre))
    print("done")

print()

for region in REGIONS:
    df = region_data_cache[region]
    _loaded_calib = None
    hba_params = alaton_params = benth_params = xgb_payload = wavelet_payload = lstm_payload = ffnn_payload = None
    calib_ctx = wavelet_calib_ctx = ffnn_calib_ctx = None
    print(f"\n{'─'*60}\n  Region {region}\n{'─'*60}")

    for test_year in TEST_YEARS:
        calib_year = test_year - 1
        df_train, df_test = get_train_test_for_year(df, test_year)
        if len(df_train) < 365 or len(df_test) == 0:
            print(f"  [{test_year}] Insufficient data — skipping")
            continue

        test_dates = df_test["date"].tolist()
        T_actual = df_test["T"].values.astype(float)
        if len(T_actual) < 365:
            T_actual = np.pad(T_actual, (0, 365 - len(T_actual)), constant_values=np.nan)
            test_dates += [test_dates[-1] + pd.Timedelta(days=i + 1) for i in range(365 - len(test_dates))]
        else:
            T_actual = T_actual[:365]
            test_dates = test_dates[:365]

        if calib_year != _loaded_calib:
            print(f"  [{test_year}] calib={calib_year}", end="  ", flush=True)
            try:
                hba_params = load_hba(region, calib_year)
            except FileNotFoundError:
                hba_params = None
            try:
                alaton_params = load_alaton(region, calib_year)
            except FileNotFoundError:
                alaton_params = None
            try:
                benth_params = load_benth(region, calib_year)
            except FileNotFoundError:
                benth_params = None
            try:
                xgb_payload = load_paper_xgb(region, calib_year)
            except FileNotFoundError:
                xgb_payload = None
            try:
                wavelet_payload = load_wavelet_fnn(region, calib_year)
            except FileNotFoundError:
                wavelet_payload = None
            try:
                lstm_payload = load_lstm(region, calib_year)
            except FileNotFoundError:
                lstm_payload = None
            try:
                ffnn_payload = load_feedforward_nn(region, calib_year)
            except FileNotFoundError:
                ffnn_payload = None

            calib_ctx = compute_calib_context(df_train)
            wavelet_calib_ctx = compute_wavelet_calib_context(region, calib_year, df_train)
            ffnn_calib_ctx = compute_ffnn_calib_context(region, calib_year)

            _loaded_calib = calib_year
            print("PKLs loaded")
        else:
            print(f"  [{test_year}]", end="  ", flush=True)

        f_hba = forecast_hba(hba_params, test_dates) if hba_params else np.full(365, np.nan)
        f_alaton = forecast_alaton(alaton_params, df_train, test_dates) if alaton_params else np.full(365, np.nan)
        f_benth = forecast_benth(benth_params, df_train, test_dates) if benth_params else np.full(365, np.nan)
        f_xgb = forecast_paper_xgb(xgb_payload, test_dates, calib_ctx) if xgb_payload else np.full(365, np.nan)
        if wavelet_payload and wavelet_calib_ctx:
            f_wavelet = forecast_wavelet_fnn(wavelet_payload, test_dates, wavelet_calib_ctx)
        else:
            f_wavelet = np.full(365, np.nan)
        f_lstm = forecast_lstm(lstm_payload, test_dates, calib_ctx) if lstm_payload else np.full(365, np.nan)
        f_ffnn = forecast_feedforward_nn(ffnn_payload, test_dates, ffnn_calib_ctx) if ffnn_payload else np.full(365, np.nan)

        forecasts = {
            "HBA": f_hba,
            "Alaton": f_alaton,
            "Benth": f_benth,
            "XGB": f_xgb,
            "WaveletFNN": f_wavelet,
            "LSTM": f_lstm,
            "FeedForwardNN": f_ffnn,
        }

        for m, f_arr in forecasts.items():
            for d, T_hat, T_act in zip(test_dates, f_arr, T_actual):
                all_forecasts.append({
                    "region": region,
                    "year": test_year,
                    "date": d,
                    "actual": T_act,
                    "model": m,
                    "forecast": T_hat
                })

        obs_metadata.append({"region": region, "year": test_year})
        for m, f_arr in forecasts.items():
            ape_results["CAT"][m].append(compute_ape(T_actual, f_arr, "CAT"))
            ape_results["HDD"][m].append(compute_ape(T_actual, f_arr, "HDD"))
            ape_results["CDD"][m].append(compute_ape(T_actual, f_arr, "CDD"))
            mape_res[m].append(compute_mape(T_actual, f_arr))
            rmse_res[m].append(compute_rmse(T_actual, f_arr))
            mae_res[m].append(compute_mae(T_actual, f_arr))
            wmape_res[m].append(compute_wmape(T_actual, f_arr))

        xgb_cat = ape_results["CAT"]["XGB"][-1].get("90", float("nan"))
        wav_cat = ape_results["CAT"]["WaveletFNN"][-1].get("90", float("nan"))
        print(f"    XGB CAT-90={xgb_cat*100:.1f}%  WaveletFNN CAT-90={wav_cat*100:.1f}%")

n_obs = len(obs_metadata)
print(f"\n{'='*70}\n  Complete. {n_obs} obs ({len(REGIONS)} regions × {len(TEST_YEARS)} years)\n{'='*70}")

# ===========================================================================
# SECTION 8: RESULTS OUTPUT
# ===========================================================================

cluster_labels = cluster_regions(region_moments_list)
print("\n# K-MEANS CLUSTERS (§4.5)")
for ri, region in enumerate(REGIONS):
    m = region_moments_list[ri]
    print(f"  Region {region}: cluster {cluster_labels[ri]}  mean={m[0]:.1f}°C  std={m[1]:.1f}")

print("\n# Building metric tables ...")
df_ape_cat = metric_table(ape_results["CAT"], obs_metadata)
df_ape_hdd = metric_table(ape_results["HDD"], obs_metadata)
df_ape_cdd = metric_table(ape_results["CDD"], obs_metadata)
df_mape = metric_table(mape_res, obs_metadata)
df_rmse = metric_table(rmse_res, obs_metadata)
df_mae = metric_table(mae_res, obs_metadata)
df_wmape = metric_table(wmape_res, obs_metadata)

rd_cat = rank_models(ape_results["CAT"], obs_metadata)
rd_hdd = rank_models(ape_results["HDD"], obs_metadata)
rd_cdd = rank_models(ape_results["CDD"], obs_metadata)

for lbl, rd in [("CAT", rd_cat), ("HDD", rd_hdd), ("CDD", rd_cdd)]:
    rc = rank_count_table(rd)
    ar = avg_rank_table(rd)
    print(f"\n# RANK COUNT — {lbl} APE\n{rc.to_string()}")
    print(f"\n# AVG RANK — {lbl} APE\n{ar.to_string()}")
    rc.to_csv(f"bk_rank_counts_{lbl.lower()}.csv")
    ar.to_csv(f"bk_rank_avg_{lbl.lower()}.csv")

print("\n# FRIEDMAN + NEMENYI (§4.4.1)")
friedman_dfs = {}
for lbl, ad in [("CAT", ape_results["CAT"]), ("HDD", ape_results["HDD"]), ("CDD", ape_results["CDD"])]:
    rd = rank_models(ad, obs_metadata)
    dff = run_friedman_nemenyi(rd)
    friedman_dfs[lbl] = dff
    print(f"\n  {lbl} APE\n{dff.to_string()}")
pd.concat(friedman_dfs, axis=1).to_csv("bk_friedman_nemenyi.csv")

print("\n# BLOCKED BOOTSTRAP WILCOXON (§4.4.2)")
boot_dfs = {}
for lbl, ad in [("CAT", ape_results["CAT"]), ("HDD", ape_results["HDD"]), ("CDD", ape_results["CDD"])]:
    print(f"  {lbl} APE ...", end=" ", flush=True)
    dfb = blocked_bootstrap_wilcoxon(ad, obs_metadata, cluster_labels)
    boot_dfs[lbl] = dfb
    print(f"done\n{dfb.to_string()}")
pd.concat(boot_dfs, axis=1).to_csv("bk_bootstrap_wilcoxon.csv")

df_ape_cat.to_csv("bk_ape_cat.csv")
df_ape_hdd.to_csv("bk_ape_hdd.csv")
df_ape_cdd.to_csv("bk_ape_cdd.csv")
df_mape.to_csv("bk_mape_cat.csv")
df_rmse.to_csv("bk_rmse_hdd.csv")
df_mae.to_csv("bk_mae_hdd.csv")
df_wmape.to_csv("bk_wmape_cat.csv")
pd.DataFrame(all_forecasts).to_csv("bk_forecasts.csv", index=False)
pd.DataFrame(obs_metadata).to_csv("bk_obs_metadata.csv", index=False)

print("\n# SAVED OUTPUTS")
for f in [
    "bk_ape_cat.csv", "bk_ape_hdd.csv", "bk_ape_cdd.csv", "bk_mape_cat.csv", "bk_rmse_hdd.csv",
    "bk_mae_hdd.csv", "bk_wmape_cat.csv", "bk_rank_counts_cat.csv", "bk_rank_counts_hdd.csv",
    "bk_rank_counts_cdd.csv", "bk_rank_avg_cat.csv", "bk_rank_avg_hdd.csv", "bk_rank_avg_cdd.csv",
    "bk_friedman_nemenyi.csv", "bk_bootstrap_wilcoxon.csv", "bk_forecasts.csv", "bk_obs_metadata.csv"
]:
    print(f"  Saved -> {f}")
print(f"\nDone. {n_obs} observations, {len(all_forecasts)} forecast rows.")
