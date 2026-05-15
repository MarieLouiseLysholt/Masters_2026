"""
Selected model-parameter tables by region.

Creates a compact booktabs-style table matching the Forecast Accuracy/CRPS
tables. For models with an explicit grid/model-selection step, the selected
hyperparameters are reported. For models calibrated without a grid search, the
cell states the calibration specification so the absence of grid-selected
hyperparameters is explicit.

Outputs:
  T16_selected_parameters_ml_by_region.png
  T16a_selected_parameters_ml_classical_by_region.png
  T16b_selected_parameters_ml_neural_by_region.png
  T16_xgb_selected_parameters_by_region.png
  T16_lstm_selected_parameters_by_region.png
  T16_svr_selected_parameters_by_region.png
  T16_rf_selected_parameters_by_region.png
  T16_ffnn_selected_parameters_by_region.png
  T16_knn_selected_parameters_by_region.png
  T17_selected_parameters_benth_by_region.png
  T18_selected_parameters_alaton_by_region.png
  csv_selected_parameters_ml_by_region.csv
  csv_selected_parameters_benth_by_region.csv
  csv_selected_parameters_alaton_by_region.csv
"""

from pathlib import Path
import ast
import pickle
import pickletools
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "03_Models"
PKL_DIR = MODEL_DIR / "01_PKL Files"
OUTPUT_DIR = ROOT / "99_Thesis Graphs" / "03_Results" / "01_Forecast Accuracy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 260
REGIONS = [11, 24, 27, 28, 32, 44, 52, 53]
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

ML_COLUMNS = ["XGB", "LSTM", "SVR", "RF", "FFNN", "KNN"]
ML_CLASSICAL_COLUMNS = ["XGB", "SVR", "RF", "KNN"]
ML_NEURAL_COLUMNS = ["LSTM", "FFNN"]
ML_FULL_NAMES = {
    "XGB": "Extreme Gradient Boosting (XGBoost)",
    "LSTM": "Long Short-Term Memory Network (LSTM)",
    "SVR": "Support Vector Regression (SVR)",
    "RF": "Random Forest Regression",
    "FFNN": "Feed-Forward Neural Network (FFNN)",
    "KNN": "k-Nearest Neighbours Regression (KNN)",
}
ML_FILE_STEMS = {
    "XGB": "xgb",
    "LSTM": "lstm",
    "SVR": "svr",
    "RF": "rf",
    "FFNN": "ffnn",
    "KNN": "knn",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
})


def _compat_numpy_frombuffer(buf, dtype, shape, order, *extra):
    if not extra:
        import numpy.core.numeric as _np_numeric
        return _np_numeric._frombuffer(buf, dtype, shape, order)
    arr = np.frombuffer(buf, dtype=dtype)
    if order == "K":
        order = "C"
    return arr.reshape(shape, order=order)


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {"numpy._core.numeric", "numpy.core.numeric"} and name == "_frombuffer":
            return _compat_numpy_frombuffer
        return super().find_class(module, name)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return CompatUnpickler(f).load()


def fmt_num(v, dec=3):
    try:
        if pd.isna(v) or not np.isfinite(float(v)):
            return "—"
        return f"{float(v):.{dec}f}"
    except (TypeError, ValueError):
        return str(v)


def compact_params(params: dict, keys: list[str], aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or {}
    parts = []
    for key in keys:
        if key not in params:
            continue
        val = params[key]
        if val is None:
            continue
        label = aliases.get(key, key)
        if isinstance(val, float):
            val = f"{val:g}"
        parts.append(f"{label}={val}")
    return "\n".join(parts) if parts else "—"


def clean_param_value(val):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:g}"
    if isinstance(val, tuple):
        return "(" + ", ".join(str(v) for v in val) + ")"
    return str(val)


def _simple_pickle_value(opname: str, arg):
    if opname in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4"}:
        return int(arg)
    if opname == "BINFLOAT":
        return float(arg)
    if opname in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
        return arg
    if opname == "NEWTRUE":
        return True
    if opname == "NEWFALSE":
        return False
    if opname == "NONE":
        return None
    return None


def _extract_pickle_dict_after_key(path: Path, key_name: str) -> dict:
    """Extract a simple dict saved early in a pickle without loading objects."""
    ops = list(pickletools.genops(path.read_bytes()))
    key_pos = next(
        i for i, (op, arg, _) in enumerate(ops)
        if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"} and arg == key_name
    )
    start = next(i for i in range(key_pos + 1, len(ops)) if ops[i][0].name == "EMPTY_DICT")
    mark = next(i for i in range(start + 1, len(ops)) if ops[i][0].name == "MARK")
    stack = []
    for op, arg, _ in ops[mark + 1:]:
        name = op.name
        if name == "SETITEMS":
            break
        if name == "MEMOIZE":
            continue
        if name == "TUPLE1":
            stack.append((stack.pop(),))
            continue
        if name == "TUPLE2":
            b, a = stack.pop(), stack.pop()
            stack.append((a, b))
            continue
        if name == "TUPLE3":
            c, b, a = stack.pop(), stack.pop(), stack.pop()
            stack.append((a, b, c))
            continue
        value = _simple_pickle_value(name, arg)
        if value is not None or name in {"NEWTRUE", "NEWFALSE", "NONE"}:
            stack.append(value)
    return dict(zip(stack[0::2], stack[1::2]))


def _extract_top_level_pickle_value(path: Path, key_name: str):
    ops = list(pickletools.genops(path.read_bytes()))
    for i, (op, arg, _) in enumerate(ops):
        if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"} and arg == key_name:
            for next_op, next_arg, _ in ops[i + 1:]:
                if next_op.name == "MEMOIZE":
                    continue
                value = _simple_pickle_value(next_op.name, next_arg)
                if value is not None or next_op.name in {"NEWTRUE", "NEWFALSE", "NONE"}:
                    return value
                break
    return None


def read_arimaf_params() -> dict[int, str]:
    path = MODEL_DIR / "arimaf_selection_summary_2026-04-30.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).set_index("region")
    out = {}
    for region, row in df.iterrows():
        out[int(region)] = f"p={int(row['p'])}, q={int(row['q'])}\nK={int(row['K'])}"
    return out


def read_benth_params() -> dict[int, str]:
    path = MODEL_DIR / "benth_selection_summary_2026-05-10.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).set_index("region")
    out = {}
    for region, row in df.iterrows():
        out[int(region)] = (
            f"I1,J1={int(row['I1'])},{int(row['J1'])}\n"
            f"I2,J2={int(row['I2'])},{int(row['J2'])}\n"
            f"alpha={fmt_num(row['sel_alpha'], 3)}"
        )
    return out


def read_svm_params() -> dict[int, str]:
    path = PKL_DIR / "09_SVM" / "svm_tune_summary_2026-05-02.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        params = ast.literal_eval(row["best_params"])
        out[int(row["region"])] = compact_params(
            params,
            ["svr__kernel", "svr__C", "svr__epsilon", "svr__gamma", "svr__degree", "svr__tol"],
            {
                "svr__kernel": "kernel",
                "svr__C": "C",
                "svr__epsilon": "eps",
                "svr__gamma": "gamma",
                "svr__degree": "deg",
                "svr__tol": "tol",
            },
        )
    return out


def read_svm_param_rows() -> pd.DataFrame:
    path = PKL_DIR / "09_SVM" / "svm_tune_summary_2026-05-02.csv"
    rows = []
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            params = ast.literal_eval(row["best_params"])
            rows.append({
                "Region": int(row["region"]),
                "kernel": clean_param_value(params.get("svr__kernel")),
                "C": clean_param_value(params.get("svr__C")),
                "epsilon": clean_param_value(params.get("svr__epsilon")),
                "gamma": clean_param_value(params.get("svr__gamma")),
                "degree": clean_param_value(params.get("svr__degree")),
                "tol": clean_param_value(params.get("svr__tol")),
            })
    return pd.DataFrame(rows)


def read_sklearn_pickle_params(model_dir: str, stem: str, keys: list[str],
                               aliases: dict[str, str]) -> dict[int, str]:
    out = {}
    for region in REGIONS:
        path = PKL_DIR / model_dir / f"best_mod_{region}_{stem}.pkl"
        if not path.exists():
            continue
        try:
            obj = load_pickle(path)
            params = obj.get_params() if hasattr(obj, "get_params") else {}
            out[region] = compact_params(params, keys, aliases)
        except Exception as exc:
            out[region] = f"read failed:\n{type(exc).__name__}"
    return out


def read_xgb_params() -> dict[int, str]:
    return read_sklearn_pickle_params(
        "07_XGBoost",
        "paper_xgboost",
        ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"],
        {
            "n_estimators": "n",
            "max_depth": "depth",
            "learning_rate": "eta",
            "subsample": "sub",
            "colsample_bytree": "col",
        },
    )


def read_sklearn_param_rows(model_dir: str, stem: str, param_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        path = PKL_DIR / model_dir / f"best_mod_{region}_{stem}.pkl"
        if not path.exists():
            continue
        try:
            obj = load_pickle(path)
            params = obj.get_params() if hasattr(obj, "get_params") else {}
            row = {"Region": region}
            for raw_key, label in param_map.items():
                row[label] = clean_param_value(params.get(raw_key))
            rows.append(row)
        except Exception as exc:
            rows.append({"Region": region, "error": type(exc).__name__})
    return pd.DataFrame(rows)


def read_xgb_param_rows() -> pd.DataFrame:
    return read_sklearn_param_rows(
        "07_XGBoost",
        "paper_xgboost",
        {
            "n_estimators": "n_estimators",
            "max_depth": "max_depth",
            "learning_rate": "learning_rate",
            "subsample": "subsample",
            "colsample_bytree": "colsample_bytree",
        },
    )


def read_knn_params() -> dict[int, str]:
    return read_sklearn_pickle_params(
        "08_KNN",
        "paper_knn",
        ["knn__n_neighbors", "knn__weights", "knn__metric"],
        {
            "knn__n_neighbors": "k",
            "knn__weights": "w",
            "knn__metric": "metric",
        },
    )


def read_knn_param_rows() -> pd.DataFrame:
    return read_sklearn_param_rows(
        "08_KNN",
        "paper_knn",
        {
            "knn__n_neighbors": "n_neighbors",
            "knn__weights": "weights",
            "knn__metric": "metric",
        },
    )


def read_rf_params() -> dict[int, str]:
    return read_sklearn_pickle_params(
        "06_RF",
        "paper_rf",
        [
            "rf__n_estimators", "rf__max_depth", "rf__max_features",
            "rf__min_samples_leaf", "rf__min_samples_split",
        ],
        {
            "rf__n_estimators": "n",
            "rf__max_depth": "depth",
            "rf__max_features": "maxfeat",
            "rf__min_samples_leaf": "leaf",
            "rf__min_samples_split": "split",
        },
    )


def read_rf_param_rows() -> pd.DataFrame:
    return read_sklearn_param_rows(
        "06_RF",
        "paper_rf",
        {
            "rf__n_estimators": "n_estimators",
            "rf__max_depth": "max_depth",
            "rf__max_features": "max_features",
            "rf__min_samples_leaf": "min_samples_leaf",
            "rf__min_samples_split": "min_samples_split",
        },
    )


def read_ffnn_params() -> dict[int, str]:
    out = {}
    for region in REGIONS:
        path = (
            PKL_DIR / "04_FFNN"
            / f"paper_ffnn_region_{region}_calib_2023_predict_2024.pkl"
        )
        if not path.exists():
            continue
        try:
            params = _extract_pickle_dict_after_key(path, "best_params")
            out[region] = compact_params(
                params,
                [
                    "mlp__hidden_layer_sizes", "mlp__activation",
                    "mlp__learning_rate", "mlp__learning_rate_init",
                    "mlp__tol", "mlp__validation_fraction",
                ],
                {
                    "mlp__hidden_layer_sizes": "layers",
                    "mlp__activation": "act",
                    "mlp__learning_rate": "lr sched.",
                    "mlp__learning_rate_init": "lr",
                    "mlp__tol": "tol",
                    "mlp__validation_fraction": "val",
                },
            )
        except Exception as exc:
            out[region] = f"read failed:\n{type(exc).__name__}"
    return out


def read_ffnn_param_rows() -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        path = (
            PKL_DIR / "04_FFNN"
            / f"paper_ffnn_region_{region}_calib_2023_predict_2024.pkl"
        )
        if not path.exists():
            continue
        try:
            params = _extract_pickle_dict_after_key(path, "best_params")
            rows.append({
                "Region": region,
                "hidden_layer_sizes": clean_param_value(params.get("mlp__hidden_layer_sizes")),
                "activation": clean_param_value(params.get("mlp__activation")),
                "learning_rate": clean_param_value(params.get("mlp__learning_rate")),
                "learning_rate_init": clean_param_value(params.get("mlp__learning_rate_init")),
                "tol": clean_param_value(params.get("mlp__tol")),
                "validation_fraction": clean_param_value(params.get("mlp__validation_fraction")),
            })
        except Exception as exc:
            rows.append({"Region": region, "error": type(exc).__name__})
    return pd.DataFrame(rows)


def read_lstm_params() -> dict[int, str]:
    out = {}
    for region in REGIONS:
        path = PKL_DIR / "05_LSTM" / f"lstm_region_{region}_calib_2023_predict_2024.pkl"
        if not path.exists():
            continue
        try:
            params = _extract_pickle_dict_after_key(path, "best_params")
            params["best_lr"] = _extract_top_level_pickle_value(path, "best_lr")
            params["best_input_chunk_length"] = _extract_top_level_pickle_value(
                path, "best_input_chunk_length"
            )
            params["best_output_chunk_length"] = _extract_top_level_pickle_value(
                path, "best_output_chunk_length"
            )
            params["rolling_final_epochs"] = _extract_top_level_pickle_value(
                path, "rolling_final_epochs"
            )
            out[region] = compact_params(
                params,
                [
                    "hidden_dim", "n_rnn_layers", "dropout", "activation",
                    "best_lr", "best_input_chunk_length",
                    "best_output_chunk_length", "rolling_final_epochs",
                ],
                {
                    "hidden_dim": "hidden",
                    "n_rnn_layers": "layers",
                    "dropout": "drop",
                    "activation": "act",
                    "best_lr": "lr",
                    "best_input_chunk_length": "in",
                    "best_output_chunk_length": "out",
                    "rolling_final_epochs": "epochs",
                },
            )
        except Exception as exc:
            out[region] = f"read failed:\n{type(exc).__name__}"
    return out


def read_lstm_param_rows() -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        path = PKL_DIR / "05_LSTM" / f"lstm_region_{region}_calib_2023_predict_2024.pkl"
        if not path.exists():
            continue
        try:
            params = _extract_pickle_dict_after_key(path, "best_params")
            rows.append({
                "Region": region,
                "hidden_dim": clean_param_value(params.get("hidden_dim")),
                "n_rnn_layers": clean_param_value(params.get("n_rnn_layers")),
                "dropout": clean_param_value(params.get("dropout")),
                "activation": clean_param_value(params.get("activation")),
                "learning_rate": clean_param_value(_extract_top_level_pickle_value(path, "best_lr")),
                "input_chunk_length": clean_param_value(
                    _extract_top_level_pickle_value(path, "best_input_chunk_length")
                ),
                "output_chunk_length": clean_param_value(
                    _extract_top_level_pickle_value(path, "best_output_chunk_length")
                ),
                "epochs": clean_param_value(
                    _extract_top_level_pickle_value(path, "rolling_final_epochs")
                ),
            })
        except Exception as exc:
            rows.append({"Region": region, "error": type(exc).__name__})
    return pd.DataFrame(rows)


def build_ml_table() -> pd.DataFrame:
    xgb = read_xgb_params()
    lstm = read_lstm_params()
    knn = read_knn_params()
    svm = read_svm_params()
    rf = read_rf_params()
    ffnn = read_ffnn_params()

    rows = []
    for region in REGIONS:
        rows.append({
            "Region": region,
            "XGB": xgb.get(region, "—"),
            "LSTM": lstm.get(region, "—"),
            "SVR": svm.get(region, "—"),
            "RF": rf.get(region, "—"),
            "FFNN": ffnn.get(region, "—"),
            "KNN": knn.get(region, "—"),
        })
    return pd.DataFrame(rows)


def build_benth_table() -> pd.DataFrame:
    path = MODEL_DIR / "benth_selection_summary_2026-05-10.csv"
    if not path.exists():
        return pd.DataFrame({"Region": REGIONS})
    df = pd.read_csv(path)
    df = df[df["region"].isin(REGIONS)].copy()
    df["Region"] = df["region"].map(lambda r: REGION_NAMES.get(int(r), int(r)))
    df = df.rename(columns={
        "I1": "I1",
        "J1": "J1",
        "I2": "I2",
        "J2": "J2",
        "sel_alpha": "κ",
        "sel_aic": "AIC",
        "sel_log_lik": "Log-lik.",
        "k_params": "k params",
    })
    cols = ["region", "Region", "I1", "J1", "I2", "J2", "κ", "AIC",
            "Log-lik.", "k params"]
    df = df[cols].sort_values("region").drop(columns=["region"])
    df = df.rename(columns={
        "Log-lik.": "Log-likelihood",
        "k params": "Number of parameters",
    })
    for col in ["κ"]:
        df[col] = df[col].map(lambda v: fmt_num(v, 3))
    for col in ["AIC", "Log-likelihood"]:
        df[col] = df[col].map(lambda v: fmt_num(v, 1))
    return df


def build_alaton_table() -> pd.DataFrame:
    path = MODEL_DIR / "alaton_model_summary.csv"
    if not path.exists():
        return pd.DataFrame({"Region": REGIONS})
    df = pd.read_csv(path)
    df = df[df["region"].isin(REGIONS)].copy()
    grouped = (
        df.groupby("region")
        .agg({
            "A": "median",
            "B_C_per_decade": "median",
            "C": "median",
            "phi": "median",
            "a": "median",
            "half_life_days": "median",
            "sigma_min": "median",
            "sigma_max": "median",
            "normal_rejected_5pct": "mean",
        })
        .reset_index()
        .rename(columns={
            "B_C_per_decade": "B",
            "phi": "φ",
            "a": "κ",
            "half_life_days": "Half-life",
            "sigma_min": "σ min",
            "sigma_max": "σ max",
            "normal_rejected_5pct": "Normality rejection share",
        })
    )
    grouped["Region"] = grouped["region"].map(
        lambda r: REGION_NAMES.get(int(r), int(r))
    )
    grouped = grouped.sort_values("region").drop(columns=["region"])
    grouped = grouped[[
        "Region", "A", "B", "C", "φ", "κ", "Half-life",
        "σ min", "σ max", "Normality rejection share",
    ]]
    for col in ["A", "B", "C", "φ", "κ", "Half-life", "σ min", "σ max"]:
        grouped[col] = grouped[col].map(lambda v: fmt_num(v, 3))
    grouped["Normality rejection share"] = grouped["Normality rejection share"].map(
        lambda v: fmt_num(v, 2)
    )
    return grouped


def draw_table(df: pd.DataFrame, columns: list[str], out_name: str,
               note: str, figsize=(13.5, 5.4), fontsize=7.2,
               row_height_scale=2.35, title: str | None = None,
               show_note: bool = True) -> None:
    cell_text = df[columns].astype(str).values.tolist()
    region_col = "Region" if "Region" in df.columns else "Code / Region"
    row_labels = [str(r) for r in df[region_col].tolist()]
    col_labels = columns

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, row_height_scale)

    if title:
        fig.text(
            0.5, 0.925, title,
            ha="center", va="top",
            fontsize=10,
            fontfamily="Times New Roman",
            fontweight="normal",
        )

    n_rows = len(cell_text)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        cell.visible_edges = ""
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_text_props(ha="center", fontfamily="Times New Roman")
        else:
            if row == n_rows:
                cell.visible_edges = "B"
            ha = "left" if col == -1 else "center"
            cell.set_text_props(ha=ha, fontfamily="Times New Roman")
        cell.get_text().set_linespacing(1.25)

    if show_note:
        fig.text(
            0.5, 0.025,
            note,
            ha="center",
            fontsize=7.5,
            fontfamily="Times New Roman",
        )
    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.name}")


def draw_booktabs_table(df: pd.DataFrame, columns: list[str], out_name: str,
                        figsize=(10.5, 3.4), fontsize=8.0,
                        row_height_scale=1.45,
                        header_label_map: dict[str, str] | None = None,
                        italic_header_substrings: dict[str, str] | None = None) -> None:
    """Compact booktabs-style table with Region as a normal first column."""
    table_df = df[columns].astype(str)
    cell_text = table_df.values.tolist()
    header_label_map = header_label_map or {}
    col_labels = [header_label_map.get(c, c) for c in table_df.columns]
    italic_header_substrings = italic_header_substrings or {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, row_height_scale)

    n_rows = len(cell_text)
    n_cols = len(col_labels)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        cell.visible_edges = ""
        cell.set_text_props(fontfamily="Times New Roman")
        if row == 0:
            cell.visible_edges = "B"
            cell.set_text_props(
                ha="center",
                fontfamily="Times New Roman",
                fontweight="bold",
            )
        else:
            if row == n_rows:
                cell.visible_edges = "B"
                cell.set_linewidth(0.9)
            cell.set_text_props(
                ha="center",
                fontfamily="Times New Roman",
                fontweight="normal",
            )
        cell.get_text().set_linespacing(1.05)

    for col_idx, label in enumerate(col_labels):
        if label in italic_header_substrings:
            tbl[(0, col_idx)].get_text().set_text(
                f"{label}\n$\\it{{{italic_header_substrings[label]}}}$"
            )

    # Draw a clean top rule across the full header width.
    fig.canvas.draw()
    cells = tbl.get_celld()
    x0 = cells[(0, 0)].get_x()
    x1 = cells[(0, n_cols - 1)].get_x() + cells[(0, n_cols - 1)].get_width()
    y_top = cells[(0, 0)].get_y() + cells[(0, 0)].get_height()
    ax.plot([x0, x1], [y_top, y_top], color="black", linewidth=0.9,
            transform=ax.transAxes, clip_on=False)

    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out.name}")


def draw_single_ml_tables(ml: pd.DataFrame) -> None:
    builders = {
        "XGB": read_xgb_param_rows,
        "LSTM": read_lstm_param_rows,
        "SVR": read_svm_param_rows,
        "RF": read_rf_param_rows,
        "FFNN": read_ffnn_param_rows,
        "KNN": read_knn_param_rows,
    }
    for model in ML_COLUMNS:
        stem = ML_FILE_STEMS[model]
        full_name = ML_FULL_NAMES[model]
        df = builders[model]()
        columns = [c for c in df.columns if c != "Region"]
        draw_table(
            df, columns,
            f"T16_{stem}_selected_parameters_by_region.png",
            f"{full_name} selected parameters by region. Entries report the "
            "region-level tuned specification reused across rolling "
            "calibration years.",
            figsize=(12.5, 3.8),
            fontsize=7.8,
            row_height_scale=1.42,
            title=full_name,
            show_note=False,
        )


def main() -> None:
    ml = build_ml_table()
    ml_csv = OUTPUT_DIR / "csv_selected_parameters_ml_by_region.csv"
    ml.to_csv(ml_csv, index=False)
    print(f"  -> {ml_csv.name}")
    draw_table(
        ml, ML_CLASSICAL_COLUMNS,
        "T16a_selected_parameters_ml_classical_by_region.png",
        "Selected region-level ML specifications for XGB, SVR, RF, and KNN. "
        "Entries report grid/search-selected hyperparameters.",
        figsize=(12.5, 8.0),
        fontsize=8.0,
        row_height_scale=3.45,
    )
    draw_table(
        ml, ML_NEURAL_COLUMNS,
        "T16b_selected_parameters_ml_neural_by_region.png",
        "Selected region-level neural-network specifications. LSTM and FFNN "
        "entries are extracted from rolling export payloads; the tuned "
        "region-level parameters are reused across calibration years.",
        figsize=(10.5, 8.6),
        fontsize=8.0,
        row_height_scale=4.05,
    )
    draw_single_ml_tables(ml)

    benth = build_benth_table()
    benth_csv = OUTPUT_DIR / "csv_selected_parameters_benth_by_region.csv"
    benth.to_csv(benth_csv, index=False)
    print(f"  -> {benth_csv.name}")
    draw_booktabs_table(
        benth, list(benth.columns),
        "T17_selected_parameters_benth_by_region.png",
        figsize=(12.6, 3.35),
        fontsize=7.8,
        row_height_scale=1.42,
        header_label_map={
            "Log-likelihood": "Log-\nlikelihood",
            "Number of parameters": "Number of\nparameters",
        },
    )

    alaton = build_alaton_table()
    alaton_csv = OUTPUT_DIR / "csv_selected_parameters_alaton_by_region.csv"
    alaton.to_csv(alaton_csv, index=False)
    print(f"  -> {alaton_csv.name}")
    draw_booktabs_table(
        alaton, list(alaton.columns),
        "T18_selected_parameters_alaton_by_region.png",
        figsize=(12.9, 3.35),
        fontsize=7.8,
        row_height_scale=1.42,
        header_label_map={
            "Normality rejection share": "Normality\nrejection share",
        },
    )


if __name__ == "__main__":
    main()
