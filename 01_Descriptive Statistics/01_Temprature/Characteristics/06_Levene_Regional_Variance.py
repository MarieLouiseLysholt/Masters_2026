"""
Levene test for equality of daily temperature variance across NUTS-2 regions.

Uses the regional average daily temperature series from region_avg.csv:
    daily_avg_temperature, sample from 1980-01-01 onward.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import levene


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "02_Data" / "01_Temprature" / "region_avg_exendted.csv"
OUTPUT_DIR = ROOT / "99_Thesis Graphs" / "04_Data Tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "Levene_Regional_Variance.csv"
OUT_PNG = OUTPUT_DIR / "Table_Levene_Regional_Variance.png"

TEMP_COL = "daily_avg_temperature"
START_DATE = "1980-01-01"
ALPHA = 0.05

REGION_NAMES = {
    11: "Ile-de-France",
    24: "Centre-Val de Loire",
    27: "Bourgogne-Franche-Comte",
    28: "Normandie",
    32: "Hauts-de-France",
    44: "Grand Est",
    52: "Pays de la Loire",
    53: "Bretagne",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _fmt_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "-"
    return "< 0.0001" if p_value < 1e-4 else f"{p_value:.4f}"


def draw_table(summary: pd.DataFrame, stat: float, p_value: float) -> None:
    headers = ["Region", "N", "Std. dev.", "Variance"]
    cell_text = []
    for _, row in summary.iterrows():
        cell_text.append([
            row["Region"],
            f"{int(row['N']):,}",
            f"{row['Std. dev.']:.3f}",
            f"{row['Variance']:.3f}",
        ])

    decision = "Reject equal variances" if p_value < ALPHA else "Do not reject"
    footnote = (
        f"Levene test, center=median: W = {stat:.3f}, p = {_fmt_p(p_value)}. "
        f"H0: equal regional variances. Decision at 5%: {decision}."
    )

    fig, ax = plt.subplots(figsize=(8.4, 0.55 + 0.34 * len(cell_text)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.42, 0.16, 0.20, 0.20],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1.0, 1.45)

    n_rows = len(cell_text)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.75)
        cell.visible_edges = ""
        if r == 0:
            cell.visible_edges = "TB"
            cell.set_text_props(fontfamily="Times New Roman")
        else:
            if r == n_rows:
                cell.visible_edges = "B"
            ha = "left" if c == 0 else "right"
            cell.set_text_props(ha=ha, fontfamily="Times New Roman")

    fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.1)
    fig.savefig(OUT_PNG, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df[df["date"] >= START_DATE].copy()
    df = df[df["region_code"].isin(REGION_NAMES)].copy()
    df = df.dropna(subset=[TEMP_COL])

    groups = []
    rows = []
    for code, name in REGION_NAMES.items():
        vals = df.loc[df["region_code"] == code, TEMP_COL].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        groups.append(vals)
        rows.append({
            "Region": name,
            "N": int(len(vals)),
            "Std. dev.": float(np.std(vals, ddof=1)),
            "Variance": float(np.var(vals, ddof=1)),
        })

    stat, p_value = levene(*groups, center="median")
    summary = pd.DataFrame(rows)
    summary["Levene_W"] = float(stat)
    summary["Levene_p"] = float(p_value)
    summary["Decision_5pct"] = (
        "Reject equal variances" if p_value < ALPHA else "Do not reject"
    )

    summary.to_csv(OUT_CSV, index=False)
    draw_table(summary, float(stat), float(p_value))

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
