"""
Average winter soft wheat yield by NUTS-2 region.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    ROOT
    / "02_Data"
    / "02_Wheat"
    / "ble_tendre_hiver_yield_2000_2024_regions_mainland.csv"
)
OUTPUT_DIR = ROOT / "99_Thesis Graphs" / "02_Yield"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUTPUT_DIR / "yield_avg_by_region_ranked.png"

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

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df[df["reg"].isin(REGION_NAMES)].copy()
    df["region_name"] = df["reg"].map(REGION_NAMES)

    regional = (
        df.groupby("region_name", as_index=False)["yield"]
        .mean()
        .sort_values("yield", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.barh(
        regional["region_name"],
        regional["yield"],
        color="0.72",
        edgecolor="black",
        linewidth=0.7,
    )

    for bar, value in zip(bars, regional["yield"]):
        ax.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Average yield (q/ha)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="0.88", linewidth=0.5)
    ax.set_xlim(0, regional["yield"].max() * 1.12)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to: {OUT_PNG}")


if __name__ == "__main__":
    main()
