import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

# ==========================================================
# CHANGE ONLY THESE:
# ==========================================================
EXCEL_PATH_2024_LOWTAU = r"outputs/2024/2024_low.xlsx"
EXCEL_PATH_2030_LOWTAU = r"outputs/2030/2030_low.xlsx"
EXCEL_PATH_2030_HIGHTAU = r"outputs/2030/2030_high.xlsx"
# ==========================================================

Y_MIN, Y_MAX = 100, 320
REGION_ORDER = ["ch", "eu", "us", "apac", "roa", "row"]
REGION_LABEL = {"ch": "CH", "eu": "EU", "us": "US", "apac": "APAC", "roa": "ROA", "row": "ROW"}

baseline_actual_2024 = {
    "ch": 110,
    "eu": 140,
    "us": 315,
    "apac": (120 + 120 + 200 + 200 + 130) / 5,
    "roa": 165,
    "row": 130,
}

# Swatch colors
BLUE = (20 / 255, 185 / 255, 220 / 255)        # 2024 baseline (equilibrium)
MAGENTA = (214 / 255, 90 / 255, 156 / 255)     # 2024 historical (data)
ORANGE = (245 / 255, 160 / 255, 78 / 255)      # 2030 player strategic
TEAL = (112 / 255, 196 / 255, 192 / 255)       # 2030 global welfare


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_last_lambda(excel_path: str) -> pd.Series:
    regions = pd.read_excel(excel_path, sheet_name="regions")

    region_col = _find_col(regions, ["region", "r", "name"])
    price_col = _find_col(regions, ["lam", "lambda", "price", "p", "nodal_price"])

    if region_col is None:
        raise ValueError(f"Could not find region column. Columns: {list(regions.columns)}")
    if price_col is None:
        raise ValueError(f"Could not find price/lambda column. Columns: {list(regions.columns)}")

    if "iter" in regions.columns:
        regions = regions[regions["iter"] == regions["iter"].max()].copy()

    regions[region_col] = regions[region_col].astype(str).str.strip().str.lower()

    lam = (
        regions[[region_col, price_col]]
        .groupby(region_col)[price_col]
        .mean()
        .reindex(REGION_ORDER)
        .astype(float)
    )
    return lam


def main() -> None:
    for p in [EXCEL_PATH_2024_LOWTAU, EXCEL_PATH_2030_LOWTAU, EXCEL_PATH_2030_HIGHTAU]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Excel file not found: {p}")

    eq_2024 = read_last_lambda(EXCEL_PATH_2024_LOWTAU).values.astype(float)
    eq_2030_global_welfare = read_last_lambda(EXCEL_PATH_2030_LOWTAU).values.astype(float)
    eq_2030_player_strategic = read_last_lambda(EXCEL_PATH_2030_HIGHTAU).values.astype(float)

    x = np.arange(len(REGION_ORDER))
    labels = [REGION_LABEL[r] for r in REGION_ORDER]
    actual_vals = np.array([baseline_actual_2024[r] for r in REGION_ORDER], dtype=float)

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 8.4))

    # X spacing
    x_scale = 1.3
    x_plot = x * x_scale

    # range "hug" bar (rounded, grey, transparent)
    all_vals = np.vstack([actual_vals, eq_2024, eq_2030_global_welfare, eq_2030_player_strategic])
    ymins = all_vals.min(axis=0)
    ymaxs = all_vals.max(axis=0)

    bar_w = 0.17
    for i in range(len(x_plot)):
        h = float(ymaxs[i] - ymins[i])
        if h <= 0:
            continue
        patch = FancyBboxPatch(
            (x_plot[i] - bar_w / 2, float(ymins[i])),
            bar_w,
            h,
            boxstyle="round,pad=0.0",
            mutation_scale=10,
            linewidth=0,
            facecolor=(0.3, 0.3, 0.3),
            alpha=0.22,
            zorder=1,
        )
        ax.add_patch(patch)

    # markers (exclude from auto-legend; we'll build custom legends)
    h_actual = ax.scatter(
        x_plot, actual_vals, marker="o", s=100, color=MAGENTA, label="_nolegend_", zorder=3
    )
    h_2024 = ax.scatter(
        x_plot, eq_2024, marker="^", s=130, color=BLUE, label="_nolegend_", zorder=3
    )
    h_2030_gw = ax.scatter(
        x_plot, eq_2030_global_welfare, marker="D", s=115, color=TEAL, label="_nolegend_", zorder=3
    )
    h_2030_ps = ax.scatter(
        x_plot, eq_2030_player_strategic, marker="*", s=170, color=ORANGE, label="_nolegend_", zorder=3
    )

    ax.set_xticks(x_plot)
    ax.set_xticklabels(labels)  # non-bold
    ax.set_ylabel("PV module price (USD/kW)")  # non-bold
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(True, axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.set_xlim(x_plot.min() - 0.55, x_plot.max() + 0.55)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.tick_params(axis="both", width=1.8, length=7)

    # --- Two-column legend with headers (2024 / 2030), normal weight text ---
    fp = FontProperties(weight="normal")
    tp = FontProperties(weight="normal")

    leg2024 = ax.legend(
        handles=[h_actual, h_2024],
        labels=["Historical (data)", "Baseline (model)"],
        title="2024",
        loc="upper center",
        bbox_to_anchor=(0.28, -0.12),
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor=(0.2, 0.2, 0.2),
        handletextpad=0.8,
        borderpad=0.8,
        prop=fp,
    )
    leg2024.set_title("2024", prop=tp)
    ax.add_artist(leg2024)

    leg2030 = ax.legend(
        handles=[h_2030_gw, h_2030_ps],
        labels=["Global welfare", "Player strategic"],
        title="2030",
        loc="upper center",
        bbox_to_anchor=(0.72, -0.12),
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor=(0.2, 0.2, 0.2),
        handletextpad=0.8,
        borderpad=0.8,
        prop=fp,
    )
    leg2030.set_title("2030", prop=tp)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out_dir = os.path.dirname(os.path.abspath(EXCEL_PATH_2024_LOWTAU)) or "."
    out_path = os.path.join(out_dir, "price_plot.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
