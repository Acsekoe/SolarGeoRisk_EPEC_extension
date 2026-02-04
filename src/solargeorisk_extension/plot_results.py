from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def write_default_plots(*, output_path: str, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)

    try:
        df_regions = pd.read_excel(output_path, sheet_name="regions")
        df_flows = pd.read_excel(output_path, sheet_name="flows")
    except Exception as exc:
        print(f"[PLOT_WARN] could not read results workbook: {exc}")
        return

    if {"r", "Q_offer", "Qcap"}.issubset(df_regions.columns):
        df = df_regions.copy().sort_values("r")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df["r"], df["Qcap"], label="Qcap", alpha=0.4)
        ax.bar(df["r"], df["Q_offer"], label="Q_offer", alpha=0.9)
        ax.set_title("Capacity offer by region")
        ax.set_ylabel("GW")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "q_offer.png"), dpi=150)
        plt.close(fig)

    if {"r", "x_dem", "lam"}.issubset(df_regions.columns):
        df = df_regions.copy().sort_values("r")
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(df["r"], df["x_dem"], color="tab:blue", alpha=0.8)
        ax1.set_ylabel("x_dem (GW)")
        ax2 = ax1.twinx()
        ax2.plot(df["r"], df["lam"], color="tab:red", marker="o")
        ax2.set_ylabel("lam (price)")
        ax1.set_title("Demand and price by region")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "demand_price.png"), dpi=150)
        plt.close(fig)

    if {"exp", "imp", "x"}.issubset(df_flows.columns):
        pivot = df_flows.pivot(index="exp", columns="imp", values="x")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Flows x (GW)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "flows_heatmap.png"), dpi=150)
        plt.close(fig)

