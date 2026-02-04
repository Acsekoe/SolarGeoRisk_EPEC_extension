"""
Unified plotting module for SolarGeoRisk EPEC extension.
Includes:
- Default result plots (bar charts, heatmaps)
- Price plots per region (2024 vs 2030)
- Capacity chord diagrams (triptych)
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Patch, PathPatch, Wedge, Circle
from matplotlib.path import Path as MplPath

# ==========================================================
# CONFIGURATION & CONSTANTS
# ==========================================================

# -- Excel Paths (for Price/Chord plots) --
# Default paths (can be overridden or are relative to workspace)
EXCEL_PATH_2024_LOW = r"outputs/2024/2024_low.xlsx"
EXCEL_PATH_2030_LOW = r"outputs/2030/2030_low.xlsx"
EXCEL_PATH_2030_HIGH = r"outputs/2030/2030_high.xlsx"

# -- Regions --
REGION_ORDER = ["ch", "eu", "us", "apac", "roa", "row"]
REGION_LABEL = {
    "ch": "CH", "eu": "EU", "us": "US", "apac": "APAC", "roa": "ROA", "row": "ROW"
}
DEST_ORDER = ["unused", "ch", "eu", "us", "apac", "roa", "row"]
APAC_MEMBERS = {"my", "vn", "in", "th", "kr"}

# -- Colors --
# Common palette
COLOR_CH = (38 / 255, 45 / 255, 99 / 255)
COLOR_EU = (20 / 255, 185 / 255, 220 / 255)      # 2024 baseline (equilibrium) / EU
COLOR_US = (245 / 255, 160 / 255, 78 / 255)      # 2030 player strategic / US
COLOR_APAC = (112 / 255, 196 / 255, 192 / 255)   # 2030 global welfare / APAC
COLOR_ROA = (100 / 255, 185 / 255, 133 / 255)
COLOR_ROW = (214 / 255, 90 / 255, 156 / 255)     # 2024 historical (data)
COLOR_UNUSED = (0.95, 0.95, 0.95)

COLORS: Dict[str, Tuple[float, float, float]] = {
    "ch": COLOR_CH,
    "eu": COLOR_EU,
    "us": COLOR_US,
    "apac": COLOR_APAC,
    "roa": COLOR_ROA,
    "row": COLOR_ROW,
    "unused": COLOR_UNUSED,
}

# Aliases for Price Plot specific usages (matching original script semantics)
COLOR_PRICE_2024_BASE = COLOR_EU
COLOR_PRICE_HISTORICAL = COLOR_ROW
COLOR_PRICE_2030_STRAT = COLOR_US
COLOR_PRICE_2030_WELFARE = COLOR_APAC

# -- Data --
BASELINE_ACTUAL_2024 = {
    "ch": 110.0,
    "eu": 140.0,
    "us": 315.0,
    "apac": (120 + 120 + 200 + 200 + 130) / 5.0,
    "roa": 165.0,
    "row": 130.0,
}

QCAP_EXIST_GW: Dict[str, Dict[str, float]] = {
    "2024": {"ch": 931.0, "eu": 22.0, "us": 23.0, "apac": 110.0, "roa": 0.0, "row": 293.0},
    "2030": {"ch": 1068.0, "eu": 25.0, "us": 26.4, "apac": 126.24, "roa": 0.0, "row": 336.36},
}

# -- Chord Plot Settings --
EXPORT_ALPHA = 0.45
UNUSED_ALPHA = 0.70
MIN_SHARE_OF_EXPORTER = 0.006
MIN_ABS_FLOW = 0.0
R_OUT = 1.0
RING_WIDTH = 0.16
R_IN = R_OUT - RING_WIDTH
EXP_START, EXP_END = 90.0, 270.0
DEST_START, DEST_END = 270.0, 450.0
GAP_DEG = 3.0
UNUSED_EXTRA_GAP = 6.0
DRAW_SEPARATOR = True
UNUSED_EDGE_COLOR = "0.45"
UNUSED_EDGE_LW = 1.4
UNUSED_EDGE_LS = (0, (4, 3))


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = list(df.columns)
    norm = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in norm:
            return norm[key]
        if c in cols:
            return c
    return None

def _read_sheet_last_iter(excel_path: str, sheet_name: str) -> pd.DataFrame:
    try:
        xls = pd.ExcelFile(excel_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
    if sheet_name not in xls.sheet_names:
        raise ValueError(f"Missing sheet '{sheet_name}' in {excel_path}. Sheets: {xls.sheet_names}")
    
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if "iter" in df.columns:
        it = pd.to_numeric(df["iter"], errors="coerce")
        if it.notna().any():
            df = df[it == it.max()].copy()
    return df

def _norm_code(x: object) -> str:
    s = str(x).strip().lower()
    return s.replace("_", "").replace(" ", "")

def _map_to_6(code: object) -> str:
    c = _norm_code(code)
    if c in APAC_MEMBERS:
        return "apac"
    if c in REGION_ORDER:
        return c
    return "row"

def _polar_xy(angle_deg: float, r: float) -> Tuple[float, float]:
    a = np.deg2rad(angle_deg)
    return (r * float(np.cos(a)), r * float(np.sin(a)))

def _arc_points(a0: float, a1: float, r: float, n: int = 18) -> list[Tuple[float, float]]:
    return [_polar_xy(a, r) for a in np.linspace(a0, a1, max(2, n))]

def _build_arc_spans(
    names: list[str],
    values: pd.Series,
    start_deg: float,
    end_deg: float,
    gap_deg: float,
    extra_gap_after: Dict[str, float] | None = None,
) -> Dict[str, Tuple[float, float]]:
    extra_gap_after = extra_gap_after or {}
    total_span = end_deg - start_deg
    base_gaps = gap_deg * (len(names) - 1) if len(names) > 1 else 0.0
    extra_gaps = sum(extra_gap_after.get(nm, 0.0) for nm in names[:-1])
    avail = max(0.0, total_span - base_gaps - extra_gaps)

    vals = np.maximum(np.array([values.get(nm, 0.0) for nm in names], float), 0.0)
    tot = float(vals.sum())

    spans: Dict[str, Tuple[float, float]] = {}
    a = start_deg
    for i, (nm, v) in enumerate(zip(names, vals)):
        d = 0.0 if tot <= 0 else avail * (v / tot)
        spans[nm] = (a, a + d)
        a += d
        if i < len(names) - 1:
            a += gap_deg + extra_gap_after.get(nm, 0.0)
    return spans

def _add_ribbon(ax, a0: float, a1: float, b0: float, b1: float, rgb, alpha: float) -> None:
    r_attach = R_IN - 0.010
    chord_len = abs((a0 + a1) / 2.0 - (b0 + b1) / 2.0)
    r_ctrl = 0.18 + 0.10 * (1.0 - np.exp(-chord_len / 70.0))

    p0, p1 = _polar_xy(a0, r_attach), _polar_xy(b0, r_attach)
    p2, p3 = _polar_xy(b1, r_attach), _polar_xy(a1, r_attach)
    c0, c1 = _polar_xy(a0, r_ctrl), _polar_xy(b0, r_ctrl)
    c2, c3 = _polar_xy(b1, r_ctrl), _polar_xy(a1, r_ctrl)

    verts = [p0]
    codes = [MplPath.MOVETO]
    verts += [c0, c1, p1]
    codes += [MplPath.CURVE4] * 3

    for pt in _arc_points(b0, b1, r_attach)[1:]:
        verts.append(pt)
        codes.append(MplPath.LINETO)

    verts += [c2, c3, p3]
    codes += [MplPath.CURVE4] * 3

    for pt in _arc_points(a1, a0, r_attach)[1:]:
        verts.append(pt)
        codes.append(MplPath.LINETO)

    verts.append((0.0, 0.0))
    codes.append(MplPath.CLOSEPOLY)

    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor=rgb,
            edgecolor=(0, 0, 0, 0.10),
            lw=0.25,
            alpha=alpha,
            zorder=2,
        )
    )

def _mid_angle(span: Tuple[float, float]) -> float:
    return 0.5 * (span[0] + span[1])

def _read_last_lambda(excel_path: str) -> pd.Series:
    regions = _read_sheet_last_iter(excel_path, "regions")
    
    region_col = _find_col(regions, ["region", "r", "name"])
    price_col = _find_col(regions, ["lam", "lambda", "price", "p", "nodal_price"])

    if region_col is None:
        # Fallback to index if no col found? Or just raise
        raise ValueError(f"Could not find region column. Columns: {list(regions.columns)}")
    if price_col is None:
        raise ValueError(f"Could not find price/lambda column. Columns: {list(regions.columns)}")

    regions[region_col] = regions[region_col].astype(str).str.strip().str.lower()

    lam = (
        regions[[region_col, price_col]]
        .groupby(region_col)[price_col]
        .mean()
        .reindex(REGION_ORDER)
        .astype(float)
    )
    return lam

def _load_flows_6x6(excel_path: str) -> pd.DataFrame:
    flows = _read_sheet_last_iter(excel_path, "flows")
    exp_col = _find_col(flows, ["exp", "e", "from"])
    imp_col = _find_col(flows, ["imp", "i", "to"])
    x_col = _find_col(flows, ["x"])
    if exp_col is None or imp_col is None or x_col is None:
        raise ValueError(f"'flows' missing exp/imp/x columns. Columns: {list(flows.columns)}")

    tmp = flows[[exp_col, imp_col, x_col]].copy()
    tmp[exp_col] = tmp[exp_col].map(_map_to_6)
    tmp[imp_col] = tmp[imp_col].map(_map_to_6)
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce").fillna(0.0)
    tmp = tmp.groupby([exp_col, imp_col], as_index=False)[x_col].sum()
    return tmp.rename(columns={exp_col: "exp", imp_col: "imp", x_col: "x"})

# ==========================================================
# EXISTING FUNCTIONALITY (for run_gs.py)
# ==========================================================

def write_default_plots(*, output_path: str, plots_dir: str) -> None:
    """
    Standard simple plots generated at the end of a run.
    """
    os.makedirs(plots_dir, exist_ok=True)

    try:
        df_regions = pd.read_excel(output_path, sheet_name="regions")
        df_flows = pd.read_excel(output_path, sheet_name="flows")
    except Exception as exc:
        print(f"[PLOT_WARN] could not read results workbook: {exc}")
        return

    # Normalize columns
    df_regions.columns = [str(c).strip() for c in df_regions.columns]
    
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


# ==========================================================
# NEW PLOTTING FUNCTIONS
# ==========================================================

def plot_prices(output_folder: str = "plots") -> None:
    """
    Generates the 'price_plot.png' comparing 2024 baseline vs 2030 scenarios.
    """
    print("Generating Price Plot...")
    os.makedirs(output_folder, exist_ok=True)
    
    # Check files
    for p in [EXCEL_PATH_2024_LOW, EXCEL_PATH_2030_LOW, EXCEL_PATH_2030_HIGH]:
        if not os.path.exists(p):
            print(f"[WARN] Price plot skipped. Missing file: {p}")
            return

    try:
        eq_2024 = _read_last_lambda(EXCEL_PATH_2024_LOW).values.astype(float)
        eq_2030_global_welfare = _read_last_lambda(EXCEL_PATH_2030_LOW).values.astype(float)
        eq_2030_player_strategic = _read_last_lambda(EXCEL_PATH_2030_HIGH).values.astype(float)
    except Exception as e:
        print(f"[WARN] Failed to read data for price plot: {e}")
        return

    x = np.arange(len(REGION_ORDER))
    labels = [REGION_LABEL[r] for r in REGION_ORDER]
    actual_vals = np.array([BASELINE_ACTUAL_2024[r] for r in REGION_ORDER], dtype=float)

    # Style/Layout
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 13,
    })

    fig, ax = plt.subplots(figsize=(7.6, 8.4))

    x_scale = 1.3
    x_plot = x * x_scale

    # Range hug bar
    all_vals = np.vstack([actual_vals, eq_2024, eq_2030_global_welfare, eq_2030_player_strategic])
    ymins = all_vals.min(axis=0)
    ymaxs = all_vals.max(axis=0)
    
    bar_w = 0.17
    for i in range(len(x_plot)):
        h = float(ymaxs[i] - ymins[i])
        if h <= 0: continue
        patch = FancyBboxPatch(
            (x_plot[i] - bar_w / 2, float(ymins[i])), bar_w, h,
            boxstyle="round,pad=0.0", mutation_scale=10, linewidth=0,
            facecolor=(0.3, 0.3, 0.3), alpha=0.22, zorder=1
        )
        ax.add_patch(patch)

    # Markers
    h_actual = ax.scatter(x_plot, actual_vals, marker="o", s=100, color=COLOR_PRICE_HISTORICAL, label="_nolegend_", zorder=3)
    h_2024 = ax.scatter(x_plot, eq_2024, marker="^", s=130, color=COLOR_PRICE_2024_BASE, label="_nolegend_", zorder=3)
    h_2030_gw = ax.scatter(x_plot, eq_2030_global_welfare, marker="D", s=115, color=COLOR_PRICE_2030_WELFARE, label="_nolegend_", zorder=3)
    h_2030_ps = ax.scatter(x_plot, eq_2030_player_strategic, marker="*", s=170, color=COLOR_PRICE_2030_STRAT, label="_nolegend_", zorder=3)

    ax.set_xticks(x_plot)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PV module price (USD/kW)")
    ax.set_ylim(100, 320)
    ax.grid(True, axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.set_xlim(x_plot.min() - 0.55, x_plot.max() + 0.55)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.tick_params(axis="both", width=1.8, length=7)

    fp = FontProperties(weight="normal")
    tp = FontProperties(weight="normal")

    leg2024 = ax.legend(
        handles=[h_actual, h_2024],
        labels=["Historical (data)", "Baseline (model)"],
        title="2024",
        loc="upper center",
        bbox_to_anchor=(0.28, -0.12),
        ncol=1,
        frameon=True, fancybox=True, framealpha=0.95, edgecolor=(0.2, 0.2, 0.2),
        handletextpad=0.8, borderpad=0.8, prop=fp,
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
        frameon=True, fancybox=True, framealpha=0.95, edgecolor=(0.2, 0.2, 0.2),
        handletextpad=0.8, borderpad=0.8, prop=fp,
    )
    leg2030.set_title("2030", prop=tp)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    out_path = os.path.join(output_folder, "price_plot.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _draw_chord_panel(ax: plt.Axes, excel_path: str, year: str) -> None:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(excel_path)

    cap_gw = QCAP_EXIST_GW.get(year, {})
    cap = pd.Series(cap_gw).reindex(REGION_ORDER).fillna(0.0)

    left_exporters = [r for r in REGION_ORDER if float(cap.get(r, 0.0)) > 0.0]
    cap_left = cap.reindex(left_exporters).fillna(0.0)

    try:
        flows_all = _load_flows_6x6(excel_path)
    except Exception as e:
        print(f"[WARN] Could not load flows for {year}: {e}")
        return

    flows_all["x"] = pd.to_numeric(flows_all["x"], errors="coerce").fillna(0.0)
    flows_all = flows_all[flows_all["x"] > 0].copy()

    used_by_exp = flows_all.groupby("exp")["x"].sum().reindex(left_exporters).fillna(0.0)
    flows_plot = flows_all.copy()

    unused_rows = []
    for r in left_exporters:
        unused = max(0.0, float(cap_left[r]) - float(used_by_exp.get(r, 0.0)))
        if unused > 0:
            unused_rows.append({"exp": r, "imp": "unused", "x": unused})
    if unused_rows:
        flows_plot = pd.concat([flows_plot, pd.DataFrame(unused_rows)], ignore_index=True)

    flows_plot["share_exp"] = flows_plot.apply(
        lambda row: (float(row["x"]) / float(cap.get(row["exp"], 0.0)))
        if float(cap.get(row["exp"], 0.0)) > 0.0
        else 0.0,
        axis=1,
    )
    keep = (flows_plot["imp"] == "unused") | (flows_plot["share_exp"] >= MIN_SHARE_OF_EXPORTER)
    flows_plot = flows_plot[(flows_plot["x"] >= MIN_ABS_FLOW) & keep].copy()

    recv = flows_plot.groupby("imp")["x"].sum().reindex(DEST_ORDER).fillna(0.0)

    exp_spans = _build_arc_spans(left_exporters, cap_left, EXP_START, EXP_END, GAP_DEG)
    dest_spans = _build_arc_spans(DEST_ORDER, recv, DEST_START, DEST_END, GAP_DEG, {"unused": UNUSED_EXTRA_GAP})

    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.add_patch(Circle((0, 0), R_OUT, facecolor="none", edgecolor="0.35", lw=0.9))
    if DRAW_SEPARATOR:
        ax.plot([0, 0], [-1.06, 1.06], linestyle=(0, (6, 6)), lw=1.0, color="0.35")

    for r in left_exporters:
        a0, a1 = exp_spans[r]
        if a1 <= a0 + 1e-9: continue
        ax.add_patch(Wedge((0, 0), R_OUT, a0, a1, width=RING_WIDTH, facecolor=COLORS[r], edgecolor="white", lw=0.9, zorder=3))

    for d in DEST_ORDER:
        b0, b1 = dest_spans[d]
        if b1 <= b0 + 1e-9: continue
        if d == "unused":
            ax.add_patch(Wedge(
                (0, 0), R_OUT, b0, b1, width=RING_WIDTH,
                facecolor=COLORS[d], edgecolor=UNUSED_EDGE_COLOR,
                lw=UNUSED_EDGE_LW, linestyle=UNUSED_EDGE_LS, zorder=3
            ))
        else:
            ax.add_patch(Wedge(
                (0, 0), R_OUT, b0, b1, width=RING_WIDTH,
                facecolor=COLORS[d], edgecolor="white", lw=0.9, zorder=3
            ))

    exp_cursor = {r: exp_spans[r][0] for r in left_exporters}
    dest_cursor = {d: dest_spans[d][0] for d in DEST_ORDER}
    recv_safe = recv.reindex(DEST_ORDER).fillna(0.0)

    flows_non_unused = flows_plot[flows_plot["imp"] != "unused"].sort_values("x", ascending=False).copy()
    flows_unused = flows_plot[flows_plot["imp"] == "unused"].copy()
    exp_mid_map = {r: _mid_angle(exp_spans[r]) for r in left_exporters}
    flows_unused["exp_mid"] = flows_unused["exp"].map(exp_mid_map).fillna(0.0)
    flows_unused = flows_unused.sort_values("exp_mid", ascending=True)

    def draw_flow(exp: str, dest: str, v: float) -> None:
        if exp not in exp_spans or dest not in DEST_ORDER: return
        if float(cap.get(exp, 0.0)) <= 0.0 or float(recv_safe.get(dest, 0.0)) <= 0.0 or v <= 0.0: return

        exp_span = exp_spans[exp][1] - exp_spans[exp][0]
        dest_span = dest_spans[dest][1] - dest_spans[dest][0]
        if exp_span <= 1e-9 or dest_span <= 1e-9: return

        da = exp_span * (v / float(cap.get(exp, 0.0)))
        db = dest_span * (v / float(recv_safe.get(dest, 0.0)))

        a0 = exp_cursor[exp]
        b0 = dest_cursor[dest]
        a1 = min(exp_spans[exp][1], a0 + da)
        b1 = min(dest_spans[dest][1], b0 + db)
        exp_cursor[exp] = a1
        dest_cursor[dest] = b1

        if a1 <= a0 + 1e-6 or b1 <= b0 + 1e-6: return
        alpha = UNUSED_ALPHA if dest == "unused" else EXPORT_ALPHA
        _add_ribbon(ax, a0, a1, b0, b1, COLORS[exp], alpha)

    for _, rr in flows_non_unused.iterrows():
        draw_flow(str(rr["exp"]), str(rr["imp"]), float(rr["x"]))
    for _, rr in flows_unused.iterrows():
        draw_flow(str(rr["exp"]), "unused", float(rr["x"]))

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)


def plot_capacity_chord(output_folder: str = "plots") -> None:
    """
    Generates the 'capacity_allocation_triptych.png' chord diagram.
    """
    print("Generating Capacity Chord Plot...")
    os.makedirs(output_folder, exist_ok=True)

    scenarios = [
        ("2024", EXCEL_PATH_2024_LOW),
        ("2030", EXCEL_PATH_2030_LOW),
        ("2030", EXCEL_PATH_2030_HIGH),
    ]
    
    # Pre-check for existence
    valid_scenarios = []
    for s in scenarios:
        if os.path.exists(s[1]):
            valid_scenarios.append(s)
        else:
            print(f"[WARN] Skipped chord panel for {s[0]}, missing {s[1]}")
    
    if not valid_scenarios:
        print("[WARN] No valid data found for chord plots. Skipping.")
        return

    # Original layout assumes 3 panels. If files missing, this might look weird, 
    # but we'll try to stick to the requested triptych if possible, or just fail gracefully.
    # If all 3 are missing we returned above. If 1 is missing it might crash or be empty.
    # Let's assume strict requirement for triptych for now.
    
    if len(valid_scenarios) < 3:
        print("[WARN] Need 3 valid input files for the triptych. Found fewer.")
        # Proceed anyway? Let's try, code might just produce empty panels if not handled.
        # But _draw_chord_panel checks existence, so it's fine. 
    
    # Re-verify strictly for the 3-panel layout
    if not all(os.path.exists(p) for _, p in scenarios):
         print("[WARN] Some files missing. Chord plot may be incomplete.")


    panel_headers = ["2024 Baseline", "2030 Global welfare", "2030 Player strategic"]
    LEGEND_FONTSIZE = 13
    TITLE_FONTSIZE = LEGEND_FONTSIZE

    fig = plt.figure(figsize=(14.8, 5.6))
    ax_left = fig.add_axes([0.02, 0.26, 0.31, 0.68])
    ax_mid  = fig.add_axes([0.345, 0.26, 0.31, 0.68])
    ax_right= fig.add_axes([0.67, 0.26, 0.31, 0.68])
    axes = [ax_left, ax_mid, ax_right]

    for ax, (year, path), header in zip(axes, scenarios, panel_headers):
        if os.path.exists(path):
            try:
                _draw_chord_panel(ax, path, year)
            except Exception as e:
                print(f"Error drawing panel {header}: {e}")
        
        ax.text(
            0.5, 0.985, header, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=TITLE_FONTSIZE,
            fontweight="normal", color="0.15", clip_on=False
        )

    # Legend
    handles: List[Patch] = []
    labels: List[str] = []
    for r in ["ch", "eu", "us", "apac", "roa", "row"]:
        handles.append(Patch(facecolor=COLORS[r], edgecolor="white", linewidth=0.8))
        labels.append(r.upper())

    handles.append(Patch(
        facecolor=COLORS["unused"], edgecolor=UNUSED_EDGE_COLOR,
        linewidth=UNUSED_EDGE_LW, linestyle=UNUSED_EDGE_LS
    ))
    labels.append("UNUSED CAPACITY")

    leg = fig.legend(
        handles, labels, loc="lower center", ncol=len(labels),
        frameon=True, fancybox=True, edgecolor="0.55",
        bbox_to_anchor=(0.5, 0.10), handlelength=1.8, columnspacing=1.15,
        fontsize=LEGEND_FONTSIZE,
    )
    leg.get_frame().set_linewidth(0.9)

    out_path = os.path.join(output_folder, "capacity_allocation_triptych.png")
    fig.savefig(out_path, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # If run directly, try to generate the ad-hoc plots
    # assuming we are in the project root or src folder
    plot_prices()
    plot_capacity_chord()
