"""
Three-panel chord plot (no region labels on arcs):

Layout:
- THREE circles next to each other in ONE LINE (left -> right)

Panels (as placed):
LEFT:   2024 baseline (model)
MIDDLE: 2030 global welfare
RIGHT:  2030 player strategic

This version:
- Titles are tight above their corresponding circles.
- Titles are NON-BOLD and same size as legend text.
- Legend is ONE LINE centered under the circles.

Rules (unchanged):
- LEFT half uses exogenous Qcap_exist (GW) for that year.
- RIGHT half shows served demand destinations + UNUSED CAPACITY.
- UNUSED ribbon per exporter = Qcap_exist(exp) − total shipments(exp).
- If installed capacity = 0, DO NOT display the region on the LEFT side
  (no exporter arc, no unused ribbon from that exporter). It may still appear on RIGHT as destination.
- Remove ALL region labels around the circle.
- UNUSED wedge distinct: light fill + dashed grey outline.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, PathPatch, Patch
from matplotlib.path import Path as MplPath

# ==========================================================
# CHANGE ONLY THESE (if needed):
# ==========================================================
EXCEL_PATH_2024_LOW = r"outputs/2024/2024_low.xlsx"
EXCEL_PATH_2030_LOW = r"outputs/2030/2030_low.xlsx"
EXCEL_PATH_2030_HIGH = r"outputs/2030/2030_high.xlsx"
# ==========================================================

REGION_ORDER = ["ch", "eu", "us", "apac", "roa", "row"]
DEST_ORDER = ["unused", "ch", "eu", "us", "apac", "roa", "row"]
APAC_MEMBERS = {"my", "vn", "in", "th", "kr"}

QCAP_EXIST_GW: Dict[str, Dict[str, float]] = {
    "2024": {"ch": 931.0, "eu": 22.0, "us": 23.0, "apac": 110.0, "roa": 0.0, "row": 293.0},
    "2030": {"ch": 1068.0, "eu": 25.0, "us": 26.4, "apac": 126.24, "roa": 0.0, "row": 336.36},
}

COLORS: Dict[str, Tuple[float, float, float]] = {
    "ch": (38 / 255, 45 / 255, 99 / 255),
    "eu": (20 / 255, 185 / 255, 220 / 255),
    "us": (245 / 255, 160 / 255, 78 / 255),
    "apac": (112 / 255, 196 / 255, 192 / 255),
    "roa": (100 / 255, 185 / 255, 133 / 255),
    "row": (214 / 255, 90 / 255, 156 / 255),
    "unused": (0.95, 0.95, 0.95),
}

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


def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = list(df.columns)
    norm = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in norm:
            return norm[key]
        if c in cols:
            return c
    return None


def read_sheet_last_iter(excel_path: str, sheet_name: str) -> pd.DataFrame:
    xls = pd.ExcelFile(excel_path)
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


def load_flows_6x6(excel_path: str) -> pd.DataFrame:
    flows = read_sheet_last_iter(excel_path, "flows")
    exp_col = find_col(flows, ["exp", "e", "from"])
    imp_col = find_col(flows, ["imp", "i", "to"])
    x_col = find_col(flows, ["x"])
    if exp_col is None or imp_col is None or x_col is None:
        raise ValueError(f"'flows' missing exp/imp/x columns. Columns: {list(flows.columns)}")

    tmp = flows[[exp_col, imp_col, x_col]].copy()
    tmp[exp_col] = tmp[exp_col].map(_map_to_6)
    tmp[imp_col] = tmp[imp_col].map(_map_to_6)
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce").fillna(0.0)
    tmp = tmp.groupby([exp_col, imp_col], as_index=False)[x_col].sum()
    return tmp.rename(columns={exp_col: "exp", imp_col: "imp", x_col: "x"})


def _polar_xy(angle_deg: float, r: float) -> Tuple[float, float]:
    a = np.deg2rad(angle_deg)
    return (r * float(np.cos(a)), r * float(np.sin(a)))


def _arc_points(a0: float, a1: float, r: float, n: int = 18) -> list[Tuple[float, float]]:
    return [_polar_xy(a, r) for a in np.linspace(a0, a1, max(2, n))]


def build_arc_spans(
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


def add_ribbon(ax, a0: float, a1: float, b0: float, b1: float, rgb, alpha: float) -> None:
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


def mid_angle(span: Tuple[float, float]) -> float:
    return 0.5 * (span[0] + span[1])


def draw_chord_panel(ax: plt.Axes, excel_path: str, year: str) -> None:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(excel_path)

    cap = pd.Series(QCAP_EXIST_GW[year]).reindex(REGION_ORDER).fillna(0.0)

    left_exporters = [r for r in REGION_ORDER if float(cap.get(r, 0.0)) > 0.0]
    cap_left = cap.reindex(left_exporters).fillna(0.0)

    flows_all = load_flows_6x6(excel_path)
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

    exp_spans = build_arc_spans(left_exporters, cap_left, EXP_START, EXP_END, GAP_DEG)
    dest_spans = build_arc_spans(DEST_ORDER, recv, DEST_START, DEST_END, GAP_DEG, {"unused": UNUSED_EXTRA_GAP})

    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.add_patch(Circle((0, 0), R_OUT, facecolor="none", edgecolor="0.35", lw=0.9))
    if DRAW_SEPARATOR:
        ax.plot([0, 0], [-1.06, 1.06], linestyle=(0, (6, 6)), lw=1.0, color="0.35")

    for r in left_exporters:
        a0, a1 = exp_spans[r]
        if a1 <= a0 + 1e-9:
            continue
        ax.add_patch(
            Wedge((0, 0), R_OUT, a0, a1, width=RING_WIDTH, facecolor=COLORS[r], edgecolor="white", lw=0.9, zorder=3)
        )

    for d in DEST_ORDER:
        b0, b1 = dest_spans[d]
        if b1 <= b0 + 1e-9:
            continue
        if d == "unused":
            ax.add_patch(
                Wedge(
                    (0, 0),
                    R_OUT,
                    b0,
                    b1,
                    width=RING_WIDTH,
                    facecolor=COLORS[d],
                    edgecolor=UNUSED_EDGE_COLOR,
                    lw=UNUSED_EDGE_LW,
                    linestyle=UNUSED_EDGE_LS,
                    zorder=3,
                )
            )
        else:
            ax.add_patch(
                Wedge((0, 0), R_OUT, b0, b1, width=RING_WIDTH, facecolor=COLORS[d], edgecolor="white", lw=0.9, zorder=3)
            )

    exp_cursor = {r: exp_spans[r][0] for r in left_exporters}
    dest_cursor = {d: dest_spans[d][0] for d in DEST_ORDER}
    recv_safe = recv.reindex(DEST_ORDER).fillna(0.0)

    flows_non_unused = flows_plot[flows_plot["imp"] != "unused"].sort_values("x", ascending=False).copy()
    flows_unused = flows_plot[flows_plot["imp"] == "unused"].copy()
    exp_mid_map = {r: mid_angle(exp_spans[r]) for r in left_exporters}
    flows_unused["exp_mid"] = flows_unused["exp"].map(exp_mid_map).fillna(0.0)
    flows_unused = flows_unused.sort_values("exp_mid", ascending=True)

    def draw_flow(exp: str, dest: str, v: float) -> None:
        if exp not in exp_spans or dest not in DEST_ORDER:
            return
        if float(cap.get(exp, 0.0)) <= 0.0 or float(recv_safe.get(dest, 0.0)) <= 0.0 or v <= 0.0:
            return

        exp_span = exp_spans[exp][1] - exp_spans[exp][0]
        dest_span = dest_spans[dest][1] - dest_spans[dest][0]
        if exp_span <= 1e-9 or dest_span <= 1e-9:
            return

        da = exp_span * (v / float(cap.get(exp, 0.0)))
        db = dest_span * (v / float(recv_safe.get(dest, 0.0)))

        a0 = exp_cursor[exp]
        b0 = dest_cursor[dest]
        a1 = min(exp_spans[exp][1], a0 + da)
        b1 = min(dest_spans[dest][1], b0 + db)
        exp_cursor[exp] = a1
        dest_cursor[dest] = b1

        if a1 <= a0 + 1e-6 or b1 <= b0 + 1e-6:
            return

        alpha = UNUSED_ALPHA if dest == "unused" else EXPORT_ALPHA
        add_ribbon(ax, a0, a1, b0, b1, COLORS[exp], alpha)

    for _, rr in flows_non_unused.iterrows():
        draw_flow(str(rr["exp"]), str(rr["imp"]), float(rr["x"]))
    for _, rr in flows_unused.iterrows():
        draw_flow(str(rr["exp"]), "unused", float(rr["x"]))

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)


def main() -> None:
    scenarios = [
        ("2024", EXCEL_PATH_2024_LOW),
        ("2030", EXCEL_PATH_2030_LOW),
        ("2030", EXCEL_PATH_2030_HIGH),
    ]

    # Titles like the previous plot, including years, non-bold
    panel_headers = [
        "2024 Baseline",
        "2030 Global welfare",
        "2030 Player strategic",
    ]

    # Match title size to legend text size
    LEGEND_FONTSIZE = 13
    TITLE_FONTSIZE = LEGEND_FONTSIZE

    fig = plt.figure(figsize=(14.8, 5.6))

    # Three axes in one row, leave bottom margin for 1-line legend
    ax_left = fig.add_axes([0.02, 0.26, 0.31, 0.68])
    ax_mid  = fig.add_axes([0.345, 0.26, 0.31, 0.68])
    ax_right= fig.add_axes([0.67, 0.26, 0.31, 0.68])

    axes = [ax_left, ax_mid, ax_right]

    for ax, (year, path), header in zip(axes, scenarios, panel_headers):
        draw_chord_panel(ax, path, year)

        # Tight title above each circle (non-bold)
        ax.text(
            0.5,
            0.985,
            header,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=TITLE_FONTSIZE,
            fontweight="normal",
            color="0.15",
            clip_on=False,
        )

    # --- One-line legend under all circles ---
    handles: List[Patch] = []
    labels: List[str] = []
    for r in ["ch", "eu", "us", "apac", "roa", "row"]:
        handles.append(Patch(facecolor=COLORS[r], edgecolor="white", linewidth=0.8))
        labels.append(r.upper())

    handles.append(
        Patch(
            facecolor=COLORS["unused"],
            edgecolor=UNUSED_EDGE_COLOR,
            linewidth=UNUSED_EDGE_LW,
            linestyle=UNUSED_EDGE_LS,
        )
    )
    labels.append("UNUSED CAPACITY")

    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),             # ONE LINE
        frameon=True,
        fancybox=True,
        edgecolor="0.55",
        bbox_to_anchor=(0.5, 0.10),
        handlelength=1.8,
        columnspacing=1.15,
        fontsize=LEGEND_FONTSIZE,
    )
    leg.get_frame().set_linewidth(0.9)

    out_path = "capacity_allocation_triptych.png"
    fig.savefig(out_path, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
