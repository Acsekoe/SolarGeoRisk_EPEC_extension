from __future__ import annotations

import os
from typing import Dict, List, Any

import pandas as pd
from .model import ModelData


def _safe_get(d: Dict, k, default=0.0) -> float:
    try:
        v = d.get(k, default)
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)


def write_results_excel(
    *,
    data: ModelData,
    state: Dict[str, Dict],
    iter_rows: List[Dict[str, object]],
    detailed_iter_rows: List[Dict[str, object]] | None = None,
    output_path: str,
    meta: Dict[str, object] | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    Q_offer = state.get("Q_offer", {})
    x_dem = state.get("x_dem", {})
    lam = state.get("lam", {})
    x = state.get("x", {})
    tau_imp = state.get("tau_imp", {})
    tau_exp = state.get("tau_exp", {})
    obj = state.get("obj", {})

    # --- 1. Prepare DataFrames ---
    
    # Regions Sheet
    region_rows: List[Dict[str, object]] = []
    for r in data.regions:
        imports = sum(_safe_get(x, (exp, r), 0.0) for exp in data.regions)
        exports = sum(_safe_get(x, (r, imp), 0.0) for imp in data.regions)
        region_rows.append(
            {
                "r": r,
                "Q_offer": _safe_get(Q_offer, r, 0.0),
                "x_dem": _safe_get(x_dem, r, 0.0),
                "lam": _safe_get(lam, r, 0.0),
                "obj": _safe_get(obj, r, 0.0),
                "imports": float(imports),
                "exports": float(exports),
                "Qcap": float(data.Qcap[r]),
                "Dmax": float(data.Dmax[r]),
            }
        )
    df_regions = pd.DataFrame(region_rows)

    # Flows Sheet
    flow_rows: List[Dict[str, object]] = []
    for exp in data.regions:
        for imp in data.regions:
            flow_rows.append(
                {
                    "exp": exp,
                    "imp": imp,
                    "x": _safe_get(x, (exp, imp), 0.0),
                    "tau_imp": _safe_get(tau_imp, (imp, exp), 0.0),
                    "tau_exp": _safe_get(tau_exp, (exp, imp), 0.0),
                    "c_ship": float(data.c_ship[(exp, imp)]),
                    "c_man": float(data.c_man[exp]),
                }
            )
    df_flows = pd.DataFrame(flow_rows)

    # Iteration History
    df_iters = pd.DataFrame(iter_rows)
    
    # Meta
    df_meta = pd.DataFrame(list((meta or {}).items()), columns=["key", "value"])
    
    # Detailed Iterations
    df_detailed = pd.DataFrame(detailed_iter_rows) if detailed_iter_rows else pd.DataFrame()

    # --- 2. Write with XlsxWriter Engine ---
    # This avoids the double-save (pandas write -> openpyxl load -> save) pattern
    
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        
        # Define formats
        header_fmt = workbook.add_format({'bold': True, 'text_wrap': False, 'valign': 'top', 'border': 1})
        # number_fmt = workbook.add_format({'num_format': '#,##0.00'}) # optional

        def write_sheet(df: pd.DataFrame, sheet_name: str):
            if df.empty:
                return
                
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            
            # Apply header format and auto-width
            for idx, col in enumerate(df.columns):
                # Write header with format
                worksheet.write(0, idx, col, header_fmt)
                
                # Estimate width
                max_len = max(
                    len(str(col)),
                    df[col].astype(str).map(len).max() if not df.empty else 0
                )
                width = min(max(max_len + 2, 10), 30) # clamp between 10 and 30
                worksheet.set_column(idx, idx, width)
            
            # Freeze top row
            worksheet.freeze_panes(1, 0)
            
            # Add simple autofilter
            (max_row, max_col) = df.shape
            if max_row > 0:
                worksheet.autofilter(0, 0, max_row, max_col - 1)

        write_sheet(df_regions, "regions")
        write_sheet(df_flows, "flows")
        write_sheet(df_iters, "iters")
        if not df_detailed.empty:
            write_sheet(df_detailed, "detailed_iters")
        write_sheet(df_meta, "meta")

    # No need for manual save(), the context manager handles it.


