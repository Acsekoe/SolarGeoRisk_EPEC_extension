"""
Sensitivity Analysis: Initial Conditions and Player Order.

This script runs the EPEC model multiple times with varying:
1. Initial Q_offer values (starting points).
2. Player order (permutation of player update sequence).

Results are saved to 'outputs/sensitivity_<YYYYMMDD_HHMMSS>/'.
Each run generates a full results Excel file and standard plots.
"""
from __future__ import annotations

import os
import sys
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from solargeorisk_extension.model import ModelData
from solargeorisk_extension.data_prep import load_data_from_excel
from solargeorisk_extension.gauss_seidel import solve_gs
from solargeorisk_extension.results_writer import write_results_excel
from solargeorisk_extension.plot_results import write_default_plots


# --- Configuration ---

# Initial Q_offer Scenarios (as fraction of Qcap)
# Copied from run_gs.py
INIT_SCENARIOS = {
    "high_all": {"ch": 0.8, "eu": 0.8, "us": 0.8, "apac": 0.8, "roa": 0.8, "row": 0.8},
    "low_non_ch": {"ch": 0.8, "eu": 0.0, "us": 0.0, "apac": 0.8, "roa": 0.0, "row": 0.0},
    "low_eu_us_row": {"ch": 0.8, "eu": 0.0, "us": 0.0, "apac": 0.8, "roa": 0.8, "row": 0.0},
    "mid_all": {"ch": 0.5, "eu": 0.5, "us": 0.5, "apac": 0.5, "roa": 0.5, "row": 0.5},
    "low_all": {"ch": 0.2, "eu": 0.0, "us": 0.0, "apac": 0.2, "roa": 0.0, "row": 0.0},
}


# Solver Settings (Synced with run_gs.py RunConfig)
SOLVER = "knitro"
ITERS = 100
OMEGA = 0.7
TOL_REL = 1e-2
STABLE_ITERS = 3
EPS_X = 1e-3
EPS_COMP = 1e-4

# Penalty / Regularization Settings (Synced with run_gs.py default RunConfig)
RHO_IMP = 0.05
RHO_EXP = 0.05
KAPPA_Q = 1.0
RHO_PROX = 0.05
USE_QUAD = True

EXCEL_PATH = os.path.join("inputs", "input_data.xlsx")

# Workers for sequential run to respect order
WORKERS = 1


def get_order_scenarios(players: List[str]) -> Dict[str, List[str]]:
    """Generate player order scenarios."""
    base = list(players) # Expected default order: ch, eu, us, apac, roa, row (or similar from data)
    
    # Check if 'ch' is in players before creating specific scenarios
    has_ch = 'ch' in players
    
    scenarios = {
        "default": base, # typically ch is first
    }
    
    if has_ch and len(base) > 2:
        # China Last: Remove 'ch' and append
        ch_last = [p for p in base if p != 'ch'] + ['ch']
        scenarios["ch_last"] = ch_last
        
        # China Middle: Remove 'ch' and insert in middle
        others = [p for p in base if p != 'ch']
        mid_idx = len(others) // 2
        ch_mid = others[:mid_idx] + ['ch'] + others[mid_idx:]
        scenarios["ch_mid"] = ch_mid
    
    # Reverse order as a standard check
    scenarios["reverse"] = base[::-1]

    return scenarios


def run_sensitivity_batch():
    # 1. Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"sensitivity_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting Sensitivity Analysis. Output Directory: {output_dir}")

    # 2. Load Data
    print(f"Loading data from {EXCEL_PATH}...")
    data = load_data_from_excel(EXCEL_PATH)
    
    # Apply global settings if needed (eps_x, etc.)
    data.eps_x = EPS_X
    data.eps_comp = EPS_COMP

    # Apply Penalty Overrides (Standardizing with run_gs.py)
    if RHO_IMP is not None:
        for r in data.regions:
            data.rho_imp[r] = float(RHO_IMP)

    if RHO_EXP is not None:
        for r in data.regions:
            data.rho_exp[r] = float(RHO_EXP)

    if KAPPA_Q is not None and data.kappa_Q is not None:
        for r in data.regions:
            data.kappa_Q[r] = float(KAPPA_Q)
            
    # Apply Proximal Regularization
    if data.settings is None:
        data.settings = {}
    if RHO_PROX is not None:
        data.settings["rho_prox"] = float(RHO_PROX)
    data.settings["use_quad"] = USE_QUAD

    # 3. Generate Scenarios
    order_scenarios = get_order_scenarios(data.players)
    
    # Create list of (init_name, init_val, order_name, order_list)
    combinations = []
    
    for init_name, init_val in INIT_SCENARIOS.items():
        for order_name, order_list in order_scenarios.items():
            combinations.append({
                "init_name": init_name,
                "init_val": init_val,
                "order_name": order_name,
                "order_list": order_list
            })
            
    print(f"Generated {len(combinations)} combinations to run.")
    
    # 4. Run Loop
    summary_results = []
    
    for i, combo in enumerate(combinations):
        run_id = f"{i+1:03d}"
        init_name = combo["init_name"]
        order_name = combo["order_name"]
        init_val = combo["init_val"]
        order_list = combo["order_list"]
        
        run_name = f"run_{run_id}_{init_name}_{order_name}"
        print(f"\n[{i+1}/{len(combinations)}] Running {run_name}...")
        print(f"  Init: {init_name}")
        print(f"  Order: {order_name} {order_list}")
        
        # Build Initial State
        init_state = {"Q_offer": {}}
        if isinstance(init_val, dict):
             for p in data.players:
                # Use value from dict if present, else default? 
                # INIT_SCENARIOS maps player -> frac.
                if p in init_val:
                     frac = init_val[p]
                     init_state["Q_offer"][p] = float(data.Qcap[p]) * frac
                else:
                     # Fallback if scenario dict is partial?
                     init_state["Q_offer"][p] = float(data.Qcap[p]) * 0.8
        else:
            # Scalar
            for p in data.players:
                init_state["Q_offer"][p] = float(data.Qcap[p]) * float(init_val)
        
        start_time = time.perf_counter()
        
        # Define callback for progress display
        def _iter_progress(it, state, r_strat, stable_count):
            # Extract Q and lam for display
            q_map = state.get("Q_offer", {})
            lam_map = state.get("lam", {})
            
            # Format succinct string: "ch: Q=... lam=... | eu: ..."
            # Just show first few or all if few enough? 6 regions fits in one line if concise.
            # "ch(350, 120) eu(0, 300) ..."
            details = []
            for r in data.players: # Use current data.players order which is what we need
                 q = float(q_map.get(r, 0.0))
                 l = float(lam_map.get(r, 0.0))
                 details.append(f"{r}({q:.0f},{l:.0f})")
            
            info_str = " ".join(details)
            print(f"    [It {it}] r={r_strat:.4f} s={stable_count} | {info_str}      ", end="\r")

        # Run Solver 
        try:
            # Sequential solver (respecting order)
            # Manually set player order in data object
            original_players = list(data.players)
            try:
                data.players = list(order_list)
                state, iter_rows = solve_gs(
                    data,
                    iters=ITERS,
                    omega=OMEGA,
                    tol_rel=TOL_REL,
                    stable_iters=STABLE_ITERS,
                    solver=SOLVER,
                    initial_state=init_state,
                    working_directory=None, # Auto-temp dir
                    tol_obj=1e-2, # Added explicit tol_obj to match run_gs
                    iter_callback=_iter_progress,
                )
                print() # Newline after progress bar
            finally:
                data.players = original_players
            
            elapsed = time.perf_counter() - start_time
            converged = (len(iter_rows) < ITERS)
            stable_count = iter_rows[-1]["stable_count"] if iter_rows else 0
            final_r_strat = iter_rows[-1]["r_strat"] if iter_rows else 999.0
            
            print(f"  Finished in {elapsed:.2f}s. Converged? {converged} (stable={stable_count})")

            # Save Standard Results Excel
            csv_filename = f"{run_name}.xlsx"
            out_path = os.path.join(output_dir, csv_filename)
            
            # Metadata for the excel file
            meta = {
                "run_name": run_name,
                "init_condition": init_name,
                "player_order": order_name,
                "player_order_list": str(order_list),
                "solver": SOLVER,
                "omega": OMEGA,
                "elapsed_sec": elapsed
            }
            
            write_results_excel(
                data=data,
                state=state,
                iter_rows=iter_rows,
                detailed_iter_rows=[], # Can populate if needed
                output_path=out_path,
                meta={k: str(v) for k, v in meta.items()}
            )
            print(f"  Saved results to {out_path}")
            
            # Save Plots
            plot_dir = os.path.join(output_dir, f"plots_{run_name}")
            os.makedirs(plot_dir, exist_ok=True)
            write_default_plots(output_path=out_path, plots_dir=plot_dir)
            
            # Extract final Q_offer for summary
            q_offers = state.get("Q_offer", {})
            lam = state.get("lam", {})

            summary_row = {
                "run": run_name,
                "init": init_name,
                "order": order_name,
                "converged": converged,
                "stable_count": stable_count,
                "final_r_strat": final_r_strat,
                "elapsed": elapsed,
                "excel_file": csv_filename
            }
            
            # Add player Q offers to summary
            for p in data.players:
                summary_row[f"Q_{p}"] = q_offers.get(p, 0.0)
                summary_row[f"lam_{p}"] = lam.get(p, 0.0)
                
            summary_results.append(summary_row)
            
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            summary_results.append({
                "run": run_name,
                "init": init_name,
                "order": order_name,
                "converged": False,
                "error": str(e)
            })

    # 5. Write Summary CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    pd.DataFrame(summary_results).to_csv(summary_path, index=False)
    print(f"\nBatch complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    run_sensitivity_batch()
