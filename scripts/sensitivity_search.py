"""
Sensitivity Search Script for SolarGeoRisk EPEC Model.
Runs a grid search over solver parameters to find stable configurations.
"""
from __future__ import annotations

import itertools
import os
import sys
import time
import pandas as pd
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from solargeorisk_extension.data_prep import load_data_from_excel
from solargeorisk_extension.gauss_jacobi import solve_jacobi_parallel
from solargeorisk_extension.model import ModelData

# Configuration Grid
GRID = {
    "omega": [0.5, 0.7, 0.9],
    "eps_comp": [1e-4, 1e-3],
    "eps_x": [1e-4, 1e-3],
    "rho_factor": [1.0, 10.0],  # Scaling factor for default rho values
}

# Fixed Settings
ITERS = 20
SOLVER = "conopt"
# Use fewer workers to avoid overwhelming the system during sensitivity search
WORKERS = 4 
# Increased timeout to allow for slower convergence
TIMEOUT = 60 
EXCEL_PATH = os.path.join("inputs", "input_data.xlsx")
WORKDIR = "C:\\temp\\solargeorisk_sensitivity"


def run_sensitivity_search():
    print("Starting Sensitivity Search...")
    keys = list(GRID.keys())
    values = list(GRID.values())
    combinations = list(itertools.product(*values))
    
    results = []
    
    total_runs = len(combinations)
    print(f"Total combinations to test: {total_runs}")
    
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
        
    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n--- Run {idx+1}/{total_runs} ---")
        print(f"Params: {params}")
        
        # Load fresh data
        data = load_data_from_excel(EXCEL_PATH)
        
        # Apply parameters
        data.eps_comp = float(params["eps_comp"])
        data.eps_x = float(params["eps_x"])
        
        # Apply rho scaling
        factor = float(params["rho_factor"])
        for r in data.regions:
            data.rho_imp[r] *= factor
            data.rho_exp[r] *= factor
            if data.kappa_Q:
                data.kappa_Q[r] *= factor
        
        # Run Solver
        start_time = time.perf_counter()
        final_r_strat = 999.0
        stable_count = 0
        status = "OK"
        
        try:
            # Re-import to ensure no stale state if any (though function scope helps)
            state, iter_rows = solve_jacobi_parallel(
                data,
                excel_path=EXCEL_PATH,
                iters=ITERS,
                omega=params["omega"],
                tol_rel=1e-3, # Use a reasonable tolerance for check
                stable_iters=3,
                solver=SOLVER,
                workers=WORKERS,
                worker_timeout=TIMEOUT,
                use_staged_tolerances=True, # Use staging for robustness
                max_sweep_failures=6, # Allow all players to fail once (will use fallback)
                working_directory=WORKDIR
            )
            
            if iter_rows:
                last_row = iter_rows[-1]
                final_r_strat = last_row.get("r_strat", 999.0)
                stable_count = last_row.get("stable_count", 0)
            
            # Analyze State
            q_offers = state.get("Q_offer", {})
            lams = state.get("lam", {})
            
            q_vals = [float(v) for v in q_offers.values()]
            lam_vals = [float(v) for v in lams.values()]
            
            mean_q = sum(q_vals) / len(q_vals) if q_vals else 0.0
            max_q = max(q_vals) if q_vals else 0.0
            
            mean_lam = sum(lam_vals) / len(lam_vals) if lam_vals else 0.0
            max_lam = max(lam_vals) if lam_vals else 0.0
            
            # Convergence check
            converged = stable_count >= 3
            
        except Exception as e:
            print(f"Run failed: {e}")
            status = "FAILED"
            final_r_strat = 999.0
            stable_count = 0
            converged = False
            mean_q = 0.0
            max_q = 0.0
            mean_lam = 0.0
            max_lam = 0.0
        
        elapsed = time.perf_counter() - start_time
        
        res_row = {
            **params,
            "final_r_strat": final_r_strat,
            "stable_count": stable_count,
            "converged": converged,
            "status": status,
            "mean_q": mean_q,
            "max_q": max_q,
            "mean_lam": mean_lam,
            "max_lam": max_lam,
            "elapsed_sec": elapsed
        }
        results.append(res_row)
        print(f"Result: r_strat={final_r_strat:.4f}, converged={converged}")
        
        # Save intermediate results
        pd.DataFrame(results).to_csv("sensitivity_results_partial.csv", index=False)

    # Save Results
    df = pd.DataFrame(results)
    out_csv = "sensitivity_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved sensitivity results to {out_csv}")
    
    # Print Summary of Best Runs
    if not df.empty:
        print("\nTop 5 Configs by Stability (lowest r_strat):")
        print(df.sort_values("final_r_strat").head(5)[
            ["omega", "eps_comp", "eps_x", "rho_factor", "final_r_strat", "converged"]
        ].to_string())

if __name__ == "__main__":
    run_sensitivity_search()
