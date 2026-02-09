"""
Sensitivity Analysis for Equilibrium Selection

Varies initial Q_offer values and player order to explore multiple equilibria.
Sequential solving (workers=1) is used for reproducibility.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from solargeorisk_extension.data_prep import load_data_from_excel
from solargeorisk_extension.gauss_jacobi import solve_jacobi, solve_jacobi_parallel
from solargeorisk_extension.model import ModelData


@dataclass
class Scenario:
    name: str
    init_q_frac: Dict[str, float]  # fraction of Qcap for each player
    player_order: List[str] | None = None  # None = use default order


# Scenario definitions
INIT_SCENARIOS = {
    "high_all": {"ch": 0.8, "eu": 0.8, "us": 0.8, "apac": 0.8, "roa": 0.8, "row": 0.8},
    "low_non_ch": {"ch": 0.8, "eu": 0.0, "us": 0.0, "apac": 0.8, "roa": 0.0, "row": 0.0},
    "low_eu_us_row": {"ch": 0.8, "eu": 0.0, "us": 0.0, "apac": 0.8, "roa": 0.8, "row": 0.0},
    "mid_all": {"ch": 0.5, "eu": 0.5, "us": 0.5, "apac": 0.5, "roa": 0.5, "row": 0.5},
    "low_all": {"ch": 0.2, "eu": 0.0, "us": 0.0, "apac": 0.2, "roa": 0.0, "row": 0.0},
}

ORDER_SCENARIOS = {
    "default": None,  # Use data.players order
    "ch_first": ["ch", "apac", "eu", "us", "roa", "row"],
    "ch_last": ["eu", "us", "apac", "roa", "row", "ch"],
    "apac_first": ["apac", "ch", "eu", "us", "roa", "row"],
}


def run_scenario(
    data: ModelData,
    scenario: Scenario,
    iters: int = 50,
    omega: float = 0.7,
    solver: str = "conopt",
    workdir: str | None = None,
    workers: int = 1,
) -> Dict:
    """Run a single sensitivity scenario and return summary."""
    
    # Build initial state from scenario
    init_state = {
        "Q_offer": {r: scenario.init_q_frac.get(r, 0.8) * float(data.Qcap[r]) for r in data.players},
        "tau_imp": {},
        "tau_exp": {},
    }
    
    solver_opts = {"Tol_Feas_Max": 1e-4, "Tol_Optimality": 1e-4}
    
    start_time = time.perf_counter()
    
    if workers > 1:
        state, iter_rows = solve_jacobi_parallel(
            data,
            excel_path="inputs/input_data.xlsx",  # Required for parallel workers to reload data
            iters=iters,
            omega=omega,
            tol_rel=0.01,
            stable_iters=3,
            solver=solver,
            solver_options=solver_opts,
            working_directory=workdir,
            workers=workers,
            worker_timeout=60,
            initial_state=init_state,
        )
    else:
        state, iter_rows = solve_jacobi(
            data,
            iters=iters,
            omega=omega,
            tol_rel=0.01,
            stable_iters=3,
            solver=solver,
            solver_options=solver_opts,
            working_directory=workdir,
            initial_state=init_state,
            player_order=scenario.player_order,
        )
        
    elapsed = time.perf_counter() - start_time
    
    # Extract summary
    return {
        "scenario": scenario.name,
        "init_q": scenario.init_q_frac,
        "player_order": scenario.player_order or list(data.players),
        "elapsed_s": elapsed,
        "n_iters": len(iter_rows),
        "converged": len(iter_rows) < iters,
        "Q_offer": state.get("Q_offer", {}),
        "lam": state.get("lam", {}),
        "obj": state.get("obj", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for equilibrium selection")
    parser.add_argument("--iters", type=int, default=50, help="Max iterations per scenario")
    parser.add_argument("--omega", type=float, default=0.7, help="Damping factor")
    parser.add_argument("--solver", type=str, default="conopt", help="Solver to use")
    parser.add_argument("--excel", type=str, default="inputs/input_data.xlsx", help="Input Excel file")
    parser.add_argument("--out", type=str, default="outputs/sensitivity_results.csv", help="Output CSV")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (parallel if > 1)")
    parser.add_argument("--init-only", action="store_true", help="Only vary initial values, not order")
    parser.add_argument("--order-only", action="store_true", help="Only vary player order, not init values")
    args = parser.parse_args()
    
    # Load data
    data = load_data_from_excel(args.excel)
    
    # Build scenario list
    scenarios = []
    
    if args.order_only:
        # Only vary player order with default init
        for order_name, order in ORDER_SCENARIOS.items():
            scenarios.append(Scenario(
                name=f"order_{order_name}",
                init_q_frac=INIT_SCENARIOS["high_all"],
                player_order=order,
            ))
    elif args.init_only:
        # Only vary initial values with default order
        for init_name, init_frac in INIT_SCENARIOS.items():
            scenarios.append(Scenario(
                name=f"init_{init_name}",
                init_q_frac=init_frac,
                player_order=None,
            ))
    else:
        # Full factorial: all combinations
        for init_name, init_frac in INIT_SCENARIOS.items():
            for order_name, order in ORDER_SCENARIOS.items():
                scenarios.append(Scenario(
                    name=f"{init_name}__{order_name}",
                    init_q_frac=init_frac,
                    player_order=order,
                ))
    
    print(f"Running {len(scenarios)} scenarios...")
    
    # Run scenarios
    results = []
    for i, scenario in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] {scenario.name}...", end=" ", flush=True)
        result = run_scenario(
            data=data,
            scenario=scenario,
            iters=args.iters,
            omega=args.omega,
            solver=args.solver,
            workers=args.workers,
        )
        
        # Flatten for CSV
        row = {
            "scenario": result["scenario"],
            "elapsed_s": result["elapsed_s"],
            "n_iters": result["n_iters"],
            "converged": result["converged"],
        }
        for r in data.regions:
            row[f"Q_offer_{r}"] = result["Q_offer"].get(r, 0.0)
            row[f"lam_{r}"] = result["lam"].get(r, 0.0)
            row[f"obj_{r}"] = result["obj"].get(r, 0.0)
        
        results.append(row)
        print(f"lam_eu={row.get('lam_eu', 0):.1f}")
    
    # Save results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to: {args.out}")
    
    # Print summary
    print("\n=== Equilibrium Summary ===")
    df_summary = df[["scenario", "lam_eu", "lam_ch", "Q_offer_eu", "converged"]].copy()
    df_summary = df_summary.sort_values("lam_eu")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
