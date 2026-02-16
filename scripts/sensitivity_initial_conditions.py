"""
Sensitivity Analysis: Initial Conditions and Player Order.

This script runs the EPEC model multiple times with varying:
1. Initial Q_offer values (starting points).
2. Player order (permutation of player update sequence).

It reuses RunConfig from run_gs.py as the single source of truth for all
solver/penalty/convergence settings — no duplicated constants.

Results are saved to 'outputs/sensitivity_<YYYYMMDD_HHMMSS>/'.
"""
from __future__ import annotations

import os
import sys
import time
import shutil
import tempfile
import uuid
import pandas as pd
from datetime import datetime
from typing import List, Dict

# Ensure src is in pythonpath
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solargeorisk_extension.data_prep import load_data_from_excel
from solargeorisk_extension.gauss_seidel import solve_gs
from solargeorisk_extension.results_writer import write_results_excel
from solargeorisk_extension.plot_results import write_default_plots

# Import RunConfig and helpers from run_gs (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from run_gs import (
    RunConfig,
    INIT_SCENARIOS,
    _apply_data_overrides,
    _solver_options,
    _safe_float,
)


# ---------------------------------------------------------------------------
# Player-order scenarios
# ---------------------------------------------------------------------------

def get_order_scenarios(players: List[str]) -> Dict[str, List[str]]:
    """Generate player order scenarios."""
    base = list(players)
    has_ch = "ch" in players

    scenarios = {"default": base}

    if has_ch and len(base) > 2:
        ch_last = [p for p in base if p != "ch"] + ["ch"]
        scenarios["ch_last"] = ch_last

        others = [p for p in base if p != "ch"]
        mid_idx = len(others) // 2
        ch_mid = others[:mid_idx] + ["ch"] + others[mid_idx:]
        scenarios["ch_mid"] = ch_mid

    scenarios["reverse"] = base[::-1]
    return scenarios


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def run_sensitivity_batch():
    cfg = RunConfig()  # single source of truth

    # 1. Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.out_dir, f"sensitivity_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting Sensitivity Analysis.  Output: {output_dir}")

    # 2. Load data & apply overrides (same as run_gs.run())
    data = load_data_from_excel(cfg.excel_path)
    _apply_data_overrides(data, cfg)

    # 3. Build solver options (same as run_gs.run())
    solver_opts = _solver_options(
        solver=cfg.solver,
        feastol=cfg.feastol,
        opttol=cfg.opttol,
        cfg=cfg,
    )

    print(f"[CONFIG] solver={cfg.solver}  omega={cfg.omega}  tol_strat={cfg.tol_strat}")
    print(f"[CONFIG] tol_obj={cfg.tol_obj}  convergence_mode={cfg.convergence_mode}")
    print(f"[CONFIG] kappa_q={cfg.kappa_q}  rho_prox={cfg.rho_prox}  solver_options={solver_opts}")

    # 4. Generate combinations
    order_scenarios = get_order_scenarios(data.players)

    combinations = [
        {
            "init_name": init_name,
            "init_val": init_val,
            "order_name": order_name,
            "order_list": order_list,
        }
        for init_name, init_val in INIT_SCENARIOS.items()
        for order_name, order_list in order_scenarios.items()
    ]
    print(f"Generated {len(combinations)} combinations to run.\n")

    # 5. Run loop
    summary_results: list[dict] = []

    for i, combo in enumerate(combinations):
        init_name = combo["init_name"]
        order_name = combo["order_name"]
        init_val = combo["init_val"]
        order_list = combo["order_list"]
        run_name = f"run_{i+1:03d}_{init_name}_{order_name}"

        print(f"[{i+1}/{len(combinations)}] {run_name}")
        print(f"  Init: {init_name}  Order: {order_list}")

        # Build initial state (same logic as run_gs._build_initial_state)
        init_q: Dict[str, float] = {}
        for p in data.players:
            frac = float(init_val.get(p, 0.8))
            init_q[p] = frac * float(data.Qcap[p])
        init_state = {"Q_offer": init_q, "tau_imp": {}, "tau_exp": {}}

        # Create a temp workdir for this run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        workdir = os.path.join(tempfile.gettempdir(), f"solargeorisk_gams_{run_id}")
        if " " in workdir:
            workdir = os.path.join("C:\\temp", f"solargeorisk_gams_{run_id}")
        os.makedirs(workdir, exist_ok=True)

        # Progress callback
        def _iter_progress(it, state, r_strat, stable_count):
            details = []
            for r in data.players:
                q = _safe_float(state.get("Q_offer", {}).get(r))
                l = _safe_float(state.get("lam", {}).get(r))
                details.append(f"{r}({q:.0f},{l:.0f})")
            info_str = " ".join(details)
            print(f"    [It {it}] r={r_strat:.4f} s={stable_count} | {info_str}      ", end="\r")

        start_time = time.perf_counter()
        original_players = list(data.players)

        try:
            data.players = list(order_list)

            # ---- Exact same solve_gs call as run_gs.run() ----
            state, iter_rows = solve_gs(
                data,
                solver=cfg.solver,
                solver_options=solver_opts,
                iters=cfg.iters,
                omega=cfg.omega,
                tol_rel=cfg.tol_strat,
                tol_obj=cfg.tol_obj,
                stable_iters=cfg.stable_iters,
                working_directory=workdir,
                iter_callback=_iter_progress,
                initial_state=init_state,
                convergence_mode=cfg.convergence_mode,
            )
            print()  # newline after progress

            elapsed = time.perf_counter() - start_time
            converged = len(iter_rows) < cfg.iters
            stable_count = iter_rows[-1]["stable_count"] if iter_rows else 0
            final_r_strat = iter_rows[-1]["r_strat"] if iter_rows else 999.0

            print(f"  Finished in {elapsed:.1f}s.  Converged={converged} (stable={stable_count})")

            # Save results excel
            out_path = os.path.join(output_dir, f"{run_name}.xlsx")
            data.players = original_players  # restore before writing
            write_results_excel(
                data=data,
                state=state,
                iter_rows=iter_rows,
                detailed_iter_rows=[],
                output_path=out_path,
                meta={
                    "run_name": run_name,
                    "init_condition": init_name,
                    "player_order": order_name,
                    "player_order_list": str(order_list),
                    "solver": cfg.solver,
                    "omega": str(cfg.omega),
                    "convergence_mode": cfg.convergence_mode,
                    "elapsed_sec": str(elapsed),
                    "solver_options": str(solver_opts),
                },
            )

            # Save plots
            plot_dir = os.path.join(output_dir, f"plots_{run_name}")
            os.makedirs(plot_dir, exist_ok=True)
            try:
                write_default_plots(output_path=out_path, plots_dir=plot_dir)
            except Exception as e:
                print(f"  [WARN] Plots failed: {e}")

            # Summary row
            q_offers = state.get("Q_offer", {})
            lam = state.get("lam", {})
            summary_row: dict = {
                "run": run_name,
                "init": init_name,
                "order": order_name,
                "converged": converged,
                "stable_count": stable_count,
                "final_r_strat": final_r_strat,
                "elapsed": elapsed,
                "excel_file": f"{run_name}.xlsx",
            }
            for p in data.players:
                summary_row[f"Q_{p}"] = _safe_float(q_offers.get(p))
                summary_row[f"lam_{p}"] = _safe_float(lam.get(p))
            summary_results.append(summary_row)

        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()
            summary_results.append({
                "run": run_name,
                "init": init_name,
                "order": order_name,
                "converged": False,
                "error": str(e),
            })
        finally:
            data.players = original_players
            # Cleanup workdir
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass

    # 6. Write summary CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    pd.DataFrame(summary_results).to_csv(summary_path, index=False)
    print(f"\nBatch complete.  Summary: {summary_path}")


if __name__ == "__main__":
    run_sensitivity_batch()
