"""
Sensitivity Analysis: Shuffled Player Order.

Runs each init scenario N_REPEATS times with shuffle_players=True.
This tests whether the randomized player order produces consistent
equilibria across different random seeds for the same starting point.

Combinations: 5 init scenarios × 4 repeats = 20 runs.
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
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from solargeorisk_extension.data_prep import load_data_from_excel
from solargeorisk_extension.gauss_seidel import solve_gs
from solargeorisk_extension.results_writer import write_results_excel
from solargeorisk_extension.plot_results import write_default_plots

from run_gs import (
    RunConfig,
    INIT_SCENARIOS,
    _apply_data_overrides,
    _solver_options,
    _safe_float,
)

# ---------------------------------------------------------------------------
N_REPEATS = 4
# ---------------------------------------------------------------------------


def run_shuffled_sensitivity():
    cfg = RunConfig()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.out_dir, f"sensitivity_shuffled_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Shuffled Sensitivity Analysis.  Output: {output_dir}")

    # Load data & apply overrides
    data = load_data_from_excel(cfg.excel_path)
    _apply_data_overrides(data, cfg)

    solver_opts = _solver_options(
        solver=cfg.solver, feastol=cfg.feastol, opttol=cfg.opttol, cfg=cfg,
    )

    print(f"[CONFIG] solver={cfg.solver}  omega={cfg.omega}  shuffle_players=True")
    print(f"[CONFIG] tol_strat={cfg.tol_strat}  tol_obj={cfg.tol_obj}  convergence_mode={cfg.convergence_mode}")
    print(f"[CONFIG] kappa_q={cfg.kappa_q}  rho_prox={cfg.rho_prox}")
    print(f"[CONFIG] N_REPEATS={N_REPEATS}  init_scenarios={list(INIT_SCENARIOS.keys())}")

    # Build combinations: 5 scenarios × 4 repeats
    combinations = [
        {"init_name": name, "init_val": val, "repeat": rep + 1}
        for name, val in INIT_SCENARIOS.items()
        for rep in range(N_REPEATS)
    ]
    print(f"Generated {len(combinations)} runs.\n")

    summary_results: list[dict] = []
    iter_rows_all: list[dict] = []  # store iter_rows reference per run

    for i, combo in enumerate(combinations):
        init_name = combo["init_name"]
        rep = combo["repeat"]
        init_val = combo["init_val"]
        run_name = f"run_{i+1:03d}_{init_name}_rep{rep}"

        print(f"[{i+1}/{len(combinations)}] {run_name}")

        # Build initial state
        init_q: Dict[str, float] = {}
        for p in data.players:
            frac = float(init_val.get(p, 0.8))
            init_q[p] = frac * float(data.Qcap[p])
        init_state = {"Q_offer": init_q, "tau_imp": {}, "tau_exp": {}}

        # Temp workdir
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        workdir = os.path.join(tempfile.gettempdir(), f"solargeorisk_gams_{run_id}")
        if " " in workdir:
            workdir = os.path.join("C:\\temp", f"solargeorisk_gams_{run_id}")
        os.makedirs(workdir, exist_ok=True)

        def _progress(it, state, r_strat, stable_count):
            order_str = ""
            if "_sweep_order" in state:
                order_str = f" order={state['_sweep_order']}"
            details = []
            for r in data.players:
                q = _safe_float(state.get("Q_offer", {}).get(r))
                details.append(f"{r}({q:.0f})")
            print(f"    [It {it}] r={r_strat:.4f} s={stable_count}{order_str} | {' '.join(details)}")

        start_time = time.perf_counter()
        try:
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
                iter_callback=_progress,
                initial_state=init_state,
                convergence_mode=cfg.convergence_mode,
                shuffle_players=True,
            )

            elapsed = time.perf_counter() - start_time
            converged = len(iter_rows) < cfg.iters
            n_iters = len(iter_rows)
            stable_count = iter_rows[-1]["stable_count"] if iter_rows else 0
            final_r = iter_rows[-1]["r_strat"] if iter_rows else 999.0

            print(f"  Done in {elapsed:.1f}s ({n_iters} iters). Converged={converged}")

            # Save Excel
            out_path = os.path.join(output_dir, f"{run_name}.xlsx")
            write_results_excel(
                data=data, state=state, iter_rows=iter_rows,
                detailed_iter_rows=[], output_path=out_path,
                meta={
                    "run_name": run_name, "init_condition": init_name,
                    "repeat": str(rep), "shuffle_players": "True",
                    "solver": cfg.solver, "omega": str(cfg.omega),
                    "convergence_mode": cfg.convergence_mode,
                    "elapsed_sec": str(elapsed),
                },
            )

            # Plots
            plot_dir = os.path.join(output_dir, f"plots_{run_name}")
            os.makedirs(plot_dir, exist_ok=True)
            try:
                write_default_plots(output_path=out_path, plots_dir=plot_dir)
            except Exception as e:
                print(f"  [WARN] Plots: {e}")

            # Summary row
            q_offers = state.get("Q_offer", {})
            lam = state.get("lam", {})
            summary_row: dict = {
                "run": run_name, "init": init_name, "repeat": rep,
                "converged": converged, "n_iters": n_iters,
                "stable_count": stable_count, "final_r_strat": final_r,
                "elapsed": elapsed,
            }
            for p in data.players:
                summary_row[f"Q_{p}"] = _safe_float(q_offers.get(p))
                summary_row[f"lam_{p}"] = _safe_float(lam.get(p))
            summary_results.append(summary_row)

        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback; traceback.print_exc()
            summary_results.append({
                "run": run_name, "init": init_name, "repeat": rep,
                "converged": False, "error": str(e),
            })
        finally:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass

    # Write summary
    summary_path = os.path.join(output_dir, "summary.csv")
    pd.DataFrame(summary_results).to_csv(summary_path, index=False)
    print(f"\nBatch complete.  Summary: {summary_path}")


if __name__ == "__main__":
    run_shuffled_sensitivity()
