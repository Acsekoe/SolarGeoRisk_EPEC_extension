from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from solargeorisk_extension.data_prep import load_data_from_excel
from solargeorisk_extension.gauss_seidel import solve_gs
from solargeorisk_extension.plot_results import write_default_plots
from solargeorisk_extension.results_writer import write_results_excel


@dataclass(frozen=True)
class RunConfig:
    excel_path: str = os.path.join("inputs", "input_data.xlsx")
    out_dir: str = "outputs"
    plots_dir: str = "plots"

    solver: str = "conopt"
    feastol: float = 1e-4
    opttol: float = 1e-4

    iters: int = 3
    omega: float = 0.8
    tol_rel: float = 1e-2
    stable_iters: int = 3

    eps_x: float | None = None
    eps_comp: float | None = None


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _print_q_offer_and_lam(*, regions: list[str], state: dict[str, dict], tag: str = "SUMMARY") -> None:
    q_offer = state.get("Q_offer", {}) or {}
    lam = state.get("lam", {}) or {}

    if not regions:
        print(f"[{tag}] No regions configured; skipping Q_offer/lam print.")
        return

    print(f"[{tag}] Q_offer and lam by region:")
    r_width = max(2, min(24, max(len(str(r)) for r in regions)))
    for r in regions:
        q = _safe_float(q_offer.get(r, 0.0), 0.0)
        l = _safe_float(lam.get(r, 0.0), 0.0)
        print(f"  {str(r):<{r_width}}  Q_offer={q:.6g}  lam={l:.6g}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal GS diagonalization runner (demand model).")
    p.add_argument("--excel", type=str, default=None, help="Excel path (defaults to inputs/input_data.xlsx).")
    p.add_argument("--out-dir", type=str, default=None, help="Outputs folder (default: outputs).")
    p.add_argument("--plots-dir", type=str, default=None, help="Plots folder (default: plots).")

    p.add_argument("--solver", type=str, default=None, help="Solver name (default: conopt).")
    p.add_argument("--feastol", type=float, default=None, help="Solver feasibility tolerance.")
    p.add_argument("--opttol", type=float, default=None, help="Solver optimality tolerance.")

    p.add_argument("--iters", type=int, default=None, help="GS iterations.")
    p.add_argument("--omega", type=float, default=None, help="GS damping in (0,1].")
    p.add_argument("--tol-rel", type=float, default=None, help="Stop when strategy rel-change <= tol-rel.")
    p.add_argument("--stable-iters", type=int, default=None, help="Require tol satisfied this many iterations.")

    p.add_argument("--eps-x", type=float, default=None, help="Override eps_x (LLP regularization).")
    p.add_argument("--eps-comp", type=float, default=None, help="Override eps_comp (comp relaxation; 0=exact).")
    return p.parse_args()


def _resolve_excel_path(raw: str | None, default_path: str) -> str:
    if raw is None or not str(raw).strip():
        return default_path
    candidate = str(raw)
    if os.path.exists(candidate):
        return candidate
    in_inputs = os.path.join("inputs", candidate)
    if os.path.exists(in_inputs):
        return in_inputs
    return candidate


def _gams_workdir(run_id: str) -> str:
    base = tempfile.gettempdir()
    workdir = os.path.join(base, f"solargeorisk_gams_{run_id}")
    if " " in workdir:
        workdir = os.path.join("C:\\temp", f"solargeorisk_gams_{run_id}")
    os.makedirs(workdir, exist_ok=True)
    return workdir


def _solver_options(solver: str, feastol: float, opttol: float) -> Dict[str, float]:
    name = solver.strip().lower()
    if name == "conopt":
        return {"Tol_Feas_Max": feastol, "Tol_Optimality": opttol}
    if name == "knitro":
        return {"feastol": feastol, "opttol": opttol, "outlev": 1, "maxit": 2000}
    return {}


if __name__ == "__main__":
    args = _parse_args()
    cfg = RunConfig()

    excel_path = _resolve_excel_path(args.excel, cfg.excel_path)
    out_dir = args.out_dir or cfg.out_dir
    plots_dir = args.plots_dir or cfg.plots_dir

    solver = args.solver or cfg.solver
    feastol = float(args.feastol) if args.feastol is not None else cfg.feastol
    opttol = float(args.opttol) if args.opttol is not None else cfg.opttol

    iters = int(args.iters) if args.iters is not None else cfg.iters
    omega = float(args.omega) if args.omega is not None else cfg.omega
    tol_rel = float(args.tol_rel) if args.tol_rel is not None else cfg.tol_rel
    stable_iters = int(args.stable_iters) if args.stable_iters is not None else cfg.stable_iters

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    output_path = os.path.join(out_dir, f"results_{run_id}.xlsx")
    workdir = _gams_workdir(run_id)

    data = load_data_from_excel(excel_path)
    if args.eps_x is not None:
        data.eps_x = float(args.eps_x)
    if args.eps_comp is not None:
        data.eps_comp = float(args.eps_comp)

    def _iter_log(it: int, state: dict[str, dict], r_strat: float, stable_count: int) -> None:
        print(f"[ITER {it}] r_strat={r_strat:.6g} stable_count={stable_count}")
        _print_q_offer_and_lam(regions=list(data.regions), state=state, tag=f"ITER {it}")

    state, iter_rows = solve_gs(
        data,
        iters=iters,
        omega=omega,
        tol_rel=tol_rel,
        stable_iters=stable_iters,
        solver=solver,
        solver_options=_solver_options(solver, feastol, opttol),
        working_directory=workdir,
        iter_callback=_iter_log,
    )

    _print_q_offer_and_lam(regions=list(data.regions), state=state, tag="FINAL")

    write_results_excel(
        data=data,
        state=state,
        iter_rows=iter_rows,
        output_path=output_path,
        meta={
            "excel_path": excel_path,
            "solver": solver,
            "feastol": feastol,
            "opttol": opttol,
            "iters": iters,
            "omega": omega,
            "tol_rel": tol_rel,
            "stable_iters": stable_iters,
            "eps_x": float(data.eps_x),
            "eps_comp": float(data.eps_comp),
            "workdir": workdir,
        },
    )

    write_default_plots(output_path=output_path, plots_dir=plots_dir)
    print(f"[OK] wrote: {output_path}")

    # --- Run LaTeX Generation Workflow ---
    try:
        # Ensure we can import generate_latex from the same directory
        sys.path.append(os.path.dirname(__file__))
        import generate_latex
        print("Running automatic LaTeX documentation generation...")
        generate_latex.main()
    except ImportError:
        print("[WARN] Could not import generate_latex.py. Skipping LaTeX generation.")
    except Exception as e:
        print(f"[WARN] Error during LaTeX generation: {e}")
