from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from solargeorisk_extension.data_prep import load_data_from_excel
from solargeorisk_extension.gauss_jacobi import solve_jacobi, solve_jacobi_parallel
from solargeorisk_extension.gauss_seidel import solve_gs
from solargeorisk_extension.plot_results import write_default_plots
from solargeorisk_extension.results_writer import write_results_excel


@dataclass(frozen=True)
class RunConfig:
    excel_path: str = os.path.join("inputs", "input_data.xlsx")
    out_dir: str = "outputs"
    plots_dir: str = "plots"

    solver: str = "knitro"
    feastol: float = 1e-4
    opttol: float = 1e-4

    method: str = "jacobi"
    iters: int = 20
    omega: float = 0.9
    tol_rel: float = 1e-2
    stable_iters: int = 3
    workers: int = 6
    worker_timeout: float = 20

    warmup_solver: str | None = None
    warmup_iters: int = 5
    warmup_workers: int = 1

    eps_x: float | None = 1e-4
    eps_comp: float | None = 0.1

    knitro_outlev: int = 0
    knitro_maxit: int = 800
    knitro_hessopt: int = 1
    knitro_algorithm: int | None = None


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
    p = argparse.ArgumentParser(description="Diagonalization runner (EPEC).")
    p.add_argument("--excel", type=str, default=None, help="Excel path (defaults to inputs/input_data.xlsx).")
    p.add_argument("--out-dir", type=str, default=None, help="Outputs folder (default: outputs).")
    p.add_argument("--plots-dir", type=str, default=None, help="Plots folder (default: plots).")

    p.add_argument("--solver", type=str, default=None, help="Solver name (default from RunConfig).")
    p.add_argument("--feastol", type=float, default=None, help="Feasibility tolerance.")
    p.add_argument("--opttol", type=float, default=None, help="Optimality tolerance.")

    p.add_argument("--iters", type=int, default=None, help="Max sweeps.")
    p.add_argument("--omega", type=float, default=None, help="Damping in (0,1].")
    p.add_argument("--tol-rel", type=float, default=None, help="Stop when strategy rel-change <= tol-rel.")
    p.add_argument("--stable-iters", type=int, default=None, help="Require tol satisfied this many sweeps.")

    p.add_argument("--eps-x", type=float, default=None, help="Override eps_x (LLP regularization).")
    p.add_argument("--eps-comp", type=float, default=None, help="Override eps_comp (0=exact complementarity).")

    p.add_argument(
        "--method",
        type=str,
        choices=["seidel", "jacobi"],
        default=None,
        help="Diagonalization method: 'seidel' or 'jacobi' (default from RunConfig)."
    )

    # Knitro knobs (optional overrides)
    p.add_argument("--knitro-outlev", type=int, default=None, help="Knitro outlev (0 quiet).")
    p.add_argument("--knitro-maxit", type=int, default=None, help="Knitro maxit per solve.")
    p.add_argument("--knitro-hessopt", type=int, default=None, help="Knitro hessopt (e.g., 2=BFGS).")
    p.add_argument("--knitro-algorithm", type=int, default=None, help="Knitro algorithm (optional).")

    # Parallel Jacobi
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel workers for Jacobi (default: min(cpu_count, players)).")
    p.add_argument("--worker-timeout", type=float, default=None,
                   help="Timeout in seconds for each worker solve (default: 120, set low to skip lags).")
    p.add_argument("--keep-workdir", action="store_true",
                   help="Keep GAMS workdir after run (for debugging).")
    p.add_argument("--debug-workers", action="store_true",
                   help="Debug mode: verbose solver output, print worker paths.")

    # Hybrid solver: warmup phase
    p.add_argument("--warmup-solver", type=str, default=None,
                   help="Solver for warmup phase (e.g., 'knitro' for 2 sequential sweeps before main solver).")
    p.add_argument("--warmup-iters", type=int, default=None,
                   help="Number of warmup iterations (default: 3).")
    p.add_argument("--warmup-workers", type=int, default=None,
                   help="Workers for warmup phase (default: 1 = sequential).")

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


def _solver_options(
    *,
    solver: str,
    feastol: float,
    opttol: float,
    cfg: RunConfig,
    knitro_outlev: Optional[int] = None,
    knitro_maxit: Optional[int] = None,
    knitro_hessopt: Optional[int] = None,
    knitro_algorithm: Optional[int] = None,
) -> Dict[str, float]:
    """
    Keep options conservative and high-impact:
    - outlev low (less I/O)
    - maxit not huge (outer loop will re-solve anyway)
    - hessopt=2 (BFGS) often speeds up/steadies KKT-style nonconvex NLPs
    """
    name = solver.strip().lower()

    if name == "conopt":
        return {"Tol_Feas_Max": float(feastol), "Tol_Optimality": float(opttol)}

    if name == "knitro":
        # Minimal settings - just tolerances, let Knitro use defaults for everything else
        return {
            "feastol": float(feastol),
            "opttol": float(opttol),
        }

    return {}


def auto_git_push():
    """Checks for changes in 'overleaf' and pushes to origin main."""
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if not subprocess.run(["git", "status", "--porcelain", "overleaf"], cwd=root, capture_output=True, text=True, check=True).stdout.strip():
            return
        print("[GIT] Pushing updates to 'overleaf'...")
        subprocess.run(["git", "add", "overleaf"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update model equations"], cwd=root, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=root, check=True)
        print("[GIT] Done.")
    except Exception as e:
        print(f"[WARN] Git auto-push failed: {e}")


if __name__ == "__main__":
    args = _parse_args()
    cfg = RunConfig()

    excel_path = _resolve_excel_path(args.excel, cfg.excel_path)
    out_dir = args.out_dir or cfg.out_dir
    plots_dir = args.plots_dir or cfg.plots_dir

    solver = (args.solver or cfg.solver).strip()
    feastol = float(args.feastol) if args.feastol is not None else float(cfg.feastol)
    opttol = float(args.opttol) if args.opttol is not None else float(cfg.opttol)

    iters = int(args.iters) if args.iters is not None else int(cfg.iters)
    omega = float(args.omega) if args.omega is not None else float(cfg.omega)
    tol_rel = float(args.tol_rel) if args.tol_rel is not None else float(cfg.tol_rel)
    stable_iters = int(args.stable_iters) if args.stable_iters is not None else int(cfg.stable_iters)

    method = (args.method or cfg.method).lower().strip()

    # Workers: default to min(cpu_count, len(players)) for jacobi, 1 otherwise
    default_workers = min(os.cpu_count() or 4, 6)  # 6 = max players
    workers = args.workers if args.workers is not None else default_workers
    worker_timeout = args.worker_timeout if args.worker_timeout is not None else cfg.worker_timeout
    keep_workdir = args.keep_workdir

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    output_path = os.path.join(out_dir, f"results_{run_id}.xlsx")
    workdir = _gams_workdir(run_id)

    data = load_data_from_excel(excel_path)

    # Apply eps_x: CLI > RunConfig > Excel
    if args.eps_x is not None:
        data.eps_x = float(args.eps_x)
    elif cfg.eps_x is not None:
        data.eps_x = float(cfg.eps_x)

    # Apply eps_comp: CLI > RunConfig > Excel
    if args.eps_comp is not None:
        data.eps_comp = float(args.eps_comp)
    elif cfg.eps_comp is not None:
        data.eps_comp = float(cfg.eps_comp)

    print(f"[CONFIG] Method: {method}")
    print(f"[CONFIG] Solver: {solver}  feastol={feastol:g}  opttol={opttol:g}")
    print(f"[CONFIG] iters={iters} omega={omega:g} tol_rel={tol_rel:g} stable_iters={stable_iters}")
    print(f"[CONFIG] eps_x={float(data.eps_x):g} eps_comp={float(data.eps_comp):g}")
    if method == "jacobi" and workers > 1:
        print(f"[CONFIG] workers={workers} (parallel Jacobi)")
    print(f"[CONFIG] workdir={workdir}{' (keep)' if keep_workdir else ' (auto-cleanup)'}")

    # Timing
    sweep_times: list[float] = []
    timing_state = {"sweep_start": 0.0}

    def _iter_log(it: int, state: dict[str, dict], r_strat: float, stable_count: int) -> None:
        sweep_elapsed = time.perf_counter() - timing_state["sweep_start"]
        sweep_times.append(sweep_elapsed)
        print(f"[ITER {it}] r_strat={r_strat:.6g} stable_count={stable_count} sweep_time={sweep_elapsed:.2f}s")
        _print_q_offer_and_lam(regions=list(data.regions), state=state, tag=f"ITER {it}")
        timing_state["sweep_start"] = time.perf_counter()

    # Select solver function based on method and workers
    use_parallel = method == "jacobi" and workers > 1
    debug_workers = args.debug_workers

    # Override outlev for debug mode
    knitro_outlev = args.knitro_outlev
    if debug_workers and knitro_outlev is None:
        knitro_outlev = 1  # Verbose output in debug mode

    solver_opts = _solver_options(
        solver=solver,
        feastol=feastol,
        opttol=opttol,
        cfg=cfg,
        knitro_outlev=knitro_outlev,
        knitro_maxit=args.knitro_maxit,
        knitro_hessopt=args.knitro_hessopt,
        knitro_algorithm=args.knitro_algorithm,
    )
    print(f"[CONFIG] solver_options={solver_opts}")

    # Print worker workdirs in debug mode
    if use_parallel and debug_workers:
        print(f"[DEBUG] Per-player workdirs:")
        for p in data.players:
            print(f"  {p}: {os.path.join(workdir, f'worker_{p}')}")

    total_start = time.perf_counter()
    timing_state["sweep_start"] = total_start

    # === Warmup phase (optional) ===
    warmup_state = None
    warmup_iter_rows = []
    
    # Resolve warmup settings: CLI > RunConfig
    warmup_solver_arg = args.warmup_solver if args.warmup_solver is not None else cfg.warmup_solver
    
    if warmup_solver_arg:
        warmup_solver = warmup_solver_arg.strip()
        warmup_iters = int(args.warmup_iters) if args.warmup_iters is not None else int(cfg.warmup_iters)
        warmup_workers = int(args.warmup_workers) if args.warmup_workers is not None else int(cfg.warmup_workers)
        print(f"[WARMUP] Starting {warmup_iters} sweeps with {warmup_solver} (workers={warmup_workers})")
        
        warmup_opts = _solver_options(
            solver=warmup_solver,
            feastol=feastol,
            opttol=opttol,
            cfg=cfg,
        )
        print(f"[WARMUP] solver_options={warmup_opts}")
        
        if warmup_workers > 1:
            warmup_state, warmup_iter_rows = solve_jacobi_parallel(
                data,
                excel_path=excel_path,
                iters=warmup_iters,
                omega=omega,
                tol_rel=tol_rel,
                stable_iters=stable_iters,
                solver=warmup_solver,
                solver_options=warmup_opts,
                working_directory=workdir,
                iter_callback=_iter_log,
                workers=warmup_workers,
                worker_timeout=args.worker_timeout,
            )
        else:
            warmup_state, warmup_iter_rows = solve_jacobi(
                data,
                iters=warmup_iters,
                omega=omega,
                tol_rel=tol_rel,
                stable_iters=stable_iters,
                solver=warmup_solver,
                solver_options=warmup_opts,
                working_directory=workdir,
                iter_callback=_iter_log,
            )
        
        # Apply warmup state to data for main phase
        if warmup_state:
            print(f"[WARMUP] Complete. Transferring state to main solver...")
            # Update data with warmup state for warm start
            data.warmup_state = warmup_state
        
        timing_state["sweep_start"] = time.perf_counter()
    
    # === Main phase ===
    print(f"[MAIN] Starting {iters} sweeps with {solver} (workers={workers})")
    try:
        if use_parallel:
            state, iter_rows = solve_jacobi_parallel(
                data,
                excel_path=excel_path,
                iters=iters,
                omega=omega,
                tol_rel=tol_rel,
                stable_iters=stable_iters,
                solver=solver,
                solver_options=solver_opts,
                working_directory=workdir,
                iter_callback=_iter_log,
                workers=workers,
                worker_timeout=worker_timeout,
                initial_state=warmup_state,
            )
        else:
            solve_fn = solve_jacobi if method == "jacobi" else solve_gs
            state, iter_rows = solve_fn(
                data,
                iters=iters,
                omega=omega,
                tol_rel=tol_rel,
                stable_iters=stable_iters,
                solver=solver,
                solver_options=solver_opts,
                working_directory=workdir,
                iter_callback=_iter_log,
                initial_state=warmup_state,
            )
    finally:
        total_elapsed = time.perf_counter() - total_start
        print(f"\n[TIMING] Total solve time: {total_elapsed:.2f}s")
        if sweep_times:
            print(f"[TIMING] Mean sweep time: {sum(sweep_times)/len(sweep_times):.2f}s  (n={len(sweep_times)})")

    _print_q_offer_and_lam(regions=list(data.regions), state=state, tag="FINAL")

    write_results_excel(
        data=data,
        state=state,
        iter_rows=iter_rows,
        output_path=output_path,
        meta={
            "excel_path": excel_path,
            "method": method,
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
            "solver_options": str(solver_opts),
        },
    )

    write_default_plots(output_path=output_path, plots_dir=plots_dir)
    print(f"[OK] wrote: {output_path}")

    # --- Run LaTeX Generation Workflow ---
    try:
        sys.path.append(os.path.dirname(__file__))
        import generate_latex
        print("Running automatic LaTeX documentation generation...")
        generate_latex.main()
    except ImportError:
        print("[WARN] Could not import generate_latex.py. Skipping LaTeX generation.")
    except Exception as e:
        print(f"[WARN] Error during LaTeX generation: {e}")

    # --- Auto Git Push ---
    auto_git_push()

    # --- Cleanup workdir ---
    if not keep_workdir:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
            print(f"[CLEANUP] Deleted workdir: {workdir}")
        except Exception as e:
            print(f"[WARN] Could not delete workdir {workdir}: {e}")
    else:
        print(f"[KEEP] Workdir retained: {workdir}")

