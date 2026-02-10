from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Set PYTHONPATH so parallel workers (spawned subprocesses) can find the package
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if "PYTHONPATH" in os.environ:
    if src_path not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = src_path + os.pathsep + os.environ["PYTHONPATH"]
else:
    os.environ["PYTHONPATH"] = src_path

from solargeorisk_extension.data_prep import load_data_from_excel
from solargeorisk_extension.gauss_jacobi import solve_jacobi, solve_jacobi_parallel
from solargeorisk_extension.gauss_seidel import solve_gs
from solargeorisk_extension.plot_results import write_default_plots
from solargeorisk_extension.results_writer import write_results_excel



# Define project root relative to this script
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

@dataclass(frozen=True)
class RunConfig:
    excel_path: str = os.path.join(PROJECT_ROOT, "inputs", "input_data.xlsx")
    out_dir: str = os.path.join(PROJECT_ROOT, "outputs")
    plots_dir: str = os.path.join(PROJECT_ROOT, "plots")

    solver: str = "knitro"
    feastol: float = 1e-4
    opttol: float = 1e-4
    
    method: str = "jacobi"
    iters: int = 100
    omega: float = 0.7
    tol_strat: float = 1e-2  # Renamed from tol_rel for clarity, but tol_rel CLI arg maps here
    tol_obj: float = 1e-2
    stable_iters: int = 3
    eps_x: float = 1e-3
    eps_comp: float = 1e-4
    workdir: str | None = None
    convergence_mode: str = "combined"  # "strategy", "objective", or "combined"
    workers: int = 1  # 1=sequential, >1=parallel
    worker_timeout: float = 120.0
    player_order: List[str] | None = None
    init_scenario: str | None = None
    warmup_solver: str | None = None
    warmup_iters: int = 5
    warmup_workers: int = 1

    keep_workdir: bool = False
    debug_workers: bool = False

    knitro_outlev: int | None = None
    knitro_maxit: int | None = None
    knitro_hessopt: int | None = None
    knitro_algorithm: int | None = None


    rho_imp: float | None = 0.05
    rho_exp: float | None = 0.05
    kappa_q: float | None = 1
    rho_prox: float | None = 0.05
    use_quad: bool = True

    
    # Scenario name (e.g., "high_all", "low_all") to override init_q_offer
    scenario: str | None = "low_all"


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


# Scenario definitions (fractions of Qcap)
INIT_SCENARIOS = {
    "high_all": {"ch": 0.8, "eu": 0.8, "us": 0.8, "apac": 0.8, "roa": 0.8, "row": 0.8},
    "low_non_ch": {"ch": 0.8, "eu": 0.0, "us": 0.0, "apac": 0.8, "roa": 0.0, "row": 0.0},
    "low_eu_us_row": {"ch": 0.8, "eu": 0.0, "us": 0.0, "apac": 0.8, "roa": 0.8, "row": 0.0},
    "mid_all": {"ch": 0.5, "eu": 0.5, "us": 0.5, "apac": 0.5, "roa": 0.5, "row": 0.5},
    "low_all": {"ch": 0.2, "eu": 0.0, "us": 0.0, "apac": 0.2, "roa": 0.0, "row": 0.0},
}




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

    print(f"[{tag}] Q_offer, lam, and max wedges by region:")
    for r in regions:
        q_val = _safe_float(q_offer.get(r))
        l_val = _safe_float(lam.get(r))
        
        # Calculate max wedges set BY this region
        # tau_imp[r, j]: r is importer, sets tariff on j
        t_i_vals = [state.get("tau_imp", {}).get((r, j), 0.0) for j in regions if j != r]
        max_ti = max(t_i_vals) if t_i_vals else 0.0
        
        # tau_exp[r, j]: r is exporter, sets tax on j
        t_e_vals = [state.get("tau_exp", {}).get((r, j), 0.0) for j in regions if j != r]
        max_te = max(t_e_vals) if t_e_vals else 0.0
        
        print(f"  {r:<5} Q_offer={q_val:<8.4f} lam={l_val:<8.4f} mx_ti={max_ti:<8.4f} mx_te={max_te:<8.4f}")


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
    name = solver.strip().lower()

    if name == "conopt":
        return {"Tol_Feas_Max": float(feastol), "Tol_Optimality": float(opttol)}

    if name == "knitro":
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
    cfg = RunConfig()

    excel_path = _resolve_excel_path(cfg.excel_path, cfg.excel_path)
    out_dir = cfg.out_dir
    plots_dir = cfg.plots_dir

    solver = cfg.solver.strip()
    feastol = float(cfg.feastol)
    opttol = float(cfg.opttol)

    iters = int(cfg.iters)
    omega = float(cfg.omega)
    tol_strat = float(cfg.tol_strat)
    tol_obj = float(cfg.tol_obj)
    
    # Backward compatibility for script internals using tol_rel
    tol_rel = tol_strat

    stable_iters = int(cfg.stable_iters)

    method = cfg.method.lower().strip()

    workers = cfg.workers
    worker_timeout = cfg.worker_timeout
    keep_workdir = cfg.keep_workdir
    
    convergence_mode = cfg.convergence_mode

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    output_path = os.path.join(out_dir, f"results_{run_id}.xlsx")
    workdir = _gams_workdir(run_id)

    data = load_data_from_excel(excel_path)

    # Apply eps_x
    if cfg.eps_x is not None:
        data.eps_x = float(cfg.eps_x)

    # Apply eps_comp
    if cfg.eps_comp is not None:
        data.eps_comp = float(cfg.eps_comp)

    # Apply Penalty Overrides
    if cfg.rho_imp is not None:
        for r in data.regions:
            data.rho_imp[r] = float(cfg.rho_imp)

    if cfg.rho_exp is not None:
        for r in data.regions:
            data.rho_exp[r] = float(cfg.rho_exp)

    if cfg.kappa_q is not None and data.kappa_Q is not None:
        for r in data.regions:
            data.kappa_Q[r] = float(cfg.kappa_q)
            
    # Apply Proximal Regularization
    if data.settings is None:
        data.settings = {}
    if cfg.rho_prox is not None:
        data.settings["rho_prox"] = float(cfg.rho_prox)
    data.settings["use_quad"] = cfg.use_quad

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

    # Detailed iteration history (list of dicts)
    detailed_iter_rows: list[dict] = []

    def _iter_log(it: int, state: dict[str, dict], r_strat: float, stable_count: int) -> None:
        sweep_elapsed = time.perf_counter() - timing_state["sweep_start"]
        sweep_times.append(sweep_elapsed)
        print(f"[ITER {it}] r_strat={r_strat:.6g} stable_count={stable_count} sweep_time={sweep_elapsed:.2f}s")
        _print_q_offer_and_lam(regions=list(data.regions), state=state, tag=f"ITER {it}")
        
        # Collect detailed state
        # state has keys: Q_offer, tau_imp, tau_exp, lam, obj, etc.
        # We flatten this into one row per player (or just one row per iter with many cols?)
        # User asked for "per iteration and per player all strategies ... and all market outcome variables"
        # So we want a table like:
        # iter | player | Q_offer | lam | obj | ...
        
        # Extract variables
        q_map = state.get("Q_offer", {})
        lam_map = state.get("lam", {})
        mu_map = state.get("mu", {})
        beta_dem_map = state.get("beta_dem", {})
        psi_dem_map = state.get("psi_dem", {})
        obj_map = state.get("obj", {})
        x_map = state.get("x", {}) # (exp, imp) -> float
        gamma_map = state.get("gamma", {}) # (exp, imp) -> float
        # tau_imp has keys (imp, exp)
        # tau_exp has keys (exp, imp)
        
        for r in data.regions: # Loop over all regions (players included)
             row = {
                 "iter": it,
                 "r": r,
                 "stable_count": stable_count,
                 "r_strat": r_strat,
                 "Q_offer": _safe_float(q_map.get(r)),
                 "lam": _safe_float(lam_map.get(r)),
                 "mu": _safe_float(mu_map.get(r)), # Dual of capacity constraint
                 "beta_dem": _safe_float(beta_dem_map.get(r)), # Dual of Dmax constraint
                 "psi_dem": _safe_float(psi_dem_map.get(r)), # Dual of non-neg D constraint?
                 "obj": _safe_float(obj_map.get(r)) if r in data.players else 0.0,
                 # Summarize flows? Or user wants ALL vars?
                 # "all market outcome variables" implies everything.
                 # x(r, *) exports, x(*, r) imports?
                 # Writing 36 flow columns per player * 50 iters is fine.
             }
             
             # Add flows FROM r (exports)
             for dest in data.regions:
                 row[f"x_exp_to_{dest}"] = _safe_float(x_map.get((r, dest)))
                 row[f"gamma_exp_to_{dest}"] = _safe_float(gamma_map.get((r, dest)))
                 row[f"tau_exp_to_{dest}"] = _safe_float(state.get("tau_exp", {}).get((r, dest)))
             
             # Add flows TO r (imports)
             for src in data.regions:
                 row[f"x_imp_from_{src}"] = _safe_float(x_map.get((src, r)))
                 row[f"gamma_imp_from_{src}"] = _safe_float(gamma_map.get((src, r))) # Note: gamma is on (exp, imp) edge? Check def.
                 # gamma is domain [exp, imp], so key is (src, r) where src is exp, r is imp. YES.
                 row[f"tau_imp_from_{src}"] = _safe_float(state.get("tau_imp", {}).get((r, src))) # Note key (r, src) for tau_imp means r is IMPORTER
                 
             detailed_iter_rows.append(row)

        timing_state["sweep_start"] = time.perf_counter()

    # Select solver function based on method and workers
    use_parallel = method == "jacobi" and workers > 1
    debug_workers = cfg.debug_workers

    # Override outlev for debug mode
    knitro_outlev = cfg.knitro_outlev
    if debug_workers and knitro_outlev is None:
        knitro_outlev = 1  # Verbose output in debug mode

    solver_opts = _solver_options(
        solver=solver,
        feastol=feastol,
        opttol=opttol,
        cfg=cfg,
        knitro_outlev=knitro_outlev,
        knitro_maxit=cfg.knitro_maxit,
        knitro_hessopt=cfg.knitro_hessopt,
        knitro_algorithm=cfg.knitro_algorithm,
    )
    print(f"[CONFIG] solver_options={solver_opts}")

    # Print worker workdirs in debug mode
    if use_parallel and debug_workers:
        print(f"[DEBUG] Per-player workdirs:")
        for p in data.players:
            print(f"  {p}: {os.path.join(workdir, f'worker_{p}')}")

    total_start = time.perf_counter()
    timing_state["sweep_start"] = total_start

    # === Build initial state from CLI/Config scenario ===
    # This allows selecting different equilibria by starting from different points
    init_state: dict[str, dict] | None = None
    
    init_q_source = None
    
    # Priority: RunConfig.init_scenario > Default "high_all"
    scenario_name = "high_all" # Default
    if cfg.init_scenario:
        scenario_name = cfg.init_scenario
    elif cfg.scenario:
        scenario_name = cfg.scenario
        
    if scenario_name:
        print(f"[CONFIG] Using init scenario: {scenario_name}")
        if scenario_name in INIT_SCENARIOS:
            init_q_source = INIT_SCENARIOS[scenario_name]
        else:
            print(f"[WARN] Unknown scenario '{scenario_name}'. using default high_all.")
            init_q_source = INIT_SCENARIOS["high_all"]

    if init_q_source:
        init_q = {}
        for r in data.players:
            frac = init_q_source.get(r, 0.8)  # Default 80% if not specified
            init_q[r] = frac * float(data.Qcap[r])
        init_state = {"Q_offer": init_q, "tau_imp": {}, "tau_exp": {}}
        print(f"[CONFIG] Custom initial Q_offer: {init_q}")

    # === Warmup phase (optional) ===
    warmup_state = init_state  # Start warmup from custom init if provided
    warmup_iter_rows = []
    
    # Resolve warmup settings
    warmup_solver_arg = cfg.warmup_solver
    
    if warmup_solver_arg:
        warmup_solver = warmup_solver_arg.strip()
        warmup_iters = int(cfg.warmup_iters)
        warmup_workers = int(cfg.warmup_workers)
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
                excel_path=cfg.input_file, # Use corrected input path?
                iters=warmup_iters,
                omega=omega,
                tol_rel=cfg.tol_strat,
                tol_obj=cfg.tol_obj,
                stable_iters=stable_iters,
                solver=warmup_solver,
                solver_options=warmup_opts,
                working_directory=workdir,
                iter_callback=_iter_log,
                workers=warmup_workers,
                worker_timeout=cfg.worker_timeout,
                convergence_mode=convergence_mode,
            )
        else:
            warmup_state, warmup_iter_rows = solve_jacobi(
                data,
                iters=warmup_iters,
                omega=omega,
                tol_rel=cfg.tol_strat,
                tol_obj=cfg.tol_obj,
                stable_iters=stable_iters,
                solver=warmup_solver,
                solver_options=warmup_opts,
                working_directory=workdir,
                iter_callback=_iter_log,
                convergence_mode=convergence_mode,
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
        if workers > 1:
            # parallel
            state, iter_rows = solve_jacobi_parallel(
                data,
                excel_path="inputs/input_data.xlsx", # Use standard path or resolve?
                iters=iters,
                omega=omega,
                tol_rel=cfg.tol_strat,
                tol_obj=cfg.tol_obj,
                stable_iters=stable_iters,
                solver=solver,
                solver_options=solver_opts,
                working_directory=workdir,
                iter_callback=_iter_log,
                workers=workers,
                worker_timeout=cfg.worker_timeout,
                initial_state=warmup_state,
                convergence_mode=convergence_mode,
            )
        else:
            # sequential
            solve_fn = solve_jacobi if cfg.method == "jacobi" else solve_gs
            state, iter_rows = solve_fn(
                data,
                solver=solver,
                solver_options=solver_opts,
                iters=iters,
                omega=omega,
                tol_rel=cfg.tol_strat,
                tol_obj=cfg.tol_obj,
                stable_iters=stable_iters,
                working_directory=workdir,
                iter_callback=_iter_log,
                initial_state=warmup_state,
                convergence_mode=convergence_mode,
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
        detailed_iter_rows=detailed_iter_rows,
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

