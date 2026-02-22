from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
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

from solargeorisk_extension.data_prep import load_data_from_excel, load_data_intertemporal_from_excel
from solargeorisk_extension.gauss_seidel import solve_gs, solve_gs_intertemporal
from solargeorisk_extension.plot_results import write_default_plots
from solargeorisk_extension.results_writer import write_results_excel


# Define project root relative to this script
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

@dataclass(frozen=True)
class RunConfig:
    excel_path: str = os.path.join(PROJECT_ROOT, "inputs", "input_data_2024.xlsx")
    excel_path_intertemporal: str = os.path.join(PROJECT_ROOT, "inputs", "input_data_intertemporal.xlsx")
    out_dir: str = os.path.join(PROJECT_ROOT, "outputs")
    plots_dir: str = os.path.join(PROJECT_ROOT, "plots")

    solver: str = "knitro"
    feastol: float = 1e-4
    opttol: float = 1e-4
    
    method: str = "gauss_seidel"
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
    shuffle_players: bool = True
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
    kappa_q: float | None = 0.0
    rho_prox: float | None = 0.05
    use_quad: bool = True

    # Model variant: "single_year" or "intertemporal"
    model_type: str = "single_year"

    # Scenario name (e.g., "high_all", "low_all") to override init_q_offer
    scenario: str | None = "low_all"



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



def _print_state_summary(*, regions: list[str], state: dict[str, dict], tag: str = "SUMMARY") -> None:
    q_offer = state.get("Q_offer", {}) or {}
    lam = state.get("lam", {}) or {}

    if not regions:
        print(f"[{tag}] No regions configured; skipping Q_offer/lam print.")
        return

    # Support both single-key {r: val} and time-indexed {(r,t): val} state dicts
    def _val_for_region(d: dict, r: str) -> float:
        """Get scalar value for region r; if keyed by (r,t) tuples, sum across t."""
        if r in d:
            return _safe_float(d[r])
        vals = [v for (k0, *_), v in [(k if isinstance(k, tuple) else (k,), v) for k, v in d.items()] if k0 == r]
        return sum(vals) if vals else 0.0

    def _max_wedge(d: dict, r: str, regions: list) -> float:
        """Max tariff/tax set by r. Handles both (r,j) and (r,j,t) keys."""
        vals = []
        for j in regions:
            if j == r:
                continue
            # try single-year key (r, j)
            v = d.get((r, j))
            if v is not None:
                vals.append(_safe_float(v))
            else:
                # time-indexed keys (r, j, t)
                for k, w in d.items():
                    if isinstance(k, tuple) and len(k) == 3 and k[0] == r and k[1] == j:
                        vals.append(_safe_float(w))
        return max(vals) if vals else 0.0

    print(f"[{tag}] Q_offer (sum over t), lam (sum over t), and max wedges by region:")
    for r in regions:
        q_val = _val_for_region(q_offer, r)
        l_val = _val_for_region(lam, r)
        max_ti = _max_wedge(state.get("tau_imp", {}), r, regions)
        max_te = _max_wedge(state.get("tau_exp", {}), r, regions)
        print(f"  {r:<5} Q_offer={q_val:<8.4f} lam={l_val:<8.4f} mx_ti={max_ti:<8.4f} mx_te={max_te:<8.4f}")


def _gams_workdir(run_id: str, configured_workdir: str | None) -> str:
    if configured_workdir and configured_workdir.strip():
        workdir = os.path.abspath(configured_workdir.strip())
    else:
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
) -> Dict[str, float]:
    name = solver.strip().lower()

    if name == "conopt":
        return {"Tol_Feas_Max": float(feastol), "Tol_Optimality": float(opttol)}

    if name == "knitro":
        opts: Dict[str, float] = {
            "feastol": float(feastol),
            "opttol": float(opttol),
        }
        if cfg.knitro_outlev is not None:
            opts["outlev"] = float(cfg.knitro_outlev)
        if cfg.knitro_maxit is not None:
            opts["maxit"] = float(cfg.knitro_maxit)
        if cfg.knitro_hessopt is not None:
            opts["hessopt"] = float(cfg.knitro_hessopt)
        if cfg.knitro_algorithm is not None:
            opts["algorithm"] = float(cfg.knitro_algorithm)
        return opts

    return {}


def _apply_data_overrides(data, cfg: RunConfig) -> None:
    data.eps_x = float(cfg.eps_x)
    data.eps_comp = float(cfg.eps_comp)

    if cfg.rho_imp is not None:
        for r in data.regions:
            data.rho_imp[r] = float(cfg.rho_imp)

    if cfg.rho_exp is not None:
        for r in data.regions:
            data.rho_exp[r] = float(cfg.rho_exp)

    if cfg.kappa_q is not None and data.kappa_Q is not None:
        for r in data.regions:
            data.kappa_Q[r] = float(cfg.kappa_q)

    if data.settings is None:
        data.settings = {}
    if cfg.rho_prox is not None:
        data.settings["rho_prox"] = float(cfg.rho_prox)
    data.settings["use_quad"] = bool(cfg.use_quad)


def _build_initial_state(data, cfg: RunConfig) -> dict[str, dict] | None:
    scenario_name = cfg.init_scenario if cfg.init_scenario else cfg.scenario
    if not scenario_name:
        return None

    print(f"[CONFIG] Using init scenario: {scenario_name}")
    init_q_source = INIT_SCENARIOS.get(scenario_name)
    if init_q_source is None:
        print(f"[WARN] Unknown scenario '{scenario_name}'. Using fallback 'high_all'.")
        init_q_source = INIT_SCENARIOS["high_all"]

    init_q: Dict[str, float] = {}
    for r in data.players:
        frac = float(init_q_source.get(r, 0.8))
        init_q[r] = frac * float(data.Qcap[r])

    print(f"[CONFIG] Initial Q_offer: {init_q}")
    return {"Q_offer": init_q, "tau_imp": {}, "tau_exp": {}}


def _append_detailed_iter_rows(
    *,
    data,
    state: dict[str, dict],
    it: int,
    r_strat: float,
    stable_count: int,
    rows: list[dict[str, object]],
) -> None:
    q_map = state.get("Q_offer", {})
    lam_map = state.get("lam", {})
    mu_map = state.get("mu", {})
    beta_dem_map = state.get("beta_dem", {})
    psi_dem_map = state.get("psi_dem", {})
    obj_map = state.get("obj", {})
    x_map = state.get("x", {})
    gamma_map = state.get("gamma", {})
    tau_exp_map = state.get("tau_exp", {})
    tau_imp_map = state.get("tau_imp", {})

    for r in data.regions:
        row: dict[str, object] = {
            "iter": it,
            "r": r,
            "stable_count": stable_count,
            "r_strat": r_strat,
            "Q_offer": _safe_float(q_map.get(r)),
            "lam": _safe_float(lam_map.get(r)),
            "mu": _safe_float(mu_map.get(r)),
            "beta_dem": _safe_float(beta_dem_map.get(r)),
            "psi_dem": _safe_float(psi_dem_map.get(r)),
            "obj": _safe_float(obj_map.get(r)) if r in data.players else 0.0,
        }

        for dest in data.regions:
            row[f"x_exp_to_{dest}"] = _safe_float(x_map.get((r, dest)))
            row[f"gamma_exp_to_{dest}"] = _safe_float(gamma_map.get((r, dest)))
            row[f"tau_exp_to_{dest}"] = _safe_float(tau_exp_map.get((r, dest)))

        for src in data.regions:
            row[f"x_imp_from_{src}"] = _safe_float(x_map.get((src, r)))
            row[f"gamma_imp_from_{src}"] = _safe_float(gamma_map.get((src, r)))
            row[f"tau_imp_from_{src}"] = _safe_float(tau_imp_map.get((r, src)))

        rows.append(row)


def run(cfg: RunConfig) -> str:
    method = cfg.method.lower().strip()
    if method != "gauss_seidel":
        raise ValueError(f"Unsupported method '{cfg.method}'. Supported: 'gauss_seidel'.")

    is_intertemporal = cfg.model_type.strip().lower() == "intertemporal"

    if is_intertemporal:
        excel_path = _resolve_excel_path(cfg.excel_path_intertemporal, cfg.excel_path_intertemporal)
    else:
        excel_path = _resolve_excel_path(cfg.excel_path, cfg.excel_path)
    out_dir = cfg.out_dir
    plots_dir = cfg.plots_dir

    solver = cfg.solver.strip()
    feastol = float(cfg.feastol)
    opttol = float(cfg.opttol)
    iters = int(cfg.iters)
    omega = float(cfg.omega)
    tol_rel = float(cfg.tol_strat)
    tol_obj = float(cfg.tol_obj)
    stable_iters = int(cfg.stable_iters)
    convergence_mode = cfg.convergence_mode

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"results_{run_id}.xlsx")
    workdir = _gams_workdir(run_id, cfg.workdir)

    if is_intertemporal:
        data = load_data_intertemporal_from_excel(excel_path)
    else:
        data = load_data_from_excel(excel_path)
    _apply_data_overrides(data, cfg)

    print(f"[DEBUG] rho_exp (first 3): {list(data.rho_exp.items())[:3]}")
    print(f"[CONFIG] Model type: {cfg.model_type}")
    print(f"[CONFIG] Method: {method}")
    print(f"[CONFIG] Solver: {solver}  feastol={feastol:g}  opttol={opttol:g}")
    print(f"[CONFIG] iters={iters} omega={omega:g} tol_rel={tol_rel:g} stable_iters={stable_iters}")
    print(f"[CONFIG] eps_x={float(data.eps_x):g} eps_comp={float(data.eps_comp):g}")
    print(f"[CONFIG] convergence_mode={convergence_mode}")
    print(f"[CONFIG] workdir={workdir}{' (keep)' if cfg.keep_workdir else ' (auto-cleanup)'}")
    if cfg.workers != 1:
        print(f"[WARN] workers={cfg.workers} requested, but run_gs currently executes sequential GS only.")

    sweep_times: list[float] = []
    timing_state = {"sweep_start": 0.0}
    detailed_iter_rows: list[dict[str, object]] = []

    def _iter_log(it: int, state: dict[str, dict], r_strat: float, stable_count: int) -> None:
        sweep_elapsed = time.perf_counter() - timing_state["sweep_start"]
        sweep_times.append(sweep_elapsed)
        print(f"[ITER {it}] r_strat={r_strat:.6g} stable_count={stable_count} sweep_time={sweep_elapsed:.2f}s")
        # Show shuffled player order if available
        if "_sweep_order" in state:
            print(f"[ITER {it}] player order: {state['_sweep_order']}")
        _print_state_summary(regions=list(data.regions), state=state, tag=f"ITER {it}")
        _append_detailed_iter_rows(
            data=data,
            state=state,
            it=it,
            r_strat=r_strat,
            stable_count=stable_count,
            rows=detailed_iter_rows,
        )
        timing_state["sweep_start"] = time.perf_counter()

    if cfg.debug_workers and cfg.knitro_outlev is None and solver.strip().lower() == "knitro":
        # Keep debug mode behavior from previous script.
        cfg = dataclass_replace(cfg, knitro_outlev=1)

    solver_opts = _solver_options(
        solver=solver,
        feastol=feastol,
        opttol=opttol,
        cfg=cfg,
    )
    print(f"[CONFIG] solver_options={solver_opts}")

    total_start = time.perf_counter()
    timing_state["sweep_start"] = total_start
    init_state = _build_initial_state(data, cfg)
    print(f"[MAIN] Starting {iters} sweeps with {solver}")

    try:
        if is_intertemporal:
            state, iter_rows = solve_gs_intertemporal(
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
                initial_state=init_state,
                convergence_mode=convergence_mode,
                shuffle_players=cfg.shuffle_players,
            )
        else:
            state, iter_rows = solve_gs(
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
                initial_state=init_state,
                convergence_mode=convergence_mode,
                shuffle_players=cfg.shuffle_players,
            )
    finally:
        total_elapsed = time.perf_counter() - total_start
        print(f"\n[TIMING] Total solve time: {total_elapsed:.2f}s")
        if sweep_times:
            print(f"[TIMING] Mean sweep time: {sum(sweep_times)/len(sweep_times):.2f}s  (n={len(sweep_times)})")

    _print_state_summary(regions=list(data.regions), state=state, tag="FINAL")

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

    try:
        write_default_plots(output_path=output_path, plots_dir=plots_dir)
    except Exception as e:
        print(f"[WARN] Plot generation failed: {e}")
    print(f"[OK] wrote: {output_path}")

    if not cfg.keep_workdir:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
            print(f"[CLEANUP] Deleted workdir: {workdir}")
        except Exception as e:
            print(f"[WARN] Could not delete workdir {workdir}: {e}")
    else:
        print(f"[KEEP] Workdir retained: {workdir}")

    return output_path


def dataclass_replace(cfg: RunConfig, **kwargs) -> RunConfig:
    values = dict(cfg.__dict__)
    values.update(kwargs)
    return RunConfig(**values)


def main() -> None:
    run(RunConfig())


if __name__ == "__main__":
    main()
