"""
Gauss-Jacobi diagonalization solver for EPEC.

Unlike Gauss-Seidel which updates strategies immediately after each player's
best response (so subsequent players see updated values within the same sweep),
Gauss-Jacobi freezes the strategy profile at the start of each sweep, computes
ALL players' best responses against the frozen profile, then updates all
strategies simultaneously at the end of the sweep.

This can provide different convergence properties and is sometimes more stable
for certain problem structures.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Tuple

from .model import ModelData, apply_player_fixings, build_model, extract_state


def solve_jacobi(
    data: ModelData,
    *,
    iters: int = 50,
    omega: float = 0.8,
    tol_rel: float = 1e-4,
    stable_iters: int = 3,
    solver: str = "conopt",
    solver_options: Dict[str, float] | None = None,
    working_directory: str | None = None,
    iter_callback: Callable[[int, Dict[str, Dict], float, int], None] | None = None,
) -> tuple[Dict[str, Dict], List[Dict[str, object]]]:
    """
    Solve EPEC using Gauss-Jacobi diagonalization.

    Jacobi semantics: At each sweep k:
      1. Freeze theta_old = theta_current
      2. For each player r: compute best response against theta_old
      3. After ALL players solved: theta_new = (1-omega)*theta_old + omega*theta_br
      4. Check convergence using theta_new vs theta_old

    Args:
        data: Model data containing regions, players, parameters.
        iters: Maximum number of sweeps.
        omega: Damping factor in (0, 1]. 1.0 = no damping.
        tol_rel: Relative tolerance for convergence.
        stable_iters: Require this many consecutive sweeps below tolerance.
        solver: GAMS solver name.
        solver_options: Solver-specific options dict.
        working_directory: GAMS working directory (must be space-free).
        iter_callback: Optional callback(iter, state, r_strat, stable_count).

    Returns:
        Tuple of (final_state, iter_rows) matching solve_gs API.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1].")
    if tol_rel <= 0.0:
        raise ValueError("tol_rel must be > 0")
    if stable_iters < 1:
        raise ValueError("stable_iters must be >= 1")

    ctx = build_model(data, working_directory=working_directory)

    # Current strategy profile
    theta_Q: Dict[str, float] = {r: 0.8 * float(data.Qcap[r]) for r in data.players}
    theta_tau_imp: Dict[Tuple[str, str], float] = {
        (imp, exp): 0.0 for imp in data.regions for exp in data.regions
    }
    theta_tau_exp: Dict[Tuple[str, str], float] = {
        (exp, imp): 0.0 for exp in data.regions for imp in data.regions
    }

    def _rel_change(new: float, old: float, scale: float) -> float:
        return abs(new - old) / max(abs(old), scale)

    def _q_scale(r: str) -> float:
        qc = float(data.Qcap.get(r, 0.0))
        return max(0.01 * qc, 1.0)

    def _ti_scale(imp: str, exp: str) -> float:
        ub = float(data.tau_imp_ub[(imp, exp)])
        return max(0.01 * ub, 1.0)

    def _te_scale(exp: str, imp: str) -> float:
        ub = float(data.tau_exp_ub[(exp, imp)])
        return max(0.01 * ub, 1.0)

    iter_rows: List[Dict[str, object]] = []
    stable_count = 0
    last_state: Dict[str, Dict] = {}

    solve_kwargs = {"solver": solver}
    if solver_options:
        solve_kwargs["solver_options"] = solver_options

    for it in range(1, iters + 1):
        # === JACOBI: Freeze profile at sweep start ===
        theta_old_Q = deepcopy(theta_Q)
        theta_old_ti = deepcopy(theta_tau_imp)
        theta_old_te = deepcopy(theta_tau_exp)

        # Storage for best responses
        theta_br_Q: Dict[str, float] = {}
        theta_br_ti: Dict[Tuple[str, str], float] = {}
        theta_br_te: Dict[Tuple[str, str], float] = {}

        # Compute best response for each player against FROZEN profile
        for p in data.players:
            apply_player_fixings(
                ctx, data, theta_old_Q, theta_old_ti, theta_old_te, player=p
            )
            ctx.models[p].solve(**solve_kwargs)

            state = extract_state(ctx)
            last_state = state

            Q_sol = state.get("Q_offer", {})
            ti_sol = state.get("tau_imp", {})
            te_sol = state.get("tau_exp", {})

            # Store best response for Q_offer
            if p in Q_sol:
                theta_br_Q[p] = float(Q_sol[p])

            # Store best responses for tau_imp controlled by player p
            for exp in data.regions:
                key = (p, exp)
                if p == exp:
                    continue
                if key in ti_sol:
                    theta_br_ti[key] = float(ti_sol[key])

            # Store best responses for tau_exp controlled by player p
            for imp in data.regions:
                key = (p, imp)
                if p == imp:
                    continue
                if key in te_sol:
                    theta_br_te[key] = float(te_sol[key])

        # Sanity check: all players should have best responses
        assert set(theta_br_Q.keys()) == set(data.players), (
            f"Jacobi sanity check failed: theta_br_Q keys {set(theta_br_Q.keys())} "
            f"!= players {set(data.players)}"
        )

        # === JACOBI: Simultaneous update with damping ===
        for r in data.players:
            if r in theta_br_Q:
                theta_Q[r] = (1.0 - omega) * theta_old_Q[r] + omega * theta_br_Q[r]

        for key in theta_br_ti:
            theta_tau_imp[key] = (
                (1.0 - omega) * theta_old_ti[key] + omega * theta_br_ti[key]
            )

        for key in theta_br_te:
            theta_tau_exp[key] = (
                (1.0 - omega) * theta_old_te[key] + omega * theta_br_te[key]
            )

        # === Compute convergence metric: theta_new vs theta_old ===
        r_strat = 0.0
        for r in data.players:
            r_strat = max(
                r_strat, _rel_change(theta_Q[r], theta_old_Q[r], _q_scale(r))
            )
        for imp in data.regions:
            for exp in data.regions:
                if imp == exp:
                    continue
                key = (imp, exp)
                r_strat = max(
                    r_strat,
                    _rel_change(theta_tau_imp[key], theta_old_ti[key], _ti_scale(imp, exp)),
                )
        for exp in data.regions:
            for imp in data.regions:
                if exp == imp:
                    continue
                key = (exp, imp)
                r_strat = max(
                    r_strat,
                    _rel_change(theta_tau_exp[key], theta_old_te[key], _te_scale(exp, imp)),
                )

        stable_count = stable_count + 1 if r_strat <= tol_rel else 0
        iter_rows.append({
            "iter": it,
            "r_strat": float(r_strat),
            "stable_count": int(stable_count),
            "omega": float(omega),
        })

        if iter_callback is not None:
            iter_callback(it, last_state, float(r_strat), int(stable_count))

        if stable_count >= stable_iters:
            break

    return last_state, iter_rows


# =============================================================================
# PARALLEL JACOBI IMPLEMENTATION
# =============================================================================

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# Process-global cache for ctx reuse (one ctx per player per process)
_CTX_CACHE: Dict[str, object] = {}
_DATA_CACHE: object = None
_EXCEL_PATH_CACHE: str = ""


def _worker_init(excel_path: str, eps_x: float | None, eps_comp: float | None) -> None:
    """
    Initializer for worker processes. Loads data once per process.
    Called once when worker process starts.
    """
    global _DATA_CACHE, _EXCEL_PATH_CACHE, _CTX_CACHE
    from .data_prep import load_data_from_excel
    
    _EXCEL_PATH_CACHE = excel_path
    _DATA_CACHE = load_data_from_excel(excel_path)
    if eps_x is not None:
        _DATA_CACHE.eps_x = float(eps_x)
    if eps_comp is not None:
        _DATA_CACHE.eps_comp = float(eps_comp)
    _CTX_CACHE = {}  # Will be populated on demand per player


def _solve_player_br_cached(
    player: str,
    player_workdir: str,
    theta_old_Q: Dict[str, float],
    theta_old_ti: Dict[Tuple[str, str], float],
    theta_old_te: Dict[Tuple[str, str], float],
    solver: str,
    solver_options: Dict[str, float] | None,
) -> Dict[str, object]:
    """
    Worker function using process-global cached data and ctx.
    Returns strategies (Q_offer, tau_imp, tau_exp) plus lam for this player.
    """
    global _DATA_CACHE, _CTX_CACHE
    from .model import apply_player_fixings, build_model, extract_state
    
    t0 = time.perf_counter()
    data = _DATA_CACHE
    regions = list(data.regions)
    
    # Build or reuse ctx for this player
    if player not in _CTX_CACHE:
        os.makedirs(player_workdir, exist_ok=True)
        _CTX_CACHE[player] = build_model(data, working_directory=player_workdir)
    ctx = _CTX_CACHE[player]
    
    # Apply fixings for this player against frozen theta
    apply_player_fixings(ctx, data, theta_old_Q, theta_old_ti, theta_old_te, player=player)
    
    # Solve
    solve_kwargs = {"solver": solver}
    if solver_options:
        solve_kwargs["solver_options"] = solver_options
    
    try:
        ctx.models[player].solve(**solve_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Solver failed for player '{player}' in workdir '{player_workdir}': {e}"
        ) from e
    
    # Extract full state (includes lam)
    state = extract_state(ctx)
    Q_sol = state.get("Q_offer", {})
    ti_sol = state.get("tau_imp", {})
    te_sol = state.get("tau_exp", {})
    lam_sol = state.get("lam", {})
    
    result: Dict[str, object] = {
        "player": player,
        "player_workdir": player_workdir,
        "solve_time": time.perf_counter() - t0,
    }
    
    # Q_offer for player
    if player in Q_sol:
        result["Q_offer"] = float(Q_sol[player])
    
    # tau_imp controlled by player
    result["tau_imp"] = {}
    for exp in regions:
        if exp == player:
            continue
        key = (player, exp)
        if key in ti_sol:
            result["tau_imp"][key] = float(ti_sol[key])
    
    # tau_exp controlled by player
    result["tau_exp"] = {}
    for imp in regions:
        if imp == player:
            continue
        key = (player, imp)
        if key in te_sol:
            result["tau_exp"][key] = float(te_sol[key])
    
    # Full lam from this solve (for reference)
    result["lam"] = {r: float(lam_sol.get(r, 0.0)) for r in regions}
    
    return result


def solve_jacobi_parallel(
    data: "ModelData",
    *,
    excel_path: str,
    iters: int = 50,
    omega: float = 0.8,
    tol_rel: float = 1e-4,
    stable_iters: int = 3,
    solver: str = "conopt",
    solver_options: Dict[str, float] | None = None,
    working_directory: str | None = None,
    iter_callback: Callable[[int, Dict[str, Dict], float, int], None] | None = None,
    workers: int = 4,
    worker_timeout: float = 120.0,
) -> tuple[Dict[str, Dict], List[Dict[str, object]]]:
    """
    Solve EPEC using PARALLEL Gauss-Jacobi diagonalization.
    
    Uses ProcessPoolExecutor to solve all players' best responses in parallel.
    Executor is created ONCE and reused across all sweeps.
    Each worker process caches ctx per player for warm starts.
    
    Args:
        data: ModelData instance.
        excel_path: Path to input Excel file (for worker initialization).
        iters: Maximum number of sweeps.
        omega: Damping factor in (0, 1].
        tol_rel: Relative tolerance for convergence.
        stable_iters: Consecutive stable sweeps required.
        solver: GAMS solver name.
        solver_options: Solver options dict (must be JSON/pickle-friendly).
        working_directory: Base GAMS workdir (per-player dirs created under it).
        iter_callback: Optional callback(iter, state, r_strat, stable_count).
        workers: Number of parallel workers.
        worker_timeout: Timeout in seconds for each worker solve (default 120s).
    
    Returns:
        Tuple of (final_state, iter_rows) matching solve_jacobi API.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1].")
    if tol_rel <= 0.0:
        raise ValueError("tol_rel must be > 0")
    if stable_iters < 1:
        raise ValueError("stable_iters must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    
    data_regions = list(data.regions)
    data_players = list(data.players)
    data_Qcap = dict(data.Qcap)
    data_tau_imp_ub = dict(data.tau_imp_ub)
    data_tau_exp_ub = dict(data.tau_exp_ub)
    
    # Create persistent per-player workdirs
    base_workdir = working_directory or os.path.join(os.getcwd(), "_gams_workdir")
    player_workdirs = {}
    for p in data_players:
        pdir = os.path.join(base_workdir, f"worker_{p}")
        os.makedirs(pdir, exist_ok=True)
        player_workdirs[p] = pdir
    
    # Initialize strategy profile
    theta_Q: Dict[str, float] = {r: 0.8 * float(data_Qcap.get(r, 0.0)) for r in data_players}
    theta_tau_imp: Dict[Tuple[str, str], float] = {
        (imp, exp): 0.0 for imp in data_regions for exp in data_regions
    }
    theta_tau_exp: Dict[Tuple[str, str], float] = {
        (exp, imp): 0.0 for exp in data_regions for imp in data_regions
    }
    
    def _rel_change(new: float, old: float, scale: float) -> float:
        return abs(new - old) / max(abs(old), scale)
    
    def _q_scale(r: str) -> float:
        qc = float(data_Qcap.get(r, 0.0))
        return max(0.01 * qc, 1.0)
    
    def _ti_scale(imp: str, exp: str) -> float:
        ub = float(data_tau_imp_ub.get((imp, exp), 1.0))
        return max(0.01 * ub, 1.0)
    
    def _te_scale(exp: str, imp: str) -> float:
        ub = float(data_tau_exp_ub.get((exp, imp), 1.0))
        return max(0.01 * ub, 1.0)
    
    iter_rows: List[Dict[str, object]] = []
    stable_count = 0
    ref_lam: Dict[str, float] = {}  # Reference lam from first player
    
    # Limit workers to number of players
    actual_workers = min(workers, len(data_players))
    ref_player = data_players[0]  # Reference player for lam
    
    # Create executor ONCE and reuse across all sweeps
    with ProcessPoolExecutor(
        max_workers=actual_workers,
        initializer=_worker_init,
        initargs=(excel_path, float(data.eps_x), float(data.eps_comp)),
    ) as executor:
        for it in range(1, iters + 1):
            sweep_start = time.perf_counter()
            
            # === JACOBI: Freeze profile at sweep start ===
            theta_old_Q = deepcopy(theta_Q)
            theta_old_ti = deepcopy(theta_tau_imp)
            theta_old_te = deepcopy(theta_tau_exp)
            
            # Storage for best responses
            theta_br_Q: Dict[str, float] = {}
            theta_br_ti: Dict[Tuple[str, str], float] = {}
            theta_br_te: Dict[Tuple[str, str], float] = {}
            solve_times: List[float] = []
            
            # Submit all tasks
            futures = {
                executor.submit(
                    _solve_player_br_cached,
                    p,
                    player_workdirs[p],
                    theta_old_Q,
                    theta_old_ti,
                    theta_old_te,
                    solver,
                    solver_options,
                ): p
                for p in data_players
            }
            
            # Collect results with timeout
            for future in as_completed(futures):
                player = futures[future]
                try:
                    result = future.result(timeout=worker_timeout)
                    solve_times.append(result.get("solve_time", 0.0))
                    
                    # Collect Q_offer
                    if "Q_offer" in result:
                        theta_br_Q[result["player"]] = result["Q_offer"]
                    
                    # Collect tau_imp
                    for key, val in result.get("tau_imp", {}).items():
                        theta_br_ti[key] = val
                    
                    # Collect tau_exp
                    for key, val in result.get("tau_exp", {}).items():
                        theta_br_te[key] = val
                    
                    # Use lam from reference player
                    if result["player"] == ref_player:
                        ref_lam = result.get("lam", {})
                        
                except FuturesTimeoutError:
                    pdir = player_workdirs.get(player, "unknown")
                    raise RuntimeError(
                        f"Worker for player '{player}' timed out after {worker_timeout}s.\n"
                        f"  Workdir: {pdir}\n"
                        f"  Consider increasing --worker-timeout or checking solver stability."
                    )
                except Exception as e:
                    pdir = player_workdirs.get(player, "unknown")
                    raise RuntimeError(
                        f"Worker for player '{player}' failed.\n"
                        f"  Workdir: {pdir}\n"
                        f"  Check .lst/.log files in that directory.\n"
                        f"  Error: {e}"
                    ) from e
            
            sweep_time = time.perf_counter() - sweep_start
            
            # Sanity check
            assert set(theta_br_Q.keys()) == set(data_players), (
                f"Parallel Jacobi sanity check failed: theta_br_Q keys {set(theta_br_Q.keys())} "
                f"!= players {set(data_players)}"
            )
            
            # === JACOBI: Simultaneous update with damping ===
            for r in data_players:
                if r in theta_br_Q:
                    theta_Q[r] = (1.0 - omega) * theta_old_Q[r] + omega * theta_br_Q[r]
            
            for key in theta_br_ti:
                theta_tau_imp[key] = (1.0 - omega) * theta_old_ti[key] + omega * theta_br_ti[key]
            
            for key in theta_br_te:
                theta_tau_exp[key] = (1.0 - omega) * theta_old_te[key] + omega * theta_br_te[key]
            
            # === Compute convergence metric ===
            r_strat = 0.0
            for r in data_players:
                r_strat = max(r_strat, _rel_change(theta_Q[r], theta_old_Q[r], _q_scale(r)))
            for imp in data_regions:
                for exp in data_regions:
                    if imp == exp:
                        continue
                    key = (imp, exp)
                    r_strat = max(
                        r_strat,
                        _rel_change(theta_tau_imp[key], theta_old_ti[key], _ti_scale(imp, exp)),
                    )
            for exp in data_regions:
                for imp in data_regions:
                    if exp == imp:
                        continue
                    key = (exp, imp)
                    r_strat = max(
                        r_strat,
                        _rel_change(theta_tau_exp[key], theta_old_te[key], _te_scale(exp, imp)),
                    )
            
            stable_count = stable_count + 1 if r_strat <= tol_rel else 0
            
            # Timing diagnostics
            solve_sum = sum(solve_times) if solve_times else 0.0
            solve_max = max(solve_times) if solve_times else 0.0
            solve_mean = solve_sum / len(solve_times) if solve_times else 0.0
            
            iter_rows.append({
                "iter": it,
                "r_strat": float(r_strat),
                "stable_count": int(stable_count),
                "omega": float(omega),
                "sweep_time": float(sweep_time),
                "solve_time_sum": float(solve_sum),
                "solve_time_max": float(solve_max),
                "solve_time_mean": float(solve_mean),
            })
            
            # State for callback with real lam from reference player
            state = {
                "Q_offer": dict(theta_Q),
                "tau_imp": dict(theta_tau_imp),
                "tau_exp": dict(theta_tau_exp),
                "lam": dict(ref_lam),
            }
            
            if iter_callback is not None:
                iter_callback(it, state, float(r_strat), int(stable_count))
            
            if stable_count >= stable_iters:
                break
    
    # === Final evaluation solve for complete state ===
    # Do one serial solve in main process to get full equilibrium state
    final_workdir = os.path.join(base_workdir, "final_state")
    os.makedirs(final_workdir, exist_ok=True)
    
    ctx = build_model(data, working_directory=final_workdir)
    p0 = data_players[0]
    apply_player_fixings(ctx, data, theta_Q, theta_tau_imp, theta_tau_exp, player=p0)
    
    solve_kwargs = {"solver": solver}
    if solver_options:
        solve_kwargs["solver_options"] = solver_options
    ctx.models[p0].solve(**solve_kwargs)
    
    final_state = extract_state(ctx)
    
    return final_state, iter_rows

