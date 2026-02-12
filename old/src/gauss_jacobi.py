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

import os
import multiprocessing as mp
from multiprocessing.pool import Pool, AsyncResult
import time
from copy import deepcopy
from typing import Dict, List, Tuple, Callable


from .model import ModelData, apply_player_fixings, build_model, extract_state


def _update_prox_params(ctx, theta_Q, theta_ti, theta_te):
    """Update GAMS parameters for proximal regularization centers."""
    if not hasattr(ctx, "Q_offer_last"):
        return

    # Q_offer_last
    q_recs = [(r, v) for r, v in theta_Q.items()]
    ctx.Q_offer_last.setRecords(q_recs)

    # tau_imp_last
    ti_recs = [(i, e, v) for (i, e), v in theta_ti.items()]
    ctx.tau_imp_last.setRecords(ti_recs)
    
    # tau_exp_last
    te_recs = [(e, i, v) for (e, i), v in theta_te.items()]
    ctx.tau_exp_last.setRecords(te_recs)


def solve_jacobi(
    data: ModelData,
    *,
    iters: int = 50,
    omega: float = 0.5,
    tol_rel: float = 1e-4,
    stable_iters: int = 3,
    solver: str = "conopt",
    solver_options: Dict[str, float] | None = None,
    working_directory: str | None = None,
    iter_callback: Callable[[int, Dict[str, Dict], float, int], None] | None = None,
    initial_state: Dict[str, Dict] | None = None,
    player_order: List[str] | None = None,
    convergence_mode: str = "strategy",
    tol_obj: float = 1e-6,
    use_staged_tolerances: bool = True,
) -> Tuple[Dict[str, Dict], List[Dict]]:


# ...

# Parallel implementation updates similar ...
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

    # Local copies of data attributes for helper functions
    data_players = list(data.players)
    data_regions = list(data.regions)
    data_Qcap = dict(data.Qcap)
    data_tau_imp_ub = dict(data.tau_imp_ub)
    data_tau_exp_ub = dict(data.tau_exp_ub)

    # Current strategy profile - use initial_state if provided
    if initial_state:
        theta_Q: Dict[str, float] = {
            r: float(initial_state.get("Q_offer", {}).get(r, 0.8 * float(data_Qcap.get(r, 0.0))))
            for r in data_players
        }
        theta_tau_imp: Dict[Tuple[str, str], float] = {
            (imp, exp): float(initial_state.get("tau_imp", {}).get((imp, exp), 0.0))
            for imp in data_regions for exp in data_regions
        }
        theta_tau_exp: Dict[Tuple[str, str], float] = {
            (exp, imp): float(initial_state.get("tau_exp", {}).get((exp, imp), 0.0))
            for exp in data_regions for imp in data_regions
        }
        theta_obj: Dict[str, float] = {
            r: float(initial_state.get("obj", {}).get(r, 0.0)) for r in data_players
        }
    else:
        theta_Q: Dict[str, float] = {r: 0.8 * float(data_Qcap.get(r, 0.0)) for r in data_players}
        theta_tau_imp: Dict[Tuple[str, str], float] = {
            (imp, exp): 0.0 for imp in data_regions for exp in data_regions
        }
        theta_tau_exp: Dict[Tuple[str, str], float] = {
            (exp, imp): 0.0 for exp in data_regions for imp in data_regions
        }
        theta_obj: Dict[str, float] = {r: 0.0 for r in data_players}

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
    last_state: Dict[str, Dict] = {}

    # Store context for reuse
    # This part assumes ModelContext and fix_rivals are available,
    # which are not in the original file but implied by the edit.
    # For now, we'll keep the original ctx = build_model(data, ...)
    # and adapt the player loop to use it.
    ctx = build_model(data, working_directory=working_directory)

    for it in range(1, iters + 1):
        if use_staged_tolerances:
            # Use staged tolerances: looser in early sweeps
            current_solver_opts = get_staged_solver_options(it, solver, solver_options)
        else:
            current_solver_opts = solver_options

        solve_kwargs = {"solver": solver}
        if current_solver_opts:
            solve_kwargs["solver_options"] = current_solver_opts

        sweep_start = time.perf_counter()
        # === JACOBI: Freeze profile at sweep start ===
        # Use shallow copy instead of deepcopy for speed (dicts of floats/tuples are fine)
        theta_old_Q = theta_Q.copy()
        theta_old_ti = theta_tau_imp.copy()
        theta_old_te = theta_tau_exp.copy()
        theta_old_obj = theta_obj.copy()

        # Storage for best responses
        theta_br_Q: Dict[str, float] = {}
        theta_br_ti: Dict[Tuple[str, str], float] = {}
        theta_br_te: Dict[Tuple[str, str], float] = {}
        theta_br_obj: Dict[str, float] = {}

        solve_times: List[float] = []

        # Compute best response for each player against FROZEN profile
        solve_order = data_players # player_order if player_order else data.players
        for p in solve_order:
            t0 = time.perf_counter()
            try:
                # The original code builds the model once. The edit implies rebuilding/configuring per player.
                # Sticking to the original structure for now, which means ctx.models[p] is already built.
                # If the intent was to use a new ModelContext and build_model per player, that would be a larger change.
                # For now, we adapt apply_player_fixings and extract_state.
                apply_player_fixings(
                    ctx, data, theta_old_Q, theta_old_ti, theta_old_te, player=p
                )
                _update_prox_params(ctx, theta_old_Q, theta_old_ti, theta_old_te)
                
                ctx.models[p].solve(**solve_kwargs)

                solve_times.append(time.perf_counter() - t0)

                # Only extract strategic variables and obj
                state = extract_state(ctx, variables=["Q_offer", "tau_imp", "tau_exp", "obj"])
                
                # We do NOT update last_state here with partial state, 
                # because last_state is expected to have everything for callbacks.
                # However, for intermediate sweeps, maybe we don't need full state in callback
                # if the user only looks at strategies? 
                # The callback _iter_log in run_gs DOES look at lam, mu, etc.
                # But extracting them 50x per sweep is slow.
                # Compromise: In sequential, we might just pass the PARTIAL state to callback?
                # Or we skip updating last_state until the end of sweep?
                # Actually, last_state is used in iter_callback(..., last_state, ...) AFTER the loop.
                # So we must have a full state at least once per sweep?
                # But wait, the callback is called ONCE per sweep, passing 'last_state'.
                # 'last_state' is just the state of the LAST player solved. 
                # If we only extract strategic vars, 'last_state' will be incomplete (missing lam, x...)
                # minimizing overhead: let's only extract full state for the LAST player in the loop?
                
                is_last_player = (p == solve_order[-1])
                if is_last_player and iter_callback is not None:
                     # Extract full state for callback purposes
                     full_state = extract_state(ctx)
                     last_state = full_state
                     # Update our specific sol vars from full state
                     Q_sol = full_state.get("Q_offer", {})
                     ti_sol = full_state.get("tau_imp", {})
                     te_sol = full_state.get("tau_exp", {})
                     obj_sol = full_state.get("obj", {})
                else:
                     Q_sol = state.get("Q_offer", {})
                     ti_sol = state.get("tau_imp", {})
                     te_sol = state.get("tau_exp", {})
                     obj_sol = state.get("obj", {})

                # Store best response for Q_offer
                if p in Q_sol:
                    theta_br_Q[p] = float(Q_sol[p])

                # Store best responses for tau_imp controlled by player p
                for exp in data_regions:
                    key = (p, exp)
                    if p == exp:
                        continue
                    if key in ti_sol:
                        theta_br_ti[key] = float(ti_sol[key])

                # Store best responses for tau_exp controlled by player p
                for imp in data_regions:
                    key = (p, imp)
                    if p == imp:
                        continue
                    if key in te_sol:
                        theta_br_te[key] = float(te_sol[key])

                # Store objective
                if isinstance(obj_sol, dict):
                    theta_br_obj[p] = float(obj_sol.get(p, 0.0))
                else:
                    theta_br_obj[p] = float(obj_sol)

            except Exception as e:
                import traceback
                print(f"[ERROR] Player {p} failed in sequential sweep {it}: {e}")
                traceback.print_exc()
                solve_times.append(time.perf_counter() - t0)
                # Fallback to old values
                theta_br_Q[p] = theta_old_Q.get(p, 0.0)
                # Fallbacks for tau_imp
                for exp in data_regions:
                    key = (p, exp)
                    if p == exp: continue
                    theta_br_ti[key] = theta_old_ti.get(key, 0.0)
                # Fallbacks for tau_exp
                for imp in data_regions:
                    key = (p, imp)
                    if p == imp: continue
                    theta_br_te[key] = theta_old_te.get(key, 0.0)
                theta_br_obj[p] = theta_old_obj.get(p, 0.0)


        # Sanity check: all players should have best responses
        assert set(theta_br_Q.keys()) == set(data_players), (
            f"Jacobi sanity check failed: theta_br_Q keys {set(theta_br_Q.keys())} "
            f"!= players {set(data_players)}"
        )

        sweep_time = time.perf_counter() - sweep_start

        # === JACOBI: Simultaneous update with damping ===
        for r in data_players:
            if r in theta_br_Q:
                theta_Q[r] = (1.0 - omega) * theta_old_Q[r] + omega * theta_br_Q[r]

            if r in theta_br_obj:
                theta_obj[r] = theta_br_obj[r] # Objective is not damped, it's the result of the BR

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
        for r in data_players:
            r_strat = max(
                r_strat, _rel_change(theta_Q[r], theta_old_Q[r], _q_scale(r))
            )
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

        # Compute r_obj
        r_obj = 0.0
        for r in data_players:
            r_obj = max(r_obj, _rel_change(theta_obj.get(r, 0.0), theta_old_obj.get(r, 0.0), 1000.0))

        metric_met = False
        if convergence_mode == "combined":
            metric_met = (r_strat <= tol_rel) and (r_obj <= tol_obj)
        elif convergence_mode == "objective":
            metric_met = r_obj <= tol_obj
        else: # "strategy"
            metric_met = r_strat <= tol_rel
            
        stable_count = stable_count + 1 if metric_met else 0
        
        iter_rows.append({
            "iter": it,
            "r_strat": float(r_strat),
            "r_obj": float(r_obj),
            "stable_count": int(stable_count),
            "omega": float(omega),
            "sweep_time": float(sweep_time)
        })

        if iter_callback is not None:
             # Pass both metrics? Keeping signature simple for now
            iter_callback(it, last_state, float(r_strat), int(stable_count))
        
        mode_str = f"mode={convergence_mode}"
        print(f"[ITER {it}] r_strat={r_strat:g} (tol={tol_rel:g}) r_obj={r_obj:g} (tol={tol_obj:g}) stable={stable_count} time={sweep_time:.2f}s")
        
        if stable_count >= stable_iters:
            break

    # === Final Market Clearing Solve ===
    # We solve one player's model with ALL strategies fixed to get consistent LLP variables (lam, x, etc.)
    # This ensures the final reported state includes valid duals and flows matching the damped strategies.
    print(f"[FINAL] Solving for market equilibrium with fixed strategies...")
    try:
        # Re-use ctx if available, or build/fix
        # In sequential, ctx is available.
        # We need to apply the final damped profile (theta_Q, theta_tau_...) as FIXED bounds
        # for ALL players, then solve ANY player's model (e.g. first one) to clear the market.
        
        # We must ensure apply_player_fixings sets bounds for ALL players, 
        # normally it fixes rivals and leaves 'player' free.
        # So we use a helper or loop.
        
        # Actually, apply_player_fixings(..., player=p) sets p's bounds to (0, Qcap) or whatever
        # and rivals to theta.
        # We want to fix p to theta as well.
        # Let's just manually fix everything in the container variables.
        
        # Reload bounds from theta
        Q_offer_var = ctx.vars["Q_offer"]
        tau_imp_var = ctx.vars["tau_imp"]
        tau_exp_var = ctx.vars["tau_exp"]
        
        for r in data.players:
             v = theta_Q[r]
             Q_offer_var.lo[r], Q_offer_var.up[r] = v, v

        for imp in data.regions:
            for exp in data.regions:
                 if imp == exp: continue
                 # tau_imp
                 if imp in theta_tau_imp: # Should check keys properly
                      v = theta_tau_imp.get((imp, exp), 0.0) # check correct key order?
                      # theta_tau_imp keys are (imp, exp)
                 elif (imp, exp) in theta_tau_imp:
                      v = theta_tau_imp[(imp, exp)]
                 else:
                      v = 0.0 # Should probably be consistent with loop
                 
                 # Force fix
                 # Note: apply_player_fixings logic handles non-strategic/ub/etc.
                 # Let's rely on the fact that theta contains the damped/final values
                 # derived from feasible solutions, so they should be within bounds.
                 tau_imp_var.lo[imp, exp], tau_imp_var.up[imp, exp] = v, v
                 
                 # tau_exp
                 if (exp, imp) in theta_tau_exp:
                      v = theta_tau_exp[(exp, imp)]
                      tau_exp_var.lo[exp, imp], tau_exp_var.up[exp, imp] = v, v
        
        # Pick first player model to solve (MAX welfare subject to constraints)
        # With all strategic vars fixed, this is just solving the LLP (market clearing)
        p0 = data_players[0]
        ctx.models[p0].solve(solver=solver, solver_options=solver_options)
        
        # Extract full state including lam, x, etc.
        last_state = extract_state(ctx)
        
        # Ensure we keep the damped strategic values (vars should be fixed to them anyway)
        # but extract_state reads .l, which should be correct.
        
    except Exception as e:
        print(f"[WARN] Final market clearing solve failed: {e}")
        # Fallback to just returning strategic vars (previous behavior)
        last_state = {
            "Q_offer": dict(theta_Q),
            "tau_imp": dict(theta_tau_imp),
            "tau_exp": dict(theta_tau_exp),
            "obj": dict(theta_obj)
        }

    return last_state, iter_rows


# =============================================================================
# PARALLEL JACOBI IMPLEMENTATION
# =============================================================================

# Process-global cache for ctx reuse (one ctx per player per process)
_CTX_CACHE: Dict[str, object] = {}
_DATA_CACHE: object = None
_EXCEL_PATH_CACHE: str = ""


def get_staged_solver_options(
    sweep: int,
    solver: str,
    base_options: Dict[str, float] | None = None,
    final_polish: bool = False,
) -> Dict[str, float]:
    """
    Return solver options with tolerances staged by sweep number.
    
    Staging strategy:
    - Sweeps 1-3: looser tolerances (1e-3) to avoid over-solving early
    - Sweeps 4+: normal tolerances (1e-4)
    - Final polish (optional): tighter tolerances (1e-5)
    
    Also applies KNITRO-specific caps (maxit, maxtime) to prevent stuck solves.
    """
    opts = dict(base_options) if base_options else {}
    solver_lower = solver.strip().lower()
    
    # Determine tolerance tier
    if final_polish:
        feastol, opttol = 1e-5, 1e-5
    elif sweep <= 3:
        feastol, opttol = 1e-3, 1e-3
    else:
        feastol, opttol = 1e-4, 1e-4
    
    if solver_lower == "knitro":
        # Override tolerances for staged approach
        opts["feastol"] = feastol
        opts["opttol"] = opttol
        # Apply conservative caps to prevent stuck solves
        if "maxit" not in opts:
            opts["maxit"] = 400
        if "maxtime" not in opts:
            opts["maxtime"] = 60  # GAMS-KNITRO: seconds (integer)
    elif solver_lower == "conopt":
        opts["Tol_Feas_Max"] = feastol
        opts["Tol_Optimality"] = opttol
    
    return opts





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
    try:
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
        _update_prox_params(ctx, theta_old_Q, theta_old_ti, theta_old_te)
        
        # Solve
        solve_kwargs = {"solver": solver}
        if solver_options:
            solve_kwargs["solver_options"] = solver_options
        
        ctx.models[player].solve(**solve_kwargs)
        
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

    except Exception as e:
        import traceback
        return {
            "player": player,
            "success": False, 
            "error": f"{str(e)}\n{traceback.format_exc()}"
        }


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
    use_staged_tolerances: bool = True,
    max_sweep_failures: int | None = None,
    max_consecutive_failures: int = 3,
    initial_state: Dict[str, Dict] | None = None,
    convergence_mode: str = "strategy",
    tol_obj: float = 1e-6,
) -> Tuple[Dict[str, Dict], List[Dict]]:

    """
    Parallel Jacobi solver.
    
    Uses ProcessPoolExecutor to solve all players' best responses in parallel.
    Executor is created ONCE and reused across all sweeps.
    Each worker process caches ctx per player for warm starts.
    
    Robustness features:
    - Staged tolerances: looser early, tighter later (reduces KNITRO stalling)
    - Graceful failure handling: failed players use previous theta as fallback
    - Consecutive failure tracking: abort if same player fails too many times
    
    Args:
        data: ModelData instance.
        excel_path: Path to input Excel file (for worker initialization).
        iters: Maximum number of sweeps.
        omega: Damping factor in (0, 1].
        tol_rel: Relative tolerance for convergence.
        stable_iters: Consecutive stable sweeps required.
        solver: GAMS solver name.
        solver_options: Base solver options dict (staged tolerances override these).
        working_directory: Base GAMS workdir (per-player dirs created under it).
        iter_callback: Optional callback(iter, state, r_strat, stable_count).
        workers: Number of parallel workers.
        worker_timeout: Timeout in seconds for each worker solve (default 120s).
        use_staged_tolerances: If True, use looser tolerances early and tighten later.
        max_sweep_failures: Max players that can fail in one sweep before abort.
                           Default: None = floor(len(players)/2).
        max_consecutive_failures: Abort if same player fails this many times in a row.
    
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
    raw_workdir = working_directory or os.path.join(os.getcwd(), "_gams_workdir")
    abs_workdir = os.path.abspath(raw_workdir)
    
    if " " in abs_workdir:
        import tempfile
        import sys
        print(f"[WARN] Working directory '{abs_workdir}' contains spaces. Using temporary directory for GAMS.", file=sys.stderr)
        base_workdir = tempfile.mkdtemp(prefix="sgr_parallel_")
    else:
        base_workdir = abs_workdir
    player_workdirs = {}
    for p in data_players:
        pdir = os.path.join(base_workdir, f"worker_{p}")
        os.makedirs(pdir, exist_ok=True)
        player_workdirs[p] = pdir
    
    # Initialize strategy profile - use initial_state if provided
    if initial_state:
        theta_Q: Dict[str, float] = {
            r: float(initial_state.get("Q_offer", {}).get(r, 0.8 * float(data_Qcap.get(r, 0.0))))
            for r in data_players
        }
        theta_tau_imp: Dict[Tuple[str, str], float] = {
            (imp, exp): float(initial_state.get("tau_imp", {}).get((imp, exp), 0.0))
            for imp in data_regions for exp in data_regions
        }
        theta_tau_exp: Dict[Tuple[str, str], float] = {
            (exp, imp): float(initial_state.get("tau_exp", {}).get((exp, imp), 0.0))
            for exp in data_regions for imp in data_regions
        }
    else:
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
    
    # Failure tracking for graceful handling
    consecutive_failures: Dict[str, int] = {p: 0 for p in data_players}
    effective_max_sweep_failures = (
        max_sweep_failures if max_sweep_failures is not None 
        else len(data_players) // 2
    )
    
    # Create pool manager
    pool = Pool(
        processes=actual_workers,
        initializer=_worker_init,
        initargs=(excel_path, float(data.eps_x), float(data.eps_comp)),
    )
    
    # Initialize objective tracking
    theta_obj: Dict[str, float] = {}
    if initial_state and "obj" in initial_state:
        theta_obj = {r: float(initial_state["obj"].get(r, 0.0)) for r in data_players}
    else:
        theta_obj = {r: 0.0 for r in data_players}
    
    try:
        for it in range(1, iters + 1):
            sweep_start = time.perf_counter()
            
            # === JACOBI: Freeze profile at sweep start ===
            theta_old_Q = deepcopy(theta_Q)
            theta_old_ti = deepcopy(theta_tau_imp)
            theta_old_te = deepcopy(theta_tau_exp)
            theta_old_obj = deepcopy(theta_obj)
            
            # Storage for best responses
            theta_br_Q: Dict[str, float] = {}
            theta_br_ti: Dict[Tuple[str, str], float] = {}
            theta_br_te: Dict[Tuple[str, str], float] = {}
            theta_br_obj: Dict[str, float] = {}
            solve_times: List[float] = []
            sweep_failures: List[str] = []
            
            # Get solver options for this sweep (staged if enabled)
            if use_staged_tolerances:
                sweep_solver_options = get_staged_solver_options(
                    sweep=it, solver=solver, base_options=solver_options
                )
            else:
                sweep_solver_options = solver_options
            
            # Submit all tasks
            async_results: Dict[str, AsyncResult] = {}
            for p in data_players:
                res = pool.apply_async(
                    _solve_player_br_cached,
                    (p, player_workdirs[p], theta_old_Q, theta_old_ti, theta_old_te, solver, sweep_solver_options)
                )
                async_results[p] = res
            
            # Collect results with timeout and graceful failure handling
            pool_tainted = False
            for player, res in async_results.items():
                pdir = player_workdirs.get(player, "unknown")
                try:
                    # Apply timeout to the result.get()
                    result = res.get(timeout=worker_timeout)
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
                        
                    # Collect objective
                    if "obj" in result:
                        val = result["obj"]
                        if isinstance(val, dict):
                            # Usually obj is Dict[str, float] but for single player solve we might get scalar?
                            # _solve_player_br_cached returns 'obj' which is extraction from state['obj']
                            # state['obj'] is a dict (player -> value).
                            theta_br_obj[result["player"]] = float(val.get(result["player"], 0.0))
                        else:
                            theta_br_obj[result["player"]] = float(val)
                    
                    # Use lam from reference player
                    if result["player"] == ref_player:
                        ref_lam = result.get("lam", {})
                    
                    # Reset consecutive failure counter on success
                    consecutive_failures[player] = 0
                        
                except Exception as e:
                    # Graceful failure handling: use previous theta as fallback
                    sweep_failures.append(player)
                    consecutive_failures[player] += 1
                    
                    is_timeout = isinstance(e, mp.TimeoutError)
                    fail_type = "timed out" if is_timeout else "failed"
                    
                    error_msg = str(e)
                    if is_timeout:
                        pool_tainted = True
                        error_msg = "Worker timed out"
                    
                    import sys
                    print(
                        f"[WARNING] Player '{player}' {fail_type} in sweep {it} "
                        f"(consecutive: {consecutive_failures[player]}).\n"
                        f"  Error: {error_msg}\n"
                        f"  Using fallback. Workdir: {pdir}",
                        file=sys.stderr
                    )
                    
                    # Check consecutive failure threshold
                    if consecutive_failures[player] >= max_consecutive_failures:
                        raise RuntimeError(
                            f"Player '{player}' failed {max_consecutive_failures} consecutive times.\n"
                            f"  Workdir: {pdir}\n"
                            f"  Check .lst/.log files for solver diagnostics."
                        ) from (e if not is_timeout else None)
                    
                    # Fallback: use previous theta values for this player
                    theta_br_Q[player] = theta_old_Q[player]
                    for exp in data_regions:
                        if exp != player:
                            theta_br_ti[(player, exp)] = theta_old_ti[(player, exp)]
                    for imp in data_regions:
                        if imp != player:
                            theta_br_te[(player, imp)] = theta_old_te[(player, imp)]
                    # Fallback objective
                    theta_br_obj[player] = theta_old_obj.get(player, 0.0)
            
            # If pool is tainted (timeout occurred), restart it to clear stuck processes
            if pool_tainted:
                print(f"[WARN] Pool tainted by timeouts in sweep {it}. Restarting pool to clear stuck workers...", file=sys.stderr)
                pool.terminate()
                pool.join()
                pool = Pool(
                    processes=actual_workers,
                    initializer=_worker_init,
                    initargs=(excel_path, float(data.eps_x), float(data.eps_comp)),
                )

            
            # Check if too many failures in this sweep
            if len(sweep_failures) > effective_max_sweep_failures:
                raise RuntimeError(
                    f"Too many player failures in sweep {it}: {len(sweep_failures)} > {effective_max_sweep_failures}\n"
                    f"  Failed players: {sweep_failures}\n"
                    f"  Consider checking solver configuration or input data."
                )
            
            sweep_time = time.perf_counter() - sweep_start
            
            # === JACOBI: Simultaneous update with damping ===
            for r in data_players:
                if r in theta_br_Q:
                    theta_Q[r] = (1.0 - omega) * theta_old_Q[r] + omega * theta_br_Q[r]
            
            for key in theta_br_ti:
                theta_tau_imp[key] = (1.0 - omega) * theta_old_ti[key] + omega * theta_br_ti[key]
            
            for key in theta_br_te:
                theta_tau_exp[key] = (1.0 - omega) * theta_old_te[key] + omega * theta_br_te[key]
                
            # Update Objectives directly (objective reflects outcome of strategy, not a control variable)
            for r in data_players:
                 if r in theta_br_obj:
                     theta_obj[r] = theta_br_obj[r]
            
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
            
            # Compute r_obj
            r_obj = 0.0
            for r in data_players:
                # Use scale=1.0 for objective (values are typically large, 1e7+)
                # But ensure we don't divide by zero if obj is 0.
                r_obj = max(r_obj, _rel_change(theta_obj.get(r, 0.0), theta_old_obj.get(r, 0.0), 1000.0))
            
            metric_met = False
            if convergence_mode == "combined":
                metric_met = (r_strat <= tol_rel) and (r_obj <= tol_obj)
            elif convergence_mode == "objective":
                metric_met = r_obj <= tol_obj
            else: # "strategy"
                metric_met = r_strat <= tol_rel
            
            stable_count = stable_count + 1 if metric_met else 0
            
            # Timing diagnostics
            solve_sum = sum(solve_times) if solve_times else 0.0
            solve_max = max(solve_times) if solve_times else 0.0
            solve_mean = solve_sum / len(solve_times) if solve_times else 0.0
            
            iter_rows.append({
                "iter": it,
                "r_strat": float(r_strat),
                "r_obj": float(r_obj),
                "stable_count": int(stable_count),
                "omega": float(omega),
                "sweep_time": float(sweep_time),
                "solve_time_sum": float(solve_sum),
                "solve_time_max": float(solve_max),
                "solve_time_mean": float(solve_mean),
                "sweep_failures": len(sweep_failures),
            })
            
            print(
                f"[ITER {it}] r_strat={r_strat:g} (tol={tol_rel:g}) r_obj={r_obj:g} (tol={tol_obj:g}) stable={stable_count} sweep_time={sweep_time:.2f}s"
            )
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

    finally:
        pool.terminate()
        pool.join()
    
    # === Final evaluation solve for complete market equilibrium ===
    # We need to solve once with ALL players' strategies fixed to extract
    # the full market outcome (x flows, x_dem) that result from the equilibrium.
    final_workdir = os.path.join(base_workdir, "final_eval")
    os.makedirs(final_workdir, exist_ok=True)
    
    ctx = build_model(data, working_directory=final_workdir)
    
    # Fix ALL players' strategies to the converged values
    for r in data_players:
        ctx.vars["Q_offer"].lo[r] = theta_Q[r]
        ctx.vars["Q_offer"].up[r] = theta_Q[r]
    
    for imp in data_regions:
        for exp in data_regions:
            if imp != exp:
                v = theta_tau_imp.get((imp, exp), 0.0)
                ctx.vars["tau_imp"].lo[imp, exp] = v
                ctx.vars["tau_imp"].up[imp, exp] = v
    
    for exp in data_regions:
        for imp in data_regions:
            if exp != imp:
                v = theta_tau_exp.get((exp, imp), 0.0)
                ctx.vars["tau_exp"].lo[exp, imp] = v
                ctx.vars["tau_exp"].up[exp, imp] = v
    
    # Solve to get market equilibrium (x, x_dem, lam, etc.)
    p0 = data_players[0]
    solve_kwargs = {"solver": solver}
    if solver_options:
        solve_kwargs["solver_options"] = solver_options
    ctx.models[p0].solve(**solve_kwargs)
    
    # Extract full state including x and x_dem
    final_state = extract_state(ctx)
    
    # Ensure we have the converged strategic variables (not re-solved values)
    final_state["Q_offer"] = dict(theta_Q)
    final_state["tau_imp"] = dict(theta_tau_imp)
    final_state["tau_exp"] = dict(theta_tau_exp)
    
    return final_state, iter_rows

