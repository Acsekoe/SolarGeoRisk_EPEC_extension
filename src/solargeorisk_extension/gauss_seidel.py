from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from .model_single_year import ModelData, apply_player_fixings, build_model, extract_state


def solve_gs(
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
    initial_state: Dict[str, Dict] | None = None,
    convergence_mode: str = "strategy",
    tol_obj: float = 1e-6,
    shuffle_players: bool = False,
) -> tuple[Dict[str, Dict], List[Dict[str, object]]]:
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1].")
    if tol_rel <= 0.0:
        raise ValueError("tol_rel must be > 0")
    if stable_iters < 1:
        raise ValueError("stable_iters must be >= 1")

    ctx = build_model(data, working_directory=working_directory)

    # Initialize theta (strategies)
    if initial_state:
        theta_Q: Dict[str, float] = {
            r: float(initial_state.get("Q_offer", {}).get(r, 0.8 * float(data.Qcap[r])))
            for r in data.players
        }
        theta_tau_imp: Dict[Tuple[str, str], float] = {
            (imp, exp): float(initial_state.get("tau_imp", {}).get((imp, exp), 0.0))
            for imp in data.regions for exp in data.regions
        }
        theta_tau_exp: Dict[Tuple[str, str], float] = {
            (exp, imp): float(initial_state.get("tau_exp", {}).get((exp, imp), 0.0))
            for exp in data.regions for imp in data.regions
        }
        theta_obj: Dict[str, float] = {
            r: float(initial_state.get("obj", {}).get(r, 0.0)) for r in data.players
        }
    else:
        theta_Q: Dict[str, float] = {r: 0.8 * float(data.Qcap[r]) for r in data.players}
        theta_tau_imp: Dict[Tuple[str, str], float] = {(imp, exp): 0.0 for imp in data.regions for exp in data.regions}
        theta_tau_exp: Dict[Tuple[str, str], float] = {(exp, imp): 0.0 for exp in data.regions for imp in data.regions}
        theta_obj: Dict[str, float] = {r: 0.0 for r in data.players}

    def _scaled_change(new: float, old: float, scale: float) -> float:
        return abs(new - old) / max(scale, 1e-12)

    def _q_scale(r: str) -> float:
        return max(float(data.Qcap.get(r, 0.0)), 1.0)

    def _ti_scale(imp: str, exp: str) -> float:
        return max(float(data.tau_imp_ub[(imp, exp)]), 1e-3)

    def _te_scale(exp: str, imp: str) -> float:
        return max(float(data.tau_exp_ub[(exp, imp)]), 1e-3)

    iter_rows: List[Dict[str, object]] = []
    stable_count = 0
    last_state: Dict[str, Dict] = {}

    solve_kwargs = {"solver": solver}
    if solver_options:
        solve_kwargs["solver_options"] = solver_options

    def _update_prox_reference() -> None:
        # Anchor proximal terms to the strategy point at the start of each sweep.
        q_last = ctx.params.get("Q_offer_last")
        ti_last = ctx.params.get("tau_imp_last")
        te_last = ctx.params.get("tau_exp_last")
        if q_last is None or ti_last is None or te_last is None:
            return

        for r in data.regions:
            q_last[r] = float(theta_Q.get(r, float(data.Qcap[r])))

        for imp in data.regions:
            for exp in data.regions:
                if imp == exp:
                    ti_last[imp, exp] = 0.0
                else:
                    ti_last[imp, exp] = float(theta_tau_imp.get((imp, exp), 0.0))

        for exp in data.regions:
            for imp in data.regions:
                if exp == imp:
                    te_last[exp, imp] = 0.0
                else:
                    te_last[exp, imp] = float(theta_tau_exp.get((exp, imp), 0.0))

    for it in range(1, iters + 1):
        r_strat = 0.0
        
        # Snapshot for convergence check
        prev_Q = dict(theta_Q)
        prev_ti = dict(theta_tau_imp)
        prev_te = dict(theta_tau_exp)
        prev_obj = dict(theta_obj)
        sweep_order = list(data.players)
        if shuffle_players:
            random.shuffle(sweep_order)
        for p in sweep_order:
            _update_prox_reference()   # GS-consistent: anchor before each player
            apply_player_fixings(ctx, data, theta_Q, theta_tau_imp, theta_tau_exp, player=p)
            ctx.models[p].solve(**solve_kwargs)

            state = extract_state(ctx)
            last_state = state

            Q_sol = state.get("Q_offer", {})
            ti_sol = state.get("tau_imp", {})
            te_sol = state.get("tau_exp", {})
            obj_sol = state.get("obj", {})

            # Update strategies immediately (Gauss-Seidel)
            if p in Q_sol:
                br = float(Q_sol[p])
                theta_Q[p] = (1.0 - omega) * theta_Q[p] + omega * br

            for exp in data.regions:
                key = (p, exp)
                if p == exp:
                    continue
                if key in ti_sol:
                    br = float(ti_sol[key])
                    theta_tau_imp[key] = (1.0 - omega) * theta_tau_imp[key] + omega * br

            for imp in data.regions:
                key = (p, imp)
                if p == imp:
                    continue
                if key in te_sol:
                    br = float(te_sol[key])
                    theta_tau_exp[key] = (1.0 - omega) * theta_tau_exp[key] + omega * br
            
            # Update objective (no damping usually, just current value)
            if isinstance(obj_sol, dict):
                theta_obj[p] = float(obj_sol.get(p, 0.0))
            else:
                theta_obj[p] = float(obj_sol)

        # Compute convergence metrics
        for r in data.players:
            r_strat = max(r_strat, _scaled_change(theta_Q[r], prev_Q[r], _q_scale(r)))
        for imp in data.regions:
            for exp in data.regions:
                if imp == exp:
                    continue
                key = (imp, exp)
                r_strat = max(r_strat, _scaled_change(theta_tau_imp[key], prev_ti[key], _ti_scale(imp, exp)))
        for exp in data.regions:
            for imp in data.regions:
                if exp == imp:
                    continue
                key = (exp, imp)
                r_strat = max(r_strat, _scaled_change(theta_tau_exp[key], prev_te[key], _te_scale(exp, imp)))

        # Compute r_obj
        r_obj = 0.0
        for r in data.players:
             r_obj = max(r_obj, _scaled_change(theta_obj.get(r, 0.0), prev_obj.get(r, 0.0), 1000.0))

        metric_met = False
        if convergence_mode == "combined":
             metric_met = (r_strat <= tol_rel) and (r_obj <= tol_obj)
        elif convergence_mode == "objective":
             metric_met = r_obj <= tol_obj
        else: # "strategy"
             metric_met = r_strat <= tol_rel

        stable_count = stable_count + 1 if metric_met else 0
        row_data: Dict[str, object] = {
            "iter": it, 
            "r_strat": float(r_strat), 
            "r_obj": float(r_obj),
            "stable_count": int(stable_count), 
            "omega": float(omega),
        }
        if shuffle_players:
            row_data["sweep_order"] = list(sweep_order)
        iter_rows.append(row_data)

        if iter_callback is not None:
            if shuffle_players:
                last_state["_sweep_order"] = list(sweep_order)
            iter_callback(it, last_state, float(r_strat), int(stable_count))
            last_state.pop("_sweep_order", None)

        if stable_count >= stable_iters:
            break

    return last_state, iter_rows
