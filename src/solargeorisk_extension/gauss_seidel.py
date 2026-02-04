from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .model import ModelData, apply_player_fixings, build_model, extract_state


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

    theta_Q: Dict[str, float] = {r: 0.8 * float(data.Qcap[r]) for r in data.players}
    theta_tau_imp: Dict[Tuple[str, str], float] = {(imp, exp): 0.0 for imp in data.regions for exp in data.regions}
    theta_tau_exp: Dict[Tuple[str, str], float] = {(exp, imp): 0.0 for exp in data.regions for imp in data.regions}

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
        r_strat = 0.0

        prev_Q = dict(theta_Q)
        prev_ti = dict(theta_tau_imp)
        prev_te = dict(theta_tau_exp)

        for p in data.players:
            apply_player_fixings(ctx, data, theta_Q, theta_tau_imp, theta_tau_exp, player=p)
            ctx.models[p].solve(**solve_kwargs)

            state = extract_state(ctx)
            last_state = state

            Q_sol = state.get("Q_offer", {})
            ti_sol = state.get("tau_imp", {})
            te_sol = state.get("tau_exp", {})

            if p in theta_Q and p in Q_sol:
                br = float(Q_sol[p])
                theta_Q[p] = (1.0 - omega) * theta_Q[p] + omega * br

            for exp in data.regions:
                key = (p, exp)
                if p == exp:
                    continue
                if key in theta_tau_imp and key in ti_sol:
                    br = float(ti_sol[key])
                    theta_tau_imp[key] = (1.0 - omega) * theta_tau_imp[key] + omega * br

            for imp in data.regions:
                key = (p, imp)
                if p == imp:
                    continue
                if key in theta_tau_exp and key in te_sol:
                    br = float(te_sol[key])
                    theta_tau_exp[key] = (1.0 - omega) * theta_tau_exp[key] + omega * br

        for r in data.players:
            r_strat = max(r_strat, _rel_change(theta_Q[r], prev_Q[r], _q_scale(r)))
        for imp in data.regions:
            for exp in data.regions:
                if imp == exp:
                    continue
                key = (imp, exp)
                r_strat = max(r_strat, _rel_change(theta_tau_imp[key], prev_ti[key], _ti_scale(imp, exp)))
        for exp in data.regions:
            for imp in data.regions:
                if exp == imp:
                    continue
                key = (exp, imp)
                r_strat = max(r_strat, _rel_change(theta_tau_exp[key], prev_te[key], _te_scale(exp, imp)))

        stable_count = stable_count + 1 if r_strat <= tol_rel else 0
        iter_rows.append({"iter": it, "r_strat": float(r_strat), "stable_count": int(stable_count), "omega": float(omega)})

        if iter_callback is not None:
            iter_callback(it, last_state, float(r_strat), int(stable_count))

        if stable_count >= stable_iters:
            break

    return last_state, iter_rows
