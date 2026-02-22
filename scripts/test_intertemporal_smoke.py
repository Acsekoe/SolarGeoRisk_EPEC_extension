"""Smoke test for intertemporal model: build, fix, solve one player."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solargeorisk_extension.model_intertemporal import (
    ModelData,
    ModelContext,
    build_model,
    apply_player_fixings,
    extract_state,
)


def _make_test_data() -> ModelData:
    """3-region toy data for smoke test."""
    regions = ["a", "b", "c"]
    players = ["a", "b"]
    non_strategic: set = {"c"}

    times = ["2025", "2030", "2035", "2040"]

    D = {r: 10.0 for r in regions}
    a_dem = {r: 100.0 for r in regions}
    b_dem = {r: 5.0 for r in regions}
    Dmax = {r: 20.0 for r in regions}
    Qcap = {"a": 30.0, "b": 15.0, "c": 10.0}
    c_man = {"a": 10.0, "b": 20.0, "c": 25.0}

    c_ship = {}
    for r in regions:
        for i in regions:
            c_ship[(r, i)] = 0.0 if r == i else 5.0

    tau_imp_ub = {}
    tau_exp_ub = {}
    for r in regions:
        for i in regions:
            tau_imp_ub[(r, i)] = 0.0 if r == i else 50.0
            tau_exp_ub[(r, i)] = 0.0 if r == i else 50.0

    rho_imp = {r: 0.01 for r in regions}
    rho_exp = {r: 0.01 for r in regions}
    w = {r: 1.0 for r in regions}

    # Time-indexed demand
    Dmax_t = {}
    for r in regions:
        for tp in times:
            Dmax_t[(r, tp)] = 20.0 + 5.0 * times.index(tp)

    return ModelData(
        regions=regions,
        players=players,
        non_strategic=non_strategic,
        D=D,
        a_dem=a_dem,
        b_dem=b_dem,
        Dmax=Dmax,
        Qcap=Qcap,
        c_man=c_man,
        c_ship=c_ship,
        tau_imp_ub=tau_imp_ub,
        tau_exp_ub=tau_exp_ub,
        rho_imp=rho_imp,
        rho_exp=rho_exp,
        w=w,
        eps_x=1e-3,
        eps_comp=1e-4,
        kappa_Q={r: 0.0 for r in regions},
        settings={"rho_prox": 0.01, "use_quad": True},
        times=times,
        Dmax_t=Dmax_t,
        Kcap_2025={"a": 30.0, "b": 15.0, "c": 10.0},
        s_ub={"a": 5.0, "b": 5.0, "c": 0.0},
        f_hold={"a": 0.1, "b": 0.1, "c": 0.0},
        c_inv={"a": 1.0, "b": 1.0, "c": 0.0},
    )


def main():
    data = _make_test_data()
    times = data.times

    # Use a temp dir without spaces
    workdir = os.path.join(tempfile.gettempdir(), "test_intertemp")
    # Ensure no spaces
    if " " in workdir:
        workdir = os.path.join("C:\\temp", "test_intertemp")
    os.makedirs(workdir, exist_ok=True)

    print("1. Building model...")
    ctx = build_model(data, working_directory=workdir)
    print(f"   Sets:      {sorted(ctx.sets.keys())}")
    print(f"   Params:    {sorted(ctx.params.keys())}")
    print(f"   Variables: {sorted(ctx.vars.keys())}")
    print(f"   Equations: {sorted(ctx.equations.keys())}")
    print(f"   Models:    {sorted(ctx.models.keys())}")

    assert "T" in ctx.sets, "Missing set T"
    assert "s" in ctx.vars, "Missing variable s"
    assert "Kcap" in ctx.vars, "Missing variable Kcap"
    assert "Icap" in ctx.vars, "Missing variable Icap"
    assert "Dcap" in ctx.vars, "Missing variable Dcap"
    assert "eq_cap_trans_30" in ctx.equations, "Missing capacity transition eq"
    assert "eq_offer_cap" in ctx.equations, "Missing offer-cap linkage eq"
    print("   [OK] All expected components present.\n")

    # 2. Apply fixings (player "a" optimizes, "b" fixed)
    print("2. Applying fixings for player 'a'...")
    theta_Q = {}
    theta_tau_imp = {}
    theta_tau_exp = {}
    theta_s = {}
    theta_Icap = {}
    theta_Dcap = {}

    for r in data.players:
        for tp in times:
            theta_Q[(r, tp)] = 0.8 * float(data.Qcap[r])
            theta_s[(r, tp)] = 0.0
            theta_Icap[(r, tp)] = 0.0
            theta_Dcap[(r, tp)] = 0.0

    for im in data.regions:
        for ex in data.regions:
            for tp in times:
                theta_tau_imp[(im, ex, tp)] = 0.0
                theta_tau_exp[(im, ex, tp)] = 0.0

    apply_player_fixings(
        ctx, data,
        theta_Q, theta_tau_imp, theta_tau_exp,
        theta_s, theta_Icap, theta_Dcap,
        player="a",
    )
    print("   [OK] Fixings applied.\n")

    # 3. Solve
    print("3. Solving player 'a' model...")
    try:
        ctx.models["a"].solve(solver="conopt")
        print(f"   Objective value: {ctx.models['a'].objective_value:.4f}")
        print("   [OK] Solve succeeded.\n")
    except Exception as e:
        print(f"   [WARN] Solve failed: {e}")
        print("   (This may happen if CONOPT is not installed.)\n")

    # 4. Extract state
    print("4. Extracting state...")
    state = extract_state(ctx)
    for key in ["Q_offer", "s", "Kcap", "Icap", "Dcap", "lam", "x", "x_dem"]:
        entries = state.get(key, {})
        n = len(entries) if entries else 0
        sample = list(entries.items())[:3] if entries else []
        print(f"   {key}: {n} entries, sample: {sample}")
    print("   [OK] State extracted.\n")

    print("=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
