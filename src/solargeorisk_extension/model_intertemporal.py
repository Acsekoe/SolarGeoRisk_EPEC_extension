"""
Intertemporal (4-period) perfect-foresight EPEC model.

Each strategic player maximises the discounted sum of welfare across
T = {"2025", "2030", "2035", "2040"}, subject to per-period LLP KKT
conditions, dynamic capacity transitions, and production-subsidy choice.

Key features vs. the single-year model:
  - Time set  T = {"2025", "2030", "2035", "2040"}
  - Dynamic capacity: Kcap, Icap (investment), Dcap (decommission)
  - Production subsidy variable  s[r, t] >= 0
  - Demand parameters indexed by time (Dmax_t; a_dem/b_dem optionally)
  - Discounting (beta_t) and interval-length weighting (years_to_next)
"""
from __future__ import annotations

INTERTEMPORAL_IMPLEMENTED = True

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set as PySet, Tuple

import gamspy as gp
from gamspy import (
    Alias,
    Container,
    Equation,
    Model,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
    VariableType,
)

z = gp.Number(0)

_DEFAULT_TIMES = ["2025", "2030", "2035", "2040"]
_DEFAULT_YTN = {"2025": 5.0, "2030": 5.0, "2035": 5.0, "2040": 0.0}


# =============================================================================
# Step 1 — ModelData
# =============================================================================
@dataclass
class ModelData:
    regions: List[str]
    players: List[str]
    non_strategic: PySet[str]

    D: Dict[str, float]

    a_dem: Dict[str, float]
    b_dem: Dict[str, float]
    Dmax: Dict[str, float]

    Qcap: Dict[str, float]
    c_man: Dict[str, float]
    c_ship: Dict[Tuple[str, str], float]

    tau_imp_ub: Dict[Tuple[str, str], float]
    tau_exp_ub: Dict[Tuple[str, str], float]

    rho_imp: Dict[str, float]
    rho_exp: Dict[str, float]
    w: Dict[str, float]

    eps_x: float
    eps_comp: float

    kappa_Q: Dict[str, float] | None = None
    settings: Dict[str, object] | None = None

    # --- Intertemporal extensions ---
    times: List[str] | None = None

    # Time-indexed demand (region, year) → value
    a_dem_t: Dict[Tuple[str, str], float] | None = None
    b_dem_t: Dict[Tuple[str, str], float] | None = None
    Dmax_t: Dict[Tuple[str, str], float] | None = None

    # Capacity
    Kcap_2025: Dict[str, float] | None = None  # fallback to Qcap

    # Subsidy bound
    s_ub: Dict[str, float] | None = None  # default 0 (no subsidy)

    # Capacity costs
    f_hold: Dict[str, float] | None = None  # holding cost
    c_inv: Dict[str, float] | None = None   # investment cost

    # Discounting
    beta_t: Dict[str, float] | None = None  # discount per period (default 1)
    years_to_next: Dict[str, float] | None = None  # interval lengths


@dataclass
class ModelContext:
    container: Container
    sets: Dict[str, Set]
    params: Dict[str, Parameter]
    vars: Dict[str, Variable]
    equations: Dict[str, Equation]
    models: Dict[str, Model]


# =============================================================================
# Sanity checks
# =============================================================================
def _sanity_check_data(data: ModelData) -> None:
    times = data.times or _DEFAULT_TIMES

    # -- per-period demand checks --
    if data.Dmax_t is not None:
        bad = sorted(
            [k for k in data.Dmax_t if float(data.Dmax_t[k]) <= 0.0]
        )
        if bad:
            raise ValueError(f"All Dmax_t must be > 0. Invalid keys: {bad}")
    else:
        bad_dmax = sorted([r for r in data.regions if float(data.Dmax[r]) <= 0.0])
        if bad_dmax:
            raise ValueError(f"All Dmax must be > 0. Invalid regions: {bad_dmax}")

    if data.b_dem_t is not None:
        bad = sorted(
            [k for k in data.b_dem_t if float(data.b_dem_t[k]) <= 0.0]
        )
        if bad:
            raise ValueError(f"All b_dem_t must be > 0. Invalid keys: {bad}")
    else:
        bad_b = sorted([r for r in data.regions if float(data.b_dem.get(r, 1.0)) <= 0.0])
        if bad_b:
            raise ValueError(f"All b_dem must be > 0. Invalid regions: {bad_b}")

    # -- capacity init --
    kcap_init = data.Kcap_2025 or data.Qcap
    bad_k = sorted([r for r in data.regions if float(kcap_init.get(r, 0.0)) < 0.0])
    if bad_k:
        raise ValueError(f"All Kcap_2025 must be >= 0. Invalid regions: {bad_k}")

    # -- subsidy bounds --
    s_ub_map = data.s_ub or {}
    bad_s = sorted([r for r in s_ub_map if float(s_ub_map[r]) < 0.0])
    if bad_s:
        raise ValueError(f"All s_ub must be >= 0. Invalid regions: {bad_s}")

    # -- cost params --
    f_hold_map = data.f_hold or {}
    bad_fh = sorted([r for r in f_hold_map if float(f_hold_map[r]) < 0.0])
    if bad_fh:
        raise ValueError(f"All f_hold must be >= 0. Invalid regions: {bad_fh}")

    c_inv_map = data.c_inv or {}
    bad_ci = sorted([r for r in c_inv_map if float(c_inv_map[r]) < 0.0])
    if bad_ci:
        raise ValueError(f"All c_inv must be >= 0. Invalid regions: {bad_ci}")

    kappa_map = getattr(data, "kappa_Q", None) or {}
    bad_kappa = sorted([r for r in kappa_map if float(kappa_map[r]) < 0.0])
    if bad_kappa:
        raise ValueError(f"All kappa_Q must be >= 0. Invalid regions: {bad_kappa}")


# =============================================================================
# Step 2–8: build_model
# =============================================================================
def build_model(data: ModelData, working_directory: str | None = None) -> ModelContext:
    if working_directory and " " in str(working_directory):
        raise ValueError(
            f"GAMS working directory must be space-free. Got: {working_directory}"
        )

    _sanity_check_data(data)

    unknown_players = sorted(set(data.players) - set(data.regions))
    if unknown_players:
        raise ValueError(
            f"All players must be in regions. Unknown players: {unknown_players}"
        )

    settings = data.settings or {}
    use_quad = bool(settings.get("use_quad", False))

    # =====================================================================
    # Step 2 — Container, sets, time set
    # =====================================================================
    m = Container(working_directory=working_directory, debugging_level="keep")

    R = Set(m, "R", records=data.regions)
    exp = Alias(m, "exp", R)
    imp = Alias(m, "imp", R)
    j = Alias(m, "j", R)

    times = data.times or list(_DEFAULT_TIMES)
    T = Set(m, "T", records=times)
    # Note: do NOT create Alias(m, "t", T) — 't' is a GAMS built-in symbol.

    # =====================================================================
    # Step 8 — Compatibility fallback (before building params)
    # =====================================================================
    # Demand
    a_dem_t_dict: Dict[Tuple[str, str], float] = {}
    b_dem_t_dict: Dict[Tuple[str, str], float] = {}
    dmax_t_dict: Dict[Tuple[str, str], float] = {}

    if data.a_dem_t is not None:
        a_dem_t_dict = dict(data.a_dem_t)
    else:
        for r in data.regions:
            for tp in times:
                a_dem_t_dict[(r, tp)] = float(data.a_dem.get(r, 0.0))

    if data.b_dem_t is not None:
        b_dem_t_dict = dict(data.b_dem_t)
    else:
        for r in data.regions:
            for tp in times:
                b_dem_t_dict[(r, tp)] = float(data.b_dem.get(r, 1.0))

    if data.Dmax_t is not None:
        dmax_t_dict = dict(data.Dmax_t)
    else:
        for r in data.regions:
            for tp in times:
                dmax_t_dict[(r, tp)] = float(data.Dmax.get(r, 1.0))

    # Capacity init
    kcap_2025_dict: Dict[str, float] = {}
    if data.Kcap_2025 is not None:
        kcap_2025_dict = dict(data.Kcap_2025)
    else:
        kcap_2025_dict = {r: float(data.Qcap[r]) for r in data.regions}

    # Subsidy bounds
    s_ub_dict: Dict[str, float] = {}
    if data.s_ub is not None:
        s_ub_dict = dict(data.s_ub)
    else:
        s_ub_dict = {r: 0.0 for r in data.regions}

    # Capacity costs
    f_hold_dict: Dict[str, float] = {}
    if data.f_hold is not None:
        f_hold_dict = dict(data.f_hold)
    else:
        f_hold_dict = {r: 0.0 for r in data.regions}

    c_inv_dict: Dict[str, float] = {}
    if data.c_inv is not None:
        c_inv_dict = dict(data.c_inv)
    else:
        c_inv_dict = {r: 0.0 for r in data.regions}

    # Discount and interval
    beta_t_dict: Dict[str, float] = {}
    if data.beta_t is not None:
        beta_t_dict = dict(data.beta_t)
    else:
        beta_t_dict = {tp: 1.0 for tp in times}

    ytn_dict: Dict[str, float] = {}
    if data.years_to_next is not None:
        ytn_dict = dict(data.years_to_next)
    else:
        ytn_dict = dict(_DEFAULT_YTN)

    # =====================================================================
    # Step 3 — Parameters
    # =====================================================================
    # Time-invariant params (region only)
    c_man = Parameter(
        m, "c_man", domain=[R],
        records=[(r, data.c_man[r]) for r in data.regions],
    )
    c_ship = Parameter(
        m, "c_ship", domain=[exp, imp],
        records=[
            (r, i, data.c_ship[(r, i)])
            for r in data.regions for i in data.regions
        ],
    )
    rho_imp_p = Parameter(
        m, "rho_imp", domain=[R],
        records=[(r, data.rho_imp[r]) for r in data.regions],
    )
    rho_exp_p = Parameter(
        m, "rho_exp", domain=[R],
        records=[(r, data.rho_exp[r]) for r in data.regions],
    )
    w_p = Parameter(
        m, "w", domain=[R],
        records=[(r, data.w[r]) for r in data.regions],
    )

    # Time-indexed demand params [R, T]
    a_dem_t_p = Parameter(
        m, "a_dem_t", domain=[R, T],
        records=[(r, tp, a_dem_t_dict[(r, tp)]) for r in data.regions for tp in times],
    )
    b_dem_t_p = Parameter(
        m, "b_dem_t", domain=[R, T],
        records=[(r, tp, b_dem_t_dict[(r, tp)]) for r in data.regions for tp in times],
    )
    Dmax_t_p = Parameter(
        m, "Dmax_t", domain=[R, T],
        records=[(r, tp, dmax_t_dict[(r, tp)]) for r in data.regions for tp in times],
    )

    # Tariff upper bounds (time-invariant for now, but put in param for clarity)
    tau_imp_ub_p = Parameter(
        m, "tau_imp_ub", domain=[imp, exp],
        records=[
            (i, r, data.tau_imp_ub[(i, r)])
            for i in data.regions for r in data.regions
        ],
    )
    tau_exp_ub_p = Parameter(
        m, "tau_exp_ub", domain=[exp, imp],
        records=[
            (r, i, data.tau_exp_ub[(r, i)])
            for r in data.regions for i in data.regions
        ],
    )

    # Capacity / subsidy params
    Kcap_2025_p = Parameter(
        m, "Kcap_2025", domain=[R],
        records=[(r, kcap_2025_dict[r]) for r in data.regions],
    )
    s_ub_p = Parameter(
        m, "s_ub", domain=[R],
        records=[(r, s_ub_dict[r]) for r in data.regions],
    )
    f_hold_p = Parameter(
        m, "f_hold", domain=[R],
        records=[(r, f_hold_dict[r]) for r in data.regions],
    )
    c_inv_p = Parameter(
        m, "c_inv", domain=[R],
        records=[(r, c_inv_dict[r]) for r in data.regions],
    )

    # Discount & interval
    beta_p = Parameter(
        m, "beta_t", domain=[T],
        records=[(tp, beta_t_dict[tp]) for tp in times],
    )
    ytn_p = Parameter(
        m, "ytn", domain=[T],
        records=[(tp, ytn_dict[tp]) for tp in times],
    )

    # Regularization scalars
    eps_x = gp.Number(float(data.eps_x))
    eps_comp = float(data.eps_comp)
    eps_value = gp.Number(eps_comp)

    rho_prox_val = float(settings.get("rho_prox", 0.0))
    rho_prox = gp.Number(rho_prox_val)

    kappa_map = getattr(data, "kappa_Q", None) or {}
    kappa_by_r = {k: float(kappa_map.get(k, 0.0)) for k in data.regions}
    kappa_Q = Parameter(
        m, "kappa_Q", domain=[R],
        records=[(k, kappa_by_r[k]) for k in data.regions],
    )

    # Proximal reference params [R, T] and [imp, exp, T]
    Q_offer_last = Parameter(m, "Q_offer_last", domain=[R, T])
    tau_imp_last = Parameter(m, "tau_imp_last", domain=[imp, exp, T])
    tau_exp_last = Parameter(m, "tau_exp_last", domain=[exp, imp, T])

    # Initialize proximal references
    Q_offer_last[R, T] = Kcap_2025_p[R]
    tau_imp_last[imp, exp, T] = z
    tau_exp_last[exp, imp, T] = z

    # lam upper bound for variable bounding (use max a_dem over time)
    lam_ub_values: Dict[str, float] = {}
    for i in data.regions:
        max_a = max(a_dem_t_dict.get((i, tp), 0.0) for tp in times)
        lam_ub_values[i] = max_a
    lam_ub = Parameter(
        m, "lam_ub", domain=[R],
        records=[(i, lam_ub_values[i]) for i in data.regions],
    )

    # mu upper bound
    mu_ub_values: Dict[str, float] = {}
    for r in data.regions:
        mu_ub_values[r] = max(
            0.0,
            max(
                float(lam_ub_values[i])
                - (float(data.c_man[r]) + float(data.c_ship[(r, i)]))
                for i in data.regions
            ),
        )
    mu_ub = Parameter(
        m, "mu_ub", domain=[R],
        records=[(r, mu_ub_values[r]) for r in data.regions],
    )

    # gamma upper bound
    gamma_ub_values: Dict[Tuple[str, str], float] = {}
    for r in data.regions:
        for i in data.regions:
            gamma_ub_values[(r, i)] = (
                float(data.c_man[r])
                + float(data.c_ship[(r, i)])
                + float(data.tau_imp_ub[(i, r)])
                + float(data.tau_exp_ub[(r, i)])
                + float(data.eps_x) * float(kcap_2025_dict.get(r, 0.0))
                + float(mu_ub_values.get(r, 0.0))
                + float(s_ub_dict.get(r, 0.0))  # subsidy can reduce cost
            )
    gamma_ub = Parameter(
        m, "gamma_ub", domain=[exp, imp],
        records=[
            (r, i, gamma_ub_values[(r, i)])
            for r in data.regions for i in data.regions
        ],
    )

    # =====================================================================
    # Step 3 — Variables (all time-indexed)
    # =====================================================================

    # -- ULP strategic variables --
    Q_offer = Variable(m, "Q_offer", domain=[R, T], type=VariableType.POSITIVE)
    tau_imp = Variable(m, "tau_imp", domain=[imp, exp, T], type=VariableType.POSITIVE)
    tau_exp = Variable(m, "tau_exp", domain=[exp, imp, T], type=VariableType.POSITIVE)

    # Subsidy
    s_var = Variable(m, "s", domain=[R, T], type=VariableType.POSITIVE)
    s_var.up[R, T] = s_ub_p[R]

    # Capacity variables
    Kcap = Variable(m, "Kcap", domain=[R, T], type=VariableType.POSITIVE)
    Icap = Variable(m, "Icap", domain=[R, T], type=VariableType.POSITIVE)
    Dcap = Variable(m, "Dcap", domain=[R, T], type=VariableType.POSITIVE)

    # Set initial capacity (fix Kcap at t=2025)
    for r in data.regions:
        Kcap.fx[r, times[0]] = kcap_2025_dict[r]
        # No investment or decommission in 2040 (terminal)
        Icap.fx[r, times[-1]] = 0.0
        Dcap.fx[r, times[-1]] = 0.0

    # Tariff bounds
    tau_imp.up[imp, exp, T] = tau_imp_ub_p[imp, exp]
    tau_exp.up[exp, imp, T] = tau_exp_ub_p[exp, imp]

    for r in data.regions:
        tau_imp.lo[r, r, T] = 0.0
        tau_imp.up[r, r, T] = 0.0
        tau_exp.lo[r, r, T] = 0.0
        tau_exp.up[r, r, T] = 0.0

    # -- LLP market variables --
    z_llp = Variable(m, "z_llp", type=VariableType.FREE)

    x = Variable(m, "x", domain=[exp, imp, T], type=VariableType.POSITIVE)
    x_dem = Variable(m, "x_dem", domain=[R, T], type=VariableType.POSITIVE)
    x_dem.up[R, T] = Dmax_t_p[R, T]

    lam_var = Variable(m, "lam", domain=[R, T], type=VariableType.FREE)
    mu = Variable(m, "mu", domain=[R, T], type=VariableType.POSITIVE)
    gamma = Variable(m, "gamma", domain=[exp, imp, T], type=VariableType.POSITIVE)
    beta_dem = Variable(m, "beta_dem", domain=[R, T], type=VariableType.POSITIVE)
    psi_dem = Variable(m, "psi_dem", domain=[R, T], type=VariableType.POSITIVE)

    lam_var.lo[R, T] = 0.0
    lam_var.up[R, T] = lam_ub[R]
    beta_dem.up[R, T] = lam_ub[R]
    psi_dem.up[R, T] = lam_ub[R]
    mu.up[R, T] = mu_ub[R]
    gamma.up[exp, imp, T] = gamma_ub[exp, imp]

    # =====================================================================
    # Step 4 — LLP equations (time-indexed) + subsidy in stationarity
    # =====================================================================

    # --- Primal LLP Objective (not directly used in MPEC solve but kept) ---
    llp_gross_surplus = Sum(
        [R, T],
        a_dem_t_p[R, T] * x_dem[R, T]
        - (b_dem_t_p[R, T] / gp.Number(2.0)) * x_dem[R, T] * x_dem[R, T],
    )

    llp_total_cost = (
        Sum(
            [exp, imp, T],
            (c_man[exp] - s_var[exp, T] + c_ship[exp, imp]) * x[exp, imp, T],
        )
        + Sum(
            [exp, imp, T],
            (tau_imp[imp, exp, T] + tau_exp[exp, imp, T]) * x[exp, imp, T],
        )
        + Sum(
            [exp, imp, T],
            (eps_x / gp.Number(2.0)) * x[exp, imp, T] * x[exp, imp, T],
        )
    )

    eq_obj_llp = Equation(m, "eq_obj_llp")
    eq_obj_llp[...] = z_llp == llp_total_cost - llp_gross_surplus

    # --- Primal Constraints ---
    eq_bal = Equation(m, "eq_bal", domain=[imp, T])
    eq_bal[imp, T] = Sum(exp, x[exp, imp, T]) - x_dem[imp, T] == z

    eq_cap = Equation(m, "eq_cap", domain=[exp, T])
    eq_cap[exp, T] = Q_offer[exp, T] - Sum(imp, x[exp, imp, T]) >= z

    # --- Stationarity (KKT) ---
    eq_stat_x = Equation(m, "eq_stat_x", domain=[exp, imp, T])
    eq_stat_x[exp, imp, T] = (
        (c_man[exp] - s_var[exp, T] + c_ship[exp, imp]
         + tau_exp[exp, imp, T] + tau_imp[imp, exp, T])
        + eps_x * x[exp, imp, T]
        - lam_var[imp, T]
        + mu[exp, T]
        - gamma[exp, imp, T]
        == z
    )

    eq_stat_dem = Equation(m, "eq_stat_dem", domain=[imp, T])
    eq_stat_dem[imp, T] = (
        -(a_dem_t_p[imp, T] - b_dem_t_p[imp, T] * x_dem[imp, T])
        + lam_var[imp, T]
        + beta_dem[imp, T]
        - psi_dem[imp, T]
        == z
    )

    # --- Complementarity (KKT) ---
    eq_comp_mu = Equation(m, "eq_comp_mu", domain=[exp, T])
    if eps_comp == 0.0:
        eq_comp_mu[exp, T] = (
            mu[exp, T] * (Q_offer[exp, T] - Sum(imp, x[exp, imp, T])) == z
        )
    else:
        eq_comp_mu[exp, T] = (
            mu[exp, T] * (Q_offer[exp, T] - Sum(imp, x[exp, imp, T])) <= eps_value
        )

    eq_comp_gamma = Equation(m, "eq_comp_gamma", domain=[exp, imp, T])
    if eps_comp == 0.0:
        eq_comp_gamma[exp, imp, T] = gamma[exp, imp, T] * x[exp, imp, T] == z
    else:
        eq_comp_gamma[exp, imp, T] = gamma[exp, imp, T] * x[exp, imp, T] <= eps_value

    eq_comp_beta_dem = Equation(m, "eq_comp_beta_dem", domain=[imp, T])
    if eps_comp == 0.0:
        eq_comp_beta_dem[imp, T] = (
            beta_dem[imp, T] * (Dmax_t_p[imp, T] - x_dem[imp, T]) == z
        )
    else:
        eq_comp_beta_dem[imp, T] = (
            beta_dem[imp, T] * (Dmax_t_p[imp, T] - x_dem[imp, T]) <= eps_value
        )

    eq_comp_psi_dem = Equation(m, "eq_comp_psi_dem", domain=[imp, T])
    if eps_comp == 0.0:
        eq_comp_psi_dem[imp, T] = psi_dem[imp, T] * x_dem[imp, T] == z
    else:
        eq_comp_psi_dem[imp, T] = psi_dem[imp, T] * x_dem[imp, T] <= eps_value

    # =====================================================================
    # Step 5 — Capacity transitions + offer linkage
    # =====================================================================

    # 3 hard-coded transition equations
    eq_cap_trans_30 = Equation(m, "eq_cap_trans_30", domain=[R])
    eq_cap_trans_30[R] = (
        Kcap[R, "2030"] == Kcap[R, "2025"] + Icap[R, "2025"] - Dcap[R, "2025"]
    )

    eq_cap_trans_35 = Equation(m, "eq_cap_trans_35", domain=[R])
    eq_cap_trans_35[R] = (
        Kcap[R, "2035"] == Kcap[R, "2030"] + Icap[R, "2030"] - Dcap[R, "2030"]
    )

    eq_cap_trans_40 = Equation(m, "eq_cap_trans_40", domain=[R])
    eq_cap_trans_40[R] = (
        Kcap[R, "2040"] == Kcap[R, "2035"] + Icap[R, "2035"] - Dcap[R, "2035"]
    )

    # Offer linkage: Q_offer <= Kcap (explicit constraint)
    eq_offer_cap = Equation(m, "eq_offer_cap", domain=[R, T])
    eq_offer_cap[R, T] = Q_offer[R, T] <= Kcap[R, T]

    # =====================================================================
    # Collect equations
    # =====================================================================
    equations = {
        "eq_bal": eq_bal,
        "eq_cap": eq_cap,
        "eq_stat_x": eq_stat_x,
        "eq_stat_dem": eq_stat_dem,
        "eq_comp_mu": eq_comp_mu,
        "eq_comp_gamma": eq_comp_gamma,
        "eq_comp_beta_dem": eq_comp_beta_dem,
        "eq_comp_psi_dem": eq_comp_psi_dem,
        "eq_obj_llp": eq_obj_llp,
        "eq_cap_trans_30": eq_cap_trans_30,
        "eq_cap_trans_35": eq_cap_trans_35,
        "eq_cap_trans_40": eq_cap_trans_40,
        "eq_offer_cap": eq_offer_cap,
    }

    # =====================================================================
    # Step 6 — Objective = sum over time
    # =====================================================================
    models: Dict[str, Model] = {}
    for rname in data.players:
        r = rname

        # ---- Per-period welfare components (summed over T) ----

        # Consumer surplus
        cons_surplus_t = Sum(
            T,
            beta_p[T] * w_p[r] * (
                a_dem_t_p[r, T] * x_dem[r, T]
                - (b_dem_t_p[r, T] / gp.Number(2.0)) * x_dem[r, T] * x_dem[r, T]
                - lam_var[r, T] * x_dem[r, T]
            ),
        )

        # Tariff revenues
        imp_tariff_rev_t = Sum(
            [j, T],
            beta_p[T] * tau_imp[r, j, T] * x[j, r, T],
        )
        exp_tax_rev_t = Sum(
            [j, T],
            beta_p[T] * tau_exp[r, j, T] * x[r, j, T],
        )

        # Producer term
        producer_term_t = Sum(
            [j, T],
            beta_p[T] * (
                lam_var[j, T]
                - c_man[r]
                - c_ship[r, j]
                - tau_imp[j, r, T]
                - tau_exp[r, j, T]
            ) * x[r, j, T],
        )

        # Subsidy fiscal cost: government pays s*x for all exports
        subsidy_cost_t = Sum(
            [j, T],
            beta_p[T] * s_var[r, T] * x[r, j, T],
        )

        # Holding cost
        hold_cost_t = Sum(
            T,
            beta_p[T] * f_hold_p[r] * ytn_p[T] * Kcap[r, T],
        )

        # Investment cost
        inv_cost_t = Sum(
            T,
            beta_p[T] * c_inv_p[r] * ytn_p[T] * Icap[r, T],
        )

        # ---- Penalties (replicated per period) ----
        # Quadratic tariff penalties
        pen_imp_quad = Sum(
            T,
            -gp.Number(0.5) * rho_imp_p[r] * Sum(j, tau_imp[r, j, T] * tau_imp[r, j, T]),
        )
        pen_exp_quad = Sum(
            T,
            -gp.Number(0.5) * rho_exp_p[r] * Sum(j, tau_exp[r, j, T] * tau_exp[r, j, T]),
        )

        # Capacity utilization penalty
        pen_q_quad = Sum(
            T,
            -gp.Number(0.5) * kappa_Q[r] * (
                (Q_offer[r, T] - Sum(j, x[r, j, T]))
                * (Q_offer[r, T] - Sum(j, x[r, j, T]))
            ),
        )

        # Linear penalties
        pen_imp_lin = Sum(
            T,
            -rho_imp_p[r] * Sum(j, tau_imp[r, j, T]),
        )
        pen_exp_lin = Sum(
            T,
            -rho_exp_p[r] * Sum(j, tau_exp[r, j, T]),
        )
        pen_q_lin = Sum(
            T,
            -kappa_Q[r] * Q_offer[r, T],
        )

        # Proximal regularization
        pen_prox_q = Sum(
            T,
            -gp.Number(0.5) * rho_prox * (
                (Q_offer[r, T] - Q_offer_last[r, T])
                * (Q_offer[r, T] - Q_offer_last[r, T])
            ),
        )
        pen_prox_imp = Sum(
            T,
            -gp.Number(0.5) * rho_prox * Sum(
                j,
                (tau_imp[r, j, T] - tau_imp_last[r, j, T])
                * (tau_imp[r, j, T] - tau_imp_last[r, j, T]),
            ),
        )
        pen_prox_exp = Sum(
            T,
            -gp.Number(0.5) * rho_prox * Sum(
                j,
                (tau_exp[r, j, T] - tau_exp_last[r, j, T])
                * (tau_exp[r, j, T] - tau_exp_last[r, j, T]),
            ),
        )

        # ---- Assemble objective ----
        if use_quad:
            obj_welfare = (
                cons_surplus_t
                + imp_tariff_rev_t
                + exp_tax_rev_t
                + producer_term_t
                - subsidy_cost_t
                - hold_cost_t
                - inv_cost_t
                + pen_imp_quad
                + pen_exp_quad
                + pen_q_quad
                + pen_prox_q
                + pen_prox_imp
                + pen_prox_exp
            )
        else:
            obj_welfare = (
                cons_surplus_t
                + imp_tariff_rev_t
                + exp_tax_rev_t
                + producer_term_t
                - subsidy_cost_t
                - hold_cost_t
                - inv_cost_t
                + pen_imp_lin
                + pen_exp_lin
                + pen_q_lin
                + pen_prox_q
                + pen_prox_imp
                + pen_prox_exp
            )

        models[r] = Model(
            m,
            f"mpec_{r}",
            equations=list(equations.values()),
            problem=Problem.NLP,
            sense=Sense.MAX,
            objective=obj_welfare,
        )

    # =====================================================================
    # Return context
    # =====================================================================
    return ModelContext(
        container=m,
        sets={"R": R, "exp": exp, "imp": imp, "j": j, "T": T},
        params={
            "Dmax_t": Dmax_t_p,
            "a_dem_t": a_dem_t_p,
            "b_dem_t": b_dem_t_p,
            "Kcap_2025": Kcap_2025_p,
            "c_man": c_man,
            "c_ship": c_ship,
            "tau_imp_ub": tau_imp_ub_p,
            "tau_exp_ub": tau_exp_ub_p,
            "rho_imp": rho_imp_p,
            "rho_exp": rho_exp_p,
            "w": w_p,
            "kappa_Q": kappa_Q,
            "s_ub": s_ub_p,
            "f_hold": f_hold_p,
            "c_inv": c_inv_p,
            "beta_t": beta_p,
            "ytn": ytn_p,
            "Q_offer_last": Q_offer_last,
            "tau_imp_last": tau_imp_last,
            "tau_exp_last": tau_exp_last,
        },
        vars={
            "Q_offer": Q_offer,
            "tau_imp": tau_imp,
            "tau_exp": tau_exp,
            "s": s_var,
            "Kcap": Kcap,
            "Icap": Icap,
            "Dcap": Dcap,
            "x": x,
            "x_dem": x_dem,
            "lam": lam_var,
            "mu": mu,
            "gamma": gamma,
            "beta_dem": beta_dem,
            "psi_dem": psi_dem,
        },
        equations=equations,
        models=models,
    )


# =============================================================================
# Step 7 — apply_player_fixings (time-indexed)
# =============================================================================
def apply_player_fixings(
    ctx: ModelContext,
    data: ModelData,
    theta_Q: Dict[Tuple[str, str], float],
    theta_tau_imp: Dict[Tuple[str, str, str], float],
    theta_tau_exp: Dict[Tuple[str, str, str], float],
    theta_s: Dict[Tuple[str, str], float],
    theta_Icap: Dict[Tuple[str, str], float],
    theta_Dcap: Dict[Tuple[str, str], float],
    *,
    player: str,
) -> None:
    """Fix all other players' strategies; free current player's strategies."""
    times = data.times or list(_DEFAULT_TIMES)
    kcap_2025_dict = data.Kcap_2025 if data.Kcap_2025 is not None else {r: float(data.Qcap[r]) for r in data.regions}
    s_ub_dict = data.s_ub if data.s_ub is not None else {r: 0.0 for r in data.regions}

    Q_offer = ctx.vars["Q_offer"]
    tau_imp = ctx.vars["tau_imp"]
    tau_exp = ctx.vars["tau_exp"]
    s_var = ctx.vars["s"]
    Kcap = ctx.vars["Kcap"]
    Icap = ctx.vars["Icap"]
    Dcap = ctx.vars["Dcap"]

    T_set = ctx.sets["T"]

    for r in data.regions:
        for tp in times:
            if r == player:
                # Free current player's strategies
                # Q_offer bounded by Kcap (via eq_offer_cap constraint)
                Q_offer.lo[r, tp] = 0.0
                Q_offer.up[r, tp] = float("inf")  # constraint handles upper bound

                s_var.lo[r, tp] = 0.0
                s_var.up[r, tp] = float(s_ub_dict.get(r, 0.0))

                # Free capacity decisions
                if tp == times[0]:
                    # Kcap fixed at initial
                    Kcap.lo[r, tp] = float(kcap_2025_dict[r])
                    Kcap.up[r, tp] = float(kcap_2025_dict[r])
                else:
                    Kcap.lo[r, tp] = 0.0
                    Kcap.up[r, tp] = float("inf")

                if tp == times[-1]:
                    Icap.lo[r, tp] = 0.0
                    Icap.up[r, tp] = 0.0
                    Dcap.lo[r, tp] = 0.0
                    Dcap.up[r, tp] = 0.0
                else:
                    Icap.lo[r, tp] = 0.0
                    Icap.up[r, tp] = float("inf")
                    Dcap.lo[r, tp] = 0.0
                    Dcap.up[r, tp] = float("inf")

            elif r in data.players:
                # Fix other strategic player
                q_val = float(theta_Q.get((r, tp), float(kcap_2025_dict.get(r, 0.0))))
                Q_offer.lo[r, tp] = q_val
                Q_offer.up[r, tp] = q_val

                s_val = float(theta_s.get((r, tp), 0.0))
                s_var.lo[r, tp] = s_val
                s_var.up[r, tp] = s_val

                i_val = float(theta_Icap.get((r, tp), 0.0))
                Icap.lo[r, tp] = i_val
                Icap.up[r, tp] = i_val

                d_val = float(theta_Dcap.get((r, tp), 0.0))
                Dcap.lo[r, tp] = d_val
                Dcap.up[r, tp] = d_val

                # Kcap is determined by transitions — fix it
                if tp == times[0]:
                    Kcap.lo[r, tp] = float(kcap_2025_dict[r])
                    Kcap.up[r, tp] = float(kcap_2025_dict[r])
                else:
                    # Kcap implied by transitions; keep free so transition eqs work
                    Kcap.lo[r, tp] = 0.0
                    Kcap.up[r, tp] = float("inf")

            else:
                # Non-strategic: no capacity changes, no subsidy, fixed Q_offer
                v = float(kcap_2025_dict.get(r, 0.0))
                Q_offer.lo[r, tp] = v
                Q_offer.up[r, tp] = v

                Kcap.lo[r, tp] = v
                Kcap.up[r, tp] = v

                Icap.lo[r, tp] = 0.0
                Icap.up[r, tp] = 0.0
                Dcap.lo[r, tp] = 0.0
                Dcap.up[r, tp] = 0.0

                s_var.lo[r, tp] = 0.0
                s_var.up[r, tp] = 0.0

    # -- Tariffs --
    for im in data.regions:
        for ex in data.regions:
            for tp in times:
                if im == ex:
                    tau_imp.lo[im, ex, tp] = 0.0
                    tau_imp.up[im, ex, tp] = 0.0
                    continue
                ub = data.tau_imp_ub[(im, ex)]
                if im in data.non_strategic or ex in data.non_strategic:
                    tau_imp.lo[im, ex, tp] = 0.0
                    tau_imp.up[im, ex, tp] = 0.0
                elif im == player:
                    tau_imp.lo[im, ex, tp] = 0.0
                    tau_imp.up[im, ex, tp] = ub
                else:
                    v = float(theta_tau_imp.get((im, ex, tp), 0.0))
                    tau_imp.lo[im, ex, tp] = v
                    tau_imp.up[im, ex, tp] = v

    for ex in data.regions:
        for im in data.regions:
            for tp in times:
                if ex == im:
                    tau_exp.lo[ex, im, tp] = 0.0
                    tau_exp.up[ex, im, tp] = 0.0
                    continue
                ub = data.tau_exp_ub[(ex, im)]
                if ex in data.non_strategic or im in data.non_strategic:
                    tau_exp.lo[ex, im, tp] = 0.0
                    tau_exp.up[ex, im, tp] = 0.0
                elif ex == player:
                    tau_exp.lo[ex, im, tp] = 0.0
                    tau_exp.up[ex, im, tp] = ub
                else:
                    v = float(theta_tau_exp.get((ex, im, tp), 0.0))
                    tau_exp.lo[ex, im, tp] = v
                    tau_exp.up[ex, im, tp] = v


# =============================================================================
# Step 7 — extract_state (time-indexed)
# =============================================================================
def extract_state(
    ctx: ModelContext, variables: List[str] | None = None
) -> Dict[str, Dict]:
    def _maybe_var(name: str):
        if variables is not None and name not in variables:
            return {}
        v = ctx.vars.get(name)
        if v is None:
            return {}
        out = v.toDict()
        return out if isinstance(out, dict) else {}

    # Extract objective values from solved models
    obj_values = {}
    if variables is None or "obj" in variables:
        for r, model in ctx.models.items():
            try:
                obj_values[r] = float(model.objective_value)
            except (AttributeError, TypeError):
                pass

    return {
        "Q_offer": _maybe_var("Q_offer"),
        "tau_imp": _maybe_var("tau_imp"),
        "tau_exp": _maybe_var("tau_exp"),
        "s": _maybe_var("s"),
        "Kcap": _maybe_var("Kcap"),
        "Icap": _maybe_var("Icap"),
        "Dcap": _maybe_var("Dcap"),
        "x": _maybe_var("x"),
        "x_dem": _maybe_var("x_dem"),
        "lam": _maybe_var("lam"),
        "mu": _maybe_var("mu"),
        "gamma": _maybe_var("gamma"),
        "beta_dem": _maybe_var("beta_dem"),
        "psi_dem": _maybe_var("psi_dem"),
        "obj": obj_values,
    }
