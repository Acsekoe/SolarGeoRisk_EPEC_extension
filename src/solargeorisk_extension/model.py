from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Set as PySet, Tuple

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


@dataclass
class ModelContext:
    container: Container
    sets: Dict[str, Set]
    params: Dict[str, Parameter]
    vars: Dict[str, Variable]
    equations: Dict[str, Equation]
    models: Dict[str, Model]


def _sanity_check_data(data: ModelData) -> None:
    bad_b = sorted([r for r in data.regions if float(data.b_dem[r]) <= 0.0])
    if bad_b:
        raise ValueError(f"All b_dem must be > 0. Invalid regions: {bad_b}")

    bad_dmax = sorted([r for r in data.regions if float(data.Dmax[r]) <= 0.0])
    if bad_dmax:
        raise ValueError(f"All Dmax must be > 0. Invalid regions: {bad_dmax}")

    bad_qcap = sorted([r for r in data.regions if float(data.Qcap[r]) < 0.0])
    if bad_qcap:
        raise ValueError(f"All Qcap must be >= 0. Invalid regions: {bad_qcap}")

    kappa_map = getattr(data, "kappa_Q", None) or {}
    bad_kappa = sorted([r for r in kappa_map if float(kappa_map[r]) < 0.0])
    if bad_kappa:
        raise ValueError(f"All kappa_Q must be >= 0. Invalid regions: {bad_kappa}")


def build_model(data: ModelData, working_directory: str | None = None) -> ModelContext:
    if working_directory and " " in str(working_directory):
        raise ValueError(f"GAMS working directory must be space-free. Got: {working_directory}")

    _sanity_check_data(data)

    unknown_players = sorted(set(data.players) - set(data.regions))
    if unknown_players:
        raise ValueError(f"All players must be in regions. Unknown players: {unknown_players}")

    # Profit scenario removed. Default to welfare maximization for all players.
    settings = data.settings or {}
    use_quad = bool(settings.get("use_quad", False))

    m = Container(working_directory=working_directory)

    R = Set(m, "R", records=data.regions)
    exp = Alias(m, "exp", R)
    imp = Alias(m, "imp", R)
    j = Alias(m, "j", R)

    D = Parameter(m, "D", domain=[R], records=[(k, data.D[k]) for k in data.regions])
    Dmax = Parameter(m, "Dmax", domain=[R], records=[(k, data.Dmax[k]) for k in data.regions])
    a_dem = Parameter(m, "a_dem", domain=[R], records=[(k, data.a_dem[k]) for k in data.regions])
    b_dem = Parameter(m, "b_dem", domain=[R], records=[(k, data.b_dem[k]) for k in data.regions])

    Qcap = Parameter(m, "Qcap", domain=[R], records=[(k, data.Qcap[k]) for k in data.regions])
    c_man = Parameter(m, "c_man", domain=[R], records=[(k, data.c_man[k]) for k in data.regions])

    c_ship = Parameter(
        m,
        "c_ship",
        domain=[exp, imp],
        records=[(r, i, data.c_ship[(r, i)]) for r in data.regions for i in data.regions],
    )

    tau_imp_ub = Parameter(
        m,
        "tau_imp_ub",
        domain=[imp, exp],
        records=[(i, r, data.tau_imp_ub[(i, r)]) for i in data.regions for r in data.regions],
    )
    tau_exp_ub = Parameter(
        m,
        "tau_exp_ub",
        domain=[exp, imp],
        records=[(r, i, data.tau_exp_ub[(r, i)]) for r in data.regions for i in data.regions],
    )

    rho_imp = Parameter(m, "rho_imp", domain=[R], records=[(k, data.rho_imp[k]) for k in data.regions])
    rho_exp = Parameter(m, "rho_exp", domain=[R], records=[(k, data.rho_exp[k]) for k in data.regions])
    w = Parameter(m, "w", domain=[R], records=[(k, data.w[k]) for k in data.regions])

    eps_x = gp.Number(float(data.eps_x))
    eps_comp = float(data.eps_comp)
    eps_value = gp.Number(eps_comp)


    # =========================================================================
    # UPPER LEVEL PROBLEM (ULP) - STRATEGIC VARIABLES
    # =========================================================================
    Q_offer = Variable(m, "Q_offer", domain=[R], type=VariableType.POSITIVE)
    tau_imp = Variable(m, "tau_imp", domain=[imp, exp], type=VariableType.POSITIVE)
    tau_exp = Variable(m, "tau_exp", domain=[exp, imp], type=VariableType.POSITIVE)

    Q_offer.up[R] = Qcap[R]
    tau_imp.up[imp, exp] = tau_imp_ub[imp, exp]
    tau_exp.up[exp, imp] = tau_exp_ub[exp, imp]

    for r in data.regions:
        tau_imp.lo[r, r] = 0.0
        tau_imp.up[r, r] = 0.0
        tau_exp.lo[r, r] = 0.0
        tau_exp.up[r, r] = 0.0

    # =========================================================================
    # LOWER LEVEL PROBLEM (LLP) - MARKET VARIABLES
    # =========================================================================
    z_llp = Variable(m, "z_llp", type=VariableType.FREE) # Primal LLP Objective Value

    x = Variable(m, "x", domain=[exp, imp], type=VariableType.POSITIVE)
    x_dem = Variable(m, "x_dem", domain=[R], type=VariableType.POSITIVE)
    x_dem.up[R] = Dmax[R]

    lam = Variable(m, "lam", domain=[R], type=VariableType.FREE)
    mu = Variable(m, "mu", domain=[R], type=VariableType.POSITIVE)
    gamma = Variable(m, "gamma", domain=[exp, imp], type=VariableType.POSITIVE)
    beta_dem = Variable(m, "beta_dem", domain=[R], type=VariableType.POSITIVE)
    psi_dem = Variable(m, "psi_dem", domain=[R], type=VariableType.POSITIVE)

    lam.lo[R] = 0.0

    lam_ub_values = {i: float(data.a_dem[i]) for i in data.regions}
    lam_ub = Parameter(m, "lam_ub", domain=[R], records=[(i, lam_ub_values[i]) for i in data.regions])
    lam.up[R] = lam_ub[R]
    beta_dem.up[R] = lam_ub[R]
    psi_dem.up[R] = lam_ub[R]

    mu_ub_values = {
        r: max(
            0.0,
            max(
                float(lam_ub_values[i]) - (float(data.c_man[r]) + float(data.c_ship[(r, i)])) for i in data.regions
            ),
        )
        for r in data.regions
    }
    mu_ub = Parameter(m, "mu_ub", domain=[R], records=[(r, mu_ub_values[r]) for r in data.regions])
    mu.up[R] = mu_ub[R]

    gamma_ub = Parameter(
        m,
        "gamma_ub",
        domain=[exp, imp],
        records=[
            (
                r,
                i,
                float(data.c_man[r])
                + float(data.c_ship[(r, i)])
                + float(data.tau_imp_ub[(i, r)])
                + float(data.tau_exp_ub[(r, i)])
                + float(data.eps_x) * float(data.Qcap[r])
                + float(mu_ub_values.get(r, 0.0)),
            )
            for r in data.regions
            for i in data.regions
        ],
    )
    gamma.up[exp, imp] = gamma_ub[exp, imp]

    # =========================================================================
    # LOWER LEVEL PROBLEM (LLP) - Market Clearing
    # =========================================================================

    # --- Primal Objective Function (Min z_llp) ---
    # Max Welfare = Gross Consumer Surplus - Total Costs
    # Equivalent to Min Cost - Gross Consumer Surplus
    # Gross Consumer Surplus = (a * x_dem - 0.5 * b * x_dem^2)
    
    llp_gross_surplus = Sum(
        R, 
        a_dem[R] * x_dem[R] - (b_dem[R] / gp.Number(2.0)) * x_dem[R] * x_dem[R]
    )

    llp_total_cost = (
        Sum([exp, imp], (c_man[exp] + c_ship[exp, imp]) * x[exp, imp])  # Production + Shipping
        + Sum([exp, imp], (tau_imp[imp, exp] + tau_exp[exp, imp]) * x[exp, imp]) # Tariffs
        + Sum([exp, imp], (eps_x / gp.Number(2.0)) * x[exp, imp] * x[exp, imp]) # Regularization
    )

    eq_obj_llp = Equation(m, "eq_obj_llp")
    eq_obj_llp[...] = z_llp == llp_total_cost - llp_gross_surplus

    # --- Primal Constraints ---
    eq_bal = Equation(m, "eq_bal", domain=[imp])
    eq_bal[imp] = Sum(exp, x[exp, imp]) - x_dem[imp] == z

    eq_cap = Equation(m, "eq_cap", domain=[exp])
    eq_cap[exp] = Q_offer[exp] - Sum(imp, x[exp, imp]) >= z

    # --- Stationarity Conditions (KKT) ---
    eq_stat_x = Equation(m, "eq_stat_x", domain=[exp, imp])
    eq_stat_x[exp, imp] = (
        (c_man[exp] + c_ship[exp, imp] + tau_exp[exp, imp] + tau_imp[imp, exp])
        + eps_x * x[exp, imp]
        - lam[imp]
        + mu[exp]
        - gamma[exp, imp]
        == z
    )

    eq_stat_dem = Equation(m, "eq_stat_dem", domain=[imp])
    eq_stat_dem[imp] = -(a_dem[imp] - b_dem[imp] * x_dem[imp]) + lam[imp] + beta_dem[imp] - psi_dem[imp] == z

    # --- Complementarity Conditions (KKT) ---
    eq_comp_mu = Equation(m, "eq_comp_mu", domain=[exp])
    if eps_comp == 0.0:
        eq_comp_mu[exp] = mu[exp] * (Q_offer[exp] - Sum(imp, x[exp, imp])) == z
    else:
        eq_comp_mu[exp] = mu[exp] * (Q_offer[exp] - Sum(imp, x[exp, imp])) <= eps_value

    eq_comp_gamma = Equation(m, "eq_comp_gamma", domain=[exp, imp])
    if eps_comp == 0.0:
        eq_comp_gamma[exp, imp] = gamma[exp, imp] * x[exp, imp] == z
    else:
        eq_comp_gamma[exp, imp] = gamma[exp, imp] * x[exp, imp] <= eps_value

    eq_comp_beta_dem = Equation(m, "eq_comp_beta_dem", domain=[imp])
    if eps_comp == 0.0:
        eq_comp_beta_dem[imp] = beta_dem[imp] * (Dmax[imp] - x_dem[imp]) == z
    else:
        eq_comp_beta_dem[imp] = beta_dem[imp] * (Dmax[imp] - x_dem[imp]) <= eps_value

    eq_comp_psi_dem = Equation(m, "eq_comp_psi_dem", domain=[imp])
    if eps_comp == 0.0:
        eq_comp_psi_dem[imp] = psi_dem[imp] * x_dem[imp] == z
    else:
        eq_comp_psi_dem[imp] = psi_dem[imp] * x_dem[imp] <= eps_value

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
    }

    kappa_map = getattr(data, "kappa_Q", None) or {}
    kappa_by_r = {k: float(kappa_map.get(k, 0.0)) for k in data.regions}
    kappa_Q = Parameter(m, "kappa_Q", domain=[R], records=[(k, kappa_by_r[k]) for k in data.regions])



    # =========================================================================
    # UPPER LEVEL PROBLEM (ULP) - Strategic Players
    # =========================================================================
    models: Dict[str, Model] = {}
    for rname in data.players:
        r = rname

        imp_tariff_rev = Sum(j, tau_imp[r, j] * x[j, r])
        exp_tax_rev = Sum(j, tau_exp[r, j] * x[r, j])

        pen_imp_quad = -gp.Number(0.5) * rho_imp[r] * Sum(j, tau_imp[r, j] * tau_imp[r, j])
        pen_exp_quad = -gp.Number(0.5) * rho_exp[r] * Sum(j, tau_exp[r, j] * tau_exp[r, j])
        pen_q_quad = -gp.Number(0.5) * kappa_Q[r] * (Q_offer[r] * Q_offer[r])

        pen_imp_lin = -rho_imp[r] * Sum(j, tau_imp[r, j])
        pen_exp_lin = -rho_exp[r] * Sum(j, tau_exp[r, j])
        pen_q_lin = -kappa_Q[r] * Q_offer[r]

        producer_term = Sum(
            j,
            (lam[j] - c_man[r] - c_ship[r, j] - tau_imp[j, r]) * x[r, j],
        )

        cons_surplus = (
            a_dem[r] * x_dem[r]
            - (b_dem[r] / gp.Number(2.0)) * x_dem[r] * x_dem[r]
            - lam[r] * x_dem[r]
        )

        if use_quad:
            obj_welfare = (
                w[r] * cons_surplus
                + imp_tariff_rev
                + exp_tax_rev
                + producer_term
                + pen_imp_quad
                + pen_exp_quad
                + pen_q_quad
            )
        else:
            obj_welfare = (
                w[r] * cons_surplus
                + imp_tariff_rev
                + exp_tax_rev
                + producer_term
                + pen_imp_lin
                + pen_exp_lin
                + pen_q_lin
            )

        models[r] = Model(
            m,
            f"mpec_{r}",
            equations=list(equations.values()),
            problem=Problem.NLP,
            sense=Sense.MAX,
            objective=obj_welfare,
        )

    return ModelContext(
        container=m,
        sets={"R": R, "exp": exp, "imp": imp, "j": j},
        params={
            "D": D,
            "Dmax": Dmax,
            "a_dem": a_dem,
            "b_dem": b_dem,
            "Qcap": Qcap,
            "c_man": c_man,
            "c_ship": c_ship,
            "tau_imp_ub": tau_imp_ub,
            "tau_exp_ub": tau_exp_ub,
            "rho_imp": rho_imp,
            "rho_exp": rho_exp,
            "w": w,
            "kappa_Q": kappa_Q,
        },
        vars={
            "Q_offer": Q_offer,
            "tau_imp": tau_imp,
            "tau_exp": tau_exp,
            "x": x,
            "x_dem": x_dem,
            "lam": lam,
            "mu": mu,
            "gamma": gamma,
            "beta_dem": beta_dem,
            "psi_dem": psi_dem,
        },
        equations=equations,
        models=models,
    )


def apply_player_fixings(
    ctx: ModelContext,
    data: ModelData,
    theta_Q: Dict[str, float],
    theta_tau_imp: Dict[Tuple[str, str], float],
    theta_tau_exp: Dict[Tuple[str, str], float],
    *,
    player: str,
) -> None:
    Q_offer = ctx.vars["Q_offer"]
    tau_imp = ctx.vars["tau_imp"]
    tau_exp = ctx.vars["tau_exp"]

    for r in data.regions:
        if r == player:
            Q_offer.lo[r], Q_offer.up[r] = 0.0, data.Qcap[r]
        elif r in data.players:
            v = float(theta_Q[r])
            Q_offer.lo[r], Q_offer.up[r] = v, v
        else:
            v = float(data.Qcap[r])
            Q_offer.lo[r], Q_offer.up[r] = v, v

    for imp in data.regions:
        for exp in data.regions:
            if imp == exp:
                tau_imp.lo[imp, exp] = 0.0
                tau_imp.up[imp, exp] = 0.0
                continue
            ub = data.tau_imp_ub[(imp, exp)]
            if imp in data.non_strategic or exp in data.non_strategic:
                tau_imp.lo[imp, exp] = 0.0
                tau_imp.up[imp, exp] = 0.0
            elif imp == player:
                tau_imp.lo[imp, exp] = 0.0
                tau_imp.up[imp, exp] = ub
            else:
                v = float(theta_tau_imp[(imp, exp)])
                tau_imp.lo[imp, exp] = v
                tau_imp.up[imp, exp] = v

    for exp in data.regions:
        for imp in data.regions:
            if exp == imp:
                tau_exp.lo[exp, imp] = 0.0
                tau_exp.up[exp, imp] = 0.0
                continue
            ub = data.tau_exp_ub[(exp, imp)]
            if exp in data.non_strategic or imp in data.non_strategic:
                tau_exp.lo[exp, imp] = 0.0
                tau_exp.up[exp, imp] = 0.0
            elif exp == player:
                tau_exp.lo[exp, imp] = 0.0
                tau_exp.up[exp, imp] = ub
            else:
                v = float(theta_tau_exp[(exp, imp)])
                tau_exp.lo[exp, imp] = v
                tau_exp.up[exp, imp] = v


def extract_state(ctx: ModelContext) -> Dict[str, Dict]:
    def _maybe_var(name: str):
        v = ctx.vars.get(name)
        if v is None:
            return {}
        out = v.toDict()
        return out if isinstance(out, dict) else {}

    return {
        "Q_offer": _maybe_var("Q_offer"),
        "tau_imp": _maybe_var("tau_imp"),
        "tau_exp": _maybe_var("tau_exp"),
        "x": _maybe_var("x"),
        "x_dem": _maybe_var("x_dem"),
        "lam": _maybe_var("lam"),
        "mu": _maybe_var("mu"),
        "gamma": _maybe_var("gamma"),
        "beta_dem": _maybe_var("beta_dem"),
        "psi_dem": _maybe_var("psi_dem"),
    }

