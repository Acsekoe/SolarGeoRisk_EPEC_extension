from __future__ import annotations

import numbers
from typing import Dict, List, Tuple

import pandas as pd

from .model_single_year import ModelData
from .model_intertemporal import ModelData as IntertemporalModelData


def _require_columns(df: pd.DataFrame, cols: List[str], sheet: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet '{sheet}': {missing}")


def _norm_region(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _get_optional_float(row: pd.Series, col: str, default: float) -> float:
    if col not in row.index:
        return float(default)
    value = row[col]
    if pd.isna(value):
        return float(default)
    return float(value)


def _get_setting_float(settings: Dict[str, object], key: str, default: float) -> float:
    if key not in settings:
        return float(default)
    try:
        value = float(settings[key])
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(value):
        return float(default)
    return float(value)


def _get_setting_bool(settings: Dict[str, object], key: str, default: bool) -> bool:
    value = settings.get(key, default)
    if value is None or pd.isna(value):
        return bool(default)
    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
        raise ValueError(f"Invalid boolean for setting '{key}': {value!r}")

    if isinstance(value, numbers.Integral):
        iv = int(value)
        if iv == 0:
            return False
        if iv == 1:
            return True
        raise ValueError(f"Invalid boolean for setting '{key}': {value!r} (expected 0/1)")

    if isinstance(value, numbers.Real):
        fv = float(value)
        if pd.isna(fv):
            return bool(default)
        if fv == 0.0:
            return False
        if fv == 1.0:
            return True
        raise ValueError(f"Invalid boolean for setting '{key}': {value!r} (expected 0/1)")

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Invalid boolean for setting '{key}': {value!r}")


def load_data_from_excel(path: str) -> ModelData:
    # regions
    try:
        df_regions = pd.read_excel(path, sheet_name="regions")
    except ValueError as exc:
        raise ValueError("Missing required sheet: regions") from exc

    _require_columns(df_regions, ["r"], "regions")
    regions = [_norm_region(v) for v in df_regions["r"].tolist()]
    regions = [r for r in regions if r]
    if not regions:
        raise ValueError("No regions found in sheet 'regions'.")

    # params_region
    try:
        df_params = pd.read_excel(path, sheet_name="params_region")
    except ValueError as exc:
        raise ValueError("Missing required sheet: params_region") from exc

    _require_columns(df_params, ["r", "Qcap_exist (GW)", "c_man (USD/kW)", "a_dem", "b_dem"], "params_region")

    d_col = next((c for c in ["D", "D (GW)"] if c in df_params.columns), None)
    dmax_col = next((c for c in ["Dmax", "Dmax (GW)", "Dmax_2025 (GW)", "Dmax_2025(GW)", "Dmax_2025"] if c in df_params.columns), None)
    if dmax_col is None and d_col is None:
        raise ValueError("Missing demand-cap column in sheet 'params_region'. Add 'Dmax' or legacy 'D'.")

    df_params = df_params.copy()
    df_params["r"] = df_params["r"].map(_norm_region)
    df_params = df_params[df_params["r"].isin(regions)]

    row_map: Dict[str, pd.Series] = {}
    for _, row in df_params.iterrows():
        r = row["r"]
        if r and r not in row_map:
            row_map[r] = row

    missing_regions = [r for r in regions if r not in row_map]
    if missing_regions:
        raise ValueError(f"params_region missing rows for regions: {missing_regions}")

    Qcap: Dict[str, float] = {}
    D: Dict[str, float] = {}
    Dmax: Dict[str, float] = {}
    a_dem: Dict[str, float] = {}
    b_dem: Dict[str, float] = {}
    c_man: Dict[str, float] = {}
    for r, row in row_map.items():
        qcap_v = row["Qcap_exist (GW)"]
        if pd.isna(qcap_v):
            raise ValueError(f"Column 'Qcap_exist (GW)' has missing value for region '{r}'.")
        qcap_f = float(qcap_v)
        if qcap_f < 0.0:
            raise ValueError(f"Column 'Qcap_exist (GW)' must be >= 0 for region '{r}'. Got: {qcap_f}")
        Qcap[r] = qcap_f

        dmax_name = dmax_col or d_col
        assert dmax_name is not None
        dmax_v = row[dmax_name]
        if pd.isna(dmax_v):
            raise ValueError(f"Column '{dmax_name}' has missing value for region '{r}'.")
        dmax_f = float(dmax_v)
        if dmax_f <= 0.0:
            raise ValueError(f"Demand cap must be > 0 for region '{r}'. Got: {dmax_f} from '{dmax_name}'.")
        Dmax[r] = dmax_f

        if d_col is not None:
            d_v = row[d_col]
            if pd.isna(d_v):
                raise ValueError(f"Column '{d_col}' has missing value for region '{r}'.")
            d_f = float(d_v)
        else:
            d_f = dmax_f
        if d_f <= 0.0:
            raise ValueError(f"Column 'D' must be > 0 for region '{r}'. Got: {d_f}")
        D[r] = d_f

        a_v = row["a_dem"]
        b_v = row["b_dem"]
        # Allow NaN a_dem/b_dem for regions that are pure importers (no demand function).
        # The model build() handles NaN via its compatibility fallback.
        a_f = float(a_v) if not pd.isna(a_v) else float("nan")
        b_f = float(b_v) if not pd.isna(b_v) else float("nan")
        if not pd.isna(a_f) and a_f <= 0.0:
            raise ValueError(f"Column 'a_dem' must be > 0 for region '{r}'. Got: {a_f}")
        if not pd.isna(b_f) and b_f <= 0.0:
            raise ValueError(f"Column 'b_dem' must be > 0 for region '{r}'. Got: {b_f}")
        a_dem[r] = a_f
        b_dem[r] = b_f


        c_v = row["c_man (USD/kW)"]
        if pd.isna(c_v):
            raise ValueError(f"Column 'c_man (USD/kW)' has missing value for region '{r}'.")
        c_man[r] = float(c_v)

    # c_ship
    try:
        df_ship = pd.read_excel(path, sheet_name="c_ship")
    except ValueError as exc:
        raise ValueError("Missing required sheet: c_ship") from exc

    if df_ship.shape[1] < 2:
        raise ValueError("Sheet 'c_ship' must have an exporter column and at least one importer column.")

    df_ship = df_ship.copy()
    first_col = df_ship.columns[0]
    df_ship = df_ship.rename(columns={first_col: "exporter"})
    df_ship["exporter"] = df_ship["exporter"].map(_norm_region)
    importers = [str(c).strip().lower() for c in df_ship.columns if c != "exporter"]
    df_ship.columns = ["exporter"] + importers

    missing_rows = [r for r in regions if r not in set(df_ship["exporter"].tolist())]
    if missing_rows:
        raise ValueError(f"c_ship is missing exporter rows for regions: {missing_rows}")
    missing_cols = [r for r in regions if r not in importers]
    if missing_cols:
        raise ValueError(f"c_ship is missing importer columns for regions: {missing_cols}")

    df_ship = df_ship.set_index("exporter", drop=True)
    c_ship: Dict[Tuple[str, str], float] = {}
    for exp in regions:
        for imp in regions:
            value = df_ship.at[exp, imp]
            if pd.isna(value):
                if exp == imp:
                    c_ship[(exp, imp)] = 0.0
                else:
                    raise ValueError(f"c_ship missing value for ({exp}, {imp}).")
            else:
                c_ship[(exp, imp)] = float(value)

    # settings
    try:
        df_settings = pd.read_excel(path, sheet_name="settings")
    except ValueError as exc:
        raise ValueError("Missing required sheet: settings") from exc
    _require_columns(df_settings, ["setting", "value"], "settings")

    settings: Dict[str, object] = {}
    for _, row in df_settings.iterrows():
        key = str(row["setting"]).strip().lower()
        if not key or key == "nan":
            continue
        settings[key] = row["value"]

    welfare_flag = _get_setting_bool(settings, "scenario_welfare", True)
    cap_ch_flag = _get_setting_bool(settings, "scenario_capitalistic_china", False)
    if (welfare_flag and cap_ch_flag) or (not welfare_flag and not cap_ch_flag):
        raise ValueError("Exactly one of scenario_welfare or scenario_capitalistic_china must be TRUE in settings.")
    settings["scenario_name"] = "welfare" if welfare_flag else "capitalistic_china"

    tau_max_default = _get_setting_float(settings, "tau_max", 100.0)
    eps_comp = _get_setting_float(settings, "eps_comp", 1e-6)
    eps_x = _get_setting_float(settings, "eps_x", 1e-3)

    cap_raw = settings.get("capitalistic_players", None)
    cap_tokens: List[str] = []
    if cap_raw is not None and not pd.isna(cap_raw):
        text = str(cap_raw).strip()
        if text and text.lower() != "nan":
            cap_tokens = [t.strip().lower() for t in text.split(",") if t.strip()]
    cap_set = set(cap_tokens) & set(regions)
    settings["capitalistic_players_set"] = sorted(cap_set)

    players_raw = settings.get("players", None)
    players_list: List[str] = []
    if players_raw is not None and str(players_raw).strip():
        seen = set()
        for t in [t.strip().lower() for t in str(players_raw).split(",")]:
            if t in regions and t not in seen:
                players_list.append(t)
                seen.add(t)
    if not players_list:
        players_list = list(regions)

    max_c_ship = max(c_ship.values()) if c_ship else 0.0

    rho_imp_default = _get_setting_float(settings, "rho_imp", 1e-3)
    rho_exp_default = _get_setting_float(settings, "rho_exp", 1e-3)

    rho_imp: Dict[str, float] = {}
    rho_exp: Dict[str, float] = {}
    w: Dict[str, float] = {}
    tau_imp_max: Dict[str, float] = {}
    tau_exp_max: Dict[str, float] = {}
    kappa_Q: Dict[str, float] = {r: 0.0 for r in regions}

    if "w" in df_params.columns:
        for r in regions:
            row = row_map[r]
            value = row["w"]
            if pd.isna(value):
                raise ValueError(f"Column 'w' has missing value for region '{r}'.")
            value_f = float(value)
            if value_f < 0.0:
                raise ValueError(f"Column 'w' must be >= 0 for region '{r}'. Got: {value_f}")
            w[r] = value_f
    else:
        w = {r: 1.0 for r in regions}

    kappa_col = next((c for c in df_params.columns if str(c).strip().lower() == "kappa_q"), None)
    if kappa_col is not None:
        for r in regions:
            row = row_map[r]
            value = row[kappa_col]
            value_f = 0.0 if pd.isna(value) else float(value)
            if value_f < 0.0:
                raise ValueError(f"Column '{kappa_col}' must be >= 0 for region '{r}'. Got: {value_f}")
            kappa_Q[r] = value_f

    for r in regions:
        row = row_map[r]
        rho_imp[r] = _get_optional_float(row, "rho_imp", float(rho_imp_default))
        rho_exp[r] = _get_optional_float(row, "rho_exp", float(rho_exp_default))
        if float(rho_imp[r]) < 0.0:
            raise ValueError(f"rho_imp must be >= 0 for region '{r}'. Got: {rho_imp[r]}")
        if float(rho_exp[r]) < 0.0:
            raise ValueError(f"rho_exp must be >= 0 for region '{r}'. Got: {rho_exp[r]}")
        tau_imp_max[r] = _get_optional_float(row, "tau_imp_max", tau_max_default)
        tau_exp_max[r] = _get_optional_float(row, "tau_exp_max", tau_max_default)

    tau_imp_ub: Dict[Tuple[str, str], float] = {}
    tau_exp_ub: Dict[Tuple[str, str], float] = {}
    for imp in regions:
        for exp in regions:
            if imp == exp:
                tau_imp_ub[(imp, exp)] = 0.0
                tau_exp_ub[(exp, imp)] = 0.0
            else:
                tau_imp_ub[(imp, exp)] = float(tau_imp_max[imp])
                tau_exp_ub[(exp, imp)] = float(tau_exp_max[exp])

    return ModelData(
        regions=regions,
        players=players_list,
        non_strategic=set(),
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
        eps_x=float(eps_x),
        eps_comp=float(eps_comp),
        kappa_Q=kappa_Q,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Intertemporal data loader
# ---------------------------------------------------------------------------
_TIMES = ["2025", "2030", "2035", "2040"]

# Map clean year label → possible column names in the Excel sheet
_DMAX_COL_CANDIDATES = {
    "2025": ["Dmax_2025 (GW)", "Dmax_2025(GW)", "Dmax_2025"],
    "2030": ["Dmax_2030 (GW)", "Dmax_2030(GW)", "Dmax_2030"],
    "2035": ["Dmax_2035 (GW)", "Dmax_2035(GW)", "Dmax_2035"],
    "2040": ["Dmax_2040 (GW)", "Dmax_2040(GW)", "Dmax_2040"],
}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first candidate column name that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data_intertemporal_from_excel(path: str) -> IntertemporalModelData:
    """Load intertemporal ModelData from an Excel file.

    The sheet layout is identical to the single-year format, with these
    additional columns in *params_region*:

      Dmax_2025 (GW), Dmax_2030 (GW), Dmax_2035 (GW), Dmax_2040 (GW)
        – time-indexed demand caps (required; fallback to single Dmax if absent)

      f_hold       – holding cost per GW per year (optional, default 0)
      c_inv        – investment cost per GW per year (optional, default 0)
      s_ub         – subsidy upper bound per GW (optional, default 0)
      Kcap_2025    – initial capacity override (optional, fallback to Qcap_exist)
    """

    def _opt_float(row: "pd.Series", col: str, default: float) -> float:
        """Read an optional float column from a pandas Series, returning default if missing/NaN."""
        if col not in row.index:
            return default
        val = row[col]
        try:
            f = float(val)
            return default if pd.isna(f) else f
        except (TypeError, ValueError):
            return default

    # Reuse the single-year loader for the shared fields
    sy_data = load_data_from_excel(path)

    # Re-read params_region for the extra columns
    df_params = pd.read_excel(path, sheet_name="params_region").copy()
    df_params["r"] = df_params["r"].map(_norm_region)
    df_params = df_params[df_params["r"].isin(sy_data.regions)]

    row_map: Dict[str, pd.Series] = {}
    for _, row in df_params.iterrows():
        r = row["r"]
        if r and r not in row_map:
            row_map[r] = row

    regions = sy_data.regions

    # --- Time-indexed Dmax ---
    Dmax_t: Dict[Tuple[str, str], float] = {}
    for tp in _TIMES:
        col = _find_col(df_params, _DMAX_COL_CANDIDATES[tp])
        for r in regions:
            if col is not None:
                raw = row_map[r].get(col, float("nan"))
                val = float(raw) if not pd.isna(raw) else float(sy_data.Dmax[r])
            else:
                val = float(sy_data.Dmax[r])
            if val <= 0.0:
                raise ValueError(
                    f"Dmax_t for region '{r}', year '{tp}' must be > 0. Got: {val}"
                )
            Dmax_t[(r, tp)] = val

    # --- Optional per-region capacity-cost columns ---
    f_hold: Dict[str, float] = {}
    c_inv: Dict[str, float] = {}
    s_ub: Dict[str, float] = {}
    Kcap_2025: Dict[str, float] = {}

    for r in regions:
        row = row_map[r]
        f_hold[r] = _opt_float(row, "f_hold", 0.0)
        c_inv[r] = _opt_float(row, "c_inv", 0.0)
        s_ub[r] = _opt_float(row, "s_ub", 0.0)
        kcap_raw = _opt_float(row, "Kcap_2025", float("nan"))
        if not pd.isna(kcap_raw):
            Kcap_2025[r] = float(kcap_raw)
        else:
            Kcap_2025[r] = float(sy_data.Qcap[r])

    # Assemble IntertemporalModelData from single-year base + new fields
    return IntertemporalModelData(
        regions=sy_data.regions,
        players=sy_data.players,
        non_strategic=sy_data.non_strategic,
        D=sy_data.D,
        a_dem=sy_data.a_dem,
        b_dem=sy_data.b_dem,
        Dmax=sy_data.Dmax,
        Qcap=sy_data.Qcap,
        c_man=sy_data.c_man,
        c_ship=sy_data.c_ship,
        tau_imp_ub=sy_data.tau_imp_ub,
        tau_exp_ub=sy_data.tau_exp_ub,
        rho_imp=sy_data.rho_imp,
        rho_exp=sy_data.rho_exp,
        w=sy_data.w,
        eps_x=sy_data.eps_x,
        eps_comp=sy_data.eps_comp,
        kappa_Q=sy_data.kappa_Q,
        settings=sy_data.settings,
        times=list(_TIMES),
        Dmax_t=Dmax_t,
        Kcap_2025=Kcap_2025,
        s_ub=s_ub,
        f_hold=f_hold,
        c_inv=c_inv,
    )
