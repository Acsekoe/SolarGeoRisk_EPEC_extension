import pandas as pd
import numpy as np
import xlsxwriter

# ==========================================
# 1. SETUP & HARDCODED BASELINES
# ==========================================

# New Region List (Merging apac + roa -> apac_all)
# ch = China, eu = Europe, us = USA, apac_all = All Asia (excl. CH), row = Rest of World
REGIONS = ["ch", "eu", "us", "apac_all", "row"]

# 2030 Baseline Costs (USD/kW) - Used for Learning Rate
COST_2030_BASE = {
    "ch": 105.0, 
    "eu": 125.0, 
    "us": 130.0, 
    "apac_all": 110.0, # Average of old APAC/ROA
    "row": 120.0
}

# Shipping Matrix (Aggregating APAC/ROA)
# We take the average of costs to/from APAC and ROA to approximate the new region
SHIPPING_RAW_2024 = {
    "ch":   {"ch": 0,  "eu": 16, "us": 11, "apac": 17, "roa": 9,  "row": 14},
    "eu":   {"ch": 16, "eu": 0,  "us": 18, "apac": 21, "roa": 14, "row": 8},
    "us":   {"ch": 24, "eu": 9,  "us": 0,  "apac": 16, "roa": 13, "row": 14},
    "apac": {"ch": 12, "eu": 17, "us": 14, "apac": 0,  "roa": 42, "row": 19}, # Old APAC
    "roa":  {"ch": 14, "eu": 14, "us": 13, "apac": 19, "roa": 0,  "row": 18}, # Old ROA
    "row":  {"ch": 14, "eu": 8,  "us": 14, "apac": 24, "roa": 18, "row": 0},
}

def get_aggregated_shipping():
    """Calculates 5x5 matrix merging apac/roa -> apac_all"""
    new_matrix = {}
    
    # Helper to get cost from raw matrix
    def get_raw(src, dst):
        return SHIPPING_RAW_2024.get(src, {}).get(dst, 25.0)

    for exp in REGIONS:
        new_matrix[exp] = {}
        for imp in REGIONS:
            if exp == imp:
                new_matrix[exp][imp] = 0.0
                continue
                
            # --- DERIVE COSTS ---
            # If exporter is apac_all, average the costs from old APAC and ROA
            if exp == "apac_all":
                # We need to look up where we are sending TO
                target_imp = imp
                if imp == "apac_all": target_imp = "apac" # Self-trade (handled by 0 above usually)
                
                c1 = get_raw("apac", target_imp)
                c2 = get_raw("roa", target_imp)
                cost = (c1 + c2) / 2
                
            # If importer is apac_all, average the costs TO old APAC and TO old ROA
            elif imp == "apac_all":
                c1 = get_raw(exp, "apac")
                c2 = get_raw(exp, "roa")
                cost = (c1 + c2) / 2
                
            # Standard Case
            else:
                cost = get_raw(exp, imp)
            
            new_matrix[exp][imp] = round(cost, 2)
    return new_matrix

SHIPPING_MATRIX = get_aggregated_shipping()

# ==========================================
# 2. PROJECTION LOGIC (2035 / 2040)
# ==========================================

def get_projections(year):
    if year == 2035:
        # LEARNING RATE: ~20% reduction from 2030
        data = {
            "c_man": {k: v * 0.80 for k, v in COST_2030_BASE.items()}, 
            
            # --- CAPACITY (GW) ---
            # apac_all = 180 GW (India 100 + ASEAN 70 + Other 10)
            # Source: IEA Supply Chain Report (PLI schemes + existing ASEAN)
            "Qcap": {"ch": 1500, "eu": 80, "us": 80, "apac_all": 180, "row": 500}, 
            
            # --- DEMAND REF (GW/yr) ---
            # apac_all = 150 GW (IEA Renewables 2024: ~135GW/yr avg 2025-2030 -> growing to 150)
            "D_ref": {"ch": 350, "eu": 100, "us": 90, "apac_all": 150, "row": 150},
            
            # PRICES ($/kW) - tracking cost decline
            "P_ref": {"ch": 180, "eu": 210, "us": 220, "apac_all": 190, "row": 200},
            "Dmax_mult": 3.0
        }
    elif year == 2040:
        # LEARNING RATE: Further 15% reduction
        data = {
            "c_man": {k: v * 0.80 * 0.85 for k, v in COST_2030_BASE.items()},
            
            # --- CAPACITY (GW) ---
            # apac_all grows to 220 GW (India saturation + ASEAN expansion)
            "Qcap": {"ch": 1800, "eu": 120, "us": 120, "apac_all": 220, "row": 700},
            
            # --- DEMAND REF (GW/yr) ---
            "D_ref": {"ch": 400, "eu": 120, "us": 110, "apac_all": 200, "row": 200},
            
            "P_ref": {"ch": 150, "eu": 180, "us": 190, "apac_all": 160, "row": 170},
            "Dmax_mult": 3.0
        }
    else:
        raise ValueError("Year not supported")
    
    # Calculate Linear Demand (P = a - bQ)
    epsilon = -0.5
    data["a_dem"] = {}
    data["b_dem"] = {}
    
    for r in REGIONS:
        P = data["P_ref"][r]
        Q = data["D_ref"][r]
        b_val = -1 * P / (epsilon * Q)
        a_val = P + b_val * Q
        data["a_dem"][r] = round(a_val, 4)
        data["b_dem"][r] = round(b_val, 4)
        
    return data

# ==========================================
# 3. EXCEL GENERATION
# ==========================================

def create_scenario_file(year, filename):
    proj = get_projections(year)
    
    # 1. Regions DataFrame
    # Note: We rename apac_all to "Asia Pacific (excl. China)" for readability in plots
    df_regions = pd.DataFrame([
        {"r": "ch", "name": "China"},
        {"r": "eu", "name": "Europe"},
        {"r": "us", "name": "North America"},
        {"r": "apac_all", "name": "Asia Pacific (excl. China)"},
        {"r": "row", "name": "Rest of World"}
    ])
    
    # 2. Params DataFrame
    params_data = []
    for r in REGIONS:
        row = {
            "r": r,
            "Qcap_exist (GW)": proj["Qcap"][r],
            "Dmax (GW)": proj["D_ref"][r] * proj["Dmax_mult"],
            "c_man (USD/kW)": proj["c_man"][r],
            "rho_imp": 0.05, "rho_exp": 0.05, 
            "tau_imp_max": 100, "tau_exp_max": 100,
            "w": 1, "k_shed_mult": 1,
            "a_dem": proj["a_dem"][r],
            "b_dem": proj["b_dem"][r],
            "kappa_Q": 0.0001
        }
        params_data.append(row)
    df_params = pd.DataFrame(params_data)
    
    # 3. Shipping DataFrame
    ship_data = []
    for exp in REGIONS:
        row = {"exporter": exp}
        for imp in REGIONS:
            row[imp] = SHIPPING_MATRIX[exp][imp]
        ship_data.append(row)
    df_ship = pd.DataFrame(ship_data)
    
    # 4. Settings
    df_settings = pd.DataFrame([
        {"setting": "scenario_welfare", "value": "TRUE"},
        {"setting": "scenario_capitalistic_china", "value": "FALSE"},
        {"setting": "players", "value": "ch,eu,us,apac_all"}, # APAC is now a player
        {"setting": "rho_prox", "value": 0.05},
        {"setting": "use_quad", "value": "TRUE"}
    ])

    # --- Write to Excel ---
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book
    
    # Formats
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
    
    # Write Sheets
    def write_sheet(df, name):
        df.to_excel(writer, sheet_name=name, index=False)
        ws = writer.sheets[name]
        for idx, col in enumerate(df.columns):
            ws.write(0, idx, col, header_fmt)

    write_sheet(df_regions, 'regions')
    write_sheet(df_params, 'params_region')
    write_sheet(df_ship, 'c_ship')
    write_sheet(df_settings, 'settings')
    
    # Add Comments for Defensibility
    ws_params = writer.sheets['params_region']
    
    if year == 2035:
        # Comment on APAC Capacity
        ws_params.write_comment(4, 1, "Source: IEA Supply Chain Report / India PLI.\nAggregates India (100GW) + ASEAN (70GW) + Others.")
        # Comment on APAC Demand
        ws_params.write_comment(4, 2, "Source: IEA Renewables 2024.\nBased on 135GW/yr avg growth 2025-2030, extrapolated to 150GW.")
    
    writer.close()
    print(f"Generated {filename} (Aggregated Region: apac_all)")

if __name__ == "__main__":
    create_scenario_file(2035, "input_data_2035_aggregated.xlsx")
    create_scenario_file(2040, "input_data_2040_aggregated.xlsx")