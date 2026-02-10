
import os
import pandas as pd
import numpy as np

def aggregate_inputs():
    # Setup paths
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    input_path = os.path.join(project_root, "inputs", "input_data.xlsx")
    output_path = os.path.join(project_root, "inputs", "input_data_aggregated.xlsx")
    
    print(f"Reading from: {input_path}")
    dfs = pd.read_excel(input_path, sheet_name=None)
    
    # Mapping
    mapping = {
        "ch": "ch",
        "eu": "eu",
        "us": "eu",
        "apac": "apac",
        "roa": "apac",
        "row": "apac"
    }
    
    # Column Names config
    COL_QCAP = "Qcap_exist (GW)"
    COL_DMAX = "Dmax (GW)"
    COL_CMAN = "c_man (USD/kW)"
    
    # --- 1. Params Aggregation ---
    print("Aggregating params...")
    df_params = dfs["params_region"].copy()
    df_params["new_r"] = df_params["r"].map(mapping)
    
    agg_rows = []
    
    # Identify non-numeric columns to drop or handle
    # We only aggregate numeric ones. 'r' and 'new_r' handled manually.
    
    for new_r, group in df_params.groupby("new_r"):
        # Sums
        Qcap_agg = group[COL_QCAP].sum()
        Dmax_agg = group[COL_DMAX].sum()
        
        # Demand Aggregation A-B
        # Q = (a/b) - (1/b)P
        inv_b_sum = (1.0 / group["b_dem"]).sum()
        ab_sum = (group["a_dem"] / group["b_dem"]).sum()
        
        b_agg = 1.0 / inv_b_sum
        a_agg = ab_sum * b_agg
        
        # Weights (by Qcap)
        total_q = group[COL_QCAP].sum()
        if total_q > 0:
            weights = group[COL_QCAP] / total_q
        else:
            weights = pd.Series([1.0/len(group)]*len(group), index=group.index)
            
        # Weighted Means
        c_man_agg = (group[COL_CMAN] * weights).sum()
        
        # Optional columns
        rho_imp_agg = (group["rho_imp"] * weights).sum() if "rho_imp" in group else 0
        rho_exp_agg = (group["rho_exp"] * weights).sum() if "rho_exp" in group else 0
        kappa_Q_agg = (group["kappa_Q"] * weights).sum() if "kappa_Q" in group else 0
        
        # Simple Means for logic/multipliers
        w_agg = group["w"].mean() if "w" in group else 1.0
        k_shed = group["k_shed_mult"].mean() if "k_shed_mult" in group else 1.0
        
        # Limits (Weighted or Max? Weighted seems consistent for 'average' region behavior)
        tau_imp_max_agg = (group["tau_imp_max"] * weights).sum() if "tau_imp_max" in group else 100
        tau_exp_max_agg = (group["tau_exp_max"] * weights).sum() if "tau_exp_max" in group else 100

        row = {
            "r": new_r,
            COL_QCAP: Qcap_agg,
            COL_DMAX: Dmax_agg, # run_gs.py handles D vs Dmax check
            "D (GW)": Dmax_agg, # Provide D just in case
            COL_CMAN: c_man_agg,
            "a_dem": a_agg,
            "b_dem": b_agg,
            "rho_imp": rho_imp_agg,
            "rho_exp": rho_exp_agg,
            "kappa_Q": kappa_Q_agg,
            "w": w_agg,
            "k_shed_mult": k_shed,
            "tau_imp_max": tau_imp_max_agg,
            "tau_exp_max": tau_exp_max_agg
        }
        agg_rows.append(row)
        
    df_params_agg = pd.DataFrame(agg_rows)
    
    # --- 2. Shipping Aggregation ---
    print("Aggregating shipping...")
    df_ship = dfs["c_ship"].copy()
    
    # Identify keys (columns that are regions)
    cols = [c for c in df_ship.columns if c != "exporter"]
    
    # Melt
    df_melt = df_ship.melt(id_vars=["exporter"], value_vars=cols, var_name="importer", value_name="cost")
    
    # Map
    df_melt["new_exp"] = df_melt["exporter"].map(mapping)
    df_melt["new_imp"] = df_melt["importer"].map(mapping)
    
    # Aggregate (Mean cost between groups)
    df_ship_agg_long = df_melt.groupby(["new_exp", "new_imp"])["cost"].mean().reset_index()
    
    # Pivot back
    df_ship_agg = df_ship_agg_long.pivot(index="new_exp", columns="new_imp", values="cost").reset_index()
    df_ship_agg = df_ship_agg.rename(columns={"new_exp": "exporter"})
    
    # Fill NA with 0 for self-loops if missing, or high cost? 
    # Usually pivot results in NaN if no link. But here we mapped all, so should be fine.
    # Ensure all regions derived are present in columns
    present_regions = sorted(list(set(mapping.values())))
    for r in present_regions:
        if r not in df_ship_agg.columns:
            df_ship_agg[r] = 0.0 # Should not happen if data is complete
            
    # --- 3. Regions List ---
    # ch, eu, apac
    regions_df = pd.DataFrame([
        {"r": "ch", "name": "China", "strategic": 1},
        {"r": "eu", "name": "Europe_West", "strategic": 1},
        {"r": "apac", "name": "APAC_Rest", "strategic": 1}
    ])
    
    # --- 4. Settings ---
    # Copy settings if exist
    df_settings = dfs.get("settings", pd.DataFrame())
    
    with pd.ExcelWriter(output_path) as writer:
        if "README" in dfs: dfs["README"].to_excel(writer, sheet_name="README", index=False)
        regions_df.to_excel(writer, sheet_name="regions", index=False)
        df_params_agg.to_excel(writer, sheet_name="params_region", index=False)
        df_ship_agg.to_excel(writer, sheet_name="c_ship", index=False)
        if not df_settings.empty: df_settings.to_excel(writer, sheet_name="settings", index=False)

    print(f"Created: {output_path}")

if __name__ == "__main__":
    aggregate_inputs()
