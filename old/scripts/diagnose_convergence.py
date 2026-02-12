
import pandas as pd
import numpy as np
import os
import sys

def diagnose_convergence(results_path):
    print(f"Loading results from: {results_path}")
    try:
        df = pd.read_excel(results_path, sheet_name="detailed_iters")
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return

    # Pivot to get time series for each variable per region
    # We want to see: iter vs (Q, lam, obj) for each region
    
    regions = df['r'].unique()
    iters = df['iter'].unique()
    print(f"Regions: {regions}")
    print(f"Iterations: {len(iters)}")
    
    # Check for cycles / oscillations
    # 1. Delta Strategy vs Delta Objective
    # For each player, compute L2 norm of strategy change vs change in objective
    
    # Strategy vector per player: Q_offer, tau_imp (all), tau_exp (all)
    # We need to construct a vector representation
    
    report = []
    
    print("\n--- Strategy vs Objective Flatness ---")
    
    for r in regions:
        player_df = df[df['r'] == r].sort_values('iter')
        
        # Extract strategy columns
        # Q_offer
        strat_cols = ['Q_offer']
        # tau_imp keys
        tau_imp_cols = [c for c in player_df.columns if c.startswith('tau_imp_from_')]
        tau_exp_cols = [c for c in player_df.columns if c.startswith('tau_exp_to_')]
        
        all_strat_cols = strat_cols + tau_imp_cols + tau_exp_cols
        
        # Vectors (N_iter x N_vars)
        strat_vecs = player_df[all_strat_cols].values
        obj_vecs = player_df['obj'].values
        
        # Differences
        d_strat = np.diff(strat_vecs, axis=0) # (N-1 x Vars)
        d_obj = np.diff(obj_vecs) # (N-1)
        
        # Norms
        norm_d_strat = np.linalg.norm(d_strat, axis=1)
        abs_d_obj = np.abs(d_obj)
        
        # Ratio: d_obj / d_strat
        # Avoid div by zero
        ratio = np.zeros_like(abs_d_obj)
        mask = norm_d_strat > 1e-6
        ratio[mask] = abs_d_obj[mask] / norm_d_strat[mask]
        
        avg_ratio = np.mean(ratio)
        max_strat_move = np.max(norm_d_strat)
        
        print(f"Player {r}: Avg d_obj/d_strat = {avg_ratio:.4e}, Max d_strat = {max_strat_move:.4e}")
        
        # Check for cycles (autocorrelation of d_strat?)
        # Simple check: distance between x[t] and x[t-2] vs x[t] and x[t-1]
        # if dist(t, t-2) << dist(t, t-1), likely period-2 cycle
        
        if len(strat_vecs) > 10:
            recent_strat = strat_vecs[-10:]
            d1 = np.linalg.norm(recent_strat[1:] - recent_strat[:-1], axis=1).mean()
            d2 = np.linalg.norm(recent_strat[2:] - recent_strat[:-2], axis=1).mean()
            d3 = np.linalg.norm(recent_strat[3:] - recent_strat[:-3], axis=1).mean()
            
            print(f"  Cycle Check (Last 10 iters): Dist(1-lag)={d1:.4f}, Dist(2-lag)={d2:.4f}, Dist(3-lag)={d3:.4f}")
            if d2 < 0.5 * d1:
                print(f"  -> POSSIBLY 2-CYCLE DETECTED")
            if d3 < 0.5 * d1:
                print(f"  -> POSSIBLY 3-CYCLE DETECTED")

    print("\n--- Wedge Accounting Check ---")
    print("(Run manual check on model.py for confirmation)")
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default to most recent results if not provided?
        # Just use the one user mentioned
        path = r"outputs/results_20260210_123707_cdd099.xlsx"
        # Convert relative to absolute if needed
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        path = os.path.join(project_root, path)
        
    diagnose_convergence(path)
