
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path if needed (though not strictly necessary for simple plotting if dependencies are installed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def plot_convergence(results_path: str, output_dir: str):
    print(f"Reading results from: {results_path}")
    
    try:
        df = pd.read_excel(results_path, sheet_name="detailed_iters")
    except ValueError:
        print("Sheet 'detailed_iters' not found. Please ensure run_gs.py was run with a version that logs detailed iterations.")
        return

    if df.empty:
        print("detailed_iters sheet is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(results_path))[0]
    
    # Ensure iter is numeric
    df["iter"] = pd.to_numeric(df["iter"])
    
    # Get list of regions
    regions = sorted(df["r"].unique())
    
    # --- Plot 1: Q_offer ---
    plt.figure(figsize=(10, 6))
    for r in regions:
        subset = df[df["r"] == r].sort_values("iter")
        plt.plot(subset["iter"], subset["Q_offer"], label=r, marker='o', markersize=3)
    
    plt.title("Convergence of Q_offer")
    plt.xlabel("Iteration")
    plt.ylabel("Q_offer (GW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{base_name}_Q_offer.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close()

    # --- Plot 2: Lambda (Price) ---
    plt.figure(figsize=(10, 6))
    for r in regions:
        subset = df[df["r"] == r].sort_values("iter")
        plt.plot(subset["iter"], subset["lam"], label=r, marker='o', markersize=3)
    
    plt.title("Convergence of Market Price (lambda)")
    plt.xlabel("Iteration")
    plt.ylabel("Lambda ($/kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{base_name}_lam.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close()

    # --- Plot 3: Max Tariffs (Imp/Exp) ---
    # We need to compute max per iteration per region from the flow columns?
    # Or did we log max_tau in detailed_iters?
    # run_gs.py logs 'tau_exp_to_{dest}' and 'tau_imp_from_{src}' columns.
    
    # We'll compute max per row.
    
    # Identify tau columns
    tau_exp_cols = [c for c in df.columns if c.startswith("tau_exp_to_")]
    tau_imp_cols = [c for c in df.columns if c.startswith("tau_imp_from_")]
    
    if tau_imp_cols:
        df["max_tau_imp"] = df[tau_imp_cols].max(axis=1)
        
        plt.figure(figsize=(10, 6))
        for r in regions:
            subset = df[df["r"] == r].sort_values("iter")
            plt.plot(subset["iter"], subset["max_tau_imp"], label=r, marker='x', markersize=3)
        plt.title("Max Import Tariff (tau_imp) per Region")
        plt.xlabel("Iteration")
        plt.ylabel("Tariff ($/kW)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{base_name}_tau_imp_max.png")
        plt.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.close()

    if tau_exp_cols:
        df["max_tau_exp"] = df[tau_exp_cols].max(axis=1)
        
        plt.figure(figsize=(10, 6))
        for r in regions:
            subset = df[df["r"] == r].sort_values("iter")
            plt.plot(subset["iter"], subset["max_tau_exp"], label=r, marker='x', markersize=3)
        plt.title("Max Export Tax (tau_exp) per Region")
        plt.xlabel("Iteration")
        plt.ylabel("Tax ($/kW)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{base_name}_tau_exp_max.png")
        plt.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot convergence from detailed_iters sheet.")
    parser.add_argument("results_file", help="Path to the results Excel file")
    parser.add_argument("--out", "-o", default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: File not found: {args.results_file}")
        sys.exit(1)
        
    plot_convergence(args.results_file, args.out)
