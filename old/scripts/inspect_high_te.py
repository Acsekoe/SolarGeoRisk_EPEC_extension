
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from solargeorisk_extension.data_prep import load_data_from_excel

def inspect(excel_path, results_path):
    print(f"Loading data from {excel_path}...")
    data = load_data_from_excel(excel_path)
    
    print(f"Loading results from {results_path}...")
    xls = pd.ExcelFile(results_path)
    if 'detailed_iters' in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name='detailed_iters')
        last_iter = df['iter'].max()
        print(f"Last iteration: {last_iter}")
        
        # Filter for last iter
        df_last = df[df['iter'] == last_iter]
        
        # Check tau_exp_to_* columns
        cols = [c for c in df.columns if c.startswith('tau_exp_to_')]
        for _, row in df_last.iterrows():
            r = row['r']
            for c in cols:
                val = row[c]
                dest = c.replace('tau_exp_to_', '')
                if val > 1.0: # Arbitrary threshold
                    print(f"[HIGH] {r} -> {dest}: tau_exp = {val:.4f}")
                    # Check flow
                    x_col_name = f"x_exp_to_{dest}"
                    if x_col_name in row:
                        print(f"       Flow x = {row[x_col_name]:.4f}")
                    # Check upper bound
                    if (r, dest) in data.tau_exp_ub:
                        ub = data.tau_exp_ub[(r, dest)]
                        print(f"       Upper Bound: {ub:.4f}")
    else:
        print("No detailed_iter sheet found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # scan for latest
        outputs = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        files = [os.path.join(outputs, f) for f in os.listdir(outputs) if f.startswith("results_") and f.endswith(".xlsx")]
        if not files:
            print("No result files found.")
            sys.exit(1)
        latest = max(files, key=os.path.getmtime)
        results_path = latest
    else:
        results_path = sys.argv[1]
    
    input_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "input_data.xlsx")
    inspect(input_path, results_path)
