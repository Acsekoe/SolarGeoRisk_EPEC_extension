
import pandas as pd
import os
import sys

def check_results(file_path):
    print(f"Checking {file_path}...")
    try:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.isnull().values.any():
                print(f"[WARN] Sheet '{sheet_name}' contains NaNs!")
                print(df[df.isnull().any(axis=1)])
            
            # Check for numeric columns with Inf
            num_df = df.select_dtypes(include=['float64', 'int64'])
            if hasattr(num_df, 'apply'): # Check for Inf
                 import numpy as np
                 if np.isinf(num_df).values.any():
                     print(f"[WARN] Sheet '{sheet_name}' contains Infs!")
                     print(df[np.isinf(num_df).any(axis=1)])
            
            print(f"Sheet '{sheet_name}' check complete.")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # scan for latest
        outputs = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        files = [os.path.join(outputs, f) for f in os.listdir(outputs) if f.startswith("results_") and f.endswith(".xlsx")]
        if not files:
            print("No result files found.")
            sys.exit(1)
        latest = max(files, key=os.path.getmtime)
        check_results(latest)
    else:
        check_results(sys.argv[1])
