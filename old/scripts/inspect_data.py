
import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from solargeorisk_extension.data_prep import load_data_from_excel

def inspect_params(excel_path):
    print(f"Loading data from {excel_path}...")
    data = load_data_from_excel(excel_path)
    
    print("\n[DEMAND PARAMETERS]")
    print(f"{'Region':<10} {'a_dem':<10} {'b_dem':<10} {'Dmax':<10}")
    for r in data.regions:
        print(f"{r:<10} {data.a_dem[r]:<10.4f} {data.b_dem[r]:<10.4f} {data.Dmax[r]:<10.4f}")

    print("\n[COST PARAMETERS: ch -> row]")
    r, j = "ch", "row"
    print(f"c_man[{r}] = {data.c_man[r]}")
    if (r, j) in data.c_ship:
        print(f"c_ship[{r},{j}] = {data.c_ship[(r, j)]}")
    else:
        print(f"c_ship[{r},{j}] not found!")

    print("\n[UPPER BOUNDS]")
    if (r, j) in data.tau_exp_ub:
        print(f"tau_exp_ub[{r},{j}] = {data.tau_exp_ub[(r, j)]}")
    
    # Also check rho_exp
    print(f"\n[PENALTY PARAMETERS]")
    for r in data.regions:
        print(f"rho_exp[{r}] = {data.rho_exp[r]}")


if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "input_data.xlsx")
    inspect_params(input_path)
