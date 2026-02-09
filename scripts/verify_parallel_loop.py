"""
Verify Parallel Reproducibility

Runs the 'high_all' scenario 3 times with 6 workers to check if the result (λ_eu ~ 198) is reproducible.
"""
import sys
import os

# Set PYTHONPATH to include src so workers can find the package
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = src_path + os.pathsep + os.environ["PYTHONPATH"]
else:
    os.environ["PYTHONPATH"] = src_path

# Also add to sys.path for this process
sys.path.append(src_path)

import argparse
import pandas as pd
from sensitivity_equilibrium import run_scenario, Scenario, INIT_SCENARIOS, load_data_from_excel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    # Load shared data
    data = load_data_from_excel("inputs/input_data.xlsx")
    
    # Define the robust scenario
    scenario = Scenario(
        name="high_all_parallel_test",
        init_q_frac=INIT_SCENARIOS["high_all"],
        player_order=None,
    )

    results = []
    n_runs = 3
    
    print(f"Running scenario 'high_all' {n_runs} times with {args.workers} workers...")
    
    import tempfile
    import shutil
    
    # Create a temp dir for GAMS (must be space-free)
    # We use a centralized temp dir
    gams_tmp_dir = tempfile.mkdtemp(prefix="sgr_parallel_")
    print(f"Using GAMS workdir: {gams_tmp_dir}")
    
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")
        res = run_scenario(
            data=data,
            scenario=scenario,
            iters=args.iters,
            omega=0.7,
            solver="conopt",
            workers=args.workers,
            workdir=gams_tmp_dir,  # Pass space-free path
        )
        
        row = {
            "run": i+1,
            "lam_eu": res["lam"].get("eu", 0.0),
            "obj_eu": res["obj"].get("eu", 0.0),
            "converged": res["converged"],
            "iters": res["n_iters"],
        }
        results.append(row)
        print(f"Result: lam_eu={row['lam_eu']:.4f}, converged={row['converged']}")

    # Summary
    df = pd.DataFrame(results)
    print("\n=== Parallel Consistency Check ===")
    print(df.to_string(index=False))
    
    # Check if consistent
    lams = df["lam_eu"].tolist()
    is_consistent = all(abs(x - lams[0]) < 1e-3 for x in lams)
    print(f"\nConsistent: {is_consistent}")
    if is_consistent:
        print("SUCCESS: Parallel execution is reproducible.")
    else:
        print("WARNING: Parallel execution gave different results.")

if __name__ == "__main__":
    main()
