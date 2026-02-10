"""
Sequential Jacobi Diagnostics

Run a single sequential Jacobi sweep to verify optimizations.
"""

import argparse
import os
import sys
import tempfile
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solargeorisk_extension.data_prep import load_data_from_excel
from solargeorisk_extension.gauss_jacobi import solve_jacobi

def main():
    parser = argparse.ArgumentParser(description="Test sequential Jacobi sweep")
    parser.add_argument("--excel", type=str, default=None,
                        help="Excel path (defaults to inputs/input_data.xlsx)")
    parser.add_argument("--solver", type=str, default="knitro",
                        help="Solver name (default: knitro)")
    args = parser.parse_args()

    # Paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    excel_path = args.excel or os.path.join(repo_root, "inputs", "input_data.xlsx")
    
    if not os.path.exists(excel_path):
        print(f"[ERROR] Excel file not found: {excel_path}")
        return 1

    # Create workdir in TEMP
    workdir = os.path.join(
        tempfile.gettempdir(),
        f"seq_jacobi_diag_{int(time.time())}"
    )
    os.makedirs(workdir, exist_ok=True)
    print(f"[DIAG] Workdir: {workdir}")

    # Load data
    print(f"[DIAG] Loading data from: {excel_path}")
    data = load_data_from_excel(excel_path)
    
    # Enable query in settings if needed (for penalty check)
    data.settings = data.settings or {}
    data.settings["use_quad"] = True # Test with quadratic penalties as requested by user

    print(f"\n[DIAG] Running 1 sequential sweep...")
    t0 = time.perf_counter()
    
    # Callback to verify partial state extraction didn't break logging
    def iter_log(it, state, r_strat, stable_count):
        print(f"[CB] it={it} r_strat={r_strat:.4g}")
        # Verify state has strategic keys
        print(f"     Q_offer keys: {list(state.get('Q_offer', {}).keys())[:3]}")

    try:
        final_state, iter_rows = solve_jacobi(
            data,
            iters=2, # Run 2 sweeps to test loop transition
            omega=0.5,
            tol_rel=1e-2,
            stable_iters=1,
            solver=args.solver,
            working_directory=workdir,
            iter_callback=iter_log,
            use_staged_tolerances=True
        )
        elapsed = time.perf_counter() - t0
        print(f"\n[SUCCESS] Completed in {elapsed:.2f}s")
        print(f"Final Q_offer (sample): {list(final_state['Q_offer'].items())[:3]}")
        
        # Test result writing
        from solargeorisk_extension.results_writer import write_results_excel
        out_path = os.path.join(workdir, "test_results.xlsx")
        print(f"\n[DIAG] Writing results to {out_path}...")
        write_results_excel(
            data=data,
            state=final_state,
            iter_rows=iter_rows,
            output_path=out_path,
            meta={"test": "true"}
        )
        print("[SUCCESS] Excel written successfully.")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
