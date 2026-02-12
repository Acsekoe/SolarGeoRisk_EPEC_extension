"""
Parallel Jacobi Diagnostics

Run a single parallel Jacobi sweep to identify worker failures.
Captures and prints which player fails and relevant workdir paths.

Usage:
    python diagnostics/test_parallel_sweep.py --workers 6
    python diagnostics/test_parallel_sweep.py --workers 6 --keep-workdir
"""

import argparse
import os
import sys
import tempfile
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solargeorisk_extension.data_prep import load_data_from_excel


def main():
    parser = argparse.ArgumentParser(description="Test parallel Jacobi sweep")
    parser.add_argument("--excel", type=str, default=None,
                        help="Excel path (defaults to inputs/input_data.xlsx)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel workers (default: 6)")
    parser.add_argument("--solver", type=str, default="knitro",
                        help="Solver name (default: knitro)")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Keep workdir after run")
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
        f"parallel_jacobi_diag_{int(time.time())}"
    )
    os.makedirs(workdir, exist_ok=True)
    print(f"[DIAG] Workdir: {workdir}")

    # Load data
    print(f"[DIAG] Loading data from: {excel_path}")
    data = load_data_from_excel(excel_path)
    print(f"[DIAG] Players: {list(data.players)}")
    print(f"[DIAG] Regions: {list(data.regions)}")

    # Print per-player workdirs
    print(f"\n[DIAG] Per-player workdirs:")
    for p in data.players:
        pdir = os.path.join(workdir, f"worker_{p}")
        print(f"  {p}: {pdir}")
    print(f"  final_state: {os.path.join(workdir, 'final_state')}")

    # Import parallel solver
    from solargeorisk_extension.gauss_jacobi import solve_jacobi_parallel

    solver_opts = {
        "feastol": 1e-4,
        "opttol": 1e-4,
        "outlev": 1,  # Verbose for diagnostics
        "maxit": 800,
        "hessopt": 1,
    }

    print(f"\n[DIAG] Running 1 sweep with {args.workers} workers...")
    print(f"[DIAG] Solver: {args.solver}")
    print(f"[DIAG] Solver options: {solver_opts}")
    print("-" * 60)

    def iter_callback(it, state, r_strat, stable_count):
        q_offer = state.get("Q_offer", {})
        lam = state.get("lam", {})
        print(f"\n[ITER {it}] r_strat={r_strat:.6g} stable_count={stable_count}")
        for r in sorted(q_offer.keys()):
            q = q_offer.get(r, 0)
            l = lam.get(r, 0)
            print(f"  {r}: Q_offer={q:.4f} lam={l:.4f}")

    try:
        t0 = time.perf_counter()
        state, iter_rows = solve_jacobi_parallel(
            data,
            excel_path=excel_path,
            iters=1,
            omega=0.8,
            tol_rel=1e-2,
            stable_iters=1,
            solver=args.solver,
            solver_options=solver_opts,
            working_directory=workdir,
            iter_callback=iter_callback,
            workers=args.workers,
        )
        elapsed = time.perf_counter() - t0
        print("-" * 60)
        print(f"\n[SUCCESS] Sweep completed in {elapsed:.2f}s")
        
        # Print final lam to verify it's nonzero
        lam = state.get("lam", {})
        print(f"\n[FINAL] lam values:")
        for r in sorted(lam.keys()):
            print(f"  {r}: {lam.get(r, 0):.4f}")
        
        if all(v == 0 for v in lam.values()):
            print("\n[WARNING] All lam values are zero!")
        else:
            print("\n[OK] lam values are nonzero")
            
    except Exception as e:
        print("-" * 60)
        print(f"\n[ERROR] Parallel sweep failed:")
        print(f"  {e}")
        print(f"\n[DIAG] Check .lst and .log files in workdir:")
        print(f"  {workdir}")
        
        # List workdir contents
        if os.path.exists(workdir):
            print(f"\n[DIAG] Workdir contents:")
            for root, dirs, files in os.walk(workdir):
                level = root.replace(workdir, "").count(os.sep)
                indent = "  " * level
                print(f"{indent}{os.path.basename(root)}/")
                for f in files[:10]:  # Limit to 10 files
                    print(f"{indent}  {f}")
                if len(files) > 10:
                    print(f"{indent}  ... and {len(files) - 10} more files")
        return 1
    finally:
        if not args.keep_workdir:
            import shutil
            try:
                shutil.rmtree(workdir, ignore_errors=True)
                print(f"\n[CLEANUP] Deleted workdir")
            except Exception as e:
                print(f"\n[WARN] Could not delete workdir: {e}")
        else:
            print(f"\n[KEEP] Workdir retained: {workdir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
