"""
KNITRO Concurrency Diagnostics

Test whether concurrent KNITRO solves work correctly.
Helps identify license/concurrency issues vs model-specific stalling.

Usage:
    python diagnostics/check_knitro_concurrency.py --workers 4
    python diagnostics/check_knitro_concurrency.py --workers 6 --keep-workdir
"""

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _single_solve(player: str, workdir: str, excel_path: str, solver: str, solver_opts: dict) -> dict:
    """Run a single player best-response solve."""
    from solargeorisk_extension.data_prep import load_data_from_excel
    from solargeorisk_extension.model import apply_player_fixings, build_model, extract_state
    
    t0 = time.perf_counter()
    data = load_data_from_excel(excel_path)
    
    # Initial theta (should be fast since problem is well-scaled)
    theta_Q = {r: 0.8 * float(data.Qcap[r]) for r in data.players}
    theta_ti = {(imp, exp): 0.0 for imp in data.regions for exp in data.regions}
    theta_te = {(exp, imp): 0.0 for exp in data.regions for imp in data.regions}
    
    os.makedirs(workdir, exist_ok=True)
    ctx = build_model(data, working_directory=workdir)
    apply_player_fixings(ctx, data, theta_Q, theta_ti, theta_te, player=player)
    
    try:
        ctx.models[player].solve(solver=solver, solver_options=solver_opts)
        state = extract_state(ctx)
        lam = state.get("lam", {})
        elapsed = time.perf_counter() - t0
        return {
            "player": player,
            "success": True,
            "time": elapsed,
            "lam_sum": sum(lam.values()),
            "workdir": workdir,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "player": player,
            "success": False,
            "time": elapsed,
            "error": str(e),
            "workdir": workdir,
        }


def main():
    parser = argparse.ArgumentParser(description="Test KNITRO concurrency")
    parser.add_argument("--excel", type=str, default=None,
                        help="Excel path (defaults to inputs/input_data.xlsx)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers (default: 4)")
    parser.add_argument("--solver", type=str, default="knitro",
                        help="Solver name (default: knitro)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Per-worker timeout in seconds (default: 60)")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Keep workdir after run")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    excel_path = args.excel or os.path.join(repo_root, "inputs", "input_data.xlsx")
    
    if not os.path.exists(excel_path):
        print(f"[ERROR] Excel file not found: {excel_path}")
        return 1

    workdir = os.path.join(tempfile.gettempdir(), f"knitro_concurrency_{int(time.time())}")
    os.makedirs(workdir, exist_ok=True)
    print(f"[DIAG] Workdir: {workdir}")
    print(f"[DIAG] Testing {args.workers} concurrent {args.solver} solves")
    print("-" * 60)

    solver_opts = {"feastol": 1e-4, "opttol": 1e-4, "maxit": 200, "outlev": 0}
    
    # Load data to get player list
    from solargeorisk_extension.data_prep import load_data_from_excel
    data = load_data_from_excel(excel_path)
    players = list(data.players)[:args.workers]  # Use first N players
    
    print(f"[DIAG] Using players: {players}")

    results = []
    t0 = time.perf_counter()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _single_solve,
                player,
                os.path.join(workdir, f"worker_{player}"),
                excel_path,
                args.solver,
                solver_opts,
            ): player
            for player in players
        }
        
        for future in as_completed(futures):
            player = futures[future]
            try:
                result = future.result(timeout=args.timeout)
                results.append(result)
            except FuturesTimeoutError:
                results.append({
                    "player": player,
                    "success": False,
                    "time": args.timeout,
                    "error": f"Timeout after {args.timeout}s",
                    "workdir": os.path.join(workdir, f"worker_{player}"),
                })
            except Exception as e:
                results.append({
                    "player": player,
                    "success": False,
                    "time": 0,
                    "error": str(e),
                    "workdir": os.path.join(workdir, f"worker_{player}"),
                })

    total_time = time.perf_counter() - t0
    print("-" * 60)
    
    # Summary
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    
    print(f"\n[SUMMARY] Total: {len(results)}, Success: {len(successes)}, Failed: {len(failures)}")
    print(f"[SUMMARY] Wall time: {total_time:.2f}s")
    
    if successes:
        avg_time = sum(r["time"] for r in successes) / len(successes)
        print(f"[SUMMARY] Avg solve time: {avg_time:.2f}s")
    
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"  {r['player']}: {status} ({r['time']:.2f}s)")
        if not r["success"]:
            print(f"    Error: {r.get('error', 'unknown')}")
            print(f"    Workdir: {r['workdir']}")
    
    if failures:
        print(f"\n[ACTION] Check .lst/.log files in failed workdirs above")
        if "license" in str(failures).lower():
            print("[HINT] License-related failures suggest concurrent license limit reached")
    else:
        print(f"\n[OK] All {args.workers} concurrent {args.solver} solves succeeded")

    if not args.keep_workdir:
        import shutil
        try:
            shutil.rmtree(workdir, ignore_errors=True)
            print(f"\n[CLEANUP] Deleted workdir")
        except Exception as e:
            print(f"\n[WARN] Could not delete workdir: {e}")
    else:
        print(f"\n[KEEP] Workdir retained: {workdir}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
