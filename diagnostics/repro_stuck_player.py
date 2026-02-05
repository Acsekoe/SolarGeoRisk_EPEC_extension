"""
Reproduce Stuck Player Diagnostics

Run repeated best-response solves for a single player to:
1. Reproduce stalling behavior
2. Test different solver options
3. Identify problematic configurations

Usage:
    python diagnostics/repro_stuck_player.py --player DE
    python diagnostics/repro_stuck_player.py --player DE --runs 5 --solver knitro
    python diagnostics/repro_stuck_player.py --player DE --maxit 100 --algorithm 4
"""

import argparse
import os
import sys
import tempfile
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def run_single_br(
    player: str,
    workdir: str,
    excel_path: str,
    solver: str,
    solver_opts: dict,
    run_number: int,
) -> dict:
    """Run a single best-response solve and time it."""
    from solargeorisk_extension.data_prep import load_data_from_excel
    from solargeorisk_extension.model import apply_player_fixings, build_model, extract_state
    
    t0 = time.perf_counter()
    data = load_data_from_excel(excel_path)
    
    # Use fixed theta (like first sweep)
    theta_Q = {r: 0.8 * float(data.Qcap[r]) for r in data.players}
    theta_ti = {(imp, exp): 0.0 for imp in data.regions for exp in data.regions}
    theta_te = {(exp, imp): 0.0 for exp in data.regions for imp in data.regions}
    
    run_workdir = os.path.join(workdir, f"run_{run_number}")
    os.makedirs(run_workdir, exist_ok=True)
    
    ctx = build_model(data, working_directory=run_workdir)
    apply_player_fixings(ctx, data, theta_Q, theta_ti, theta_te, player=player)
    
    solve_start = time.perf_counter()
    try:
        ctx.models[player].solve(solver=solver, solver_options=solver_opts)
        solve_time = time.perf_counter() - solve_start
        
        state = extract_state(ctx)
        q_offer = state.get("Q_offer", {}).get(player, 0)
        lam = state.get("lam", {})
        
        return {
            "run": run_number,
            "success": True,
            "solve_time": solve_time,
            "total_time": time.perf_counter() - t0,
            "Q_offer": q_offer,
            "lam_sum": sum(lam.values()),
            "workdir": run_workdir,
        }
    except Exception as e:
        solve_time = time.perf_counter() - solve_start
        return {
            "run": run_number,
            "success": False,
            "solve_time": solve_time,
            "total_time": time.perf_counter() - t0,
            "error": str(e),
            "workdir": run_workdir,
        }


def main():
    parser = argparse.ArgumentParser(description="Reproduce stuck player behavior")
    parser.add_argument("--player", type=str, required=True,
                        help="Player to test (e.g., DE, FR, ES)")
    parser.add_argument("--excel", type=str, default=None,
                        help="Excel path (defaults to inputs/input_data.xlsx)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of test runs (default: 3)")
    parser.add_argument("--solver", type=str, default="knitro",
                        help="Solver name (default: knitro)")
    
    # Solver option overrides
    parser.add_argument("--maxit", type=int, default=400,
                        help="Maximum iterations (default: 400)")
    parser.add_argument("--feastol", type=float, default=1e-4,
                        help="Feasibility tolerance (default: 1e-4)")
    parser.add_argument("--opttol", type=float, default=1e-4,
                        help="Optimality tolerance (default: 1e-4)")
    parser.add_argument("--algorithm", type=int, default=None,
                        help="KNITRO algorithm (1=barrier, 4=SQP, etc.)")
    parser.add_argument("--maxtime", type=float, default=60.0,
                        help="Max time per solve in seconds (default: 60)")
    
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Keep workdir after run")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    excel_path = args.excel or os.path.join(repo_root, "inputs", "input_data.xlsx")
    
    if not os.path.exists(excel_path):
        print(f"[ERROR] Excel file not found: {excel_path}")
        return 1

    workdir = os.path.join(tempfile.gettempdir(), f"repro_{args.player}_{int(time.time())}")
    os.makedirs(workdir, exist_ok=True)
    
    # Build solver options
    solver_opts = {
        "feastol": args.feastol,
        "opttol": args.opttol,
        "maxit": args.maxit,
        "outlev": 1,  # Verbose for debugging
    }
    if args.algorithm is not None:
        solver_opts["algorithm"] = args.algorithm
    if args.maxtime:
        solver_opts["maxtime"] = int(args.maxtime)

    print(f"[DIAG] Player: {args.player}")
    print(f"[DIAG] Solver: {args.solver}")
    print(f"[DIAG] Options: {solver_opts}")
    print(f"[DIAG] Workdir: {workdir}")
    print(f"[DIAG] Runs: {args.runs}")
    print("-" * 60)

    results = []
    for i in range(1, args.runs + 1):
        print(f"\n[RUN {i}/{args.runs}] Starting...")
        result = run_single_br(
            args.player, workdir, excel_path,
            args.solver, solver_opts, i
        )
        results.append(result)
        
        status = "OK" if result["success"] else "FAIL"
        print(f"[RUN {i}] {status} - solve_time: {result['solve_time']:.2f}s")
        if result["success"]:
            print(f"         Q_offer: {result['Q_offer']:.4f}, lam_sum: {result['lam_sum']:.4f}")
        else:
            print(f"         Error: {result.get('error', 'unknown')}")
    
    print("-" * 60)
    
    # Summary
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    
    print(f"\n[SUMMARY] Total: {len(results)}, Success: {len(successes)}, Failed: {len(failures)}")
    
    if successes:
        times = [r["solve_time"] for r in successes]
        print(f"[SUMMARY] Solve times: min={min(times):.2f}s, max={max(times):.2f}s, avg={sum(times)/len(times):.2f}s")
        
        # Check for stalling (large variance in times)
        if len(times) > 1 and max(times) > 3 * min(times):
            print(f"[WARNING] High variance in solve times - possible stalling detected")
    
    if failures:
        print(f"\n[ACTION] Check .lst/.log files in workdir: {workdir}")
        print(f"[HINT] Try different options:")
        print(f"  --algorithm 1  (interior-point)")
        print(f"  --algorithm 4  (SQP)")
        print(f"  --maxit 200    (reduce iterations)")
        print(f"  --feastol 1e-3 (loosen tolerance)")
    else:
        print(f"\n[OK] Player {args.player} completed all {args.runs} runs successfully")

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
