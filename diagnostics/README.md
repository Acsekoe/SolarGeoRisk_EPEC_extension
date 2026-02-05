# Diagnostics

This folder contains diagnostic tools for debugging parallel Jacobi and KNITRO solver issues.
**Safe to delete** - removing this folder will not affect the main project.

## Files

- `test_parallel_sweep.py` - Run single parallel sweep to test worker failure detection
- `check_knitro_concurrency.py` - Test concurrent KNITRO solve capability (license issues)
- `repro_stuck_player.py` - Reproduce stalling for a specific player with different options

## Usage

### Test Parallel Sweep
```powershell
python diagnostics/test_parallel_sweep.py --workers 6
python diagnostics/test_parallel_sweep.py --workers 6 --solver knitro --keep-workdir
```

### Check KNITRO Concurrency
```powershell
# Test if N concurrent KNITRO solves work
python diagnostics/check_knitro_concurrency.py --workers 4
python diagnostics/check_knitro_concurrency.py --workers 6 --timeout 90
```

### Reproduce Stuck Player
```powershell
# Test specific player with default options
python diagnostics/repro_stuck_player.py --player DE --runs 3

# Try different KNITRO algorithms
python diagnostics/repro_stuck_player.py --player DE --algorithm 4  # SQP
python diagnostics/repro_stuck_player.py --player DE --algorithm 1  # Interior-point

# Loosen tolerances
python diagnostics/repro_stuck_player.py --player DE --feastol 1e-3 --opttol 1e-3

# Reduce iterations
python diagnostics/repro_stuck_player.py --player DE --maxit 200
```

## Cleanup

```powershell
Remove-Item -Recurse -Force diagnostics/
```
