# Diagnostics

This folder contains diagnostic tools for debugging parallel Jacobi and solver issues.

## Files

- `test_parallel_sweep.py` - Run single parallel sweep to test worker failure detection

## Usage

```powershell
# Test parallel sweep with 6 workers
python diagnostics/test_parallel_sweep.py --workers 6

# Keep workdir for inspection
python diagnostics/test_parallel_sweep.py --workers 6 --keep-workdir

# Use specific solver
python diagnostics/test_parallel_sweep.py --workers 6 --solver conopt
```

## Cleanup

This entire folder can be safely deleted without affecting the main project:

```powershell
Remove-Item -Recurse -Force diagnostics/
```
