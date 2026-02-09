---
description: Run the EPEC model solver
---

// turbo-all

1. Run the main solver script:
```bash
python scripts/run_gs.py
```

2. If you need to run with custom parameters, use CLI args:
```bash
python scripts/run_gs.py --iters 50 --omega 0.7 --workers 4
```

3. Run sensitivity analysis:
```bash
python scripts/sensitivity_search.py
```

4. Generate LaTeX documentation only:
```bash
python scripts/generate_latex.py
```
