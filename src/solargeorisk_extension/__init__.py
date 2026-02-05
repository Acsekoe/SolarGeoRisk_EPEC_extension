"""SolarGeoRisk EPEC Extension package."""
from .gauss_jacobi import solve_jacobi, solve_jacobi_parallel
from .gauss_seidel import solve_gs

__all__ = ["solve_gs", "solve_jacobi", "solve_jacobi_parallel"]
