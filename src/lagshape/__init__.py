"""lagshape: Geometry and visualization of linear time-series processes."""

from .process import ARMAProcess
from .geometry import (
    lag_matrix, LagCloud, LinearPredictor, Convergence,
    predictor_convergence, ar_inf_weights,
    ar2_stationary, ma2_invertible, ar2_complex_roots,
    ar2_triangle, ma2_triangle, ar2_complex_boundary,
    CloudComparison, compare_clouds,
)

__version__ = "0.2.0"

__all__ = [
    # Process
    "ARMAProcess",
    # Geometry
    "lag_matrix", "LagCloud", "LinearPredictor", "Convergence",
    "predictor_convergence", "ar_inf_weights",
    "ar2_stationary", "ma2_invertible", "ar2_complex_roots",
    "ar2_triangle", "ma2_triangle", "ar2_complex_boundary",
    "CloudComparison", "compare_clouds",
    # Plots
    "plot_cloud", "plot_series", "plot_acf", "plot_sweep",
    "plot_regions", "plot_convergence", "plot_ellipses",
]

# Plot functions load lazily so Plotly is imported only when a plot is requested.
# Both `from lagshape import plot_cloud` and `import lagshape.plots` work seamlessly.
_PLOTS = {
    "plot_cloud", "plot_series", "plot_acf", "plot_sweep",
    "plot_regions", "plot_convergence", "plot_ellipses",
}


def __getattr__(name):
    if name in _PLOTS:
        from . import plots
        return getattr(plots, name)
    raise AttributeError(f"module 'lagshape' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_PLOTS))