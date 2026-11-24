from .process import ARMAProcess
from .geometry import (
    lag_matrix, LagCloud, LinearPredictor, Convergence,
    predictor_convergence, ar_inf_weights,
    ar2_stationary, ma2_invertible, ar2_complex_roots,
    ar2_triangle, ma2_triangle, ar2_complex_boundary,
    CloudComparison, compare_clouds,
)

__all__ = [
    "ARMAProcess", "lag_matrix", "LagCloud", "LinearPredictor", "Convergence",
    "predictor_convergence", "ar_inf_weights",
    "ar2_stationary", "ma2_invertible", "ar2_complex_roots",
    "ar2_triangle", "ma2_triangle", "ar2_complex_boundary",
    "CloudComparison", "compare_clouds",
]
__version__ = "1.1.0"

# Plot functions load lazily, so plotly is only needed if you actually plot.
# `from lagshape import plot_cloud` works; so does `import lagshape.plots`.
_PLOTS = {"plot_cloud", "plot_series", "plot_acf", "plot_sweep",
          "plot_regions", "plot_convergence", "plot_ellipses"}


def __getattr__(name):
    if name in _PLOTS:
        from . import plots
        return getattr(plots, name)
    raise AttributeError(f"module 'lagshape' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_PLOTS))
