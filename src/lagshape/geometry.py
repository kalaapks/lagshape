"""Geometry of linear time-series processes.

Take the vector of a value and its past, (Y_t, Y_{t-1}, ..., Y_{t-k}).  Plotted
over many t this is a cloud of dots; for a linear process the cloud is an
ellipse (k=1), an ellipsoid (k=2), or its higher-dimensional version.  This
module turns that cloud into objects you can compute with:

* :class:`LagCloud`        - the cloud's covariance, axes, volume, boundary
* :class:`LinearPredictor` - the best straight-line guess of Y_t from the past
* :func:`predictor_convergence`, :func:`ar_infinity_weights`
                           - how the guess behaves as more lags are allowed
* region functions         - stationarity / invertibility maps in (phi1, phi2)
* :func:`compare_clouds`   - theory-versus-data comparison

Coordinate order everywhere is (Y_t, Y_{t-1}, ..., Y_{t-k}).  Drawing code
may reorder them (the HTML pages plot Y_{t-1} first).

Existing tools do the heavy lifting: ``scipy.linalg`` (Toeplitz, eigh, solve),
statsmodels (autocovariances, Levinson-Durbin, AR(inf) weights).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import eigh, solve, toeplitz
from scipy.special import gamma as _gamma
from statsmodels.tsa.arima_process import ArmaProcess as _SMArma
from statsmodels.tsa.stattools import levinson_durbin

from .process import ARMAProcess  #the dot in the beg means from the same package where this file lies (that is lagshape for us)

__all__ = [
    "lag_matrix", "LagCloud", "LinearPredictor", "Convergence",
    "predictor_convergence", "ar_inf_weights",
    "ar2_stationary", "ma2_invertible", "ar2_complex_roots",
    "ar2_triangle", "ma2_triangle", "ar2_complex_boundary",
    "CloudComparison", "compare_clouds",
]


# =============================================================== helpers
def lag_matrix(y: Sequence[float], lags: int) -> np.ndarray:
    """Stack a series into rows (Y_t, Y_{t-1}, ..., Y_{t-lags}) for t = lags..n-1.

    >>> lag_matrix([1, 2, 3, 4], 1).tolist()
    [[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]]
    """
    y = np.asarray(y, dtype=float).ravel()
    if lags < 0:
        raise ValueError("lags must be >= 0")
    if not np.all(np.isfinite(y)):
        raise ValueError("data contains NaN or infinite values")
    if len(y) < lags + 2:
        raise ValueError(f"need at least {lags + 2} observations for lags={lags}")
    n = len(y) - lags
    return np.column_stack([y[lags - j: lags - j + n] for j in range(lags + 1)])


def _fix_signs(vecs: np.ndarray) -> np.ndarray:
    """Eigenvector signs are arbitrary; make the largest component positive."""
    idx = np.argmax(np.abs(vecs), axis=0)
    signs = np.sign(vecs[idx, np.arange(vecs.shape[1])])
    for i in range(len(signs)):
        if signs[i] == 0:
            signs[i] = 1.0
    return vecs * signs


#predictor
@dataclass(frozen=True)
class LinearPredictor:
    """Best linear guess of Y_t from its last ``lags`` values.

    Y_hat_t = intercept + weights[0]*Y_{t-1} + weights[1]*Y_{t-2} + ...
    """
    weights: np.ndarray
    intercept: float
    error_variance: float
    total_variance: float

    @property
    def lags(self) -> int:
        return len(self.weights)

    @property
    def error_std(self) -> float:
        return math.sqrt(self.error_variance)

    @property
    def r_squared(self) -> float:
        """Share of Var(Y_t) the guess explains."""
        return 1.0 - self.error_variance / self.total_variance

    def predict(self, y: Sequence[float]) -> np.ndarray:
        """Guesses for y[lags:], each from the ``lags`` values before it."""
        m = lag_matrix(y, self.lags)
        return self.intercept + m[:, 1:] @ self.weights


#the cloud
class LagCloud:
    """The cloud of (Y_t, Y_{t-1}, ..., Y_{t-k}), described by mean and covariance.

    Build it from theory (:meth:`from_process`) or from data (:meth:`from_data`);
    every method works the same on both.

    >>> c = LagCloud.from_process(ARMAProcess(ar=[0.7]), lags=1)
    >>> c.cov.round(3).tolist()
    [[1.961, 1.373], [1.373, 1.961]]
    >>> c.axis_lengths.round(2).tolist()
    [1.83, 0.77]
    """

    def __init__(self, cov: np.ndarray, mean: np.ndarray | None = None,
                 source: str = "custom", n_obs: int | None = None,
                 points: np.ndarray | None = None):
        cov = np.asarray(cov, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
            raise ValueError("cov must be a square matrix of size >= 2")
        if not np.allclose(cov, cov.T, atol=1e-10 * max(1.0, np.abs(cov).max())):
            raise ValueError("cov must be symmetric")
        self.cov = (cov + cov.T) / 2
        self.mean = np.zeros(cov.shape[0]) if mean is None else np.asarray(mean, dtype=float)
        self.source = source
        self.n_obs = n_obs
        self.points = points
        vals, vecs = eigh(self.cov)
        order = np.argsort(vals)[::-1]
        self._eigvals = np.clip(vals[order], 0.0, None)   # tiny negatives -> 0
        self._eigvecs = _fix_signs(vecs[:, order])

    # ---- constructors
    @classmethod
    def from_process(cls, process: ARMAProcess, lags: int) -> "LagCloud":
        """Exact theoretical cloud of a stationary process."""
        if lags < 1:
            raise ValueError("lags must be >= 1")
        cov = toeplitz(process.autocov(lags))
        return cls(cov, source="process")

    @classmethod
    def from_data(cls, y: Sequence[float], lags: int) -> "LagCloud":
        """Measured cloud of a data series (columns are de-meaned, ddof=1)."""
        if lags < 1:
            raise ValueError("lags must be >= 1")
        pts = lag_matrix(y, lags)
        if pts.shape[0] < pts.shape[1] + 2:
            raise ValueError("too few observations for this many lags")
        return cls(np.cov(pts, rowvar=False), mean=pts.mean(axis=0),
                   source="data", n_obs=pts.shape[0], points=pts)

    #parameters for the cloud
    #these are finally new stuff, so far we did wrappers and cleanups
    @property #lags
    def lags(self) -> int:
        return self.cov.shape[0] - 1  

    @property #number of dim in phase space = lag + 1
    def dim(self) -> int:
        return self.cov.shape[0]

    @property #Correlation matrix
    def corr(self) -> np.ndarray:
        s = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(s, s)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Variances along the principal axes, largest first."""
        return self._eigvals.copy()

    @property
    def axes(self) -> np.ndarray:
        """Principal-axis directions as columns"""
        return self._eigvecs.copy()

    @property
    def axis_lengths(self) -> np.ndarray:
        """Semi-axis lengths at 1 standard deviation (sqrt of eigenvalues)."""
        return np.sqrt(self._eigvals)

    @property
    def elongation(self) -> float:
        """Longest axis / shortest axis (1 = round, large = needle-like)."""
        lo = self.axis_lengths[-1]
        return math.inf if lo <= 0 else float(self.axis_lengths[0] / lo)

    @property
    def det(self) -> float:
        return float(np.prod(self._eigvals))

    def volume(self, radius: float = 1.0) -> float:
        """Volume (area if dim=2) of the ellipsoid at ``radius`` standard deviations. This is the standard formulae using Jacobian."""
        d = self.dim
        unit_ball = math.pi ** (d / 2) / _gamma(d / 2 + 1)
        return float(unit_ball * radius**d * np.prod(self.axis_lengths))

    def boundary_points(self, n: int = 200, radius: float = 1.0) -> np.ndarray:
        """Points on the ellipse (dim 2, shape (n, 2)) or ellipsoid surface
        (dim 3, mesh of shape (n//2, n, 3)), centred on the mean. For drawing."""
        A = self._eigvecs * self.axis_lengths * radius       # columns = scaled axes
        if self.dim == 2:
            t = np.linspace(0, 2 * np.pi, n)
            circle = np.vstack([np.cos(t), np.sin(t)])
            return (A @ circle).T + self.mean
        if self.dim == 3:
            u = np.linspace(0, 2 * np.pi, n)
            v = np.linspace(0, np.pi, max(n // 2, 2))     #for sym, if o to 2npi has n many, we want half as many for 0 to npi
            uu, vv = np.meshgrid(u, v)
            sphere = np.stack([np.sin(vv) * np.cos(uu), np.sin(vv) * np.sin(uu), np.cos(vv)], axis=-1)     #standard polar coordinate
            return sphere @ A.T + self.mean
        raise ValueError("boundary_points only supports dim 2 or 3 (lags 1 or 2)")

    #prediction
    def predictor(self) -> LinearPredictor:
        """Best linear guess of Y_t (first coordinate) from the other coordinates."""
        s_pp = self.cov[1:, 1:]
        c = self.cov[1:, 0]
        w = solve(s_pp, c, assume_a="pos")
        err = float(self.cov[0, 0] - c @ w)   #can become neg a bit during calculation
        intercept = float(self.mean[0] - w @ self.mean[1:])
        return LinearPredictor(w, intercept, max(err, 0.0), float(self.cov[0, 0]))

    def __repr__(self) -> str:
        extra = f", n_obs={self.n_obs}" if self.n_obs is not None else ""
        return f"LagCloud(lags={self.lags}, source={self.source!r}{extra})"


#predictor's change as lags are added, the standard Durbin-Levinson algorithm
@dataclass(frozen=True)
class Convergence:
    """Best-guess weights for 1, 2, ..., max_lags lags of a process.

    lags[i]            : number of lags used (i+1)
    weights[i]         : weight array for that many lags
    last_weight[i]     : weight on the oldest lag (this is the PACF)
    error_variance[i]  : prediction error variance (falls towards sigma^2)
    """
    lags: np.ndarray
    weights: list
    last_weight: np.ndarray
    error_variance: np.ndarray


def predictor_convergence(process: ARMAProcess, max_lags: int = 20) -> Convergence:              #lags beyond 20 is possible ofc, it is simply my choice to keep default at 20, even 1000 will take milliseconds to calculate, recall durbin levinson has O(n^2)
    """Best-guess weights for every lag count up to ``max_lags`` (Levinson-Durbin)."""
    if max_lags < 1:
        raise ValueError("max_lags must be >= 1")
    _, _, _, sigma, phi = levinson_durbin(process.autocov(max_lags), nlags=max_lags, isacov=True)
    ks = np.arange(1, max_lags + 1)
    weights = [phi[1:k + 1, k].copy() for k in ks]
    return Convergence(ks, weights, np.array([w[-1] for w in weights]), np.asarray(sigma[1:]).copy())
    #instead of p.array([w[-1] for w in weights]), we could have used the pacf directly that durbin gives (instaed of keeping _, in the third) but lets avoid using using pacfs acfs etc that are part of the lib we are building, we directly used the last entries that are pacfs


def ar_inf_weights(process: ARMAProcess, nterms: int = 20) -> np.ndarray:            
    """Weights pi_1, pi_2, ... with Y_t = sum_j pi_j Y_{t-j} + e_t (needs invertibility).

    >>> ar_inf_weights(ARMAProcess(ma=[0.7]), 3).round(3).tolist()
    [0.7, -0.49, 0.343]
    """
    #notice we are back to ARMAProcess, so invertibility is needed for AR representation
    if not process.is_invertible:
        raise ValueError("AR(infinity) form exists only for invertible processes")
    sm = _SMArma(np.r_[1.0, -process.ar], np.r_[1.0, process.ma])
    return -np.asarray(sm.arma2ar(nterms + 1))[1:]


#parameter regions
def ar2_stationary(phi1, phi2):
    """True where an AR(2) with these coefficients is stationary (the triangle)."""
    phi1, phi2 = np.asarray(phi1, float), np.asarray(phi2, float)
    return (phi1 + phi2 < 1) & (phi2 - phi1 < 1) & (np.abs(phi2) < 1)  #&combines arrays, it is element wise AND


def ma2_invertible(theta1, theta2):
    """True where an MA(2) with these coefficients is invertible (the triangle)."""
    t1, t2 = np.asarray(theta1, float), np.asarray(theta2, float)
    return (t1 + t2 > -1) & (t2 - t1 > -1) & (np.abs(t2) < 1)


def ar2_complex_roots(phi1, phi2):
    """True where the AR(2) has complex roots (pseudo-cycles): phi1^2 + 4 phi2 < 0."""
    phi1, phi2 = np.asarray(phi1, float), np.asarray(phi2, float)
    return phi1**2 + 4 * phi2 < 0


def ar2_triangle() -> np.ndarray:
    """Corners of the AR(2) stationarity triangle in (phi1, phi2)."""
    return np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 1.0]])


def ma2_triangle() -> np.ndarray:
    """Corners of the MA(2) invertibility triangle in (theta1, theta2)."""
    return np.array([[-2.0, 1.0], [2.0, 1.0], [0.0, -1.0]])


def ar2_complex_boundary(n: int = 200) -> np.ndarray:
    """Curve phi2 = -phi1^2/4 separating real from complex roots, inside the triangle."""
    p1 = np.linspace(-2, 2, n)
    return np.column_stack([p1, -p1**2 / 4])


#theory vs data
@dataclass(frozen=True)
class CloudComparison:
    """How far cloud ``a`` is from reference cloud ``b`` (e.g. data vs theory)."""
    cov_rel_error: float          # ||cov_a - cov_b|| / ||cov_b||  (0 = identical)
    axis_length_ratio: np.ndarray # a / b, per axis (largest first)
    elongation_ratio: float
    volume_ratio: float
    axis_angles_deg: np.ndarray   # angle between matching axes (unreliable if eigenvalues nearly tie)


def compare_clouds(a: LagCloud, b: LagCloud) -> CloudComparison:
    """Compare two clouds of the same dimension; ``b`` is the reference.
    
    Perfcet match will give 0, 1 per axis, 1, 1, 0^degree per axis. A 0.1 in cov_rel_error means the two cov matrices differ by 10%
    """
    if a.dim != b.dim:
        raise ValueError("clouds must have the same number of lags")
    dots = np.abs(np.sum(a.axes * b.axes, axis=0)).clip(0, 1)   #dot product o eigen vectors
    with np.errstate(divide="ignore", invalid="ignore"):
        return CloudComparison(
            cov_rel_error=float(np.linalg.norm(a.cov - b.cov) / np.linalg.norm(b.cov)),
            axis_length_ratio=a.axis_lengths / b.axis_lengths,
            elongation_ratio=a.elongation / b.elongation,
            volume_ratio=a.volume() / b.volume(),
            axis_angles_deg=np.degrees(np.arccos(dots)),
        )