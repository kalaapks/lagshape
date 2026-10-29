"""ARMA process: the exact numbers behind an AR / MA / ARMA model.

Convention (same as the HTML visualizers):

    Y_t = ar[0]*Y_{t-1} + ... + ar[p-1]*Y_{t-p}  +  e_t + ma[0]*e_{t-1} + ... + ma[q-1]*e_{t-q}

with e_t = sigma * (unit-variance white noise).

All heavy maths is delegated to statsmodels' ``ArmaProcess``; this class only
handles the sign convention, the sigma scaling, and a cleaner interface.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Sequence

import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

# A root is treated as "on the unit circle" if it is within this distance of it.
_TOL = 1e-8


class ARMAProcess:
    """A (zero-mean) ARMA(p, q) process.

    Parameters
    ----------
    ar : sequence of float, optional
        AR coefficients [phi_1, ..., phi_p]. Empty means no AR part.
    ma : sequence of float, optional
        MA coefficients [theta_1, ..., theta_q]. Empty means no MA part.
    sigma : float
        Standard deviation of the noise e_t. Must be > 0.

    Examples
    --------
    >>> p = ARMAProcess(ar=[0.7])
    >>> round(p.variance, 2)
    1.96
    >>> p.acf(3).round(3).tolist()
    [1.0, 0.7, 0.49, 0.343]
    """

    def __init__(self, ar: Sequence[float] = (), ma: Sequence[float] = (), sigma: float = 1.0):
        ar_a = np.asarray(ar, dtype=float).ravel()
        ma_a = np.asarray(ma, dtype=float).ravel()
        if not (np.all(np.isfinite(ar_a)) and np.all(np.isfinite(ma_a))):
            raise ValueError("ar and ma coefficients must be finite numbers")
        if not (np.isfinite(sigma) and sigma > 0):
            raise ValueError("sigma must be a positive finite number")
        self.ar = ar_a
        self.ma = ma_a
        self.sigma = float(sigma)
        # statsmodels convention: lag polynomials 1 - phi_1 z - ... and 1 + theta_1 z + ...
        self._sm = ArmaProcess(np.r_[1.0, -ar_a], np.r_[1.0, ma_a])

    # ------------------------------------------------------------------ info
    @property
    def p(self) -> int:
        return len(self.ar)

    @property
    def q(self) -> int:
        return len(self.ma)

    def __repr__(self) -> str:
        return f"ARMAProcess(ar={self.ar.tolist()}, ma={self.ma.tolist()}, sigma={self.sigma})"

    # ----------------------------------------------------------------- roots
    @property
    def ar_roots(self) -> np.ndarray:
        """Roots of the AR polynomial 1 - phi_1 z - ... (stationary iff all |root| > 1)."""
        return np.asarray(self._sm.arroots)

    @property
    def ma_roots(self) -> np.ndarray:
        """Roots of the MA polynomial 1 + theta_1 z + ... (invertible iff all |root| > 1)."""
        return np.asarray(self._sm.maroots)

    @staticmethod
    def _margin(roots: np.ndarray) -> float:
        return math.inf if roots.size == 0 else float(np.min(np.abs(roots)) - 1.0)

    @property
    def ar_margin(self) -> float:
        """min|AR root| - 1. Positive = stationary, ~0 = on the edge, inf = no AR part."""
        return self._margin(self.ar_roots)

    @property
    def ma_margin(self) -> float:
        """min|MA root| - 1. Positive = invertible, ~0 = on the edge, inf = no MA part."""
        return self._margin(self.ma_roots)

    @property
    def margin(self) -> float:
        """The smaller of ar_margin and ma_margin (distance to the nearest unit root)."""
        return min(self.ar_margin, self.ma_margin)

    @property
    def is_stationary(self) -> bool:
        """True if all AR roots are outside the unit circle (beyond a tiny tolerance)."""
        return self.ar_margin > _TOL

    @property
    def is_invertible(self) -> bool:
        """True if all MA roots are outside the unit circle (beyond a tiny tolerance)."""
        return self.ma_margin > _TOL

    @property
    def has_complex_roots(self) -> bool:
        """True if the AR part has complex roots (pseudo-cycles / spirals)."""
        return bool(np.any(np.abs(self.ar_roots.imag) > _TOL))

    # ---------------------------------------------------------------- theory
    def _require_stationary(self) -> None:
        if not self.is_stationary:
            raise ValueError(
                "Process is not stationary (AR root on/inside the unit circle); "
                "theoretical variance and correlations do not exist."
            )

    def autocov(self, nlags: int = 10) -> np.ndarray:
        """Exact autocovariances gamma(0..nlags), length nlags+1."""
        self._require_stationary()
        if nlags < 0:
            raise ValueError("nlags must be >= 0")
        return self.sigma**2 * np.asarray(self._sm.acovf(nlags + 1))

    def acf(self, nlags: int = 10) -> np.ndarray:
        """Exact autocorrelations rho(0..nlags), length nlags+1."""
        self._require_stationary()
        if nlags < 0:
            raise ValueError("nlags must be >= 0")
        return np.asarray(self._sm.acf(nlags + 1))

    def pacf(self, nlags: int = 10) -> np.ndarray:
        """Exact partial autocorrelations at lags 0..nlags, length nlags+1."""
        self._require_stationary()
        if nlags < 0:
            raise ValueError("nlags must be >= 0")
        x = np.asarray(self._sm.pacf(nlags + 1))
        y =np.where(x < 1e-12, 0, x)
        return y

    @property
    def variance(self) -> float:
        """Var(Y_t); ``inf`` if the process is not stationary."""
        if not self.is_stationary:
            return math.inf
        return float(self.autocov(0)[0])

    def spectral_density(self, freqs: np.ndarray) -> np.ndarray:
        """Spectral density f(w) = sigma^2/(2pi) * |theta(e^{-iw})|^2 / |phi(e^{-iw})|^2.

        ``freqs`` are angular frequencies in [0, pi]. Normalised so that
        the integral of f over [-pi, pi] equals Var(Y_t).
        """
        self._require_stationary()
        w = np.asarray(freqs, dtype=float)
        z = np.exp(-1j * w)
        num = np.polyval(np.r_[self.ma[::-1], 1.0], z)   # 1 + theta_1 z + ...
        den = np.polyval(np.r_[-self.ar[::-1], 1.0], z)  # 1 - phi_1 z - ...
        return self.sigma**2 / (2 * np.pi) * np.abs(num) ** 2 / np.abs(den) ** 2

    # ------------------------------------------------------------ simulation
    def _default_burnin(self) -> int:
        """Burn-in long enough for the start-up effect to fade to ~1e-10."""
        if not self.is_stationary:
            return 0
        r = float(np.min(np.abs(self.ar_roots))) if self.p else math.inf
        decay = 100 if math.isinf(r) else math.ceil(math.log(1e-10) / -math.log(r)) if r > 1 + 1e-12 else 10_000
        return int(min(max(decay, 100), 100_000)) + self.q + self.p

    def simulate(
        self,
        n: int,
        seed: int | None = None,
        burnin: int | None = None,
        innovations: Callable[[np.random.Generator, int], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Simulate ``n`` observations.

        Parameters
        ----------
        n : number of observations returned.
        seed : makes the result repeatable.
        burnin : initial observations discarded. Default: chosen automatically
            from the AR roots so start-up effects have faded. (Non-stationary
            processes start at zero with no burn-in and a warning.)
        innovations : optional function ``f(rng, size)`` returning unit-variance
            noise (default: standard normal). Use it for e.g. Student-t noise.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        rng = np.random.default_rng(seed)
        draw = innovations or (lambda g, size: g.standard_normal(size))
        if not self.is_stationary:
            warnings.warn("Process is not stationary: simulation starts at zero and may explode.")
        b = self._default_burnin() if burnin is None else int(burnin)

        def distrvs(size):
            m = int(np.prod(size))               # statsmodels passes a tuple; users get a plain int
            e = np.asarray(draw(rng, m), dtype=float)
            if e.shape != (m,):
                raise ValueError(f"innovations must return an array of length {m}, got shape {e.shape}")
            return e.reshape(size)

        return np.asarray(self._sm.generate_sample(nsample=n, scale=self.sigma, distrvs=distrvs, burnin=b))
