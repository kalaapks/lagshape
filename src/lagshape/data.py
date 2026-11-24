"""From real (or practice) data to the lagshape geometry.

Workflow
--------
    prices  = load_prices("SPY", start="2015-01-01")      # or read_series("file.csv")
    returns = to_returns(prices)
    print(diagnose(returns))                              # what kind of series is this?
    select_order(returns)                                 # which (p, q) does the data prefer?
    fit = fit_arma(returns, (1, 0))                       # fit_arma gives an ARMAProcess (fit.model)
    compare_lags(returns, fit)                            # cloud of data vs cloud of model, lags 1..5
    cloud_test(returns, fit, lags=1)                      # is the mismatch bigger than chance?
    walk_forward(returns, lags=2)                         # can the best-guess plane predict unseen data?
    rolling_cloud(returns, lags=1, window=250)            # does the cloud's shape drift over time?

No internet?  ``example_series("roll")`` etc. give practice series where the true model is known.

Existing tools do the work: pandas (tables), statsmodels (ARIMA fitting, ADF, Ljung-Box),
scipy.stats (skew/kurtosis, normal cdf), yfinance (optional, downloads only).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from .geometry import LagCloud, compare_clouds
from .process import ARMAProcess

__all__ = [
    "read_series", "load_prices", "to_returns", "example_series",
    "Diagnostics", "diagnose", "Fit", "fit_arma", "select_order",
    "compare_lags", "CloudTest", "cloud_test", "WalkForward", "walk_forward", "rolling_cloud",
]


# ================================================================== helpers
def _values(y) -> np.ndarray:
    a = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError("data contains NaN or infinite values; clean it first (e.g. series.dropna())")
    return a


def _index(y, positions: np.ndarray):
    """Labels for ``positions`` (dates if y is a pandas Series, else the integers)."""
    return y.index[positions] if isinstance(y, (pd.Series, pd.DataFrame)) else positions


def _as_model(m) -> ARMAProcess:
    if isinstance(m, Fit):
        return m.model
    if isinstance(m, ARMAProcess):
        return m
    raise TypeError("expected an ARMAProcess or a Fit (from fit_arma)")


# ================================================================== loading
def read_series(path: str, column: str | None = None, date_col: str | int | None = 0) -> pd.Series:
    """Read one column of a CSV file as a pandas Series.

    ``date_col`` is the column holding dates (default: the first one; None = no dates).
    If ``column`` is omitted the file must have exactly one other column.
    """
    df = pd.read_csv(path, index_col=date_col, parse_dates=True if date_col is not None else False)
    if column is None:
        if df.shape[1] != 1:
            raise ValueError(f"file has several columns, pick one with column=...: {list(df.columns)}")
        column = df.columns[0]
    return df[column].astype(float).dropna()


def load_prices(tickers, start: str | None = None, end: str | None = None):
    """Download adjusted closing prices with yfinance (``pip install yfinance``).

    One ticker (a string) gives a Series, several give a DataFrame.
    Needs internet access; dates are "YYYY-MM-DD".
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("load_prices needs yfinance:  pip install yfinance  (or use read_series on a CSV)") from exc
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise ValueError("no data returned; check the ticker and your internet connection")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame) and close.shape[1] == 1 and isinstance(tickers, str):
        close = close.iloc[:, 0]
    return close.dropna()


def to_returns(prices, kind: str = "log"):
    """Prices -> returns: ``kind="log"`` (log differences) or ``"simple"`` (percent changes)."""
    p = prices if isinstance(prices, (pd.Series, pd.DataFrame)) else pd.Series(np.asarray(prices, dtype=float))
    if kind == "log":
        return np.log(p).diff().dropna()
    if kind == "simple":
        return p.pct_change().dropna()
    raise ValueError("kind must be 'log' or 'simple'")


# ================================================================== practice data
def example_series(kind: str = "roll", n: int = 2000, seed: int = 0) -> pd.Series:
    """Synthetic series where the truth is known (business-day index starting 2020-01-01).

    kind
    ----
    "white"        pure noise (sigma 1). Nothing to find.
    "random_walk"  a price path (log returns N(0.0003, 0.01)). Prices are not stationary; their returns are noise.
    "spread"       mean-reverting spread: AR(1), phi = 0.95 (half-life about 13.5 days).
    "cycle"        AR(2), phi = (1.2, -0.6): complex roots, a pseudo-cycle.
    "roll"         RETURNS with bid-ask bounce (Roll model): efficient shocks sd 0.3, spread c = 0.6.
                   Exactly an MA(1) with theta about -0.38, rho_1 = -1/3.
    "arch"         GARCH(1,1) returns (alpha 0.08, beta 0.90, daily vol 1%): uncorrelated, but squares are
                   strongly autocorrelated and tails are fat.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    if kind == "white":
        y = rng.standard_normal(n)
    elif kind == "random_walk":
        y = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    elif kind == "spread":
        y = ARMAProcess(ar=[0.95]).simulate(n, seed=seed)
    elif kind == "cycle":
        y = ARMAProcess(ar=[1.2, -0.6]).simulate(n, seed=seed)
    elif kind == "roll":
        eps = rng.normal(0, 0.3, n + 1)
        q = rng.choice([-1.0, 1.0], n + 1)
        y = eps[1:] + 0.3 * (q[1:] - q[:-1])           # c/2 = 0.3
    elif kind == "arch":
        w, a, b = 2e-6, 0.08, 0.90
        h, y, e_prev = w / (1 - a - b), np.empty(n), 0.0
        for t in range(n):
            h = w + a * e_prev**2 + b * h if t else h
            e_prev = np.sqrt(h) * rng.standard_normal()
            y[t] = e_prev
    else:
        raise ValueError("kind must be one of: white, random_walk, spread, cycle, roll, arch")
    return pd.Series(y, index=idx, name=kind)


# ================================================================== diagnostics
@dataclass(frozen=True)
class Diagnostics:
    """Quick health check of a series (see ``diagnose``). ``print`` it for a readable report."""
    n: int
    mean: float
    std: float
    skew: float
    excess_kurtosis: float
    adf_pvalue: float            # small => no unit root found (looks stationary)
    ljung_box_pvalue: float      # small => the series itself is autocorrelated
    ljung_box_sq_pvalue: float   # small => squared (demeaned) series is autocorrelated: volatility clustering
    nlags: int

    def __str__(self) -> str:
        yes = lambda p: "yes" if p < 0.05 else "no"
        return "\n".join([
            f"observations            {self.n}",
            f"mean / std              {self.mean:.4g} / {self.std:.4g}",
            f"skewness                {self.skew:.3f}",
            f"excess kurtosis         {self.excess_kurtosis:.2f}    (0 = normal; much larger = fat tails)",
            f"ADF p-value             {self.adf_pvalue:.4f}  -> looks stationary?             {yes(self.adf_pvalue)}",
            f"Ljung-Box p (lag {self.nlags:>2})   {self.ljung_box_pvalue:.4f}  -> autocorrelated?                {yes(self.ljung_box_pvalue)}",
            f"Ljung-Box p, squares    {self.ljung_box_sq_pvalue:.4f}  -> volatility clustering?           {yes(self.ljung_box_sq_pvalue)}",
        ])


def diagnose(y, nlags: int = 20) -> Diagnostics:
    """Mean, spread, tail weight, stationarity (ADF) and autocorrelation tests (Ljung-Box)
    on the series and on its squares (the usual check for volatility clustering)."""
    v = _values(y)
    if len(v) < 3 * nlags:
        raise ValueError(f"need at least {3 * nlags} observations for nlags={nlags}")
    lb = lambda x: float(acorr_ljungbox(x, lags=[nlags]).iloc[0]["lb_pvalue"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)       # statsmodels announces a future change to adfuller's return type
        adf_p = float(adfuller(v, autolag="AIC")[1])
    return Diagnostics(
        n=len(v), mean=float(v.mean()), std=float(v.std(ddof=1)), skew=float(stats.skew(v)),
        excess_kurtosis=float(stats.kurtosis(v)), adf_pvalue=adf_p,
        ljung_box_pvalue=lb(v - v.mean()), ljung_box_sq_pvalue=lb((v - v.mean()) ** 2), nlags=nlags)


# ================================================================== fitting
@dataclass(frozen=True)
class Fit:
    """A fitted ARMA model. ``model`` is the fitted zero-mean ``ARMAProcess``; ``mean`` is the fitted level."""
    model: ARMAProcess
    mean: float
    order: tuple
    n: int
    aic: float
    bic: float
    loglik: float
    table: pd.DataFrame = field(repr=False)          # estimates, standard errors, p-values
    residuals: np.ndarray = field(repr=False)

    def __str__(self) -> str:
        return (f"ARMA{self.order} fit on {self.n} observations   AIC {self.aic:.2f}  BIC {self.bic:.2f}\n"
                f"{self.table.round(4).to_string()}")


def fit_arma(y, order: tuple = (1, 0), demean: bool = True) -> Fit:
    """Fit an ARMA(p, q) by maximum likelihood (statsmodels ARIMA) and return it as lagshape objects.

    Same sign convention as the rest of the library: Y_t = sum phi_i Y_{t-i} + e_t + sum theta_j e_{t-j}.
    """
    p, q = order
    v = _values(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(v, order=(p, 0, q), trend="c" if demean else "n").fit()
    names = list(res.param_names)
    par = pd.Series(np.asarray(res.params), index=names)
    ar = [par[f"ar.L{i}"] for i in range(1, p + 1)]
    ma = [par[f"ma.L{j}"] for j in range(1, q + 1)]
    table = pd.DataFrame({"estimate": np.asarray(res.params), "std_err": np.asarray(res.bse),
                          "p_value": np.asarray(res.pvalues)}, index=names)
    return Fit(model=ARMAProcess(ar=ar, ma=ma, sigma=float(np.sqrt(par["sigma2"]))),
               mean=float(par["const"]) if demean else 0.0, order=(p, q), n=len(v),
               aic=float(res.aic), bic=float(res.bic), loglik=float(res.llf),
               table=table, residuals=np.asarray(res.resid))


def select_order(y, max_p: int = 3, max_q: int = 3, ic: str = "aic") -> pd.DataFrame:
    """Fit every ARMA(p, q) with p <= max_p, q <= max_q; table sorted best-first by ``ic`` ("aic" or "bic").

    Information criteria reward fit but punish extra parameters (BIC punishes harder).
    """
    if ic not in ("aic", "bic"):
        raise ValueError("ic must be 'aic' or 'bic'")
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                f = fit_arma(y, (p, q))
                rows.append((p, q, f.aic, f.bic, f.loglik))
            except Exception:
                rows.append((p, q, np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["p", "q", "aic", "bic", "loglik"]).sort_values(ic).reset_index(drop=True)


# ================================================================== theory vs data
def compare_lags(y, model, lags: Sequence[int] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Compare the data cloud with the model cloud at lags 1, 2, 3, 4, 5.

    One row per lag count. ``cov_rel_error`` is 0 for a perfect match; ``r2_data`` / ``r2_model`` are the
    shares of variance the best linear guess from that many lags explains, in the data and in the model.
    A match at lags=1 alone is weak evidence; a match across many lags is much stronger.
    """
    m, v = _as_model(model), _values(y)
    rows = []
    for k in lags:
        d, t = LagCloud.from_data(v, k), LagCloud.from_process(m, k)
        c = compare_clouds(d, t)
        rows.append(dict(lags=k, cov_rel_error=c.cov_rel_error, elongation_data=d.elongation,
                         elongation_model=t.elongation, volume_ratio=c.volume_ratio,
                         r2_data=d.predictor().r_squared, r2_model=t.predictor().r_squared))
    return pd.DataFrame(rows).set_index("lags")


@dataclass(frozen=True)
class CloudTest:
    """Result of ``cloud_test``: where the data's mismatch falls among mismatches of series the model itself produces."""
    observed: float
    simulated: np.ndarray = field(repr=False)
    p_value: float
    lags: int

    def __str__(self) -> str:
        q = np.percentile(self.simulated, [50, 95])
        return (f"lags={self.lags}: observed cloud mismatch {self.observed:.4f}\n"
                f"series generated by the model itself: median {q[0]:.4f}, 95th percentile {q[1]:.4f}  "
                f"({len(self.simulated)} simulations)\n"
                f"p-value {self.p_value:.3f}  (small = the data's cloud is further from the model than chance explains)")


def cloud_test(y, model, lags: int = 1, n_sim: int = 200, seed: int | None = None,
               refit: tuple | None = None, innovations: Callable | None = None) -> CloudTest:
    """Is the mismatch between the data's cloud and the model's cloud bigger than chance?

    Simulates ``n_sim`` series of the same length from ``model``, measures each one's mismatch with the model's
    cloud (``cov_rel_error``), and reports the share that are at least as bad as the data's.

    refit : give an order like (1, 0) when ``model`` was fitted on this same data. Each simulated series is then
            re-fitted with that order before comparing (a parametric bootstrap). Without it, the test is too
            kind to a fitted model. Slower: expect roughly 0.1-0.3 s per simulation, so lower n_sim
            (e.g. 100) for a first look.
    innovations : noise generator for the simulations (see ``ARMAProcess.simulate``).
    The test looks only at second-order structure (covariances), not at tails or volatility clustering.
    """
    m, v = _as_model(model), _values(y)
    obs = compare_clouds(LagCloud.from_data(v, lags), LagCloud.from_process(m, lags)).cov_rel_error
    rng = np.random.default_rng(seed)
    theory = LagCloud.from_process(m, lags)
    sims = []
    for _ in range(n_sim):
        ys = m.simulate(len(v), seed=int(rng.integers(2**32)), innovations=innovations)
        ref = theory
        if refit is not None:
            try:
                ref = LagCloud.from_process(fit_arma(ys, refit).model, lags)
            except Exception:
                continue
        sims.append(compare_clouds(LagCloud.from_data(ys, lags), ref).cov_rel_error)
    sims = np.array(sims)
    if len(sims) == 0:
        raise RuntimeError("all refits failed")
    return CloudTest(float(obs), sims, float((1 + np.sum(sims >= obs)) / (len(sims) + 1)), lags)


# ================================================================== prediction on unseen data
@dataclass(frozen=True)
class WalkForward:
    """Out-of-sample one-step forecasts from ``walk_forward``. ``to_frame()`` gives the table."""
    lags: int
    index: object = field(repr=False)
    actual: np.ndarray = field(repr=False)
    forecast: np.ndarray = field(repr=False)
    benchmark: np.ndarray = field(repr=False)
    r2_oos: float
    rmse: float
    rmse_benchmark: float
    hit_rate: float
    cw_pvalue: float

    @property
    def n_test(self) -> int:
        return len(self.actual)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"actual": self.actual, "forecast": self.forecast, "benchmark": self.benchmark}, index=self.index)

    def __str__(self) -> str:
        return (f"walk-forward, lags={self.lags}, {self.n_test} unseen points\n"
                f"out-of-sample R^2 vs mean forecast  {self.r2_oos:+.4f}   (>0: beats 'predict the historical mean')\n"
                f"RMSE model / benchmark              {self.rmse:.5g} / {self.rmse_benchmark:.5g}\n"
                f"sign hit rate                       {self.hit_rate:.3f}\n"
                f"Clark-West p-value (one-sided)      {self.cw_pvalue:.3f}   (small: the lags really add predictive power)")


def walk_forward(y, lags: int = 1, train_size: int | None = None, window: int | None = None,
                 refit_every: int = 1) -> WalkForward:
    """Honest forecasting test: at each time t, build the best-guess line/plane from the PAST only, predict y_t.

    The guess comes from ``LagCloud.from_data(past, lags).predictor()``, i.e. the data cloud's hyperplane.
    train_size : points used before the first forecast (default: half the series).
    window     : None = expanding window; an int = use only the last ``window`` points (rolling).
    refit_every: rebuild the hyperplane every this-many steps (speed vs. freshness).
    Benchmark: forecast y_t by the mean of the same past points. Out-of-sample R^2 = 1 - SSE_model / SSE_benchmark.
    The Clark-West test is the right significance test here because the benchmark is nested in the model.
    """
    v = _values(y)
    n = len(v)
    train_size = train_size if train_size is not None else n // 2
    need = lags + 30
    if train_size < need or (window is not None and window < need) or train_size >= n:
        raise ValueError(f"train_size (and window) must be at least {need} and train_size < len(y)")
    csum = np.r_[0.0, np.cumsum(v)]
    ts = np.arange(train_size, n)
    fc, bm = np.empty(len(ts)), np.empty(len(ts))
    pr = None
    for i, t in enumerate(ts):
        start = 0 if window is None else max(0, t - window)
        if pr is None or i % refit_every == 0:
            pr = LagCloud.from_data(v[start:t], lags).predictor()
        fc[i] = pr.intercept + pr.weights @ v[t - lags:t][::-1]    # (Y_{t-1}, ..., Y_{t-lags})
        bm[i] = (csum[t] - csum[start]) / (t - start)
    act = v[ts]
    e_m, e_b = act - fc, act - bm
    nz = act != 0
    f = e_b**2 - e_m**2 + (bm - fc) ** 2                            # Clark-West adjusted loss gap
    cw_t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f))) if f.std(ddof=1) > 0 else 0.0
    return WalkForward(
        lags=lags, index=_index(y, ts), actual=act, forecast=fc, benchmark=bm,
        r2_oos=float(1 - np.sum(e_m**2) / np.sum(e_b**2)),
        rmse=float(np.sqrt(np.mean(e_m**2))), rmse_benchmark=float(np.sqrt(np.mean(e_b**2))),
        hit_rate=float(np.mean(np.sign(fc[nz]) == np.sign(act[nz]))),
        cw_pvalue=float(1 - stats.norm.cdf(cw_t)))


# ================================================================== drift over time
# def rolling_cloud(y, lags: int = 1, window: int = 250, step: int = 1) -> pd.DataFrame:
#     """Re-measure the cloud in a moving window; one row per window, labelled by its last date/position.

#     Columns: variance, rho1 (lag-1 correlation), weight_1..weight_k (best-guess weights),
#     r_squared, error_std, elongation. Plot a column to see whether the shape holds still.
#     """
#     v = _values(y)
#     need = lags + 30
#     if window < need:
#         raise ValueError(f"window must be at least {need}")
#     ends = np.arange(window, len(v) + 1, step)
#     rows = []
#     for t in ends:
#         c = LagCloud.from_data(v[t - window:t], lags)
#         pr = c.predictor()
#         row = dict(variance=c.cov[0, 0], rho1=c.corr[0, 1])
#         row.update({f"weight_{j + 1}": w for j, w in enumerate(pr.weights)})
#         row.update(r_squared=pr.r_squared, error_std=pr.error_std, elongation=c.elongation)
#         rows.append(row)
#     return pd.DataFrame(rows, index=_index(y, ends - 1))

#updated version:


def rolling_cloud(y, lags: int = 1, window: int = 250, step: int = 1,
                  plot: bool = False, plot_steps: Sequence[int] = (1, 50, 100, 150, 200, 250)) -> pd.DataFrame:
    """Re-measure the cloud in a moving window; one row per window, labelled by its last date/position.

    Columns: variance, rho1 (lag-1 correlation), weight_1..weight_k (best-guess weights),
    r_squared, error_std, elongation. Plot a column to see whether the shape holds still.

    plot : if True, displays a 2x3 matplotlib grid of the 2D clouds (lags=1)
           at the requested step indices in `plot_steps`.
    """
    v = _values(y)
    need = lags + 30
    if window < need:
        raise ValueError(f"window must be at least {need}")
    
    ends = np.arange(window, len(v) + 1, step)
    rows = []
    clouds_to_plot = {}

    for i, t in enumerate(ends):
        sub_v = v[t - window:t]
        c = LagCloud.from_data(sub_v, lags)
        pr = c.predictor()
        row = dict(variance=c.cov[0, 0], rho1=c.corr[0, 1])
        row.update({f"weight_{j + 1}": w for j, w in enumerate(pr.weights)})
        row.update(r_squared=pr.r_squared, error_std=pr.error_std, elongation=c.elongation)
        rows.append(row)

        # Collect snapshot for plotting (step index is 1-based: 1, 50, 100, ...)
        current_step_num = i + 1
        if plot and current_step_num in plot_steps:
            clouds_to_plot[current_step_num] = c

    df = pd.DataFrame(rows, index=_index(y, ends - 1))

    # Automatic Matplotlib plotting
    if plot:
        if lags != 1:
            warnings.warn("Automatic 2x3 plot grid only supports lags=1 (2D). Skipping plot.")
        else:
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
            axes = axes.flatten()

            # Global coordinate limit so all 6 subplots have the exact same scale
            max_val = np.max(np.abs(v)) * 0.95

            for idx, target_step in enumerate(plot_steps[:6]):
                ax = axes[idx]
                if target_step in clouds_to_plot:
                    c = clouds_to_plot[target_step]
                    pts = c.points  # shape (window-1, 2): column 0 is Y_t, column 1 is Y_{t-1}
                    
                    # 1. Scatter points: x = Y_{t-1}, y = Y_t
                    ax.scatter(pts[:, 1], pts[:, 0], s=12, alpha=0.35, color="#4a5568", edgecolors="none")

                    # 2. Draw 2-sigma ellipse
                    # boundary_points returns points in library order (Y_t, Y_{t-1})
                    b_pts = c.boundary_points(n=100, radius=2.0)
                    ax.plot(b_pts[:, 1], b_pts[:, 0], color="#319795", lw=2, label="2σ ellipse")

                    # 3. Draw best-guess regression line
                    pr = c.predictor()
                    xl = np.array([-max_val, max_val])
                    yl = pr.intercept + pr.weights[0] * xl
                    ax.plot(xl, yl, color="#805ad5", lw=1.8, linestyle="--", label="predictor")

                    ax.set_title(f"Step {target_step} (Window {target_step * step})\n"
                                 f"ρ₁ = {c.corr[0,1]:.2f} | elong = {c.elongation:.2f}", fontsize=10)
                else:
                    ax.text(0.5, 0.5, f"Step {target_step}\n(Not reached)", ha="center", va="center")

                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)
                ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
                ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
                ax.set_xlabel("$Y_{t-1}$")
                ax.set_ylabel("$Y_t$")
                if idx == 0:
                    ax.legend(loc="upper left", fontsize=8)

            plt.suptitle(f"Rolling Cloud Evolution (lags=1, window={window})", fontsize=14, y=1.02)
            plt.tight_layout()
            plt.show()

    return df