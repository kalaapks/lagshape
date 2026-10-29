"""Interactive plotly pictures for lagshape.

Every function returns a plotly ``Figure``; call ``.show()`` to open it
(in a plain IPython session it opens in your browser), or ``.write_html("x.html")``.

Pictures of the dot cloud
    plot_cloud      dots + covariance ellipse/ellipsoid + axes + best-guess line/plane (lags 1 or 2)
    plot_series     the time series, with theory bands if a model is given
    plot_acf        sample vs theoretical autocorrelation (or partial autocorrelation)
    plot_sweep      slider over a coefficient: watch the ellipse stretch and turn

New views
    plot_regions        stationarity / invertibility triangle of AR(2) / MA(2)
    plot_convergence    best-guess weights as more lags are allowed, vs the AR(inf) weights
    plot_ellipses       overlay the ellipses of several clouds (models and/or data)

Plot coordinates follow: x = Y_{t-1}, y = Y_t, z = Y_{t-2}.
(The library's own order is (Y_t, Y_{t-1}, Y_{t-2}); this module reorders.)

Change the look globally with ``lagshape.plots.TEMPLATE = "plotly_white"``.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise ImportError("lagshape.plots needs plotly:  pip install plotly") from exc

from statsmodels.tsa.stattools import acf as _sm_acf
from statsmodels.tsa.stattools import pacf as _sm_pacf

from .geometry import (
    LagCloud, ar2_complex_boundary, ar2_stationary, ar2_triangle, ar_inf_weights,
    ma2_invertible, ma2_triangle, predictor_convergence,
)
from .process import ARMAProcess

__all__ = [
    "plot_cloud", "plot_series", "plot_acf", "plot_sweep",
    "plot_regions", "plot_convergence", "plot_ellipses", "TEMPLATE",
]

TEMPLATE = "plotly_dark"

#colours
_DOTS, _THEORY, _SAMPLE = "#9aa7b4", "#4fd1c5", "#f6ad55"
_AXES, _PRED, _LIMIT = "#fc8181", "#b794f4", "#ffffff"
_CYCLE = ["#4fd1c5", "#f6ad55", "#b794f4", "#fc8181", "#63b3ed", "#68d391", "#f687b3", "#ecc94b"]

# library order (Y_t, Y_t-1, Y_t-2)  but standard plot order (Y_t-1, Y_t, Y_t-2)!! crazy right, someone thought this is natural
_PERM = {2: [1, 0], 3: [1, 0, 2]}
_LABEL = ["Y<sub>t</sub>", "Y<sub>t-1</sub>", "Y<sub>t-2</sub>"]


#helpers
def _resolve(source, model, n, seed):
    """source = ARMAProcess (simulate it) or an array of data."""
    if isinstance(source, ARMAProcess):
        if model is not None:
            raise ValueError("pass either an ARMAProcess as the source or a model=, not both")
        return source.simulate(n, seed=seed), source
    y = np.asarray(source, dtype=float).ravel()
    return y, model


def _style(fig, title, **layout):
    layout.setdefault("legend", dict(orientation="h", y=-0.12))
    fig.update_layout(template=TEMPLATE, title=dict(text=title, x=0.02), margin=dict(l=50, r=20, t=60, b=40), **layout)
    return fig


def _ellipse_trace(cloud: LagCloud, radius, center, color, name, dash=None, opacity=0.22):
    """Ellipse (2D) or ellipsoid surface (3D) of ``cloud``, re-centred on ``center`` (library order)."""
    perm = _PERM[cloud.dim]
    b = (cloud.boundary_points(120 if cloud.dim == 2 else 48, radius) - cloud.mean + center)[..., perm]
    if cloud.dim == 2:
        return go.Scatter(x=b[:, 0], y=b[:, 1], mode="lines", name=name,
                          line=dict(color=color, width=2.5, dash=dash or "solid"), hoverinfo="skip")
    return go.Surface(x=b[..., 0], y=b[..., 1], z=b[..., 2], name=name, showlegend=True, showscale=False,
                      opacity=opacity, colorscale=[[0, color], [1, color]], hoverinfo="skip")


def _axes_trace(cloud: LagCloud, radius, center, color=_AXES, name="principal axes"):
    """The principal axes as one line trace (2D or 3D), centred on ``center``."""
    perm = _PERM[cloud.dim]
    pts = []
    for i in range(cloud.dim):
        v = cloud.axes[:, i] * cloud.axis_lengths[i] * radius
        pts += [(center - v)[perm], (center + v)[perm], np.full(cloud.dim, np.nan)]
    a = np.array(pts)
    if cloud.dim == 2:
        return go.Scatter(x=a[:, 0], y=a[:, 1], mode="lines", name=name, line=dict(color=color, width=2), hoverinfo="skip")
    return go.Scatter3d(x=a[:, 0], y=a[:, 1], z=a[:, 2], mode="lines", name=name, line=dict(color=color, width=5), hoverinfo="skip")


def _predictor_trace(cloud: LagCloud, center, R, color=_PRED):
    """Best-guess line (2D) or plane (3D). Passes through ``center``."""
    pr = cloud.predictor()
    w = pr.weights
    c0 = center[0] - w @ center[1:]            # intercept so the plane goes through the centre
    terms = [f"{wi:.3g}·Y<sub>t-{i + 1}</sub>" for i, wi in enumerate(w)]
    if abs(c0) > 1e-6 * max(1.0, np.abs(center).max()):
        terms.insert(0, f"{c0:.3g}")
    name = "best guess: Y<sub>t</sub> = " + " + ".join(terms).replace("+ -", "- ")
    if cloud.dim == 2:
        x = center[1] + np.array([-R, R])
        return go.Scatter(x=x, y=c0 + w[0] * x, mode="lines", name=name, line=dict(color=color, width=2.5, dash="dash"), hoverinfo="skip")
    gx = center[1] + np.linspace(-R, R, 16)          # Y_t-1
    gz = center[2] + np.linspace(-R, R, 16)          # Y_t-2
    X, Z = np.meshgrid(gx, gz)
    Y = c0 + w[0] * X + w[1] * Z
    return go.Surface(x=X, y=Y, z=Z, name=name, showlegend=True, showscale=False, opacity=0.35,
                      colorscale=[[0, color], [1, color]], hoverinfo="skip")


def _equal_axes_2d(fig, center, R):
    fig.update_xaxes(range=[center[1] - R, center[1] + R], title=_LABEL[1])
    fig.update_yaxes(range=[center[0] - R, center[0] + R], title=_LABEL[0], scaleanchor="x", scaleratio=1)


#for actual page
def plot_cloud(source, 
               lags: int = 1, *, n: int = 500,
               seed: int | None = None,
               model: ARMAProcess | None = None, 
               radius: float = 2.0,
               axes: bool = True, 
               predictor: bool = True,
               axis_range: float | None = None, 
               title: str | None = None) -> go.Figure:
    """Dot cloud of (Y_{t-1}, Y_t) (lags=1, 2D) or (Y_{t-1}, Y_t, Y_{t-2}) (lags=2, 3D).

    Parameters:
    
    source : ARMAProcess (n points are simulated) or an array of data.
    model  : optional ARMAProcess to overlay on data. Its exact ellipse is drawn solid, the data's own ellipse dashed/transparent.
    radius : ellipse size in standard deviations (2 = the usual "2-sigma" ellipse).
    axes, predictor : draw principal axes / best-guess line or plane.
    axis_range : half-width of the plot box; default automatic.
    """
    if lags not in (1, 2):
        raise ValueError("plot_cloud draws lags=1 (2D) or lags=2 (3D); other lags cannot be drawn")
    y, model = _resolve(source, model, n, seed)
    sample = LagCloud.from_data(y, lags)
    theory = LagCloud.from_process(model, lags) if model is not None else None
    ref = theory if theory is not None else sample
    center = sample.mean                                  # library order
    perm = _PERM[lags + 1]

    pts = sample.points[:, perm]
    R = axis_range
    if R is None:
        ell = [c.boundary_points(40, radius) - c.mean + center for c in (sample, theory) if c is not None]
        R = 1.08 * max(np.abs(pts - center[perm]).max(), *(np.abs(e - center).max() for e in ell))
 
#========================the 1.08 is us choosing to add extra 8% margin


    fig = go.Figure()
    if lags == 1:
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name="data" if model is None else "simulated / data",
                                 marker=dict(size=4, color=_DOTS, opacity=0.6), hoverinfo="skip"))
    else:
        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name="data",
                                   marker=dict(size=2.5, color=_DOTS, opacity=0.6), hoverinfo="skip"))
    if theory is not None:
        fig.add_trace(_ellipse_trace(theory, radius, center, _THEORY, f"model {radius:g}σ ellipse"))
        fig.add_trace(_ellipse_trace(sample, radius, center, _SAMPLE, f"data {radius:g}σ ellipse", dash="dash", opacity=0.12))
    else:
        fig.add_trace(_ellipse_trace(sample, radius, center, _THEORY, f"{radius:g}σ ellipse"))
    if axes:
        fig.add_trace(_axes_trace(ref, radius, center))
    if predictor:
        fig.add_trace(_predictor_trace(ref, center, R))

    if title is None:
        title = (f"Lag cloud, lags={lags}" + (f" — {model}" if model is not None else "")
                 + f"<br><sup>elongation {ref.elongation:.2f}, axis lengths "
                   f"{', '.join(f'{a:.2f}' for a in ref.axis_lengths)}</sup>")
    _style(fig, title, height=650 if lags == 2 else 620)
    if lags == 1:
        _equal_axes_2d(fig, center, R)
    else:
        rng = lambda i: [center[i] - R, center[i] + R]
        fig.update_layout(scene=dict(
            xaxis=dict(title=_LABEL[1], range=rng(1)), yaxis=dict(title=_LABEL[0], range=rng(0)),
            zaxis=dict(title=_LABEL[2], range=rng(2)), aspectmode="cube",
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9))))
    return fig


def plot_series(source, *,
                n: int = 500, seed: int | None = None,
                model: ARMAProcess | None = None,
                band: float = 2.0,
                yrange: float | tuple[float, float] | list[float] | None = None,
                title: str | None = None) -> go.Figure:
    """The series over time. With a model: dashed lines at +-``band`` theoretical std devs.
        Parameters:
        
        yrange : float, tuple, list, or None
                If a single number `c` (e.g., 4 or 5), fixes the y-axis to [-c, c].
                If a tuple/list `(y_min, y_max)`, sets the explicit boundaries.
                If None, Plotly auto-scales.
    """
    y, model = _resolve(source, model, n, seed)
    fig = go.Figure(go.Scatter(x=np.arange(len(y)), y=y, mode="lines", name="series", line=dict(color=_THEORY, width=1.5)))
    if model is not None and model.is_stationary:
        s = band * np.sqrt(model.variance)
        for v in (-s, s):
            fig.add_hline(y=v, line=dict(color=_AXES, dash="dash", width=1))
        fig.add_hline(y=0, line=dict(color=_DOTS, width=1))
    _style(fig, title or "Time series" + (f" — {model}" if model is not None else ""), height=380)
    fig.update_xaxes(title="t")

    # Configure y-axis range
    if isinstance(yrange, (int, float)):
        y_limits = [-yrange, yrange]
    else:
        y_limits = yrange
    fig.update_yaxes(title="Y<sub>t</sub>", range=y_limits)
    return fig


def plot_acf(source, *, model: ARMAProcess | None = None, nlags: int = 20, partial: bool = False,
             n: int = 500, seed: int | None = None, title: str | None = None) -> go.Figure:
    """Sample (bars) vs theoretical (dots) autocorrelation, or partial autocorrelation.

    The shaded band is +-1.96/sqrt(N): bars inside it are consistent with zero.
    """
    y, model = _resolve(source, model, n, seed)
    sample = (_sm_pacf(y, nlags=nlags, method="ywm") if partial else _sm_acf(y, nlags=nlags, fft=True))[1:]
    lags = np.arange(1, nlags + 1)
    what = "PACF" if partial else "ACF"
    fig = go.Figure(go.Bar(x=lags, y=sample, name=f"sample {what}", marker_color=_SAMPLE, opacity=0.8))
    if model is not None:
        th = (model.pacf(nlags) if partial else model.acf(nlags))[1:]
        fig.add_trace(go.Scatter(x=lags, y=th, mode="markers", name=f"theoretical {what}",
                                 marker=dict(color=_THEORY, size=9, symbol="diamond")))
    b = 1.96 / np.sqrt(len(y))
    fig.add_hrect(y0=-b, y1=b, fillcolor=_DOTS, opacity=0.15, line_width=0)
    _style(fig, title or f"{'Partial ' if partial else ''}autocorrelation", height=400)
    fig.update_xaxes(title="lag", dtick=max(1, nlags // 10))
    fig.update_yaxes(title=what, range=[-1.05, 1.05])
    return fig


def plot_sweep(make_process: Callable[[float], ARMAProcess], values: Sequence[float], *,
               n: int = 400, seed: int = 0, radius: float = 2.0, label: str = "φ",
               axis_range: float | None = None, title: str | None = None) -> go.Figure:
    """Slider over one coefficient (2D cloud, lags=1) — the Python version of your page's slider.

    ``make_process`` maps a value to a process, e.g. ``lambda phi: ARMAProcess(ar=[phi])``.
    The same noise is reused for every value, so dots move smoothly as the slider moves.
    All values must give stationary processes.
    """
    values = list(values)
    frames_data, titles, R = [], [], 0.0
    for v in values:
        p = make_process(v)
        c = LagCloud.from_process(p, 1)
        y = p.simulate(n, seed=seed, burnin=500)                 # fixed burn-in -> same noise each frame
        pts = LagCloud.from_data(y, 1).points[:, _PERM[2]]
        e = c.boundary_points(120, radius)[:, _PERM[2]]
        ax = _axes_trace(c, radius, c.mean)
        pr = _predictor_trace(c, c.mean, 1.0)                     # x range rescaled below
        frames_data.append((pts, e, ax, pr, c, p))
        R = max(R, np.abs(pts).max(), np.abs(e).max())
        titles.append(f"{label} = {v:.3g} &nbsp;|&nbsp; Var = {p.variance:.3g} &nbsp;|&nbsp; "
                      f"ρ₁ = {p.acf(1)[1]:.3f} &nbsp;|&nbsp; elongation = {c.elongation:.2f}")
    R = (axis_range if axis_range is not None else 1.08 * R)

    def traces(i):
        pts, e, ax, pr, c, p = frames_data[i]
        w = c.predictor().weights[0]
        xl = np.array([-R, R])
        return [
            go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name="dots", marker=dict(size=4, color=_DOTS, opacity=0.6), hoverinfo="skip"),
            go.Scatter(x=e[:, 0], y=e[:, 1], mode="lines", name=f"{radius:g}σ ellipse", line=dict(color=_THEORY, width=2.5), hoverinfo="skip"),
            ax,
            go.Scatter(x=xl, y=w * xl, mode="lines", name="best guess", line=dict(color=_PRED, width=2.5, dash="dash"), hoverinfo="skip"),
        ]

    fig = go.Figure(data=traces(0), frames=[go.Frame(data=traces(i), name=str(i), traces=[0, 1, 2, 3],
                                                     layout=go.Layout(title_text=titles[i])) for i in range(len(values))])
    steps = [dict(method="animate", label=f"{v:.2f}",
                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for i, v in enumerate(values)]
    play = dict(type="buttons", showactive=False, x=0.0, y=-0.2, xanchor="left", yanchor="top", buttons=[
        dict(label="▶ play", method="animate", args=[None, dict(frame=dict(duration=90, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
        dict(label="⏸ pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))])])
    _style(fig, title or titles[0], height=700, legend=dict(orientation="v", x=1.01, y=1.0), sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix=f"{label} = "), pad=dict(t=50, l=120))],
           updatemenus=[play])
    _equal_axes_2d(fig, np.zeros(2), R)
    return fig


# ============================= B: new views
def plot_regions(kind: str = "ar2", *, points: Sequence | None = None, title: str | None = None) -> go.Figure:
    """The AR(2) stationarity triangle (or MA(2) invertibility triangle) in coefficient space.

    The curve inside separates real roots (smooth decay) from complex roots (pseudo-cycles).
    ``points``: list of (c1, c2) pairs or ARMAProcess objects to mark; points outside the
    triangle (not stationary / not invertible) are shown in red.
    """
    if kind not in ("ar2", "ma2"):
        raise ValueError("kind must be 'ar2' or 'ma2'")
    ar = kind == "ar2"
    tri = ar2_triangle() if ar else ma2_triangle()
    p1 = np.linspace(-2, 2, 200)
    curve = np.column_stack([p1, -p1**2 / 4]) if ar else np.column_stack([p1, p1**2 / 4])
    c1, c2 = ("φ₁", "φ₂") if ar else ("θ₁", "θ₂")
    tri_c = np.vstack([tri, tri[:1]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tri_c[:, 0], y=tri_c[:, 1], mode="lines", fill="toself", fillcolor="rgba(79,209,197,0.25)",
                             line=dict(color=_THEORY, width=2), name="real roots (inside triangle)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curve[:, 0], y=curve[:, 1], mode="lines", fill="toself", fillcolor="rgba(246,173,85,0.35)",
                             line=dict(color=_SAMPLE, width=2), name="complex roots (pseudo-cycles)", hoverinfo="skip"))
    for k, pt in enumerate(points or []):
        if isinstance(pt, ARMAProcess):
            if ar and not (pt.p == 2 and pt.q == 0) or (not ar and not (pt.q == 2 and pt.p == 0)):
                raise ValueError(f"for kind={kind!r} give a pure {'AR' if ar else 'MA'}(2) process")
            a, b = (pt.ar if ar else pt.ma)
        else:
            a, b = pt
        ok = bool(ar2_stationary(a, b) if ar else ma2_invertible(a, b))
        fig.add_trace(go.Scatter(x=[a], y=[b], mode="markers+text", text=[f"({a:.2f}, {b:.2f})"], textposition="top center",
                                 marker=dict(size=11, color=_LIMIT if ok else "#f56565", line=dict(width=1.5, color="#000")),
                                 name=f"point {k + 1}" + ("" if ok else " (outside)")))
    _style(fig, title or ("AR(2) stationarity region" if ar else "MA(2) invertibility region"), height=520)
    fig.update_xaxes(title=c1, range=[-2.6, 2.6], zeroline=True)
    fig.update_yaxes(title=c2, range=[-1.5, 1.5], zeroline=True)
    return fig


def plot_convergence(model: ARMAProcess, max_lags: int = 15, *, show_lags: Sequence[int] | None = None,
                     title: str | None = None) -> go.Figure:
    """Best-guess weights as more lags are allowed.

    Left:   weights for several lag counts, against the AR(inf) weights (white, if invertible).
    Middle: prediction error variance by number of lags (floor = sigma^2).
    Right:  the weight on the oldest lag at each step (this is the PACF).
    """
    conv = predictor_convergence(model, max_lags)
    if show_lags is None:
        show_lags = sorted({k for k in (1, 2, 3, 5, max_lags) if k <= max_lags})
    fig = make_subplots(rows=1, cols=3, subplot_titles=("best-guess weights", "prediction error variance", "weight on oldest lag (PACF)"))
    if model.is_invertible:      # drawn first so the k-lag curves sit on top of it
        fig.add_trace(go.Scatter(x=np.arange(1, max_lags + 1), y=ar_inf_weights(model, max_lags), mode="lines+markers",
                                 name="AR(∞) limit", line=dict(color=_LIMIT, width=7), opacity=0.35,
                                 marker=dict(size=11)), row=1, col=1)
    colours = sample_colorscale("Viridis", np.linspace(0.05, 0.95, max(len(show_lags), 2)))
    for c, k in zip(colours, show_lags):
        fig.add_trace(go.Scatter(x=np.arange(1, k + 1), y=conv.weights[k - 1], mode="lines+markers",
                                 name=f"{k} lag{'s' if k > 1 else ''}", line=dict(color=c, width=2), marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=conv.lags, y=conv.error_variance, mode="lines+markers", name="error variance",
                             line=dict(color=_SAMPLE)), row=1, col=2)
    fig.add_hline(y=model.sigma**2, line=dict(color=_LIMIT, dash="dot"), annotation_text="σ²", row=1, col=2)
    fig.add_trace(go.Bar(x=conv.lags, y=conv.last_weight, name="PACF", marker_color=_THEORY), row=1, col=3)
    _style(fig, title or f"Best linear guess vs number of lags — {model}", height=470, legend=dict(orientation="v", x=1.01, y=1.0))
    fig.update_xaxes(title_text="j (how far back)", row=1, col=1)
    fig.update_xaxes(title_text="number of lags", row=1, col=2)
    fig.update_xaxes(title_text="lag", row=1, col=3)
    fig.update_yaxes(title_text="weight", row=1, col=1)
    return fig


def plot_ellipses(clouds: Mapping[str, LagCloud] | Sequence[LagCloud], *, radius: float = 2.0,
                  title: str | None = None) -> go.Figure:
    """Overlay the ellipses (2D) or ellipsoids (3D) of several LagClouds — models and/or data.

    ``clouds`` is a dict ``{"name": cloud}`` or a plain list. All must have the same lags (1 or 2).
    """
    if not isinstance(clouds, Mapping):
        clouds = {f"cloud {i + 1}": c for i, c in enumerate(clouds)}
    dims = {c.dim for c in clouds.values()}
    if len(dims) != 1 or dims.pop() not in (2, 3):
        raise ValueError("all clouds must have the same lags, equal to 1 or 2")
    fig = go.Figure()
    for i, (name, c) in enumerate(clouds.items()):
        fig.add_trace(_ellipse_trace(c, radius, c.mean, _CYCLE[i % len(_CYCLE)], name, opacity=0.2))
    first = next(iter(clouds.values()))
    R = 1.08 * max(np.abs(c.boundary_points(40, radius) - c.mean + c.mean).max() for c in clouds.values())
    _style(fig, title or f"{radius:g}σ ellipses", height=600)
    if first.dim == 2:
        _equal_axes_2d(fig, np.zeros(2), R)
    else:
        fig.update_layout(scene=dict(xaxis_title=_LABEL[1], yaxis_title=_LABEL[0], zaxis_title=_LABEL[2], aspectmode="cube"))
    return fig
