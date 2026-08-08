"""Is the band honest? — the most important plots in the project.

The model does not output a salary, it outputs a **band**: "₹18L–₹24L, and I am
80% confident." That second half is a promise, and a promise nobody checks is a
promise nobody should believe.

Checking it is called **coverage**. Take a hundred people, give each their own
80% band, and count how many of their real salaries land inside their own band.
If the answer is 80, the model is calibrated. If the answer is 55, the model is
**over-confident**: the bands are too narrow, and every recruiter using them is
being told the market is tighter than it is. If the answer is 97, the model is
over-cautious: the bands are honest but so wide they give no advice.

Almost nobody plots this. It takes three functions, and it is the difference
between a project that reports a number and a project that audits itself.

Everything here is **model-agnostic** — plain arrays in, figure out. Nothing
imports the model, so these plots work equally on the quantile model, the
conformal-adjusted bands, a competitor's output, or a hand-built example in a
test.

The three plots:

* :func:`coverage_plot` — one confidence level. Promised vs actual, side by side.
* :func:`coverage_by_quantile` — every level at once, against the
  perfect-calibration diagonal. The diagnostic.
* :func:`band_width_plot` — is the band the right *shape*? A band equally wide
  for a junior and a CTO is a bad band even when its coverage is perfect.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .style import (
    AXIS,
    CRITICAL,
    GOOD,
    GRID,
    MUTED,
    SERIES,
    WARNING,
    new_axes,
    note,
    percent_formatter,
    rupee_axis,
    rupee_label,
)

#: How far empirical coverage may sit from the promise before we call it
#: mis-calibrated. Judgement call, stated once here rather than re-decided per
#: chart: 3 points either way. On 1,000 test rows the pure sampling wobble on an
#: 80% figure is about ±2.5 points, so a smaller tolerance would flag noise.
CALIBRATION_TOLERANCE = 0.03


def empirical_coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """What fraction of true salaries actually fell inside their own band?

    Deliberately reimplemented here in five lines rather than imported from the
    model package. These plots must stay usable on any pair of arrays — including
    ones invented in a test — and a plotting module that drags the model in
    behind it is a plotting module you cannot point at someone else's numbers.

    Boundaries count as inside, matching ``model.metrics.coverage``. The two
    definitions must agree, and there is a test that they do.
    """
    truth = np.asarray(y_true, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if not (truth.shape == low.shape == high.shape):
        raise ValueError(
            f"shape mismatch: y_true {truth.shape}, lower {low.shape}, upper {high.shape}"
        )
    if truth.size == 0:
        raise ValueError("cannot measure coverage on an empty set")
    if np.any(low > high):
        raise ValueError("some lower bounds sit above their upper bounds; the band is inverted")
    return float(np.mean((truth >= low) & (truth <= high)))


def _verdict(empirical: float, nominal: float, tolerance: float) -> tuple[str, str]:
    """Turn two numbers into a sentence and a colour.

    The sentence comes first and the colour second, always. Colour alone cannot
    tell a colourblind reader — or a greyscale printout, or anyone glancing at a
    thumbnail — that a model is lying to them.
    """
    gap = empirical - nominal
    if abs(gap) <= tolerance:
        return f"calibrated — within {tolerance:.0%} of the promise", GOOD
    if gap < 0:
        return (
            f"OVER-CONFIDENT by {abs(gap) * 100:.1f} points —\nthe band is too narrow",
            CRITICAL,
        )
    return (
        f"conservative by {gap * 100:.1f} points —\nthe band is honest but wide",
        WARNING,
    )


def coverage_plot(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    nominal: float = 0.8,
    ax: Axes | None = None,
    tolerance: float = CALIBRATION_TOLERANCE,
) -> tuple[Figure, Axes]:
    """Does an "80% band" actually contain 80% of true salaries?

    Two bars: what the model **promised**, and what it **delivered**. Same axis,
    same scale, stacked one above the other, because the entire content of this
    chart is one comparison and any layout that makes the reader hunt for it has
    failed.

    Read it in one second: if the lower bar is shorter than the upper one, the
    model is over-confident and the band is too narrow. The verdict is written
    on the chart in words, and the bar is coloured to match — words first,
    colour as reinforcement.

    Args:
        y_true: Actual salaries.
        lower: Lower edge of each person's band.
        upper: Upper edge of each person's band.
        nominal: The confidence the model claimed, as a fraction. 0.8 = "80%".
        ax: Draw into this axes instead of making a figure.
        tolerance: How far from the promise still counts as calibrated.

    Returns:
        ``(figure, axes)``.
    """
    if not 0.0 < nominal < 1.0:
        raise ValueError(f"nominal must be a fraction strictly between 0 and 1, got {nominal}")

    empirical = empirical_coverage(y_true, lower, upper)
    n = np.asarray(y_true, dtype=float).size
    verdict, colour = _verdict(empirical, nominal, tolerance)

    fig, ax = new_axes(ax, figsize=(7.0, 3.2))

    values = [empirical, nominal]
    labels = ["actual\n(measured)", "promised\n(nominal)"]
    colours = [colour, AXIS]
    bars = ax.barh(
        [0, 1],
        values,
        height=0.5,
        color=colours,
        # 2px of surface between the fills so the two bars never read as one.
        edgecolor="white",
        linewidth=2,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

    # A vertical rule at the promise, running through both bars, so the shortfall
    # is a visible distance rather than a subtraction the reader has to do.
    ax.axvline(nominal, color=MUTED, linewidth=1.2, linestyle="--", zorder=4)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.12)
    # Empty space below the lower bar, deliberately, so the verdict has somewhere
    # to sit inside the axes without landing on top of the data.
    ax.set_ylim(-1.5, 1.5)
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.set_xlabel("share of people whose true salary fell inside their own band")
    ax.set_title(f"Coverage of the {nominal:.0%} band  (n = {n:,})")
    ax.grid(axis="y", visible=False)
    note(ax, verdict, loc="lower left", color=colour)
    return fig, ax


def coverage_by_quantile(
    y_true: ArrayLike,
    bands: Mapping[float, tuple[ArrayLike, ArrayLike]],
    *,
    ax: Axes | None = None,
    show_uncertainty: bool = True,
) -> tuple[Figure, Axes]:
    """Sweep every confidence level and compare promise to reality.

    **How to read this plot, if you have never seen one.**

    The horizontal axis is what the model *promised*: 50%, 60%, ... 95%. The
    vertical axis is what it *delivered* — the fraction of true salaries that
    actually landed inside the band at that level. The dashed diagonal is
    ``delivered = promised``: a perfectly calibrated model draws its points
    exactly on that line.

    * **Points below the diagonal → the model is OVER-CONFIDENT.** It promised
      80% and delivered 62%. Its bands are too narrow. This is the dangerous
      failure: a recruiter is being handed a tighter range than the evidence
      supports, and will negotiate as if the uncertainty were smaller than it is.
    * **Points above the diagonal → the model is conservative.** It promised 80%
      and delivered 94%. Nobody is misled, but the bands are wider than they need
      to be, and a band from ₹8L to ₹60L is advice nobody can act on.
    * **Points on the diagonal at every level → calibrated.** This is what
      conformal prediction is *for*, and this plot is how you check that it
      worked rather than assuming it did.

    Args:
        y_true: Actual salaries. One array, shared by every level.
        bands: ``{nominal_level: (lower_array, upper_array)}``. Typically
            ``{0.5: (...), 0.6: (...), ..., 0.95: (...)}``. Keeping this a plain
            mapping is what keeps the plot model-agnostic — anything that can
            produce two arrays per level can be checked here.
        ax: Draw into this axes instead of making a figure.
        show_uncertainty: Draw ±1.96 standard errors on each point. Coverage is
            itself an estimate from a finite test set; without this, a reader
            will over-interpret a two-point wobble as mis-calibration.

    Returns:
        ``(figure, axes)``.
    """
    if not bands:
        raise ValueError("bands is empty — pass at least one nominal level")

    nominals = sorted(float(level) for level in bands)
    for level in nominals:
        if not 0.0 < level < 1.0:
            raise ValueError(f"nominal levels must be fractions in (0, 1), got {level}")

    empiricals = [empirical_coverage(y_true, *bands[level]) for level in nominals]
    n = np.asarray(y_true, dtype=float).size

    fig, ax = new_axes(ax, figsize=(6.0, 5.6))

    # The reference first, underneath everything: the line the points are being
    # judged against should never be drawn on top of the evidence.
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
    # Shade the half-plane that means "over-confident". The failure direction is
    # not symmetric in consequence, so the chart should not look symmetric.
    ax.fill_between([0, 1], [0, 0], [0, 1], color=GRID, alpha=0.55, linewidth=0, zorder=1)

    if show_uncertainty:
        # Standard error of a proportion: sqrt(p(1-p)/n). 1.96 of them is the
        # familiar 95% interval. Nothing exotic — but leaving it off invites the
        # reader to treat sampling noise as a finding.
        errors = [1.96 * np.sqrt(p * (1 - p) / n) for p in empiricals]
        ax.errorbar(
            nominals,
            empiricals,
            yerr=errors,
            fmt="none",
            ecolor=SERIES[0],
            elinewidth=1.2,
            capsize=3,
            alpha=0.6,
            zorder=3,
        )

    ax.plot(nominals, empiricals, color=SERIES[0], linewidth=2.0, zorder=4)
    ax.scatter(nominals, empiricals, color=SERIES[0], s=44, zorder=5)

    low = min([*nominals, *empiricals]) - 0.08
    high = max([*nominals, *empiricals]) + 0.08
    lo, hi = max(0.0, low), min(1.0, high)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(percent_formatter())
    ax.yaxis.set_major_formatter(percent_formatter())

    ax.set_xlabel("promised coverage — what the model claimed")
    ax.set_ylabel("actual coverage — what really happened")
    ax.set_title(f"Calibration of the bands  (n = {n:,})")
    note(
        ax,
        "dashed line = perfect calibration\n"
        "below it = OVER-CONFIDENT, bands too narrow\n"
        "above it = conservative, bands too wide",
        loc="upper left",
    )
    return fig, ax


def band_width_plot(
    prediction: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    ax: Axes | None = None,
    relative: bool = False,
    n_bins: int = 10,
) -> tuple[Figure, Axes]:
    """How does band width change with predicted salary? It should change.

    Coverage says the band is *honest*. This says whether it is *useful*.

    A band of ±₹4L is a reasonable answer for someone the model puts at ₹18L and
    a meaningless one for someone it puts at ₹1.2Cr — senior pay genuinely varies
    more, in rupees, than junior pay does. If this plot comes out flat, the model
    is applying one uncertainty to everybody, which means it is over-confident at
    the top of the market and over-cautious at the bottom, while possibly scoring
    perfect *average* coverage and hiding both.

    The cloud is one dot per person; the line is the median width within each
    decile of predicted salary. The line is the finding, the cloud is the
    evidence that the line is not an artefact of a handful of points.

    Args:
        prediction: The point prediction (band midpoint) per person, in rupees.
        lower: Lower band edge per person.
        upper: Upper band edge per person.
        ax: Draw into this axes instead of making a figure.
        relative: Plot width as a *fraction of the prediction* instead of in
            rupees. Worth flipping on: a band that widens in rupees but is flat
            in percentage terms is behaving exactly as a log-space model should,
            and this is how you tell the two apart. (Two separate charts, never
            two y-axes on one — a dual-axis chart invents relationships.)
        n_bins: Number of prediction bins for the median line.

    Returns:
        ``(figure, axes)``.
    """
    pred = np.asarray(prediction, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if not (pred.shape == low.shape == high.shape):
        raise ValueError(
            f"shape mismatch: prediction {pred.shape}, lower {low.shape}, upper {high.shape}"
        )
    if np.any(high < low):
        raise ValueError("some lower bounds sit above their upper bounds; the band is inverted")

    keep = np.isfinite(pred) & np.isfinite(low) & np.isfinite(high) & (pred > 0)
    pred, low, high = pred[keep], low[keep], high[keep]
    if pred.size == 0:
        raise ValueError("no finite, positive predictions to plot")

    width = high - low
    y = width / pred if relative else width

    fig, ax = new_axes(ax)
    ax.scatter(pred, y, s=10, color=SERIES[0], alpha=0.28, linewidth=0, zorder=2)

    # Median width per decile of prediction. Quantile bins rather than equal-width
    # bins, so every point on the line rests on the same number of people —
    # equal-width bins would put three people in the top bin and draw a confident
    # line through them.
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    if edges.size > 2:
        centres, median_widths = [], []
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            in_bin = (pred >= start) & (pred <= end if end == edges[-1] else pred < end)
            if in_bin.sum() >= 3:
                centres.append(float(np.median(pred[in_bin])))
                median_widths.append(float(np.median(y[in_bin])))
        if centres:
            ax.plot(centres, median_widths, color=SERIES[1], linewidth=2.4, zorder=4)
            ax.scatter(centres, median_widths, color=SERIES[1], s=34, zorder=5)
            ratio = median_widths[-1] / median_widths[0] if median_widths[0] else float("nan")
            flat = "a flat band is a bad band" if ratio < 1.2 else "band widens with salary — good"
            note(
                ax,
                f"orange = median width per decile\n"
                f"widest decile ÷ narrowest = {ratio:.1f}×\n{flat}",
                loc="upper left",
            )

    rupee_axis(ax, "x")
    if relative:
        ax.yaxis.set_major_formatter(percent_formatter())
        ax.set_ylabel("band width as a share of the prediction")
    else:
        rupee_axis(ax, "y")
        ax.set_ylabel("band width (upper − lower)")
    ax.set_xlabel("predicted salary")
    median_label = rupee_label(float(np.median(width)))
    ax.set_title(f"Band width vs predicted salary  (median width {median_label})")
    return fig, ax
