"""One prediction, taken apart — the chart version of `explain/shapley.py`.

The claim this plot exists to prove: **"the model can show its working."**

## The one design decision, and it is the whole plot

A **waterfall** chart shows a starting value, a run of steps that push it up and
down, and where it lands. It is the natural shape for SHAP — but drawn naively in
rupees it would be **wrong here**, and quietly so.

The model works in ``log(salary)``. Contributions add up in log units and
*multiply* in rupees, so a rupee waterfall's steps do not telescope to the
prediction. Anyone building one has to fudge something to make the bars meet, and
`explain/shapley.py` (Idea 2) is explicit that fudging is the one thing not to do.

So the bars are drawn on a **log-scaled rupee axis**. Each step multiplies the
running total, the last one lands exactly on the prediction, and the arithmetic
is honest with nothing adjusted. The reading rule that falls out is the useful
one anyway: **equal bar lengths are equal percentage effects**, which is how pay
actually moves — "a 20% bump" means the same thing to a fresher and to a
principal engineer, "₹4L" does not.

The axis still carries lakh/crore labels (`style.rupee_axis`), so the reader
sees rupees while the geometry stays multiplicative.

## What the labels say, and why two numbers

Each bar is labelled ``×1.21  ≈ +₹4.1L``.

* ``×1.21`` is exact and matches the bar you are looking at.
* ``≈ +₹4.1L`` is `Contribution.rupees` — the same approximate figure the API
  and the sentence report, so the chart and the JSON never disagree.

The note inside the axes says that the rupee figures do not sum. That sentence is
not optional: the chart is the artefact that gets screenshotted into a hiring
thread with none of the surrounding prose, and a stack of rupee bars is read as a
sum unless something on the image says otherwise.

Follows the three package rules (`plots/__init__.py`): Agg backend, an optional
``ax``, and no file written as a side effect.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator

from ..explain.shapley import Explanation
from .style import (
    INK,
    MUTED,
    SECONDARY_INK,
    SERIES,
    new_axes,
    note,
    rupee_axis,
    rupee_label,
)

#: Contributions drawn individually before the rest are pooled. Eight is about
#: what fits on one screen with readable labels — and `docs/findings.md` §4.1
#: found only nine features in this model move held-out error at all, so a chart
#: with twenty rows would be nineteen rows of noise around one real one.
DEFAULT_TOP_N = 8


def contribution_waterfall(
    explanation: Explanation,
    *,
    ax: Axes | None = None,
    top_n: int = DEFAULT_TOP_N,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Draw one prediction as a chain of multiplicative steps.

    Reads top to bottom: start at the baseline, apply the largest factor, then
    the next, and land on the prediction. Blue pushes the salary up, orange
    pushes it down.

    Args:
        explanation: An `explain.shapley.Explanation` — from
            `explain_prediction`, typically with the four experience encodings
            already merged into one row.
        ax: Draw into this axes instead of making a figure.
        top_n: How many contributions get their own bar. Everything smaller is
            pooled into a final "everything else" step, so the chain still lands
            exactly on the prediction rather than stopping short of it.
        figsize: Size of a new figure. Height defaults to whatever fits the
            number of bars — a fixed height either crushes ten labels together
            or leaves half the canvas empty for three.

    Returns:
        ``(figure, axes)``.

    Raises:
        ValueError: if ``top_n`` is not positive, or the explanation has no
            contributions to draw.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")
    if not explanation.contributions:
        raise ValueError("this explanation has no contributions to draw")

    # Contributions arrive sorted by size, so every meaningful one comes before
    # every negligible one and a single slice separates them. Anything below
    # `NEGLIGIBLE_LOG_CONTRIBUTION` — a sub-half-percent nudge — is pooled rather
    # than drawn: a bar labelled "×1.00" costs a row and says nothing. `max(1, …)`
    # keeps the chart drawable when *every* factor is that small, which is itself
    # worth seeing.
    n_real = sum(1 for c in explanation.contributions if c.direction != "no effect")
    cut = max(1, min(top_n, n_real))
    shown = list(explanation.contributions[:cut])
    rest = list(explanation.contributions[cut:])

    labels = [c.phrase for c in shown]
    # Log units, because that is the space the steps are exact in. Everything
    # below is `exp` of a running sum of these.
    steps = [c.log_contribution for c in shown]
    multipliers = [c.multiplier for c in shown]
    rupees: list[float | None] = [c.rupees for c in shown]

    if rest:
        pooled = float(sum(c.log_contribution for c in rest))
        labels.append(f"everything else ({len(rest)} factors)")
        steps.append(pooled)
        multipliers.append(float(np.exp(pooled)))
        # No rupee label on the pooled bar: it is a bag of unrelated features,
        # and "≈ +₹40,000" against a row nobody can act on is clutter.
        rupees.append(None)

    # The running total after each step, starting at the baseline. `cumsum` in
    # log space then one `exp` — the last value is the prediction, exactly.
    edges = np.exp(explanation.baseline_log + np.concatenate([[0.0], np.cumsum(steps)]))

    height = figsize or (7.6, max(3.4, 1.7 + 0.46 * len(labels)))
    fig, ax = new_axes(ax, figsize=height)

    # Top of the chart is the largest factor, so y runs downward.
    positions = np.arange(len(labels))
    for y, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        rising = end >= start
        ax.barh(
            y,
            width=end - start,
            left=start,
            height=0.62,
            color=SERIES[0] if rising else SERIES[1],
            alpha=0.85,
            edgecolor="none",
            zorder=3,
        )
        # A thin connector down to the next bar's starting edge, so the eye
        # follows the chain rather than reading the bars as independent.
        if y + 1 < len(labels):
            ax.plot(
                [end, end],
                [y + 0.31, y + 0.69],
                color=MUTED,
                linewidth=0.9,
                zorder=2,
            )

        pieces = [f"×{multipliers[y]:.2f}"]
        if rupees[y] is not None:
            pieces.append(f"≈ {'+' if rising else '−'}{rupee_label(abs(rupees[y]))}")
        ax.text(
            end,
            y,
            "  " + "  ".join(pieces) if rising else "  ".join(pieces) + "  ",
            ha="left" if rising else "right",
            va="center",
            fontsize=9,
            color=SECONDARY_INK,
            zorder=5,
        )

    # The two anchors. Baseline is quiet (it is context); the prediction is the
    # answer, so it is the darkest line on the chart.
    ax.axvline(edges[0], color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.axvline(edges[-1], color=INK, linewidth=1.6, zorder=4)

    ax.set_xscale("log")
    # matplotlib's default log ticks sit on the decades — ₹10L, then ₹1Cr. A
    # band chart usually spans well under one decade, so the default would leave
    # the axis with no labels at all. These subdivisions put a labelled tick
    # every ₹5L-ish wherever the data happens to sit.
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    rupee_axis(ax, "x")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    # Inverted, so the largest factor is the top row. The extra room underneath
    # is where the note sits, clear of the smallest bars.
    ax.set_ylim(len(labels) + 1.1, -0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("predicted midpoint  (log scale — equal widths are equal percentages)")
    ax.set_title(
        f"Why {rupee_label(explanation.prediction)}?  "
        f"starting from {rupee_label(explanation.baseline)} for a typical candidate"
    )

    # Headroom, then the sentence that keeps the chart honest on its own.
    low, high = ax.get_xlim()
    ax.set_xlim(low / 1.35, high * 1.55)
    note(
        ax,
        "blue = pushes pay up, orange = pushes it down\n"
        "× factors are exact and multiply to the prediction;\n"
        "the ≈₹ figures overlap and do NOT add up",
        loc="lower right",
    )
    return fig, ax
