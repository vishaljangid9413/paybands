"""Seeing bias, rather than asserting it.

Two plots, and between them they carry the most sensitive claim the project
makes. Both are deliberately **model-agnostic** — they take plain arrays of
numbers, and import nothing from ``paybands.model`` or ``paybands.fairness``.
That is not architectural fussiness. A fairness chart that only works on our own
audit's output is a chart nobody can use to check our audit.

* :func:`residual_by_group` — the model's *errors*, split by group. A model can
  be accurate on average and still be wrong in one direction for one group
  every single time. Averages hide that; this shows it.
* :func:`gap_comparison` — the raw pay gap next to the adjusted one, with a
  confidence interval. **A raw gap is not evidence of discrimination**, and the
  plot's job is to make that distinction impossible to skip past.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .style import (
    AXIS,
    CRITICAL,
    GOOD,
    MUTED,
    SECONDARY_INK,
    SERIES,
    WARNING,
    new_axes,
    note,
    percent_formatter,
    rupee_axis,
    rupee_label,
)


def residual_by_group(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    group: ArrayLike,
    *,
    ax: Axes | None = None,
    relative: bool = False,
    min_n: int = 30,
) -> tuple[Figure, Axes]:
    """The model's errors, split by group. **The visual signature of bias.**

    A **residual** is what the model got wrong for one person::

        residual = actual salary − predicted salary

    Positive means the model *under-predicted* them: they earn more than it
    thought. Negative means it over-predicted.

    Across everybody, residuals should scatter around zero — that is what "the
    model is unbiased" means, and it is usually true by construction because the
    training process makes it true. **The question this plot asks is different:
    is it true within each group?**

    If one group's whole box sits below zero, the model systematically predicts
    too high for them; if it sits above, too low. Either is one-sided error
    against a specific group, and it is what bias looks like *before* anyone
    computes a statistic. It survives the "but the overall MAE is fine" defence,
    because the overall MAE is exactly what is hiding it: two equal and opposite
    group errors average to zero.

    **What this plot does not prove.** A group difference here can come from a
    genuine difference in the group's composition — different roles, different
    cities, different experience — that the model has represented correctly.
    Residual asymmetry is a *flag*, not a verdict. The verdict needs the adjusted
    comparison in :func:`gap_comparison`.

    Args:
        y_true: Actual salaries, in rupees.
        y_pred: Predicted salaries, in rupees. Same length.
        group: A group label per person — gender, age band, location tier,
            anything. Plain array or Series of labels.
        ax: Draw into this axes instead of making a figure.
        relative: Show the residual as a fraction of the true salary instead of
            in rupees. Worth using: being ₹3L wrong about ₹40L and ₹3L wrong
            about ₹6L are very different mistakes, and a rupee-scale box plot
            makes the high earners' errors dominate the picture.
        min_n: Groups smaller than this get a warning label. A one-sided box
            drawn from eleven people is not a finding.

    Returns:
        ``(figure, axes)``.
    """
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    labels = np.asarray(group, dtype=object)
    if not (truth.shape == pred.shape == labels.shape):
        raise ValueError(
            f"shape mismatch: y_true {truth.shape}, y_pred {pred.shape}, group {labels.shape}"
        )
    if truth.size == 0:
        raise ValueError("cannot plot residuals for an empty set")

    residual = truth - pred
    if relative:
        if np.any(truth == 0):
            raise ValueError("relative residuals divide by the true salary; one of them is 0")
        residual = residual / truth

    keep = np.isfinite(residual)
    residual, labels = residual[keep], labels[keep]

    # Sorted rather than first-seen, so re-running on a shuffled frame draws the
    # same chart. A figure whose category order moves between runs is a figure
    # you cannot diff against last week's.
    names = sorted({str(label) for label in labels})
    per_group = [residual[np.array([str(label) for label in labels]) == name] for name in names]

    fig, ax = new_axes(ax)
    positions = np.arange(len(names))

    boxes = ax.boxplot(
        per_group,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        # Outliers stay visible but recede — hiding them would be hiding data,
        # letting them shout would hide the boxes.
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": MUTED,
            "markeredgecolor": "none",
            "alpha": 0.4,
        },
        medianprops={"color": "white", "linewidth": 2.0},
        whiskerprops={"color": AXIS, "linewidth": 1.0},
        capprops={"color": AXIS, "linewidth": 1.0},
    )
    # One colour for every box on purpose. The groups are named on the axis, so
    # hue would be carrying no information the labels do not already carry — and
    # a rainbow here would quietly imply the groups are ranked.
    for patch in boxes["boxes"]:
        patch.set_facecolor(SERIES[0])
        patch.set_alpha(0.75)
        patch.set_edgecolor("none")

    # Zero: the line every box should straddle.
    ax.axhline(0.0, color=SECONDARY_INK, linewidth=1.4, zorder=1)

    # The group mean, marked separately from the median. The median says where
    # the typical error is; the mean is what a fairness metric will actually
    # report, and when the two disagree it is because a tail is doing the work.
    means = [float(np.mean(values)) if values.size else float("nan") for values in per_group]
    ax.scatter(positions, means, marker="D", s=34, color=SERIES[1], zorder=6)

    counts = [int(values.size) for values in per_group]
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{name}\nn={n:,}" for name, n in zip(names, counts, strict=True)])

    if relative:
        ax.yaxis.set_major_formatter(percent_formatter())
        ax.set_ylabel("residual as a share of actual salary")
    else:
        rupee_axis(ax, "y")
        ax.set_ylabel("residual  (actual − predicted)")
    ax.set_title("Model error by group — is it one-sided for anyone?")
    # Headroom at the top so the explanatory note is not sitting on a whisker.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 0.32 * (top - bottom))

    lines = [
        "orange diamond = group mean error",
        "a box sitting wholly above or below zero is",
        "systematic one-sided error for that group",
    ]
    thin = [name for name, n in zip(names, counts, strict=True) if n < min_n]
    if thin:
        lines.append(f"⚠ fewer than {min_n} people in: {', '.join(thin)}")
    note(ax, "\n".join(lines), loc="upper right")
    return fig, ax


def gap_comparison(
    raw_gap: float,
    adjusted_gap: float,
    ci: tuple[float, float],
    *,
    raw_ci: tuple[float, float] | None = None,
    known_gap: float | None = None,
    labels: tuple[str, str] = ("raw gap", "adjusted gap"),
    ax: Axes | None = None,
    rupees: bool = False,
) -> tuple[Figure, Axes]:
    """Raw gap vs adjusted gap, with a confidence interval. Read the docstring.

    **The distinction this plot exists to make.**

    The *raw gap* is the plain difference in average pay between two groups. It
    is real, it matters, and **it is not by itself evidence of discrimination**,
    because it does not hold anything else constant. If one group is on average
    newer to the industry, or more concentrated in lower-paying roles or cities,
    a raw gap appears without anyone ever having been paid unfairly for the same
    work.

    The *adjusted gap* is what remains after comparing like with like — same
    experience, same role, same location. That is the number that speaks to
    "equal pay for equal work". It is also the harder number, because whether you
    have controlled for the right things is a judgement, not a calculation.

    Both belong on the chart, next to each other, because **either one alone
    misleads**. The raw gap alone overstates the case. The adjusted gap alone
    hides a real and important fact — that the groups are not distributed
    equally across the well-paid roles in the first place, which is its own
    problem even if every individual pay decision was defensible.

    **Why the interval is not optional.** An adjusted gap of 2% with an interval
    running from −3% to +7% is a measurement that has not ruled out zero. Drawn
    as a bare bar it would look like a finding. Drawn with its interval crossing
    the zero line, it looks like what it is.

    Args:
        raw_gap: The unadjusted gap, as a fraction. 0.08 means the disadvantaged
            group is paid 8% less. Use ``rupees=True`` if you are passing rupee
            amounts instead.
        adjusted_gap: The gap after controls, same units.
        ci: ``(low, high)`` confidence interval for the **adjusted** gap.
        raw_ci: Optional interval for the raw gap.
        known_gap: The true gap, when you know it. On synthetic data you do — the
            generator injected it — and drawing it here is how the fairness audit
            gets validated against ground truth rather than against a hunch.
        labels: Names for the two estimates.
        ax: Draw into this axes instead of making a figure.
        rupees: Format the axis in lakhs/crores instead of percent.

    Returns:
        ``(figure, axes)``.
    """
    low, high = float(ci[0]), float(ci[1])
    if low > high:
        raise ValueError(f"ci is inverted: low {low} is above high {high}")
    if not low <= adjusted_gap <= high:
        raise ValueError(
            f"adjusted_gap {adjusted_gap} sits outside its own interval ({low}, {high}) — "
            "one of the two is being computed on a different scale"
        )

    fig, ax = new_axes(ax, figsize=(7.0, 3.4))

    estimates = [float(raw_gap), float(adjusted_gap)]
    intervals = [raw_ci, (low, high)]
    positions = [1, 0]  # raw on top, adjusted below — reading order is downward

    for position, estimate, interval, colour in zip(
        positions, estimates, intervals, (MUTED, SERIES[0]), strict=True
    ):
        if interval is not None:
            ax.plot(
                [interval[0], interval[1]],
                [position, position],
                color=colour,
                linewidth=2.4,
                solid_capstyle="butt",
                zorder=3,
            )
            for edge in interval:
                ax.plot(
                    [edge, edge], [position - 0.09, position + 0.09], color=colour, linewidth=2.0
                )
        ax.scatter([estimate], [position], color=colour, s=90, zorder=5)
        fmt = rupee_label(estimate) if rupees else f"{estimate:+.1%}"
        ax.text(
            estimate,
            position + 0.22,
            fmt,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=colour,
        )

    # Zero: "no gap". Solid and dark, because it is the hypothesis every interval
    # on this chart is being tested against.
    ax.axvline(0.0, color=SECONDARY_INK, linewidth=1.4, zorder=2)

    if known_gap is not None:
        ax.axvline(float(known_gap), color=WARNING, linestyle="--", linewidth=1.6, zorder=2)
        ax.text(
            float(known_gap),
            1.5,
            f" injected truth {known_gap:+.1%}"
            if not rupees
            else f" truth {rupee_label(known_gap)}",
            ha="left",
            va="center",
            fontsize=9,
            color=WARNING,
        )

    # `positions` is [1, 0] — raw on top, adjusted below — so the tick labels go
    # in that same order rather than the natural one. The extra room underneath
    # is where the verdict sits, clear of both intervals.
    ax.set_yticks(positions)
    ax.set_yticklabels([labels[0], labels[1]])
    ax.set_ylim(-1.7, 1.9)
    if rupees:
        rupee_axis(ax, "x")
        ax.set_xlabel("pay gap")
    else:
        ax.xaxis.set_major_formatter(percent_formatter(decimals=0))
        ax.set_xlabel("pay gap  (negative = the group is paid less)")
    ax.set_title("Raw vs adjusted pay gap")
    ax.grid(axis="y", visible=False)

    crosses_zero = low <= 0.0 <= high
    verdict = (
        "the adjusted interval crosses zero —\nno gap demonstrated once like is compared with like"
        if crosses_zero
        else "the adjusted interval excludes zero —\na gap remains after controlling for role,\n"
        "experience and location"
    )
    note(ax, verdict, loc="lower right", color=GOOD if crosses_zero else CRITICAL)
    return fig, ax
