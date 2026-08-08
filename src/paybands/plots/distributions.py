"""What the salary data actually looks like — and why we take its logarithm.

Two claims in the README depend on the pictures in this file, and neither claim
should be taken on trust:

1. **"Salaries are skewed, so we model log(salary)."** Anyone can assert that.
   :func:`salary_distribution_comparison` shows it: the raw histogram has a long
   right tail (skew **+2.87** on the survey extract), and the same data on a log
   axis is very nearly symmetric (skew **−0.23**). That pair of numbers is the
   justification, and it is printed on the chart rather than in a caption.

2. **"Experience is the strongest signal, but it flattens."**
   :func:`salary_by_experience` shows the curve — *and shows how many people are
   behind each point*, which is the part almost every version of this chart
   leaves out. On our data the ``20+`` bucket holds only about **36 people**, and
   its flat median is far more likely to be small-sample noise than a finding
   about senior pay. A chart that hides ``n`` invites exactly that misreading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from ..data.schema import TARGET
from ..features.experience import BUCKET_LABELS, bucket_years
from .style import (
    MUTED,
    SERIES,
    WARNING,
    WIDE_FIGSIZE,
    new_axes,
    note,
    rupee_axis,
    rupee_label,
    subplots,
)

#: Below this many people in a bucket, the median is more noise than signal and
#: the chart says so out loud. 50 is a judgement call, not a law — it is roughly
#: where the bootstrap band on this data stops being narrow enough to argue from.
THIN_BUCKET_N = 50


def skewness(values: ArrayLike) -> float:
    """Fisher–Pearson skewness: how lopsided a distribution is.

    Zero means symmetric. Positive means a long tail to the **right** — a few
    very large values dragging the mean above the median, which is precisely the
    shape of a salary distribution. Negative means the tail is on the left.

    Written out in numpy rather than imported from scipy for one reason: it is
    three lines of arithmetic and seeing them makes the number stop being
    magic. It is the average cubed deviation, divided by the cubed standard
    deviation. Cubing is what makes it lopsidedness rather than spread — cubing
    keeps the sign, so a value far to the right and a value far to the left do
    not cancel out the way they would if we squared them.

    Rules of thumb: |skew| under 0.5 is roughly symmetric, over 1 is strongly
    skewed. Our raw salaries score +2.87.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("skewness needs at least three values")
    deviation = x - x.mean()
    spread = x.std()
    if spread == 0:
        raise ValueError("skewness is undefined when every value is identical")
    return float(np.mean(deviation**3) / spread**3)


def _clean_salaries(salaries: ArrayLike) -> np.ndarray:
    """Drop NaNs and non-positive values, and refuse an empty result.

    Non-positive salaries have to go before anything log-shaped happens, and
    dropping them silently inside the plot would mean the histogram quietly
    describes a different population than the one you passed in. So: drop, then
    say how many, via the caller.
    """
    values = np.asarray(salaries, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("no positive, finite salaries to plot")
    return values


def salary_distribution(
    salaries: ArrayLike,
    *,
    log: bool = False,
    ax: Axes | None = None,
    bins: int = 40,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of salaries, raw or on a log axis, with the skew annotated.

    **What ``log=True`` actually does.** It does not plot ``log(salary)`` in log
    units — an axis reading "13.2" helps nobody. It keeps the axis in rupees and
    spaces it logarithmically, so ₹5L→₹10L takes the same width as ₹10L→₹20L.
    The bin edges are spaced the same way. That is exactly what modelling
    ``log(salary)`` does to the target, drawn: *equal multiplicative steps get
    equal room*, which is how pay actually moves ("a 30% raise", never "a ₹2.4L
    raise").

    The mean and median are both marked. On the raw chart they are far apart and
    the mean sits well to the right — that gap *is* the skew, and it is why an
    average salary is a misleading summary of a salary distribution.

    Args:
        salaries: Annual salaries in rupees. Non-positive and missing values are
            dropped.
        log: Log-spaced axis and bins instead of linear.
        ax: Draw into this axes instead of making a figure.
        bins: Number of histogram bins.
        title: Override the default title.

    Returns:
        ``(figure, axes)``.
    """
    values = _clean_salaries(salaries)
    fig, ax = new_axes(ax)

    # The skew we report is the skew of the thing being plotted: raw rupees on
    # the linear chart, log rupees on the log chart. Reporting the raw skew on
    # both panels would make the log panel look like it changed nothing.
    skew = skewness(np.log(values) if log else values)

    if log:
        edges = np.logspace(np.log10(values.min()), np.log10(values.max()), bins + 1)
    else:
        edges = np.linspace(values.min(), values.max(), bins + 1)

    ax.hist(
        values,
        bins=edges,
        color=SERIES[1] if log else SERIES[0],
        # A hairline gap in the surface colour between bars, so adjacent bars
        # read as separate counts rather than one continuous blob.
        edgecolor="white",
        linewidth=0.6,
    )
    if log:
        ax.set_xscale("log")
    rupee_axis(ax, "x")

    # The mean and the median, as two rules through the histogram. On raw
    # salaries they are far apart and the mean sits to the right — that distance
    # IS the skew, drawn.
    #
    # They are labelled in the note rather than beside their own lines: on a
    # skewed distribution the two lines can land almost on top of each other (log
    # panel) or hard against the left edge (raw panel), and a direct label that
    # sometimes overlaps its neighbour is worse than a small legend that never
    # does. Solid = median, dashed = mean, said once in the note.
    median = float(np.median(values))
    mean = float(np.mean(values))
    ax.axvline(median, color=MUTED, linewidth=1.4, linestyle="-", zorder=3)
    ax.axvline(mean, color=MUTED, linewidth=1.4, linestyle="--", zorder=3)

    shape = "near-symmetric" if abs(skew) < 0.5 else "long right tail"
    note(
        ax,
        f"skew {skew:+.2f} — {shape}\n"
        f"solid line = median {rupee_label(median)}\n"
        f"dashed line = mean {rupee_label(mean)}",
        loc="upper right" if not log else "upper left",
    )

    ax.set_xlabel("annual salary" + (" (log-spaced axis)" if log else ""))
    ax.set_ylabel("people")
    ax.set_title(title or ("Salary — log axis" if log else "Salary — raw"))
    return fig, ax


def salary_distribution_comparison(
    salaries: ArrayLike,
    *,
    bins: int = 40,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Raw and logged, side by side. **The visual case for the log transform.**

    This is one chart doing one job: showing that a distribution which breaks
    averages on the left becomes a well-behaved, roughly bell-shaped one on the
    right, purely by changing how the axis is spaced. Nothing was removed, no
    outliers were trimmed — the same 1,022 people appear in both panels.

    Note that this function is only six lines of its own. It builds the figure,
    then calls :func:`salary_distribution` twice with an ``ax`` argument. That is
    the whole payoff of every plotting function accepting an ``ax``.
    """
    fig, axes = subplots(1, 2, figsize=figsize or WIDE_FIGSIZE)
    salary_distribution(salaries, log=False, ax=axes[0], bins=bins)
    salary_distribution(salaries, log=True, ax=axes[1], bins=bins)
    fig.suptitle(
        "Same salaries, two axes — why the model predicts log(salary)",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    return fig, (axes[0], axes[1])


# ────────────────────────────────────────── the experience curve, with n


def bootstrap_median_ci(
    values: ArrayLike,
    *,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """How uncertain is this median? Answered by resampling.

    **The idea, in one paragraph.** We have 36 salaries for the ``20+`` bucket
    and we computed their median. If we had asked 36 *different* veterans, we
    would have got a different median — but how different? We cannot go and ask
    them, so we do the next best thing: draw 36 salaries *from the 36 we have*,
    with replacement, and take that median. Do it a thousand times and the spread
    of those thousand medians is an honest estimate of how much our one median
    could have wobbled. That is the **bootstrap**, and it needs no assumption
    about the shape of the distribution — which matters here, because salary
    distributions are exactly the shape that breaks the textbook formulas.

    The band it produces widens automatically when a bucket is thin. That is the
    entire reason this plot uses it rather than showing the interquartile range:
    the IQR describes how *spread out people are*, which stays wide even with
    100,000 rows. This describes how *unsure we are about the middle*, which is
    the question the reader is actually asking.

    Args:
        values: The sample.
        confidence: 0.95 gives the 2.5th–97.5th percentile of the resampled
            medians.
        n_boot: How many resamples. 1000 is plenty for a plot.
        seed: Fixed so the same data always draws the same band. A chart that
            moves between runs is a chart nobody can review.

    Returns:
        ``(low, high)``. ``(nan, nan)`` when there is nothing to resample.
    """
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    draws = rng.choice(sample, size=(n_boot, sample.size), replace=True)
    medians = np.median(draws, axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(medians, tail)), float(np.quantile(medians, 1.0 - tail))


def salary_by_experience(
    df: pd.DataFrame,
    *,
    salary_col: str = TARGET,
    years_col: str = "years_experience",
    ax: Axes | None = None,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
    thin_n: int = THIN_BUCKET_N,
) -> tuple[Figure, Axes]:
    """Median salary per experience bucket, with uncertainty **and sample size**.

    Three things are on this chart, and the third is the one that makes it
    honest:

    * the **median** salary in each experience band — the headline curve,
      steep early and flat late, which is the finding;
    * a **bootstrap confidence band** around each median (see
      :func:`bootstrap_median_ci`) — how much that point could have moved had we
      surveyed a different sample of the same size;
    * the **number of people** in each bucket, printed under its tick.

    Why ``n`` is not optional here. On the survey extract the ``20+`` bucket
    contains about 36 people and its median lands within a rounding error of the
    ``13–20`` bucket. The tempting read is "senior pay plateaus after 20 years."
    The honest read is "we asked 36 people and cannot tell." Those are different
    claims and only the chart's sample sizes separate them — so buckets below
    ``thin_n`` are drawn with a hollow marker and called out in the note.

    Args:
        df: Any frame in the common schema — survey, synthetic, or company data.
        salary_col: Salary column. Defaults to the schema's target.
        years_col: Raw years of experience. Ignored if the frame already carries
            an ``experience_bucket`` column.
        ax: Draw into this axes instead of making a figure.
        confidence: Width of the bootstrap band.
        n_boot: Bootstrap resamples per bucket.
        seed: Bootstrap seed, so the band is reproducible.
        thin_n: Buckets with fewer rows than this are flagged as unreliable.

    Returns:
        ``(figure, axes)``.
    """
    if salary_col not in df.columns:
        raise ValueError(f"no {salary_col!r} column; got {list(df.columns)}")

    # Reuse the model's own bucket definition rather than inventing edges here.
    # If the plot binned experience differently from the features, the chart
    # would be describing a dataset the model never saw — which is the quiet way
    # a "supporting figure" ends up supporting nothing.
    if "experience_bucket" in df.columns:
        buckets = df["experience_bucket"]
    elif years_col in df.columns:
        buckets = bucket_years(df[years_col])
    else:
        raise ValueError(f"need either an 'experience_bucket' or a {years_col!r} column")

    salary = pd.to_numeric(df[salary_col], errors="coerce")
    frame = pd.DataFrame({"bucket": pd.Series(buckets).to_numpy(), "salary": salary.to_numpy()})
    frame = frame[frame["salary"].notna() & frame["bucket"].notna()]

    labels: list[str] = []
    counts: list[int] = []
    medians: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for label in BUCKET_LABELS:
        values = frame.loc[frame["bucket"] == label, "salary"].to_numpy()
        if values.size == 0:
            continue  # a bucket nobody falls into is left off, not drawn as zero
        low, high = bootstrap_median_ci(values, confidence=confidence, n_boot=n_boot, seed=seed)
        labels.append(label)
        counts.append(int(values.size))
        medians.append(float(np.median(values)))
        lows.append(low)
        highs.append(high)

    if not labels:
        raise ValueError("no rows land in any experience bucket")

    fig, ax = new_axes(ax)
    x = np.arange(len(labels))
    thin = np.array(counts) < thin_n

    ax.fill_between(x, lows, highs, color=SERIES[0], alpha=0.16, linewidth=0)
    ax.plot(x, medians, color=SERIES[0], linewidth=2.0, zorder=3)

    # Solid markers where the data supports the point; hollow where it does not.
    # Shape, not just colour, so the distinction survives a greyscale printout
    # and a colourblind reader.
    ax.scatter(
        x[~thin],
        np.array(medians)[~thin],
        color=SERIES[0],
        s=42,
        zorder=4,
    )
    ax.scatter(
        x[thin],
        np.array(medians)[thin],
        facecolor="white",
        edgecolor=WARNING,
        linewidth=2.0,
        s=52,
        zorder=4,
    )

    ax.set_xticks(x)
    # The sample size rides along with the tick label. It cannot be cropped off,
    # scrolled past, or left out of the screenshot.
    ax.set_xticklabels([f"{label}\nn={n:,}" for label, n in zip(labels, counts, strict=True)])
    rupee_axis(ax, "y")
    ax.set_xlabel("years of professional experience")
    ax.set_ylabel("median salary")
    ax.set_title("Salary by experience — with the sample size behind each point")

    thin_labels = [label for label, is_thin in zip(labels, thin, strict=True) if is_thin]
    lines = [f"shaded band = {confidence:.0%} bootstrap interval on the median"]
    if thin_labels:
        lines.append(
            f"hollow marker: fewer than {thin_n} people ({', '.join(thin_labels)}) —\n"
            "treat the level as noise, not a finding"
        )
    note(ax, "\n".join(lines), loc="upper left")
    return fig, ax
