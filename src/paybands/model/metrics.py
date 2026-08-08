"""How we score a model — always in rupees, never in log space.

There is one rule in this file that everything else follows from:

    **Train in log space if you like. Report in rupees, always.**

The model will be trained on ``log(salary)`` because salaries have a long right
tail (most people cluster low, a few earn enormously) and logs tame that. But a
score of "MSE 0.083" in log space means nothing to anyone. "Typically off by
₹2.1 lakh" means something to a recruiter, to your manager, and to you. So every
number this module produces is a rupee amount or a percentage.

What's in here:

* the four headline scores — MAE, median absolute error, MAPE, R²
* :func:`from_log`, which undoes the log transform, plus a warning about the
  subtlety hiding inside it
* :func:`pinball_loss` and :func:`coverage`, which we don't need until Phase 3
  but which are the *only* honest way to score a band rather than a point
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ─────────────────────────────────────────────── rupee formatting


def format_rupees(amount: float) -> str:
    """Format a number the way an Indian payslip does: ₹21,34,500.

    Indian digit grouping is not the Western one. The last three digits group
    together, then every *two* digits after that — so ten lakh is ₹10,00,000,
    not ₹1,000,000. Getting this right costs eight lines and is the difference
    between a tool that looks local and one that looks imported.
    """
    sign = "-" if amount < 0 else ""
    whole = f"{abs(amount):.0f}"

    if len(whole) <= 3:
        return f"{sign}₹{whole}"

    last_three = whole[-3:]
    rest = whole[:-3]
    # Walk the remaining digits from the right in pairs.
    pairs = []
    while len(rest) > 2:
        pairs.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        pairs.insert(0, rest)
    return f"{sign}₹{','.join([*pairs, last_three])}"


# ─────────────────────────────────────────────── the log transform


def to_log(salary: ArrayLike) -> NDArray[np.float64]:
    """Rupees → log rupees. What the model actually learns to predict."""
    y = np.asarray(salary, dtype=float)
    if np.any(y <= 0):
        raise ValueError("log of a non-positive salary is undefined; clean the data first")
    return np.log(y)


def from_log(log_salary: ArrayLike) -> NDArray[np.float64]:
    """Log rupees → rupees. Run this on predictions before scoring them.

    ────────────────────────────────────────────────────────────────────────
    **The subtlety worth understanding before you use this.**

    ``exp`` and ``log`` cancel out perfectly for a single number: exp(log(x))
    is x. But they do *not* cancel out across an average.

    Take three salaries: ₹10L, ₹20L, ₹40L.

    * Their ordinary (arithmetic) mean is ₹23.3L.
    * Average their logs and exponentiate, and you get ₹20L — the **geometric
      mean**. Lower. Always lower, whenever the numbers differ at all.

    So a model trained on logs, whose predictions you exponentiate, is not
    estimating the average salary of people like this candidate. It is
    estimating something closer to their *typical* salary — the middle of the
    pack, unmoved by the one founder in the group earning ₹2Cr.

    For salary bands that is exactly what we want, and it is why this file
    reports **median** absolute error alongside the mean. But if you ever need
    a genuine expected value — "what will these 40 hires cost us in total?" —
    exponentiating is biased low, and the standard fix is Duan's smearing
    estimator. Know it exists; we don't need it here.
    ────────────────────────────────────────────────────────────────────────
    """
    return np.exp(np.asarray(log_salary, dtype=float))


# ─────────────────────────────────────────────── the four headline scores


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error — the average miss, in rupees.

    "On average we're off by this much." The most interpretable single number
    in the project, which is why it leads every report.
    """
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def median_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """The *typical* miss, in rupees.

    Report this next to MAE and compare them. MAE is dragged upward by a
    handful of catastrophic misses; the median ignores them. If MAE is ₹6L and
    the median miss is ₹2.5L, that gap is telling you the model is usually fine
    and occasionally very wrong — a completely different problem from being
    uniformly mediocre, and it needs a different fix.
    """
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.median(np.abs(yt - yp)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute *percentage* error, as a percentage.

    Being ₹3L wrong about a ₹40L salary is a small mistake. Being ₹3L wrong
    about a ₹6L salary is a disaster. MAE cannot tell those apart; MAPE can.

    Its known flaw: it punishes over-prediction more harshly than
    under-prediction, because the error is divided by the *true* value. Predict
    ₹20L for someone on ₹10L and MAPE says 100%; predict ₹10L for someone on
    ₹20L and it says 50% — the same factor-of-two error, scored differently.
    Which is why it never appears alone here.
    """
    yt, yp = _as_pair(y_true, y_pred)
    if np.any(yt == 0):
        raise ValueError("MAPE divides by the true value; a true salary of 0 makes it infinite")
    return float(np.mean(np.abs((yt - yp) / yt)) * 100)


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """R² — how much of the variation in salary the model explains, 0 to 1.

    Read it as a comparison: "how much better than always guessing the mean?"
    1.0 is perfect, 0.0 means no better than the mean, and negative means
    *worse* than the mean — which is possible and does happen.

    Treat it as a secondary number. It is unitless, so it can't tell a recruiter
    anything, and a bare R² with nothing to compare it against is the classic
    portfolio-project mistake the ROADMAP warns about. Beating the group-median
    baseline is a claim; "R² = 0.87" is not.

    **A real result from this project, worth understanding.** The global-median
    baseline scores R² = −0.131 on the survey data. Negative — worse than
    guessing. That is not a bug: R² measures squared error, and squared error is
    minimised by the *mean*, while the baseline deliberately predicts the
    *median*. On a right-skewed salary distribution those are far apart, so a
    predictor optimised for one metric loses on the other. The baseline is still
    the better model for our purpose — we care about the typical candidate, not
    about minimising squares. This is what "the metric encodes a choice" means
    in practice.
    """
    yt, yp = _as_pair(y_true, y_pred)
    ss_residual = float(np.sum((yt - yp) ** 2))
    ss_total = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_total == 0:
        raise ValueError("R² is undefined when every true value is identical")
    return 1.0 - ss_residual / ss_total


# ─────────────────────────────────────────────── scoring a BAND, not a point


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> float:
    """Score a quantile prediction. Lower is better. In rupees.

    Ready for Phase 3, where we stop predicting one number and start predicting
    the 10th and 90th percentiles that form the band.

    **Why an ordinary error metric can't score a quantile.** Suppose you're
    predicting the 90th percentile — a number that *should* sit above 90% of
    real salaries. MAE would score that as being wrong nine times out of ten and
    push it down toward the middle, destroying the thing you asked for.

    Pinball loss fixes this by charging asymmetrically. For the 90th percentile
    it charges 0.9 per rupee of under-prediction and only 0.1 per rupee of
    over-prediction — so the cheapest possible prediction is one that lands
    above the truth exactly 90% of the time. Which is the definition of the 90th
    percentile. The metric and the goal are the same thing.

    Args:
        quantile: Between 0 and 1. 0.5 recovers plain MAE, halved.
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be strictly between 0 and 1, got {quantile}")
    yt, yp = _as_pair(y_true, y_pred)
    error = yt - yp
    # Where we under-predicted (error > 0) charge `quantile`; where we
    # over-predicted charge `1 - quantile`. np.maximum picks whichever term is
    # positive, so this one line covers both cases.
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """What fraction of true salaries actually landed inside the predicted band?

    This is the honesty check, and it is about fifteen lines of code that almost
    nobody writes.

    If the model says "80% confident the salary is between ₹18L and ₹24L", then
    across a hundred people, roughly eighty of their real salaries should fall
    inside their own band. Run this and find out. If it returns 0.55, the model
    is overconfident — the bands are too narrow and it is lying to the
    recruiter. If it returns 0.97, the bands are so wide they're useless advice.

    Note what this does *not* measure: band width. A band of ₹0 to ₹5Cr scores
    perfect coverage and helps nobody. Always report coverage and typical band
    width together — coverage says the band is honest, width says it is useful.

    Returns:
        A fraction between 0 and 1. Compare it to the confidence you promised.
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError(f"shape mismatch: y_true {yt.shape}, lower {lo.shape}, upper {hi.shape}")
    if np.any(lo > hi):
        raise ValueError("some lower bounds are above their upper bounds; the band is inverted")
    # Inclusive on both ends: a salary landing exactly on the boundary counts as
    # covered. With continuous rupee amounts this almost never matters, but
    # leaving it undefined is how a test becomes flaky later.
    inside = (yt >= lo) & (yt <= hi)
    return float(np.mean(inside))


# ─────────────────────────────────────────────── the report object


@dataclass(frozen=True)
class MetricSet:
    """Every score for one model on one dataset, printable as a table.

    Built by :func:`evaluate`. Frozen because a results object that can be
    edited after the fact is a results object you can't trust.
    """

    n: int
    mae: float
    median_absolute_error: float
    mape: float
    r2: float
    label: str = "model"

    def __str__(self) -> str:
        return "\n".join(
            [
                f"{self.label}  (n = {self.n:,})",
                f"  MAE              {format_rupees(self.mae):>14}   average miss",
                f"  Median abs err   {format_rupees(self.median_absolute_error):>14}"
                "   typical miss",
                f"  MAPE             {self.mape:>13.1f}%   average miss as a share of salary",
                f"  R²               {self.r2:>14.3f}   share of variation explained",
            ]
        )


def evaluate(y_true: ArrayLike, y_pred: ArrayLike, *, label: str = "model") -> MetricSet:
    """Score predictions that are already in rupees."""
    yt, yp = _as_pair(y_true, y_pred)
    return MetricSet(
        n=len(yt),
        mae=mae(yt, yp),
        median_absolute_error=median_absolute_error(yt, yp),
        mape=mape(yt, yp),
        r2=r2(yt, yp),
        label=label,
    )


def evaluate_log_predictions(
    y_true_rupees: ArrayLike,
    y_pred_log: ArrayLike,
    *,
    label: str = "model",
) -> MetricSet:
    """Score a log-space model without ever showing a log-space number.

    This is the function the real model will use. It takes the truth in rupees
    and the predictions in logs, back-transforms the predictions, and scores in
    rupees.

    Deliberately there is no ``evaluate_in_log_space``. If scoring in logs were
    one import away, someone (you, at 1am, in week 3) would report "MSE 0.083"
    and nobody reading it would know whether that was good.
    """
    return evaluate(y_true_rupees, from_log(y_pred_log), label=label)


def _as_pair(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce to float arrays and check they line up.

    A length mismatch here is almost always a split gone wrong — predictions
    from one set scored against another's labels. numpy would broadcast some of
    those silently into a plausible-looking wrong answer, so we check first.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true is {yt.shape}, y_pred is {yp.shape}")
    if yt.size == 0:
        raise ValueError("cannot score an empty set of predictions")
    if np.any(np.isnan(yp)):
        raise ValueError("predictions contain NaN — the model failed on some rows")
    return yt, yp
