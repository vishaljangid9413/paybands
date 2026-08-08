"""The two deliberately stupid models the real model has to beat.

**Why bother building something you know is bad?**

Because "R² = 0.87" is not a result. It's a number with nothing behind it. A
result is "the gradient boosting model is off by ₹1.8L where a `groupby` is off
by ₹2.4L" — that's a 25% improvement, stated against something a sceptical
person can reproduce in four lines of pandas.

And sometimes the honest answer comes out the other way. If a thousand lines of
LightGBM only ties the lookup table, the model has not earned its complexity,
and the right engineering call is to ship the lookup table: it trains instantly,
never drifts in a way you can't see, and any HR person can read it. Knowing that
requires having built the lookup table first.

Two baselines, in order of stupidity:

``GlobalMedianBaseline``
    One number for everybody. The absolute floor. Nothing may score worse.

``GroupMedianBaseline``
    The median salary of people with the same role and roughly the same
    experience. No machine learning whatsoever — it is a `groupby` — and it is
    usually surprisingly hard to beat.

Both follow scikit-learn's ``fit`` / ``predict`` shape so that later, swapping
in the real model is a one-line change and every evaluation script keeps working.
They do not *inherit* from scikit-learn, though: it's an optional dependency
here, and a baseline that can't run because a 40MB library is missing defeats
the purpose.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

# ─────────────────────────────────────────────── experience buckets

#: (upper bound in years, label). A person falls in the first bucket whose upper
#: bound they do not exceed.
#:
#: **Why unequal widths.** The first years of a career are worth far more per
#: year than later ones — going from 1 to 3 years' experience moves your salary
#: much more than going from 15 to 17. Equal five-year buckets would lump the
#: steep part of that curve together and split the flat part apart, which is
#: exactly backwards. So the buckets are narrow early and wide late.
#:
#: These are a judgement call, not a fact. They match the ROADMAP's analysis,
#: and they live here in one visible list so anyone can disagree with them in
#: one place instead of hunting through code.
EXPERIENCE_BUCKETS: tuple[tuple[float, str], ...] = (
    (1, "0-1"),
    (3, "2-3"),
    (5, "4-5"),
    (8, "6-8"),
    (12, "9-12"),
    (20, "13-20"),
    (float("inf"), "20+"),
)

#: Where rows with a missing experience value go. They get their own bucket
#: rather than being dropped or filled with a guess: "we don't know" is a real
#: state, and pretending it's "0 years" would drag that bucket's median down.
UNKNOWN_BUCKET = "unknown"

#: A group needs at least this many people before we trust its median.
#:
#: **Why a minimum at all.** The median of two people is not a market rate, it
#: is those two people. If one of them is a founder who wrote down their equity,
#: every future Security hire gets quoted a fantasy number. Below the threshold
#: we deliberately throw the group away and fall back to a coarser, better-
#: populated estimate — a slightly less specific answer built on real evidence
#: beats a very specific answer built on noise.
DEFAULT_MIN_GROUP_SIZE = 10


def experience_bucket(years: float | None) -> str:
    """Turn a number of years into a bucket label."""
    # pd.isna rather than `is None` or np.isnan: missing values arrive here as
    # None, numpy NaN, or pandas' own pd.NA depending on the column's dtype, and
    # only pd.isna recognises all three.
    if years is None or pd.isna(years):
        return UNKNOWN_BUCKET
    for upper, label in EXPERIENCE_BUCKETS:
        if years <= upper:
            return label
    return EXPERIENCE_BUCKETS[-1][1]  # unreachable: the last bound is infinity


def experience_buckets(years: pd.Series) -> pd.Series:
    """Vectorised :func:`experience_bucket` for a whole column."""
    numeric = pd.to_numeric(years, errors="coerce")
    return numeric.map(experience_bucket).astype("object")


# ─────────────────────────────────────────────── baseline 0


class GlobalMedianBaseline:
    """Predicts the same number — the overall median — for every single person.

    This is the floor. A model that cannot beat "everyone earns the median" has
    learned nothing at all, and that is a genuinely useful thing to be able to
    check in one line.

    **Median, not mean.** Salaries have a long right tail: a handful of people
    earn many times what most people earn. The mean chases them upward, so it
    describes almost nobody. In this survey the mean is ₹24L and the median is
    ₹15L — the mean is above roughly two-thirds of the respondents. The median
    is the better single guess, and choosing it is the whole model.
    """

    def __init__(self) -> None:
        self.median_: float | None = None

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> GlobalMedianBaseline:
        """Learn one number. ``X`` is accepted and ignored, on purpose.

        Taking an unused ``X`` is what makes this interchangeable with every
        other model in the project — the evaluation loop doesn't need to know
        which model it is holding.
        """
        values = np.asarray(y, dtype=float)
        if values.size == 0:
            raise ValueError("cannot fit on an empty target")
        self.median_ = float(np.median(values))
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        if self.median_ is None:
            raise RuntimeError("call fit() before predict()")
        return np.full(len(X), self.median_, dtype=float)


# ─────────────────────────────────────────────── baseline 1


class GroupMedianBaseline:
    """A lookup table: median salary per (role × experience bucket).

    Four lines of pandas, no learning, no parameters to tune — and typically the
    number that a gradient boosting model beats by less than you'd hope.

    **The fallback hierarchy** is the part that takes actual thought. A lookup
    table has one failure mode: you look something up and it isn't there. A
    Security engineer with 30 years' experience walks in and that cell of the
    table is empty, or has three people in it. So predictions cascade:

    1. ``(role, experience bucket)`` — the specific answer we want, used only if
       that group has at least ``min_group_size`` people behind it.
    2. ``experience bucket`` alone — drops the role. Much better populated, and
       experience is the strongest single driver of salary anyway, so this is a
       genuinely reasonable second-best rather than a token gesture.
    3. the global median — the answer of last resort, always available.

    Each step trades precision for evidence, deliberately and in that order. The
    counts of which level fired are recorded on :attr:`fallback_counts_` after
    every ``predict``, because a table that quietly answers "global median" for
    a third of your candidates looks fine in the aggregate score and is useless
    in the room.

    **A note on log space.** You can fit this on rupees or on log(rupees) and
    get the same predictions back (after transforming), because the median of a
    list of logs is the log of the median — taking logs reorders nothing, and
    the median only cares about order. That is *not* true of the mean, which is
    a large part of why this file uses medians throughout.
    """

    def __init__(
        self,
        *,
        role_col: str = "role",
        experience_col: str = "years_experience",
        min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    ) -> None:
        if min_group_size < 1:
            raise ValueError(f"min_group_size must be at least 1, got {min_group_size}")
        self.role_col = role_col
        self.experience_col = experience_col
        self.min_group_size = min_group_size

        # Trailing underscores mark "learned during fit", the scikit-learn
        # convention. Anything without one was set by you; anything with one
        # came from the data.
        self.group_medians_: dict[tuple[str, str], float] = {}
        self.bucket_medians_: dict[str, float] = {}
        self.global_median_: float | None = None
        self.fallback_counts_: Counter[str] = Counter()

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> GroupMedianBaseline:
        for col in (self.role_col, self.experience_col):
            if col not in X.columns:
                raise ValueError(f"column {col!r} not found; available: {list(X.columns)}")

        salaries = np.asarray(y, dtype=float)
        if salaries.size != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {salaries.size}")
        if salaries.size == 0:
            raise ValueError("cannot fit on an empty target")

        # Rebuild as plain arrays so X's index can't misalign with y's. Silent
        # index misalignment between a DataFrame and a Series is one of pandas'
        # sharpest edges, and it produces NaNs rather than an error.
        work = pd.DataFrame(
            {
                "role": X[self.role_col].astype("string").fillna("unknown").to_numpy(),
                "bucket": experience_buckets(X[self.experience_col]).to_numpy(),
                "salary": salaries,
            }
        )

        self.global_median_ = float(work["salary"].median())

        # Level 2 first: bucket-only medians. No minimum size applied here —
        # this IS the fallback, so if it were also allowed to fall back the
        # cascade would have a hole in it. Buckets are coarse enough to be well
        # populated in practice.
        by_bucket = work.groupby("bucket")["salary"].median()
        self.bucket_medians_ = {str(k): float(v) for k, v in by_bucket.items()}

        # Level 1: the specific cells, keeping only those with enough people.
        by_group = work.groupby(["role", "bucket"])["salary"].agg(["median", "size"])
        trusted = by_group[by_group["size"] >= self.min_group_size]
        self.group_medians_ = {
            (str(role), str(bucket)): float(row["median"])
            for (role, bucket), row in trusted.iterrows()
        }
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        if self.global_median_ is None:
            raise RuntimeError("call fit() before predict()")

        roles = X[self.role_col].astype("string").fillna("unknown").to_numpy()
        buckets = experience_buckets(X[self.experience_col]).to_numpy()

        self.fallback_counts_ = Counter()
        predictions = np.empty(len(X), dtype=float)
        for i, (role, bucket) in enumerate(zip(roles, buckets, strict=True)):
            value, level = self._lookup(str(role), str(bucket))
            predictions[i] = value
            self.fallback_counts_[level] += 1
        return predictions

    def _lookup(self, role: str, bucket: str) -> tuple[float, str]:
        """Walk the cascade and report which rung answered."""
        assert self.global_median_ is not None  # guaranteed by predict()

        specific = self.group_medians_.get((role, bucket))
        if specific is not None:
            return specific, "role+experience"

        coarse = self.bucket_medians_.get(bucket)
        if coarse is not None:
            return coarse, "experience only"

        return self.global_median_, "global median"

    def explain(self, role: str, years: float | None) -> str:
        """Say what this candidate would be quoted, and on what evidence.

        A number a recruiter cannot argue with is a number a recruiter should
        not be given. This is the baseline's version of the SHAP explanations
        the real model gets in Phase 6.
        """
        from .metrics import format_rupees  # local import: metrics is only for display

        bucket = experience_bucket(years)
        value, level = self._lookup(str(role), bucket)
        return f"{role}, {bucket} years → {format_rupees(value)}  [matched on: {level}]"

    def coverage_report(self) -> str:
        """How often each rung of the cascade fired in the last ``predict``.

        Read this every time. A table answering "global median" for 30% of
        candidates is not a lookup table, it's a constant with extra steps.
        """
        total = sum(self.fallback_counts_.values())
        if not total:
            return "no predictions made yet"
        lines = ["fallback usage in the last predict():"]
        for level in ("role+experience", "experience only", "global median"):
            n = self.fallback_counts_.get(level, 0)
            lines.append(f"  {level:<16} {n:>6,}  ({n / total:.1%})")
        return "\n".join(lines)
