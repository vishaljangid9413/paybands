"""Measuring whether pay — or a model's predictions — treats one group worse.

**Why this file exists.**

A model learns patterns from history. If a company has historically paid one
group less for the same work, the model learns that as a pattern and applies it
to every new candidate. The output then looks objective: it is a number, from a
computer, with mathematics behind it. But it is the old unfairness, laundered —
and now it is *harder* to argue with, because "the model said so".

So we measure it. Three measurements, in increasing order of usefulness:

1. :func:`raw_gap` — what the two groups earn on average. Almost always
   non-zero, and almost never evidence of anything on its own.
2. :func:`adjusted_gap` — the difference that remains *after* holding
   experience, role, location and the rest fixed. This is the number that
   matters: same job, same experience, different pay.
3. :func:`residual_analysis` — does the *model* miss high or low for one group?
   A model can add bias of its own on top of whatever the data contained.

**This module is deliberately model-agnostic.** Every function takes plain
arrays — actual salaries, predicted salaries, group labels, a frame of
legitimate controls. Nothing here imports the band model, so the same audit
scores the lookup baseline, the gradient booster, a rule-of-thumb spreadsheet,
or a human recruiter's past offers. An audit that only works against one model
is an audit nobody will run on the thing that actually needs auditing.

**And the audit itself is audited.** :func:`validate_on_synthetic` generates
data with a pay gap *we chose*, runs this code against it, and checks that the
number comes back. That is calibrating the thermometer in boiling water before
trusting it on a patient — and it is the only way to know a bias detector works,
because on real data nobody knows the right answer to compare against.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

# ─────────────────────────────────────────────────────── what counts as a control
#
# A "control" is something we are willing to accept as a legitimate reason for
# two people to be paid differently. Experience, role, city, previous employer:
# a company can defend paying differently for these.
#
# This list is a JUDGEMENT CALL, not a fact, and it is the most contestable
# thing in the whole audit — so it lives here in plain sight rather than buried
# in a function.
#
# The trap works in both directions:
#
#   * Control for too little and you report a gap that is really just "one
#     group has fewer years of experience". That is a false accusation.
#   * Control for too much and you explain the unfairness away. If women are
#     systematically pushed into lower-paying roles, then "controlling for
#     role" subtracts exactly the discrimination you were looking for and
#     reports zero. That is a false all-clear, and it is the more dangerous
#     error because it is the comfortable one.
#
# There is no formula that settles this. State your control list out loud, as we
# do in every report this module prints, and let people argue with it.

LEGITIMATE_CONTROLS: tuple[str, ...] = (
    "years_experience",
    "role",
    "education",
    "org_size",
    "remote",
    "employment_type",
    "location_tier",
    "institute_tier",
    "prev_company_type",
)

#: Features that predict pay *and* leak group membership. Whether to control for
#: one is a genuine judgement call, not a technical question — see
#: :mod:`paybands.fairness.proxy`. Controlling for a career break says "time out
#: of the workforce is a legitimate reason to pay less"; not controlling for it
#: says "who takes that time off is itself the unfairness". Both positions are
#: arguable, they give different answers, and the honest thing is to report
#: both numbers rather than to quietly pick one.
CONTESTED_CONTROLS: tuple[str, ...] = ("career_gap_months",)

#: A numeric control with at most this many distinct values is turned into one
#: dummy variable per value instead of entering as a straight line.
#:
#: Why: `years_experience` is a number, but pay does not rise in a straight line
#: with it — the first five years are worth far more than years 15–20. Forcing a
#: straight line through a curve leaves a pattern in the residuals, and if that
#: pattern is at all correlated with group membership it lands on the group
#: coefficient and biases the headline number. Measured on our synthetic data,
#: that mistake alone shifts the recovered gap by about +0.3 percentage points.
#:
#: One dummy per value assumes nothing about the shape at all. The cost is
#: parameters, which is why genuinely continuous columns (a previous salary, an
#: age in days) stay linear. 40 is a judgement call; override it.
DEFAULT_MAX_AUTO_LEVELS = 40


# ─────────────────────────────────────────────────────── plumbing: groups


def _as_labels(group: ArrayLike) -> NDArray[np.str_]:
    """Group labels as a plain string array, so 1/1.0/'1' cannot disagree."""
    return np.asarray(pd.Series(np.asarray(group)).astype(str))


def _two_groups(
    labels: NDArray[np.str_],
    salary: NDArray[np.float64] | None,
    disadvantaged: str | None,
    reference: str | None,
) -> tuple[str, str]:
    """Work out which two group values we are comparing, and in which order.

    Order matters because it fixes the *sign*. Throughout this module a positive
    gap means "the disadvantaged group is paid less", so the two names cannot be
    interchangeable.

    If the caller does not name them and there are exactly two, we take the
    lower-paid one as disadvantaged. **Read the warning in :func:`raw_gap`
    before relying on that**: it makes the gap positive by construction, so
    "we found a positive gap" stops being a finding.
    """
    present = sorted(set(labels.tolist()))

    if disadvantaged is not None and reference is None:
        others = [g for g in present if g != disadvantaged]
        if len(others) != 1:
            raise ValueError(
                f"cannot infer the comparison group: {disadvantaged!r} vs which of "
                f"{others}? Pass `reference=` explicitly."
            )
        reference = others[0]
    elif disadvantaged is None and reference is not None:
        others = [g for g in present if g != reference]
        if len(others) != 1:
            raise ValueError(
                f"cannot infer the group under study: which of {others} against "
                f"{reference!r}? Pass `disadvantaged=` explicitly."
            )
        disadvantaged = others[0]
    elif disadvantaged is None and reference is None:
        if len(present) != 2:
            raise ValueError(
                f"found {len(present)} groups {present}, and this audit compares exactly two. "
                "Name them with `disadvantaged=` and `reference=`."
            )
        if salary is None:
            disadvantaged, reference = present
        else:
            # Auto-detect by median pay. See the warning in `raw_gap`.
            medians = {g: float(np.median(salary[labels == g])) for g in present}
            disadvantaged = min(medians, key=lambda g: medians[g])
            reference = max(medians, key=lambda g: medians[g])

    for name in (disadvantaged, reference):
        if name not in present:
            raise ValueError(f"group {name!r} does not appear in the data; found {present}")
    if disadvantaged == reference:
        raise ValueError("the two groups compared must be different")

    return str(disadvantaged), str(reference)


# ─────────────────────────────────────────────────────── plumbing: the regression


def _design_matrix(
    controls: pd.DataFrame | None,
    *,
    categorical: Sequence[str] | None = None,
    max_auto_levels: int = DEFAULT_MAX_AUTO_LEVELS,
) -> pd.DataFrame:
    """Turn a frame of controls into the numeric matrix a regression can take.

    Two rules, both worth understanding:

    * **Text columns become dummy variables** — one 0/1 column per value, with
      the first value dropped. Dropping one is not optional: keep all of them
      and they add up to the intercept column, the matrix stops being
      invertible, and the fit becomes arbitrary. The dropped value is the
      baseline every other value is measured against.
    * **Numbers with few distinct values also become dummies** — see
      :data:`DEFAULT_MAX_AUTO_LEVELS` for why forcing a straight line through a
      curved effect quietly biases the number we care about.
    """
    if controls is None or controls.shape[1] == 0:
        raise ValueError("no controls given")

    forced = set(categorical or ())
    unknown = forced - set(controls.columns)
    if unknown:
        raise ValueError(
            f"`categorical` names columns that are not in `controls`: {sorted(unknown)}"
        )

    missing = [c for c in controls.columns if controls[c].isna().any()]
    if missing:
        raise ValueError(
            f"controls contain missing values in {missing}. A regression cannot use a blank; "
            "fill them with an explicit 'unknown' category or drop those rows — but decide "
            "which, because dropping rows quietly changes who the audit is about."
        )

    parts: list[pd.DataFrame] = []
    for name in controls.columns:
        column = controls[name]
        is_number = pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)
        if is_number and name not in forced and column.nunique() > max_auto_levels:
            parts.append(column.astype(float).to_frame(name))
        else:
            parts.append(
                pd.get_dummies(column.astype(str), prefix=name, drop_first=True, dtype=float)
            )

    design = pd.concat(parts, axis=1)
    # The intercept — the baseline salary once every dummy is at its reference
    # value. Without it the fit is forced through the origin and every other
    # coefficient absorbs the difference.
    design.insert(0, "intercept", 1.0)
    return design


@dataclass(frozen=True)
class _Fit:
    """The pieces of an ordinary least squares fit that the audit needs."""

    coefficients: dict[str, float]
    standard_errors: dict[str, float]
    n: int
    n_parameters: int
    df_residual: int
    r_squared: float


def _ols(design: pd.DataFrame, y: NDArray[np.float64]) -> _Fit:
    """Ordinary least squares, written out rather than imported.

    Four lines of linear algebra and one line of statistics, kept visible
    because the *standard error* is the part that stops this module
    overclaiming, and a number you cannot see the derivation of is a number you
    will eventually misread.

    The standard error answers: "if we re-ran the world, how much would this
    coefficient wobble?" It shrinks like 1/sqrt(n), which is why a gap measured
    on 200 people is a rumour and the same gap on 20,000 is a finding.
    """
    x = design.to_numpy(dtype=float)
    n, k = x.shape

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta

    # `rank`, not `k`: if two controls happen to be perfectly collinear (a level
    # that only ever occurs with one other level, say) the matrix is short of
    # independent columns, and using k would understate the noise.
    rank = int(np.linalg.matrix_rank(x))
    df_residual = n - rank
    if df_residual <= 0:
        raise ValueError(
            f"{n} rows and {rank} independent columns leaves no degrees of freedom. "
            "The fit would be perfect and meaningless — use fewer controls or more data."
        )

    sigma_squared = float(residuals @ residuals) / df_residual
    # pinv rather than inv so a rank-deficient design degrades instead of
    # exploding; the coefficient we care about is identified either way.
    covariance = np.linalg.pinv(x.T @ x) * sigma_squared
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0 else float("nan")

    names = list(design.columns)
    return _Fit(
        coefficients=dict(zip(names, beta.tolist(), strict=True)),
        standard_errors=dict(zip(names, errors.tolist(), strict=True)),
        n=n,
        n_parameters=k,
        df_residual=df_residual,
        r_squared=r_squared,
    )


def _to_salary_array(salary: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(salary, dtype=float)
    if values.size == 0:
        raise ValueError("cannot audit an empty set of salaries")
    if np.any(~np.isfinite(values)):
        raise ValueError("salaries contain NaN or infinity; clean the data before auditing")
    if np.any(values <= 0):
        raise ValueError("salaries must be positive — the gap is measured on log(salary)")
    return values


# ─────────────────────────────────────────────────────── 1. the raw gap


@dataclass(frozen=True)
class RawGap:
    """What two groups earn, before accounting for anything at all.

    Every field is a plain number so a caller can put it in a table; the
    interpretation lives in :meth:`__str__` and in :func:`raw_gap`.
    """

    disadvantaged: str
    reference: str
    n_disadvantaged: int
    n_reference: int
    mean_disadvantaged: float
    mean_reference: float
    median_disadvantaged: float
    median_reference: float

    #: Positive = the disadvantaged group earns less. ``1 - ratio``, so 0.146
    #: reads as "earns 14.6% less".
    mean_gap: float
    median_gap: float

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Raw pay gap — '{self.disadvantaged}' vs '{self.reference}'",
                f"  mean pay      {self.mean_disadvantaged:>12,.0f}  vs "
                f"{self.mean_reference:>12,.0f}   gap {self.mean_gap:>7.1%}",
                f"  median pay    {self.median_disadvantaged:>12,.0f}  vs "
                f"{self.median_reference:>12,.0f}   gap {self.median_gap:>7.1%}",
                f"  group sizes   {self.n_disadvantaged:>12,}  vs {self.n_reference:>12,}",
                "  NOT evidence of unfair pay on its own — see the docstring.",
            ]
        )


def raw_gap(
    salary: ArrayLike,
    group: ArrayLike,
    *,
    disadvantaged: str | None = None,
    reference: str | None = None,
) -> RawGap:
    """The difference in average pay between two groups. No adjustments.

    ────────────────────────────────────────────────────────────────────────
    **Read this before quoting the number.**

    A raw gap is not evidence of discrimination, and reporting it as though it
    were is the single most common mistake in this whole area.

    Groups genuinely differ in things that pay differently. If one group
    averages four years of experience and the other nine, they will earn
    different amounts at a perfectly fair employer, and this function will
    report a large gap. It has measured a real difference in *who the people
    are*, not in *how they are treated*.

    The raw gap is still worth computing, for two reasons. It is the number the
    outside world uses — national "gender pay gap" figures are raw gaps, and a
    company reporting them needs to know its own. And the distance between the
    raw gap and the adjusted gap is itself the finding: "of the 14.6% overall
    difference, 8.1 points survive controlling for experience and role, and the
    rest is explained by the two groups doing different jobs" is a far more
    useful sentence than either number alone. (Whether *those* differences in
    who does which job are themselves fair is a separate question, and not one
    a regression can answer.)
    ────────────────────────────────────────────────────────────────────────

    Args:
        salary: Actual pay, in rupees. One entry per person.
        group: The protected attribute, same length. Compared as strings.
        disadvantaged: Which group value is under study. **If omitted, the
            lower-paid group is chosen automatically** — convenient, but it
            makes the reported gap non-negative no matter what the data says,
            so the *existence* of a positive raw gap then tells you nothing.
            Name the group explicitly whenever you plan to quote the result.
        reference: The group compared against. Inferred when there are two.

    Returns:
        A :class:`RawGap`. ``mean_gap`` of 0.146 means "earns 14.6% less".
    """
    values = _to_salary_array(salary)
    labels = _as_labels(group)
    if values.shape != labels.shape:
        raise ValueError(f"shape mismatch: salary is {values.shape}, group is {labels.shape}")

    low, high = _two_groups(labels, values, disadvantaged, reference)
    a, b = values[labels == low], values[labels == high]
    if a.size == 0 or b.size == 0:
        raise ValueError(f"one of the groups is empty: {low!r} has {a.size}, {high!r} has {b.size}")

    return RawGap(
        disadvantaged=low,
        reference=high,
        n_disadvantaged=int(a.size),
        n_reference=int(b.size),
        mean_disadvantaged=float(a.mean()),
        mean_reference=float(b.mean()),
        median_disadvantaged=float(np.median(a)),
        median_reference=float(np.median(b)),
        mean_gap=1.0 - float(a.mean()) / float(b.mean()),
        median_gap=1.0 - float(np.median(a)) / float(np.median(b)),
    )


# ─────────────────────────────────────────────────────── 2. the adjusted gap


@dataclass(frozen=True)
class AdjustedGap:
    """The like-for-like pay gap, with an honest interval around it."""

    disadvantaged: str
    reference: str
    n: int
    n_disadvantaged: int

    #: Positive = paid less. ``1 - exp(coefficient)``.
    gap: float
    ci_low: float
    ci_high: float
    confidence: float

    #: The regression's own output, kept so the arithmetic can be checked.
    coefficient: float
    standard_error: float
    p_value: float

    controls: tuple[str, ...]
    n_parameters: int
    r_squared: float

    @property
    def is_significant(self) -> bool:
        """Does the interval exclude zero?

        "Significant" means only that: the data would be surprising if the true
        gap were exactly zero. It says nothing about whether the gap is *large*.
        A statistically significant 0.4% gap is not a scandal, and a
        non-significant 9% gap on 60 employees is not an all-clear — it is a
        sample too small to tell. Read the size and the interval, in that order.
        """
        return (self.ci_low > 0) or (self.ci_high < 0)

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def __str__(self) -> str:
        verdict = (
            "unlikely to be chance" if self.is_significant else "could plausibly be chance alone"
        )
        return "\n".join(
            [
                f"Adjusted pay gap — '{self.disadvantaged}' vs '{self.reference}'",
                f"  gap {self.gap:>7.2%}   {self.confidence:.0%} confidence interval "
                f"{self.ci_low:.2%} to {self.ci_high:.2%}   ({verdict})",
                f"  n = {self.n:,} ({self.n_disadvantaged:,} in the group under study), "
                f"{self.n_parameters} parameters, R² = {self.r_squared:.3f}",
                f"  controlled for: {', '.join(self.controls)}",
            ]
        )


def adjusted_gap(
    salary: ArrayLike,
    group: ArrayLike,
    controls: pd.DataFrame,
    *,
    disadvantaged: str | None = None,
    reference: str | None = None,
    confidence: float = 0.95,
    categorical: Sequence[str] | None = None,
    max_auto_levels: int = DEFAULT_MAX_AUTO_LEVELS,
) -> AdjustedGap:
    """The pay gap between two groups who are otherwise the same. **The number.**

    We fit one regression::

        log(salary) = intercept
                    + β × [is in the disadvantaged group]
                    + (a coefficient for every control)
                    + unexplained leftover

    and read off ``β``. Because the controls are in the equation too, ``β`` is
    what is left after experience, role, location and the rest have been given
    every chance to explain the difference. Same experience, same role, same
    city — different pay. That is the claim this number supports and no other.

    **Three details that are easy to get wrong.**

    *Logs, not rupees.* Pay differences are multiplicative — people say "8%
    less", never "₹1.7 lakh less", because ₹1.7 lakh means different things at
    ₹6L and at ₹60L. Taking logs turns one consistent percentage into one
    consistent number, which is the only form a single coefficient can have.

    *``1 - exp(β)``, not ``-β``.* A coefficient of −0.0834 is an 8.0% cut, not
    an 8.3% one. At small gaps the two nearly agree, which is exactly why the
    mistake survives unnoticed until someone checks it against a known answer.

    *An interval, not a point.* On 300 employees the estimate wobbles by several
    percentage points from sample to sample. Quoting "6.2%" from such a sample
    invites a conversation the data cannot support; quoting "6.2%, and the truth
    is somewhere between 0.4% and 11.7%" invites the right one. There is no
    option here to get the point estimate without the interval, deliberately.

    **What it still cannot tell you.** A regression coefficient is not a cause.
    It says an unexplained difference remains; the explanation might be
    discrimination, or something real that nobody wrote into the controls
    (negotiation, a specialism the role code does not capture). The honest
    output is "X% unexplained by what we recorded", which is a reason to go and
    look, not a verdict.

    Args:
        salary: Actual pay in rupees. Can equally be a model's *predicted* pay —
            that measures whether the model would reproduce the gap.
        group: The protected attribute.
        controls: One column per legitimate pay driver. See
            :data:`LEGITIMATE_CONTROLS`, and think about the list rather than
            accepting it. **Do not put the protected attribute in here.**
        disadvantaged: Group under study; auto-detected when omitted.
        reference: Group compared against.
        confidence: Interval width. 0.95 is convention, not law.
        categorical: Numeric columns to force into dummy variables.
        max_auto_levels: See :data:`DEFAULT_MAX_AUTO_LEVELS`.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    values = _to_salary_array(salary)
    labels = _as_labels(group)
    if values.shape != labels.shape:
        raise ValueError(f"shape mismatch: salary is {values.shape}, group is {labels.shape}")
    if len(controls) != len(values):
        raise ValueError(f"controls has {len(controls)} rows but salary has {len(values)}")

    low, high = _two_groups(labels, values, disadvantaged, reference)
    keep = np.isin(labels, [low, high])
    values, labels = values[keep], labels[keep]
    controls = controls.loc[keep].reset_index(drop=True)

    for column in controls.columns:
        # A control that *is* the protected attribute leaves nothing to compare:
        # the group indicator becomes a copy of a control and the coefficient is
        # meaningless. Cheap to check, and an easy mistake to make when a column
        # list is assembled programmatically.
        if np.array_equal(_as_labels(controls[column]), labels):
            raise ValueError(
                f"control column {column!r} is identical to the group labels. Controlling for "
                "the protected attribute removes the very comparison you are trying to make."
            )

    design = _design_matrix(controls, categorical=categorical, max_auto_levels=max_auto_levels)
    indicator = "__is_disadvantaged__"
    controls_only = design.to_numpy(dtype=float)
    design[indicator] = (labels == low).astype(float)

    # Is there anything left to compare? Adding the group indicator must add a
    # genuinely new direction to the matrix. If it does not — if the controls
    # already pin down who is in which group — then no two people in this data
    # are alike in every control but different in group, and "same job, same
    # experience, different pay" has no examples to be about.
    #
    # This is not a hypothetical. Give one group 0–5 years of experience and the
    # other 6–12, control for experience, and there is no overlap at all. Without
    # this check the fit still returns a number (the linear algebra picks one of
    # infinitely many equally good answers) and that number is meaningless.
    if np.linalg.matrix_rank(design.to_numpy(dtype=float)) == np.linalg.matrix_rank(controls_only):
        raise ValueError(
            "the controls already determine group membership, so there is no like-for-like "
            "comparison left to make. Usually this means the two groups do not overlap on "
            "some control (no one in group A shares a value with anyone in group B) — check "
            "a cross-tabulation of the group against each control before dropping any."
        )

    fit = _ols(design, np.log(values))
    beta = fit.coefficients[indicator]
    error = fit.standard_errors[indicator]

    if not np.isfinite(error) or error == 0.0:
        raise ValueError(
            "the group indicator has no independent variation left once the controls are in. "
            "Some control predicts group membership perfectly, so there is no like-for-like "
            "comparison to make."
        )

    # Student's t, not the normal distribution: with few rows the extra width in
    # the tails is exactly the humility a small sample deserves. They converge
    # above a few hundred rows, so this costs nothing when data is plentiful.
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, fit.df_residual))
    beta_low, beta_high = beta - critical * error, beta + critical * error
    p_value = float(2.0 * stats.t.sf(abs(beta / error), fit.df_residual))

    return AdjustedGap(
        disadvantaged=low,
        reference=high,
        n=fit.n,
        n_disadvantaged=int((labels == low).sum()),
        gap=1.0 - float(np.exp(beta)),
        # `1 - exp(...)` flips the order: the most negative coefficient is the
        # biggest gap, so the interval's ends swap over.
        ci_low=1.0 - float(np.exp(beta_high)),
        ci_high=1.0 - float(np.exp(beta_low)),
        confidence=confidence,
        coefficient=float(beta),
        standard_error=float(error),
        p_value=p_value,
        controls=tuple(controls.columns),
        n_parameters=fit.n_parameters,
        r_squared=fit.r_squared,
    )


# ─────────────────────────────────────────────────────── 3. the model's own bias


@dataclass(frozen=True)
class GroupResidual:
    """How wrong the model was, for one group."""

    group: str
    n: int
    #: mean of ``log(actual) − log(predicted)``. Positive = the model predicts
    #: **less** than these people actually earn.
    mean_log_residual: float
    #: The same thing as a percentage: ``exp(mean_log_residual) − 1``.
    relative_bias: float
    mean_rupee_error: float
    median_rupee_error: float


@dataclass(frozen=True)
class ResidualCheck:
    """Whether the model's mistakes fall evenly across groups."""

    disadvantaged: GroupResidual
    reference: GroupResidual

    #: Difference in mean log residual, disadvantaged minus reference. Positive
    #: = the model under-predicts the disadvantaged group *relative to* the
    #: other. Expressed as a percentage in ``difference``.
    difference: float
    ci_low: float
    ci_high: float
    confidence: float
    p_value: float

    @property
    def is_significant(self) -> bool:
        return (self.ci_low > 0) or (self.ci_high < 0)

    def __str__(self) -> str:
        direction = "under-predicts" if self.difference > 0 else "over-predicts"
        verdict = "beyond what chance explains" if self.is_significant else "within chance"
        return "\n".join(
            [
                f"Model error by group — '{self.disadvantaged.group}' vs '{self.reference.group}'",
                f"  {self.disadvantaged.group:<12} predicted "
                f"{abs(self.disadvantaged.relative_bias):>6.2%} "
                f"{'below' if self.disadvantaged.relative_bias > 0 else 'above'} actual pay",
                f"  {self.reference.group:<12} predicted "
                f"{abs(self.reference.relative_bias):>6.2%} "
                f"{'below' if self.reference.relative_bias > 0 else 'above'} actual pay",
                f"  the model {direction} '{self.disadvantaged.group}' by "
                f"{abs(self.difference):.2%} relative to '{self.reference.group}' "
                f"({self.ci_low:.2%} to {self.ci_high:.2%}, {verdict})",
            ]
        )


def residual_analysis(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    group: ArrayLike,
    *,
    disadvantaged: str | None = None,
    reference: str | None = None,
    confidence: float = 0.95,
) -> ResidualCheck:
    """Does the model miss in a *different direction* for one group?

    A **residual** is what the model got wrong for one person:
    ``actual − predicted``. Across a whole dataset the residuals should scatter
    evenly around zero. If they average positive for one group and negative for
    another, the model is not merely noisy — it is systematically cheaper about
    one kind of person, and every offer built from it inherits that.

    We work in log space (``log(actual) − log(predicted)``) for the same reason
    the gap does: a ₹2L miss is a disaster at ₹6L and a rounding error at ₹60L,
    so an average of rupee misses is dominated by whoever earns most.

    ────────────────────────────────────────────────────────────────────────
    **This test and :func:`adjusted_gap` catch different things. You need both.**

    A model trained faithfully on unfair data will pass *this* test perfectly.
    It learned the unfairness, so it predicts the unfair salaries accurately,
    so its residuals are beautifully balanced across groups — while it quietly
    recommends lower offers for one group. Residual balance is not fairness.

    What this catches instead is bias the *model* added: too few rows from a
    group to learn their pattern, a loss function dominated by the majority, a
    proxy the model leaned on too hard. That failure is invisible to
    :func:`adjusted_gap` on the training data and very visible here.
    ────────────────────────────────────────────────────────────────────────

    Args:
        y_true: Actual salaries, rupees.
        y_pred: Predicted salaries, rupees. Any model, or none — a lookup table
            and a spreadsheet formula are audited the same way.
        group: The protected attribute.
        confidence: Interval width for the difference between groups.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    actual = _to_salary_array(y_true)
    predicted = _to_salary_array(y_pred)
    labels = _as_labels(group)
    if not (actual.shape == predicted.shape == labels.shape):
        raise ValueError(
            f"shape mismatch: y_true {actual.shape}, y_pred {predicted.shape}, group {labels.shape}"
        )

    low, high = _two_groups(labels, actual, disadvantaged, reference)
    log_residual = np.log(actual) - np.log(predicted)
    rupee_residual = actual - predicted

    def summarise(name: str) -> tuple[GroupResidual, NDArray[np.float64]]:
        mask = labels == name
        if mask.sum() < 2:
            raise ValueError(f"group {name!r} has {mask.sum()} rows; need at least 2 to compare")
        logs = log_residual[mask]
        return (
            GroupResidual(
                group=name,
                n=int(mask.sum()),
                mean_log_residual=float(logs.mean()),
                relative_bias=float(np.exp(logs.mean()) - 1.0),
                mean_rupee_error=float(rupee_residual[mask].mean()),
                median_rupee_error=float(np.median(rupee_residual[mask])),
            ),
            logs,
        )

    low_summary, low_logs = summarise(low)
    high_summary, high_logs = summarise(high)

    # Welch's t-test: it does NOT assume the two groups have the same spread.
    # They rarely do — a model is usually more certain about the group it has
    # the most examples of — and the equal-variance version would report a
    # falsely narrow interval exactly when the groups are most lopsided.
    difference = float(low_logs.mean() - high_logs.mean())
    var_low = float(low_logs.var(ddof=1)) / low_logs.size
    var_high = float(high_logs.var(ddof=1)) / high_logs.size
    standard_error = float(np.sqrt(var_low + var_high))

    if standard_error == 0.0:
        dof, critical, p_value = float(low_logs.size + high_logs.size - 2), 0.0, 1.0
    else:
        dof = (var_low + var_high) ** 2 / (
            var_low**2 / (low_logs.size - 1) + var_high**2 / (high_logs.size - 1)
        )
        critical = float(stats.t.ppf(0.5 + confidence / 2.0, dof))
        p_value = float(2.0 * stats.t.sf(abs(difference / standard_error), dof))

    margin = critical * standard_error
    # Report the difference as a percentage, consistent with every other number
    # in this module.
    return ResidualCheck(
        disadvantaged=low_summary,
        reference=high_summary,
        difference=float(np.exp(difference) - 1.0),
        ci_low=float(np.exp(difference - margin) - 1.0),
        ci_high=float(np.exp(difference + margin) - 1.0),
        confidence=confidence,
        p_value=p_value,
    )


# ─────────────────────────────────────────────────────── the report


@dataclass(frozen=True)
class FairnessReport:
    """Everything the audit found, written for someone who does not do maths.

    The audience for this string is an HR lead or a hiring manager who has to
    decide whether to act. So it says what each number means, and — just as
    importantly — what each number does *not* mean. A fairness result handed
    over without its caveats gets repeated without its caveats.
    """

    group_column: str
    raw: RawGap
    adjusted: AdjustedGap
    residual: ResidualCheck | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def explained_by_controls(self) -> float:
        """How much of the raw gap the legitimate controls account for."""
        return self.raw.mean_gap - self.adjusted.gap

    def __str__(self) -> str:
        low, high = self.adjusted.disadvantaged, self.adjusted.reference
        lines = [
            f"PAY FAIRNESS AUDIT — '{low}' compared with '{high}' (column: {self.group_column})",
            f"{'':─<86}",
            f"People included: {self.raw.n_disadvantaged + self.raw.n_reference:,} "
            f"({self.raw.n_disadvantaged:,} {low}, {self.raw.n_reference:,} {high})",
            "",
            "1. HEADLINE PAY GAP (no adjustments)                        "
            f"{self.raw.mean_gap:>8.1%}",
            f"   On average, '{low}' are paid {abs(self.raw.mean_gap):.1%} "
            f"{'less' if self.raw.mean_gap > 0 else 'more'} than '{high}'.",
            "   This number on its own proves nothing. The two groups differ in",
            "   experience, role and location, and those differences are paid for",
            "   legitimately. Read it alongside point 2, never instead of it.",
            "",
            f"2. LIKE-FOR-LIKE PAY GAP (the one that matters)             "
            f"{self.adjusted.gap:>8.1%}",
            "   Comparing people with the same experience, role and other recorded",
            f"   circumstances, '{low}' are paid {abs(self.adjusted.gap):.1%} "
            f"{'less' if self.adjusted.gap > 0 else 'more'}.",
            f"   We are {self.adjusted.confidence:.0%} confident the true figure is between "
            f"{self.adjusted.ci_low:.1%} and {self.adjusted.ci_high:.1%}.",
            "   "
            + (
                "A gap this size is unlikely to be chance."
                if self.adjusted.is_significant
                else "This is small enough that chance alone could explain it."
            ),
            f"   Accounted for by the controls: {self.explained_by_controls:.1%} of the "
            f"{self.raw.mean_gap:.1%} headline gap.",
            # Wrapped, because the control list is the part a sceptical reader
            # most needs to actually read, and a 150-character line is a line
            # people's eyes slide off.
            *textwrap.wrap(
                f"Controls used: {', '.join(self.adjusted.controls)}.",
                width=82,
                initial_indent="   ",
                subsequent_indent="      ",
            ),
        ]

        if self.residual is not None:
            direction = "LOWER" if self.residual.difference > 0 else "HIGHER"
            lines += [
                "",
                f"3. THE MODEL'S OWN ERRORS                                   "
                f"{self.residual.difference:>8.1%}",
                f"   Relative to '{high}', the model's predictions for '{low}' sit",
                f"   {abs(self.residual.difference):.1%} {direction} than their actual pay "
                f"({self.residual.ci_low:.1%} to {self.residual.ci_high:.1%}).",
                "   A balanced result here does NOT mean the model is fair: a model that",
                "   learned an unfair pattern perfectly reproduces it with perfectly even",
                "   errors. Point 2 is what tells you about fairness; this tells you whether",
                "   the model added anything of its own on top.",
            ]

        lines += [
            "",
            "WHAT THIS IS NOT",
            "   An unexplained gap is a reason to investigate, not a finding of",
            "   discrimination. Something real may be missing from the controls above.",
            "   Equally, controlling for a factor that is itself unfairly distributed",
            "   (who gets put in which role) subtracts the problem from the answer.",
        ]
        lines += [f"   {note}" for note in self.notes]
        return "\n".join(lines)


def audit(
    salary: ArrayLike,
    group: ArrayLike,
    controls: pd.DataFrame,
    *,
    y_pred: ArrayLike | None = None,
    group_column: str = "gender",
    disadvantaged: str | None = None,
    reference: str | None = None,
    confidence: float = 0.95,
    categorical: Sequence[str] | None = None,
    notes: Sequence[str] = (),
) -> FairnessReport:
    """Run every check and package the answers. The one call most people want.

    Args:
        salary: Actual pay in rupees.
        group: The protected attribute.
        controls: Legitimate pay drivers — see :data:`LEGITIMATE_CONTROLS`.
        y_pred: Model predictions in rupees. Optional; supply them and the
            report gains its third section, on the model's own bias.
        group_column: Name of the attribute, for the report header only.
        notes: Extra caveats to print, e.g. which contested controls you chose
            to include and why.
    """
    raw = raw_gap(salary, group, disadvantaged=disadvantaged, reference=reference)
    adjusted = adjusted_gap(
        salary,
        group,
        controls,
        # Pin the direction found by the raw comparison so both sections talk
        # about the same group in the same order — otherwise a sign flip
        # between sections reads as a contradiction.
        disadvantaged=raw.disadvantaged,
        reference=raw.reference,
        confidence=confidence,
        categorical=categorical,
    )
    residual = None
    if y_pred is not None:
        residual = residual_analysis(
            salary,
            y_pred,
            group,
            disadvantaged=raw.disadvantaged,
            reference=raw.reference,
            confidence=confidence,
        )
    return FairnessReport(
        group_column=group_column,
        raw=raw,
        adjusted=adjusted,
        residual=residual,
        notes=tuple(notes),
    )


# ─────────────────────────────────────────────────────── validating the audit


def validate_on_synthetic(
    injected_gaps: Sequence[float] = (0.0, 0.03, 0.08, 0.15, 0.25),
    *,
    n: int = 10_000,
    seed: int = 42,
    control_for_proxy: bool = True,
    confidence: float = 0.95,
    generator: Callable[..., tuple[pd.DataFrame, object]] | None = None,
) -> pd.DataFrame:
    """Calibrate the thermometer: inject a gap we chose, check we measure it back.

    ────────────────────────────────────────────────────────────────────────
    **This is the most important function in the module, and it is the one
    nobody writes.**

    Everything above is a measuring instrument, and an unvalidated measuring
    instrument is just an opinion with decimal places. On real company data you
    cannot validate it: nobody knows the true pay gap, so if the audit reports
    3% you have no way to tell whether the gap is 3%, or the gap is 9% and the
    audit is broken. Both look identical from the outside.

    So we generate a world where we wrote the rules —
    :func:`paybands.data.synthetic.generate` pays one group exactly ``X`` less —
    run the audit against it, and demand ``X`` back. Do this before pointing any
    of this code at real people.
    ────────────────────────────────────────────────────────────────────────

    The ``control_for_proxy`` switch is the second lesson. Career breaks in that
    dataset both depress pay *and* signal group membership. Control for them
    (``True``) and the audit recovers the injected gap. Leave them out
    (``False``) and it reports noticeably more — because part of the unfairness
    is flowing through the career-break channel and is being attributed to
    gender directly. Neither answer is wrong; they answer different questions.
    See :mod:`paybands.fairness.proxy`.

    Args:
        injected_gaps: The true gaps to test. Include 0.0 — an audit that finds
            bias in fair data is worse than no audit, because it is confidently
            wrong in the direction that gets someone accused.
        n: Rows per run. The measurement is noisy; at n=10,000 the standard
            error on the gap is about half a percentage point.
        seed: Passed to the generator, so this table is reproducible.
        control_for_proxy: Whether ``career_gap_months`` joins the controls.
        generator: Injected for testing. Defaults to the project generator.

    Returns:
        One row per injected gap: what we asked for, what came back, the
        interval, the error, and whether the truth landed inside the interval.
        That last column is the real pass/fail — a correct interval contains the
        true value about ``confidence`` of the time.
    """
    if generator is None:
        # Imported here, not at module top, on purpose. The audit must not
        # depend on our data generator — it has to run against company data,
        # survey data, anything. Only its self-test knows the generator exists,
        # and that dependency should be visible at the one place it is used.
        from paybands.data.synthetic import generate as generator  # noqa: PLC0415

    rows = []
    for injected in injected_gaps:
        frame, truth = generator(n, seed=seed, pay_gap=injected)
        control_names = [c for c in LEGITIMATE_CONTROLS if c in frame.columns]
        if control_for_proxy:
            control_names += [c for c in CONTESTED_CONTROLS if c in frame.columns]

        measured = adjusted_gap(
            frame["salary_annual"],
            frame["gender"],
            frame[control_names],
            # Read the answer sheet rather than guessing which group is which.
            # Auto-detection would pick the lower-paid group, which is exactly
            # the assumption a validation run must not smuggle in — at
            # pay_gap=0.0 it would report a positive gap by construction.
            disadvantaged=truth.disadvantaged_group,  # type: ignore[attr-defined]
            confidence=confidence,
        )
        rows.append(
            {
                "injected": injected,
                "recovered": measured.gap,
                "ci_low": measured.ci_low,
                "ci_high": measured.ci_high,
                "error": measured.gap - injected,
                "truth_inside_ci": bool(measured.ci_low <= injected <= measured.ci_high),
                "n": measured.n,
            }
        )
    return pd.DataFrame(rows)
