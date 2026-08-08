"""The proxy problem: why deleting the gender column does not remove the bias.

**The claim this file exists to demolish.**

    "We removed gender from the training data, so the model can't be biased."

You will hear this in almost every discussion of fair machine learning. It is
wrong, it is widespread, and it is comfortable — which is the worst combination
a false belief can have, because nobody looks for the evidence against it.

Here is why it fails. A model does not need the column it was denied. It needs
*anything correlated with it*. Career breaks, part-time history, a gap between
graduation year and first job, which schools, which job titles, which
neighbourhood a postcode implies. Each is an innocent-looking feature that
carries information about the protected attribute, and a model with enough of
them reassembles the pattern from the pieces. Those features are called
**proxies**.

Worse: removing the protected column makes the bias *harder to find*, not
smaller. With gender in the model you can read its coefficient. Without it, the
same effect is smeared across six features, none of which looks suspicious on
its own, and the audit has to work for it.

**What this module does about it.**

* :func:`proxy_correlation` — ranks every feature by how much it tells you
  about the protected attribute. Run it before training anything, on any
  dataset. The features at the top are the ones that will carry the bias.
* :func:`compare_with_without` — the demonstration. Fit the same model twice,
  once with the protected column and once without, and measure the gap in the
  predictions both times. On our synthetic data the gap does not disappear; it
  shrinks by a bit under two-thirds and the rest survives, delivered by career
  breaks. Meanwhile accuracy is essentially unchanged, so nothing was traded
  for it — the column was removed and the harm stayed.

Showing this is worth far more than claiming we "removed bias". One is evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .audit import (
    LEGITIMATE_CONTROLS,
    AdjustedGap,
    _as_labels,
    _design_matrix,
    _two_groups,
    adjusted_gap,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

#: The known proxy in our synthetic data. Career breaks are 45% likely in one
#: group and 6% in the other, and each year out costs 7% of pay — so the column
#: both predicts pay honestly *and* smuggles group membership. Real datasets
#: have several of these and nobody labels them; :func:`proxy_correlation` is
#: how you go looking.
DEFAULT_PROXY_COLUMNS: tuple[str, ...] = ("career_gap_months",)

#: Rough labels for an association strength. Judgement calls, printed rather
#: than hidden so that a reader can disagree with the wording instead of
#: misreading a bare 0.31.
_STRENGTH_BANDS = ((0.30, "strong"), (0.15, "moderate"), (0.05, "weak"))


# ─────────────────────────────────────────────────────── measuring association


def _correlation_ratio(values: NDArray[np.float64], labels: NDArray[np.str_]) -> float:
    """How much of a *number's* variation is explained by group membership. 0–1.

    Known as **eta** (η). Read it as: split people by group, and ask what share
    of the spread in this feature is the gap between the group averages rather
    than the spread within each group. 0 means the groups look identical on this
    feature; 1 means knowing the feature tells you the group exactly.

    For two groups this is the same number as the absolute correlation between
    the feature and a 0/1 group indicator — but written this way it also works
    for three groups or ten, so the audit does not need a separate code path for
    a non-binary attribute.
    """
    total = float(((values - values.mean()) ** 2).sum())
    if total == 0.0:
        return 0.0  # a constant feature carries no information about anything
    between = 0.0
    for name in np.unique(labels):
        group_values = values[labels == name]
        between += group_values.size * (group_values.mean() - values.mean()) ** 2
    return float(np.sqrt(min(between / total, 1.0)))


def _chi_squared(table: NDArray[np.float64]) -> float:
    """Pearson's chi-squared statistic for a contingency table.

    Written out in four lines rather than imported, because the whole idea fits
    in one sentence: compare what you observed against what independence would
    have produced, square the differences, and scale each by how big it was
    expected to be.
    """
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    return float(contribution.sum())


def _cramers_v(feature: NDArray[np.str_], labels: NDArray[np.str_]) -> float:
    """How much a *category* tells you about group membership. 0–1.

    Built on the chi-squared statistic — "how far is this table of counts from
    what it would look like if the two things were unrelated?" — then scaled by
    the table's size so that a 2×2 result and a 9×3 result can sit in the same
    column of the same report.

    We apply the standard bias correction, because chi-squared drifts upward as
    the table gets bigger: without it a job-title column with 200 values scores
    higher than a genuine proxy purely for being sprawling.
    """
    table = pd.crosstab(pd.Series(feature), pd.Series(labels)).to_numpy(float)
    n = table.sum()
    if n == 0 or min(table.shape) < 2:
        return 0.0
    chi2 = _chi_squared(table)
    phi2 = chi2 / n
    rows, cols = table.shape
    # Bergsma–Wicher correction: subtract the inflation you would expect from
    # table size alone, and floor at zero when that leaves nothing.
    phi2_corrected = max(0.0, phi2 - (rows - 1) * (cols - 1) / (n - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (n - 1)
    cols_corrected = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(rows_corrected - 1, cols_corrected - 1)
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2_corrected / denominator))


def _strength_label(association: float) -> str:
    for threshold, label in _STRENGTH_BANDS:
        if association >= threshold:
            return label
    return "negligible"


@dataclass(frozen=True)
class ProxyScan:
    """Which features leak the protected attribute, strongest first."""

    protected_column: str
    n: int
    table: pd.DataFrame

    def strongest(self, k: int = 5) -> pd.DataFrame:
        return self.table.head(k)

    def leaking(self, threshold: float = 0.15) -> list[str]:
        """Feature names worth worrying about, by the given threshold."""
        return self.table.loc[self.table["association"] >= threshold, "feature"].tolist()

    def __str__(self) -> str:
        lines = [
            f"Proxy scan for '{self.protected_column}' — n = {self.n:,}",
            "  association: 0 = tells you nothing about the group, 1 = tells you exactly",
            f"  {'feature':<24}{'kind':<14}{'assoc.':>8}  strength",
        ]
        for row in self.table.itertuples():
            lines.append(
                f"  {row.feature:<24}{row.kind:<14}{row.association:>8.3f}  {row.strength}"
            )
        lines.append("  Anything above 'weak' can carry the bias after the column is deleted.")
        return "\n".join(lines)


def proxy_correlation(
    df: pd.DataFrame,
    protected_col: str,
    candidate_cols: Sequence[str] | None = None,
    *,
    max_auto_levels: int = 40,
) -> ProxyScan:
    """Rank features by how much they reveal about the protected attribute.

    This is the reconnaissance step, and it costs one line. Run it *before*
    training, on whatever data you have. Every feature that scores high is a
    route the bias can travel down after the protected column is dropped —
    which is why "we don't collect gender" is not a fairness strategy but a
    measurement problem: you have removed your ability to check, not the effect.

    Two measures, both scaled 0 to 1 so they can share a column:

    * numbers → the correlation ratio η (see :func:`_correlation_ratio`)
    * categories → Cramér's V (see :func:`_cramers_v`)

    They are not identical quantities and a purist would not rank them against
    each other. They are close enough in spirit to sort a shortlist, and a
    shortlist is what this is for.

    **What a high score does and does not mean.** It means the feature carries
    information about the group. It does *not* mean the feature is illegitimate:
    experience carries information about age, and experience genuinely belongs
    in a pay model. The scan tells you where to look, not what to delete.
    Deleting every correlated feature usually leaves you with a model that
    cannot predict pay at all — see :func:`compare_with_without` for why
    deletion is the wrong tool in the first place.

    Args:
        df: The data.
        protected_col: The attribute to test against, e.g. ``"gender"``.
        candidate_cols: Features to score. Defaults to every other column.
        max_auto_levels: Numbers with more distinct values than this are scored
            as numbers; below it they are treated as labels, matching how
            :func:`paybands.fairness.audit.adjusted_gap` encodes them.
    """
    if protected_col not in df.columns:
        raise ValueError(f"{protected_col!r} is not a column; found {list(df.columns)}")
    if len(df) == 0:
        raise ValueError("cannot scan an empty frame")

    labels = _as_labels(df[protected_col])
    if candidate_cols is None:
        candidate_cols = [c for c in df.columns if c != protected_col]

    rows = []
    for name in candidate_cols:
        if name == protected_col:
            continue  # scoring a column against itself just prints 1.000
        if name not in df.columns:
            raise ValueError(f"candidate column {name!r} is not in the frame")

        column = df[name]
        usable = column.notna().to_numpy()
        if usable.sum() < 2:
            continue
        column, group_labels = column[usable], labels[usable]

        is_number = pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)
        if is_number and column.nunique() > max_auto_levels:
            kind = "numeric"
            association = _correlation_ratio(column.to_numpy(float), group_labels)
        else:
            kind = "categorical"
            association = _cramers_v(_as_labels(column), group_labels)

        rows.append(
            {
                "feature": name,
                "kind": kind,
                "association": association,
                "strength": _strength_label(association),
                "n_unique": int(column.nunique()),
            }
        )

    table = (
        pd.DataFrame(rows, columns=["feature", "kind", "association", "strength", "n_unique"])
        .sort_values("association", ascending=False)
        .reset_index(drop=True)
    )
    return ProxyScan(protected_column=protected_col, n=len(df), table=table)


# ─────────────────────────────────────────────────────── the demonstration


class _Regressor(Protocol):
    """The bit of the scikit-learn interface this module needs.

    Kept as a Protocol so :func:`compare_with_without` can be handed a
    ``LGBMRegressor``, a ``Ridge``, or the plain least-squares fit below without
    caring which. The demonstration should not depend on the model, and saying
    so in the type system is cheaper than saying it in a comment.
    """

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> object: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


class LeastSquares:
    """A linear model in six lines, so the default has nothing hidden in it.

    Deliberately the default. If the proxy demonstration only worked with
    gradient boosting, a reader could reasonably suspect the result was an
    artefact of a complicated model. It is not: the plainest possible fit
    reconstructs the gap, because the information is in the data rather than in
    the algorithm. Pass ``model_factory=lambda: LGBMRegressor(...)`` and the
    same thing happens.
    """

    def __init__(self) -> None:
        self.coefficients: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> LeastSquares:
        self.coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coefficients is None:
            raise RuntimeError("fit the model before predicting")
        return np.asarray(x @ self.coefficients, dtype=float)


@dataclass(frozen=True)
class ProxyComparison:
    """What happened when we removed the protected column. The headline result."""

    protected_column: str
    proxy_columns: tuple[str, ...]
    n_train: int
    n_test: int

    #: The gap actually present in the held-out salaries — what the models are
    #: being asked to reproduce.
    gap_in_data: AdjustedGap

    #: The gap the models' *predictions* carry, measured the same way.
    gap_with_protected: float
    gap_without_protected: float

    #: Average rupee error of each model, so the reader can see that removing
    #: the column bought nothing in accuracy either.
    mae_with_protected: float
    mae_without_protected: float

    @property
    def surviving_share(self) -> float:
        """Fraction of the gap that outlived deleting the column."""
        if self.gap_with_protected == 0:
            return float("nan")
        return self.gap_without_protected / self.gap_with_protected

    @property
    def accuracy_cost(self) -> float:
        """How much worse the blinded model is, as a fraction of the other's error."""
        return self.mae_without_protected / self.mae_with_protected - 1.0

    def __str__(self) -> str:
        return "\n".join(
            [
                f"THE PROXY DEMONSTRATION — dropping '{self.protected_column}'",
                f"{'':─<74}",
                f"trained on {self.n_train:,} rows, measured on {self.n_test:,} held out",
                f"the proxy channel: {', '.join(self.proxy_columns) or '(none named)'}",
                "",
                f"  gap present in the data          {self.gap_in_data.gap:>8.1%}   "
                f"({self.gap_in_data.ci_low:.1%} to {self.gap_in_data.ci_high:.1%})",
                f"  gap in predictions, WITH it      {self.gap_with_protected:>8.1%}",
                f"  gap in predictions, WITHOUT it   {self.gap_without_protected:>8.1%}   "
                f"← {self.surviving_share:.0%} of it survived",
                "",
                f"  average error, WITH it           {self.mae_with_protected:>10,.0f}",
                f"  average error, WITHOUT it        {self.mae_without_protected:>10,.0f}   "
                f"({self.accuracy_cost:+.1%})",
                "",
                "Deleting the column did not delete the disadvantage. The model rebuilt a",
                "large share of it out of the correlated features it was still allowed to",
                "see, and it did so at almost no cost to its accuracy — so nothing was even",
                "traded away for the harm that remains. 'We removed the protected column, so",
                "the model is fair' is not a weak claim; it is a false one, and this is the",
                "evidence.",
            ]
        )


def compare_with_without(
    df: pd.DataFrame,
    protected_col: str = "gender",
    *,
    target_col: str = "salary_annual",
    legitimate_controls: Sequence[str] | None = None,
    proxy_cols: Sequence[str] | None = None,
    test_size: float = 0.3,
    seed: int = 0,
    model_factory: Callable[[], _Regressor] | None = None,
    disadvantaged: str | None = None,
    reference: str | None = None,
) -> ProxyComparison:
    """Fit the same model with and without the protected column. Watch the gap survive.

    The procedure, and every step of it matters:

    1. Split the data into a training part and a held-out part. The gap is
       measured on rows the models never saw, so nobody can say the effect is
       memorisation.
    2. Fit **model A** on the legitimate controls, the proxy columns, *and* the
       protected column.
    3. Fit **model B** on exactly the same thing minus the protected column —
       the "we removed gender, so we're fine" model.
    4. Convert each model's predictions back into rupees and run
       :func:`~paybands.fairness.audit.adjusted_gap` over them, controlling for
       the **legitimate controls only**.

    Step 4 is the one people get wrong, so it is worth being explicit. We do not
    control for the proxy when measuring the gap. If we did, we would be
    subtracting the very channel we are trying to expose, and both models would
    report a clean zero — the audit would confirm the false claim instead of
    breaking it. Which is the same mistake in a different costume: a fairness
    number that has been adjusted until it is comfortable.

    **Is that fair to the model?** It is the right question. Career breaks
    really do reduce pay in this data, so part of what model B reproduces is a
    genuine effect and not laundered gender. That is precisely why this is hard,
    and precisely why "delete the column" is the wrong tool: the harmful and the
    legitimate arrive through the same wire. The answer is not a cleverer
    feature list — it is to measure the outcome, which is what this function
    does, and then to make a human decision about a career-break penalty that
    lands on one group nine times out of ten.

    Args:
        df: The data. Must contain the target, the protected column, and the
            controls.
        protected_col: The column to remove in model B.
        target_col: Salary column, in rupees.
        legitimate_controls: Defensible pay drivers. Defaults to whichever of
            :data:`~paybands.fairness.audit.LEGITIMATE_CONTROLS` are present.
        proxy_cols: Features that carry the protected attribute. Defaults to
            :data:`DEFAULT_PROXY_COLUMNS`. Find candidates with
            :func:`proxy_correlation`.
        test_size: Share held out. 0.3 is convention.
        seed: Controls the split, so the whole result is reproducible.
        model_factory: Zero-argument callable returning a fresh sklearn-style
            regressor. Defaults to :class:`LeastSquares`.
        disadvantaged: Group under study; auto-detected from pay when omitted.
        reference: Group compared against.
    """
    for column in (protected_col, target_col):
        if column not in df.columns:
            raise ValueError(f"{column!r} is not a column; found {list(df.columns)}")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}")

    controls = [
        c
        for c in (LEGITIMATE_CONTROLS if legitimate_controls is None else legitimate_controls)
        if c in df.columns
    ]
    if not controls:
        raise ValueError(
            "no legitimate controls found in the frame — the adjusted gap would then be the "
            "raw gap, which measures nothing. Pass `legitimate_controls=` explicitly."
        )
    proxies = [
        c
        for c in (DEFAULT_PROXY_COLUMNS if proxy_cols is None else proxy_cols)
        if c in df.columns and c not in controls
    ]

    frame = df.reset_index(drop=True)
    labels = _as_labels(frame[protected_col])
    salary = frame[target_col].to_numpy(float)
    low, high = _two_groups(labels, salary, disadvantaged, reference)
    keep = np.isin(labels, [low, high])
    frame, labels, salary = frame.loc[keep].reset_index(drop=True), labels[keep], salary[keep]

    # ── the split ──────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    n_test = max(2, int(round(len(frame) * test_size)))
    test_rows, train_rows = order[:n_test], order[n_test:]
    if len(train_rows) < 2:
        raise ValueError(f"{len(frame)} rows is too few to split into a train and test set")

    # ── the two feature matrices ───────────────────────────────────────
    #
    # Both are built from the WHOLE frame and then split by row. Encoding train
    # and test separately is a classic silent bug: a category that appears only
    # in the test set would produce a different set of dummy columns there, the
    # columns would no longer line up with the fitted coefficients, and the
    # predictions would be confidently meaningless.
    design_without = _design_matrix(frame[[*controls, *proxies]]).to_numpy(float)
    design_with = np.column_stack([design_without, (labels == low).astype(float)])

    log_salary = np.log(salary)
    build = model_factory if model_factory is not None else LeastSquares

    def fit_and_predict(design: NDArray[np.float64]) -> NDArray[np.float64]:
        model = build()
        model.fit(design[train_rows], log_salary[train_rows])
        return np.exp(np.asarray(model.predict(design[test_rows]), dtype=float))

    predicted_with = fit_and_predict(design_with)
    predicted_without = fit_and_predict(design_without)

    # ── measure the gap in each set of predictions ─────────────────────
    test_controls = frame.loc[test_rows, controls].reset_index(drop=True)
    test_labels = labels[test_rows]
    actual = salary[test_rows]

    def gap_of(values: NDArray[np.float64]) -> AdjustedGap:
        return adjusted_gap(values, test_labels, test_controls, disadvantaged=low, reference=high)

    # Only the gap in the real salaries gets an interval. The other two are
    # gaps in numbers a model produced, so a confidence interval there would
    # describe sampling noise in a deterministic function — a real number with
    # no real meaning. Point estimates only, on purpose.
    return ProxyComparison(
        protected_column=protected_col,
        proxy_columns=tuple(proxies),
        n_train=len(train_rows),
        n_test=len(test_rows),
        gap_in_data=gap_of(actual),
        gap_with_protected=gap_of(predicted_with).gap,
        gap_without_protected=gap_of(predicted_without).gap,
        mae_with_protected=float(np.mean(np.abs(actual - predicted_with))),
        mae_without_protected=float(np.mean(np.abs(actual - predicted_without))),
    )
