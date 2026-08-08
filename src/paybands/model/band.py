"""The band model — three gradient boosting models that together draw a range.

This is the file the whole project has been walking towards. Everything before
it predicted *one number*. This one predicts a **band**: ₹18L–₹24L, with a
middle at ₹21L, which is the only shape of answer a recruiter can actually use.

Read `baseline.py` first if you haven't. The number this file has to beat is the
one that comes out of a four-line `groupby`, and if it doesn't beat it, the
honest engineering decision is to ship the `groupby`.

---

## Idea 1 — train in log space, report in rupees

Salary is **right-skewed**: most people cluster low, a few earn enormously. Worse
for us, salary differences are *multiplicative*. Nobody has ever said "I got a
₹2.4 lakh raise" as a way of describing how well they did — they say "I got a
30% raise", because 30% means the same thing to a fresher and to a principal
engineer while ₹2.4L means wildly different things to the two.

So the model learns `log(salary)`, where multiplying becomes adding and "30%
more" is a constant step at every salary level. Predictions come back through
``exp`` before anyone sees them, and every reported score is in rupees.

**The subtlety, and it is a real one.** ``exp(log(x))`` is ``x`` for a single
number, but ``exp(mean of logs)`` is *not* the mean — it is the **geometric
mean**, and it is always lower whenever the numbers differ at all. Three
salaries of ₹10L, ₹20L and ₹40L have an arithmetic mean of ₹23.3L and a
geometric mean of ₹20L.

For a salary band that is the behaviour we want: we are describing the *typical*
person like this candidate, not an average dragged upward by one founder who
wrote down their equity. But if you ever need a true expected value — "what will
these 40 hires cost us in total?" — exponentiating a log-space model is biased
low and you need Duan's smearing estimator. `metrics.from_log` says the same
thing at more length; it is worth reading twice.

---

## Idea 2 — three models, not one model with an error bar

The obvious way to build a band is: train one model, measure how wrong it
usually is, and put ±that around every prediction. **Don't.** That band has one
width, so it is the same width for a fresher and for a CTO.

Which is wrong in both directions at once. Junior salaries genuinely cluster —
the market for a 1-year backend engineer in Bangalore is narrow, and a ±₹6L band
around it is uselessly vague. Senior and leadership salaries genuinely spread —
two CTOs can be a factor of three apart — and a ±₹6L band there is confidently
too narrow, which is worse than vague because someone will believe it.

So instead we train **three separate models**, each with LightGBM's
``objective="quantile"``:

| ``alpha`` | what it learns | role in the band |
|---|---|---|
| 0.1 | the 10th percentile | the low edge |
| 0.5 | the 50th percentile (median) | the middle |
| 0.9 | the 90th percentile | the high edge |

A **quantile** is a cut point: the 90th percentile is the number 90% of people
earn less than. Each model is fitted with **pinball loss** (see
`metrics.pinball_loss`), which charges asymmetrically — for the 0.9 model,
under-predicting costs nine times as much as over-predicting, so the cheapest
prediction it can make is one that sits above the truth 90% of the time. The
metric *is* the goal.

The gap between the 0.1 and 0.9 predictions is the band, and because all three
are full models with access to experience, role and location, that gap is free
to be narrow for the fresher and wide for the CTO. It is learned, not assumed.

Nominal coverage is 0.9 − 0.1 = **80%**. Whether the band *delivers* 80% is a
completely separate question, and the answer is usually no — which is what
`conformal.py` exists to fix.

---

## Idea 3 — quantile crossing, and why we count it rather than hide it

Nothing ties the three models together. They are fitted independently, on the
same rows, with three different loss functions. So on some rows the 0.1 model
will predict *above* the 0.9 model. The band comes out inside-out.

This is not a bug in LightGBM; it is the direct consequence of not constraining
the models jointly, and it happens most on rows where the data is thin — exactly
the rows where you were least sure anyway.

The repair is embarrassingly simple: sort the three numbers per row. That is
provably no worse than the originals under pinball loss (it is a special case of
the rearrangement result of Chernozhukov, Fernández-Val and Galichon), so there
is no reason not to do it.

The *count* is the interesting part, and this module reports it. A crossing rate
of 1% means the three models mostly agree about the shape of the world. A
crossing rate of 15% means they don't, and your band edges are being determined
by noise. That is a diagnostic worth putting in a report, not a wart to hide.

---

## Idea 4 — small data needs a small model

This dataset is about 1,000 rows. LightGBM's defaults (31 leaves, unlimited
depth, 100 trees, no regularisation) were chosen for datasets a thousand times
larger, and on 1,000 rows they will happily grow a tree with one row per leaf and
memorise the training set perfectly.

You would then see a beautiful training score and a test score worse than the
`groupby` baseline. The instinct to internalise: **model capacity should be
matched to the amount of evidence, not to the size of the library you imported.**

Concretely, `SMALL_DATA_PARAMS` below turns every dial towards "less model":
shallow trees, a floor on how few people a leaf may describe, sampling of both
rows and columns, L2 regularisation, a small learning rate — and then **early
stopping** on a held-out validation slice, which lets the data decide how many
trees are actually justified instead of us guessing.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ..features.builder import FeatureBuilder
from .metrics import coverage, format_rupees, from_log, to_log
from .split import DEFAULT_SEED

#: The three cut points, low to high. The middle one is the median, and it is
#: the number the compa-ratio divides by — so changing it to anything other than
#: 0.5 changes what "compa-ratio" means to every HR person who reads the output.
DEFAULT_QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)

#: LightGBM settings for a ~1,000-row dataset. See Idea 4 above; each line is a
#: deliberate step away from the defaults, and each has a reason next to it.
SMALL_DATA_PARAMS: dict[str, object] = {
    "objective": "quantile",  # pinball loss — the whole point of this file
    "n_estimators": 800,  # an upper bound only; early stopping picks the real number
    "learning_rate": 0.03,  # small steps, so early stopping has a fine-grained choice
    "num_leaves": 7,  # a depth-3 tree at most. Default is 31 — far too expressive here
    "max_depth": 3,  # belt and braces: caps interaction depth as well as leaf count
    "min_child_samples": 25,  # a leaf must describe 25 real people, not 3 outliers
    "subsample": 0.8,  # each tree sees 80% of rows...
    "subsample_freq": 1,  # ...and this is what actually switches that on
    "colsample_bytree": 0.7,  # ...and 70% of columns, so no one feature dominates
    "reg_lambda": 5.0,  # L2 on leaf values: pulls extreme predictions inward
    "verbose": -1,  # LightGBM is chatty about tiny data; we report our own numbers
}

#: How many trees to build when there is too little data to early-stop on. A
#: fixed, deliberately modest number — see `MIN_VALIDATION_ROWS`.
NO_EARLY_STOPPING_TREES = 300

#: Below this many validation rows, early stopping is switched off entirely.
#:
#: **Why.** Early stopping asks "did the validation score improve?" — on 15 rows
#: that question is answered by noise, and you would stop at whatever iteration
#: happened to get lucky. Fewer trees chosen blindly beats a stopping point
#: chosen by coin flip.
MIN_VALIDATION_ROWS = 40

#: How many rounds without improvement before we stop adding trees.
EARLY_STOPPING_ROUNDS = 50

#: LightGBM 4.7 renamed ``fit(eval_set=[(X, y)])`` to ``fit(eval_X=..., eval_y=...)``
#: and deprecated the old spelling. `pyproject.toml` allows 4.5 and up, so we ask
#: the installed version which one it speaks rather than picking one and emitting
#: a deprecation warning on every fit — or crashing on an older install.
_FIT_TAKES_EVAL_XY = "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters

# ─────────────────────────────────────────────── compa-ratio thresholds
#
# Standard HR vocabulary (see docs/design.md §4.2). These are policy, not
# measurement — a company can and does move them — so they sit here as named
# constants rather than being written into a comparison somewhere.

COMPA_BELOW_BAND = 0.90
COMPA_ABOVE_BAND = 1.10


def compa_label(ratio: float) -> str:
    """Turn a compa-ratio into the word HR would use for it."""
    if ratio < COMPA_BELOW_BAND:
        return "below band"
    if ratio > COMPA_ABOVE_BAND:
        return "above band"
    return "in band"


# ─────────────────────────────────────────────── what a prediction looks like


@dataclass(frozen=True)
class BandPrediction:
    """One band per row, in rupees, plus the honesty diagnostics.

    Frozen, because a predictions object you can edit after the fact is a
    predictions object nobody can audit.
    """

    lower: NDArray[np.float64]
    median: NDArray[np.float64]
    upper: NDArray[np.float64]

    #: The quantile levels these came from, so a reader never has to guess
    #: whether "the band" meant 80% or 50%.
    quantiles: tuple[float, float, float]

    #: How many rows had their three predictions out of order before we sorted
    #: them. See Idea 3 — this is a diagnostic, not a failure.
    n_crossed: int = 0

    @property
    def n(self) -> int:
        return len(self.median)

    @property
    def nominal_coverage(self) -> float:
        """What fraction of salaries this band *claims* to contain."""
        return self.quantiles[2] - self.quantiles[0]

    @property
    def width(self) -> NDArray[np.float64]:
        """Band width in rupees, per row."""
        return self.upper - self.lower

    @property
    def relative_width(self) -> NDArray[np.float64]:
        """Band width as a multiple of the median prediction.

        The comparable number across salary levels. A ₹6L band is enormous at
        ₹8L and tight at ₹60L; 0.4 is 0.4 everywhere.
        """
        return self.width / self.median

    @property
    def crossing_rate(self) -> float:
        return self.n_crossed / self.n if self.n else 0.0

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        """The band as a table, for eyeballing or writing to disk."""
        return pd.DataFrame(
            {"lower": self.lower, "median": self.median, "upper": self.upper},
            index=index,
        )

    def explain(self, i: int = 0) -> str:
        """One row, written the way it would be read aloud in a hiring meeting."""
        return (
            f"{format_rupees(self.lower[i])} – {format_rupees(self.upper[i])}  "
            f"(midpoint {format_rupees(self.median[i])}, "
            f"{self.nominal_coverage:.0%} band)"
        )


@dataclass(frozen=True)
class BandReport:
    """How good a band is — which takes exactly two numbers, never one.

    **Coverage** says whether the band is honest: of the people we said had an
    80% chance of landing inside their band, how many did?

    **Width** says whether it is useful: a band of ₹0 to ₹5Cr has perfect
    coverage and helps nobody.

    Read them together and in that order. **Narrower is better only once
    coverage holds** — a model that reports a tighter band and worse coverage
    has not improved, it has started lying more confidently.
    """

    n: int
    label: str
    target_coverage: float
    coverage: float
    mean_width: float
    median_width: float
    mean_relative_width: float
    crossing_rate: float = 0.0

    @property
    def coverage_gap(self) -> float:
        """Actual minus promised. Negative means the band is overconfident."""
        return self.coverage - self.target_coverage

    def __str__(self) -> str:
        verdict = "overconfident" if self.coverage_gap < -0.02 else "honest"
        if self.coverage_gap > 0.05:
            verdict = "conservative (wider than it needs to be)"
        return "\n".join(
            [
                f"{self.label}  (n = {self.n:,})",
                f"  promised coverage  {self.target_coverage:>13.1%}",
                f"  actual coverage    {self.coverage:>13.1%}   ← {verdict}",
                f"  mean band width    {format_rupees(self.mean_width):>13}",
                f"  median band width  {format_rupees(self.median_width):>13}",
                f"  width ÷ midpoint   {self.mean_relative_width:>13.2f}   "
                "(comparable across salary levels)",
                f"  quantile crossings {self.crossing_rate:>13.1%}   of rows, before repair",
            ]
        )


def band_report(
    y_true_rupees: ArrayLike,
    band: BandPrediction,
    *,
    label: str = "band",
) -> BandReport:
    """Score a band against the salaries people actually earn."""
    yt = np.asarray(y_true_rupees, dtype=float)
    if yt.shape != band.median.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape}, band {band.median.shape}")
    return BandReport(
        n=band.n,
        label=label,
        target_coverage=band.nominal_coverage,
        coverage=coverage(yt, band.lower, band.upper),
        mean_width=float(np.mean(band.width)),
        median_width=float(np.median(band.width)),
        mean_relative_width=float(np.mean(band.relative_width)),
        crossing_rate=band.crossing_rate,
    )


# ─────────────────────────────────────────────── the model


class SalaryBandModel:
    """Three quantile LightGBM models that together predict a salary band.

    Follows the same ``fit`` / ``predict`` shape as the baselines, so it drops
    into any evaluation loop written for them::

        model = SalaryBandModel(seed=42).fit(train_df, train_df["salary_annual"])
        band = model.predict_band(test_df)
        print(band_report(test_df["salary_annual"], band))

    ``X`` is a **common-schema frame** (see `data/schema.py`), not a feature
    matrix — the model owns its `FeatureBuilder` and fits it on the training
    rows only. That is not a convenience: a builder fitted on all the data has
    chosen its skill vocabulary and its category lists with knowledge of the
    test rows, which is **leakage**, and it makes the test score a flattering
    lie. Keeping the builder inside the model makes that mistake impossible to
    make by accident.

    Fitted attributes (trailing underscore = learned, the scikit-learn
    convention):

    * ``models_`` — the three fitted LightGBM regressors, low to high.
    * ``best_iterations_`` — how many trees each one actually kept. Read these:
      if they all sit at ``n_estimators`` the model never stopped improving and
      you capped it too low; if they sit at 20 there was almost no signal.
    * ``n_crossed_`` / ``crossing_rate_`` — from the most recent prediction.
    * ``train_index_`` — the index labels used for training. Carried purely so
      `conformal.ConformalBand` can *refuse* to calibrate on rows the model has
      already seen. See that module for why that matters so much.
    """

    def __init__(
        self,
        *,
        quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
        seed: int = DEFAULT_SEED,
        validation_size: float = 0.2,
        params: dict[str, object] | None = None,
        top_skills: int | None = None,
        include_prev_salary: bool = False,
    ) -> None:
        lo, mid, hi = quantiles
        if not 0 < lo < mid < hi < 1:
            raise ValueError(
                f"quantiles must be strictly increasing and inside (0, 1), got {quantiles}"
            )
        if not 0 < validation_size < 1:
            raise ValueError(f"validation_size must be in (0, 1), got {validation_size}")

        self.quantiles = quantiles
        self.seed = seed
        self.validation_size = validation_size
        self.params = {**SMALL_DATA_PARAMS, **(params or {})}

        builder_kwargs: dict[str, object] = {"include_prev_salary": include_prev_salary}
        if top_skills is not None:
            builder_kwargs["top_skills"] = top_skills
        self.builder = FeatureBuilder(**builder_kwargs)  # type: ignore[arg-type]

        self.models_: list[lgb.LGBMRegressor] = []
        self.best_iterations_: list[int] = []
        self.feature_names_: list[str] = []
        self.train_index_: frozenset[object] = frozenset()
        self.n_train_: int = 0
        self.n_validation_: int = 0
        self.n_crossed_: int = 0
        self.crossing_rate_: float = 0.0

    # ── properties ─────────────────────────────────────────────────────

    @property
    def nominal_coverage(self) -> float:
        """What the band claims: 0.9 − 0.1 = 80%. A claim, not yet a fact."""
        return self.quantiles[2] - self.quantiles[0]

    @property
    def is_fitted(self) -> bool:
        return bool(self.models_)

    # ── fit ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> SalaryBandModel:
        """Fit all three quantile models on log(salary).

        Args:
            X: A common-schema frame. The target column may be present — the
                `FeatureBuilder` excludes it explicitly, and there is a test
                that it never reaches the feature matrix.
            y: Annual salaries **in rupees**. Converted to logs in here so that
                no caller ever has to remember to do it, and so that nobody can
                accidentally fit one model on logs and another on rupees.

        **What happens to the validation slice.** We carve a slice out of the
        rows you hand us and use it only to decide when to stop adding trees.
        It stays inside your training data — the test set is untouched — so this
        costs nothing you were entitled to. The `FeatureBuilder` is fitted on
        *all* the rows given here, validation slice included, because that slice
        is training data too; only the test rows must be kept out.
        """
        salaries = np.asarray(y, dtype=float)
        if salaries.size != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {salaries.size}")
        if salaries.size == 0:
            raise ValueError("cannot fit on an empty target")

        y_log = to_log(salaries)  # raises on a non-positive salary, which is right

        features = self.builder.fit(X).transform(X)
        self.feature_names_ = list(self.builder.feature_names_)
        self.train_index_ = frozenset(X.index)
        self.n_train_ = len(X)

        # A local Generator, never np.random.seed(). Global random state is
        # shared by every library in the process, so anything else drawing a
        # number would silently change this "reproducible" split.
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X))
        n_val = int(round(len(X) * self.validation_size))
        n_val = min(n_val, len(X) - 1)  # never leave the fitting set empty
        val_pos, fit_pos = order[:n_val], order[n_val:]
        self.n_validation_ = int(n_val)

        use_early_stopping = n_val >= MIN_VALIDATION_ROWS
        X_fit = features.iloc[fit_pos]
        y_fit = y_log[fit_pos]

        self.models_ = []
        self.best_iterations_ = []
        for alpha in self.quantiles:
            params = dict(self.params)
            params["alpha"] = alpha
            if not use_early_stopping:
                params["n_estimators"] = min(
                    int(params.get("n_estimators", NO_EARLY_STOPPING_TREES)),
                    NO_EARLY_STOPPING_TREES,
                )

            model = lgb.LGBMRegressor(random_state=self.seed, **params)

            if use_early_stopping:
                X_val, y_val = features.iloc[val_pos], y_log[val_pos]
                eval_kwargs: dict[str, object] = (
                    {"eval_X": X_val, "eval_y": y_val}
                    if _FIT_TAKES_EVAL_XY
                    else {"eval_set": [(X_val, y_val)]}
                )
                # No eval_metric argument on purpose: LightGBM then scores the
                # validation set with the objective itself — pinball loss at
                # this very alpha. Early-stopping on the metric you are actually
                # optimising is the point; stopping on, say, RMSE would drag the
                # 0.9 model back towards the middle, undoing its whole job.
                model.fit(
                    X_fit,
                    y_fit,
                    categorical_feature=self.builder.categorical_features_,
                    callbacks=[
                        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                    **eval_kwargs,  # type: ignore[arg-type]
                )
            else:
                model.fit(
                    X_fit,
                    y_fit,
                    categorical_feature=self.builder.categorical_features_,
                )

            self.models_.append(model)
            self.best_iterations_.append(int(model.best_iteration_ or model.n_estimators_))

        return self

    # ── predict ────────────────────────────────────────────────────────

    def predict_quantiles_log(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """The three quantiles in **log space**, sorted, shape ``(n_rows, 3)``.

        This is the primitive everything else is built on, and it is the method
        `conformal.ConformalBand` requires of anything it wraps.

        **Why log space is the interface.** Conformal calibration widens a band
        by a constant amount. A constant in log space is a constant *percentage*
        in rupees — the band grows by 12% at every salary level. A constant in
        rupees would add ₹3L to the fresher's band and ₹3L to the CTO's, which
        is the fixed-width mistake from Idea 2 sneaking back in through the
        calibration step.

        The sort is the quantile-crossing repair from Idea 3. The count of rows
        that needed it is recorded on ``n_crossed_``.
        """
        if not self.is_fitted:
            raise RuntimeError("call fit() before predicting")

        features = self.builder.transform(X)
        raw = np.column_stack([m.predict(features) for m in self.models_])

        repaired = np.sort(raw, axis=1)
        # Exact float comparison is correct here: sorting permutes values, it
        # never alters them, so a row is unchanged iff it was already in order.
        self.n_crossed_ = int(np.count_nonzero(np.any(raw != repaired, axis=1)))
        self.crossing_rate_ = self.n_crossed_ / len(raw) if len(raw) else 0.0
        return repaired

    def predict_band(self, X: pd.DataFrame) -> BandPrediction:
        """The band, in rupees. The method the API will call."""
        q = self.predict_quantiles_log(X)
        return BandPrediction(
            lower=from_log(q[:, 0]),
            median=from_log(q[:, 1]),
            upper=from_log(q[:, 2]),
            quantiles=self.quantiles,
            n_crossed=self.n_crossed_,
        )

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """The midpoint alone, in rupees.

        Exists so this model is interchangeable with the baselines in any
        point-prediction scoring loop. Prefer :meth:`predict_band` — a single
        number is exactly the fake precision this project set out to avoid.
        """
        return self.predict_band(X).median

    # ── the number both use cases run on ───────────────────────────────

    def compa_ratio(self, actual_salary: ArrayLike, X: pd.DataFrame) -> NDArray[np.float64]:
        """``actual ÷ predicted median``. One division, and the centre of the tool.

        Below 0.90 → paid below band, a genuine flight risk and the pay-equity
        list. 0.90–1.10 → in band. Above 1.10 → above band, so a smaller
        increment. See `docs/design.md` §4.2; the thresholds are
        `COMPA_BELOW_BAND` and `COMPA_ABOVE_BAND` above.

        Both headline use cases are this number. **Pay equity** is everyone
        below 0.90. **Increments** are "more to the low ratios, less to the high
        ones, within budget". No second model is needed for either.

        Note the denominator is the *median*, not the midpoint of lower and
        upper. On a skewed distribution those differ, and the median is the one
        HR means by "the middle of the band" — it is the salary half the market
        is below.
        """
        actual = np.asarray(actual_salary, dtype=float)
        if actual.ndim == 0:
            actual = actual.reshape(1)
        band = self.predict_band(X)
        if actual.shape != band.median.shape:
            raise ValueError(
                f"shape mismatch: {actual.shape} salaries against {band.median.shape} rows"
            )
        return actual / band.median

    # ── reporting ──────────────────────────────────────────────────────

    def describe(self) -> str:
        """A short summary to paste into the findings log."""
        if not self.is_fitted:
            return "SalaryBandModel(unfitted)"
        stopping = (
            f"early stopping on {self.n_validation_:,} rows"
            if self.n_validation_ >= MIN_VALIDATION_ROWS
            else f"early stopping OFF ({self.n_validation_} validation rows is too few)"
        )
        trees = ", ".join(
            f"q{q:g}: {n}" for q, n in zip(self.quantiles, self.best_iterations_, strict=True)
        )
        return "\n".join(
            [
                f"SalaryBandModel — quantiles {self.quantiles}, "
                f"nominal coverage {self.nominal_coverage:.0%}",
                f"  trained on         {self.n_train_:,} rows, {len(self.feature_names_)} features",
                f"  {stopping}",
                f"  trees kept         {trees}",
                f"  last prediction    {self.crossing_rate_:.1%} of rows needed crossing repair",
            ]
        )
