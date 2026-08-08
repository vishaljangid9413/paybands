"""`FeatureBuilder` — one object that turns a common-schema table into a model
matrix, fitted on train and applied to test.

Read `experience.py`, `skills.py` and `location.py` first; this file composes
them and then makes four decisions that are worth arguing about. Each one is
documented below with the reasoning, because in six months the reasoning is the
only part you'll wish you'd written down.

---

## Decision 1 — categories stay `category`, we do not one-hot encode

The reflex from tutorials is `pd.get_dummies`. Don't, here.

One-hot encoding turns a column with K values into K binary columns. `education`
has ~8 values, which is fine. But job titles, org sizes and — when company data
lands — internal grades run to dozens or hundreds, and one-hot on those produces
a wide, mostly-zero matrix. That hurts trees specifically:

* **Each split can only ask one yes/no question.** With one-hot, a tree asking
  about a 40-value column needs up to 39 splits to isolate a group. With a native
  categorical column, LightGBM can split "these 6 values on the left, the rest on
  the right" **in a single split**, because it sorts categories by their gradient
  statistics and finds the best partition directly.
* **Rare dummies get starved.** A column that is 1 for 20 rows out of 10,000
  almost never wins a split, so the information is effectively thrown away —
  while still costing memory and training time.

So: leave them as pandas `category` dtype and pass `categorical_feature="auto"`
to LightGBM, which then handles them natively. The one-hot version is only right
for linear models and neural nets, which this project doesn't use.

## Decision 2 — categories are frozen at `fit` time, and unseen values become NaN

`fit` records the exact category list from the *training* rows. `transform`
casts to that same `CategoricalDtype`. A value that shows up only in test — a new
role, a job family nobody had before — becomes NaN.

That is the correct behaviour, and it is the same argument as unseen skills in
`skills.py`: the model has no learned response to a category it never saw, so
there is nothing honest to do but say "unknown". Crucially, freezing the
categories also freezes their *integer codes*. If the codes were re-derived per
frame, "Backend" could be code 2 in training and code 5 at prediction time, and
the model would confidently answer the wrong question. That failure is silent —
no exception, just quietly wrong salaries.

## Decision 3 — we never fill missing values with 0

LightGBM handles NaN natively: at each split it tries sending the missing rows
left and right, and keeps whichever reduces the loss. It learns what missingness
means rather than being told.

`fillna(0)` is not neutral. Zero is a **claim**: zero years of experience means
fresher, zero previous salary means unemployed. A missing value is the absence of
a claim. Conflating the two puts confident, wrong rows into training — and
because it lowers the numbers, it systematically drags predicted bands down for
people whose records happened to be incomplete. That's not a rounding error, it's
a fairness problem with a specific direction.

The one thing we *do* fill is column *presence*: if a source lacks a column
entirely, we emit an all-NaN column so the matrix is the same width everywhere.
That's shape, not content.

## Decision 4 — what we deliberately leave out

* **`prev_salary`** — off by default (`include_prev_salary=False`). See the
  previous-salary trap in `docs/design.md` §4.1: anchoring to last drawn salary
  makes an early underpayment follow someone for their whole career, and a model
  fed that column learns to do it automatically, at scale. Phase 4 trains both
  versions and compares them; the flag exists so that experiment is a one-line
  change rather than a rewrite.
* **`gender`, `age_band`** — never features. They are carried through the schema
  for the fairness *audit*, which measures whether predictions differ across
  groups. Feeding them in would be both a legal problem and a circular one.
  (Dropping them is not sufficient for fairness — proxies survive — which is the
  demonstration in Phase 4. It is still necessary.)
* **`source`** — which loader produced the row. Pure dataset identity, not a
  property of a person. Left in, the model would learn "synthetic rows pay ₹X",
  which tells us precisely nothing about a real candidate.
* **`salary_annual`** — the target. Obvious, but the assertion that it never
  appears in the feature matrix is a real test in `tests/test_features.py`,
  because "the target leaked into the features" is a mistake people genuinely
  ship, and it produces a suspiciously perfect model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import TARGET
from .experience import ExperienceFeatures
from .location import LocationFeatures
from .skills import DEFAULT_TOP_N, SkillFeatures

#: Columns treated as native LightGBM categoricals when present.
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "role",
    "education",
    "org_size",
    "remote",
    "employment_type",
    "level",
    "institute_tier",
    "prev_company_type",
)

#: Numeric columns passed straight through when present. Short list on purpose:
#: most numeric signal in this schema comes through the experience block.
#:
#: `survey_year` is here because pooling seven years of survey data without it
#: is actively harmful. The median Indian salary in this survey doubled between
#: 2019 (₹7.5L) and 2025 (₹15L); with no year column the model cannot tell a
#: well-paid 2019 respondent from an underpaid 2024 one, so six years of real
#: wage growth land in the residual and the band gets WIDER as data is added.
PASSTHROUGH_NUMERIC: tuple[str, ...] = ("performance_rating", "survey_year")

#: Numeric columns that must NOT be left as NaN when a caller omits them.
#:
#: A candidate being priced today has no survey year — they are not a survey
#: response. Passing NaN would land them wherever LightGBM sends missing values,
#: which is a weighted average of six years including 2019 prices. Filling with
#: the newest year the model was trained on asks the question the caller
#: actually means: what is this person worth at the most recent market we have
#: evidence for.
_FILL_WITH_TRAINING_MAX: frozenset[str] = frozenset({"survey_year"})

#: Never features, whatever a loader hands us. See Decision 4.
EXCLUDED_COLUMNS: frozenset[str] = frozenset({TARGET, "source", "gender", "age_band", "event_date"})


class FeatureBuilder:
    """Compose the feature blocks into one model-ready frame.

        builder = FeatureBuilder().fit(train_df)     # train ONLY
        X_train = builder.transform(train_df)
        X_test = builder.transform(test_df)          # identical columns

    Never call `fit` on the full dataset before splitting. Every learned thing in
    this pipeline — the skill vocabulary, the category lists — would then have
    been chosen with knowledge of the test rows, and the test score would stop
    measuring what you think it measures.

    Fitted attributes:

    * `feature_names_` — the exact columns, in order, that `transform` produces.
    * `categorical_features_` — the subset LightGBM should treat as categorical.
    * `skills` / `experience` / `location` — the sub-transformers, kept so you
      can inspect e.g. `builder.skills.vocabulary_` after fitting.
    """

    def __init__(
        self,
        *,
        top_skills: int = DEFAULT_TOP_N,
        city_column: str = "city",
        include_sqrt_experience: bool = True,
        include_prev_salary: bool = False,
    ) -> None:
        self.include_prev_salary = include_prev_salary
        self.experience = ExperienceFeatures(include_sqrt=include_sqrt_experience)
        self.skills = SkillFeatures(top_n=top_skills)
        self.location = LocationFeatures(city_column=city_column)

        self.feature_names_: list[str] = []
        self.categorical_features_: list[str] = []
        self.category_levels_: dict[str, list[object]] = {}
        self.numeric_features_: list[str] = []
        self.numeric_fill_: dict[str, float] = {}

    # ── fit ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        """Learn everything that needs learning, from these rows only."""
        self.experience.fit(df)
        self.skills.fit(df)
        self.location.fit(df)

        # Category vocabularies, frozen from the training rows. Sorted so the
        # codes are deterministic across runs — an unsorted `unique()` depends on
        # row order, and a model whose feature encoding depends on row order is a
        # reproducibility bug waiting to be discovered at the worst moment.
        self.category_levels_ = {}
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns and col not in EXCLUDED_COLUMNS:
                levels = sorted({str(v) for v in df[col].dropna().unique()})
                if levels:
                    self.category_levels_[col] = levels

        self.numeric_features_ = [
            col for col in PASSTHROUGH_NUMERIC if col in df.columns and col not in EXCLUDED_COLUMNS
        ]
        # Learned from the TRAINING split only, like every other vocabulary here.
        # Taking the max over the whole frame would leak the test split's newest
        # year into how training rows are encoded.
        self.numeric_fill_ = {
            col: pd.to_numeric(df[col], errors="coerce").max()
            for col in self.numeric_features_
            if col in _FILL_WITH_TRAINING_MAX
        }
        if self.include_prev_salary and "prev_salary" in df.columns:
            self.numeric_features_.append("prev_salary")

        # Order: experience → location → categoricals → other numerics → skills.
        # Fixed and explicit, because LightGBM identifies features by position as
        # well as by name; a column order that drifts between fit and transform
        # is another silently-wrong-answers bug.
        self.categorical_features_ = ["experience_bucket", *self.category_levels_]
        self.feature_names_ = [
            *self.experience.feature_names_,
            *self.location.feature_names_,
            *self.category_levels_,
            *self.numeric_features_,
            *self.skills.feature_names_,
        ]
        return self

    # ── transform ──────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline. Returns one row out per row in, same index.

        Row count is preserved unconditionally — this function never drops or
        filters. Dropping rows inside a transform silently desynchronises `X`
        from `y`, and the resulting misalignment is very hard to spot.
        """
        if not self.feature_names_:
            raise RuntimeError("call fit() before transform()")

        blocks = [
            self.experience.transform(df),
            self.location.transform(df),
            self._categorical_block(df),
            self._numeric_block(df),
            self.skills.transform(df),
        ]
        out = pd.concat(blocks, axis=1)

        if len(out) != len(df):
            raise RuntimeError(f"row count changed: {len(df)} in, {len(out)} out")
        return out[self.feature_names_]

    def _categorical_block(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col, levels in self.category_levels_.items():
            dtype = pd.CategoricalDtype(categories=levels)
            known = set(levels)
            if col in df.columns:
                # str() first so 3 and "3" don't become two different categories.
                # Anything not in `known` — unseen in training, or missing — is
                # mapped to None *before* the cast, which becomes NaN. See
                # Decision 2. Doing the check here rather than letting pandas
                # drop the value keeps the behaviour explicit and deliberate.
                values = [str(v) if not pd.isna(v) and str(v) in known else None for v in df[col]]
            else:
                # Column absent from this source entirely: an all-unknown column,
                # so the matrix keeps its width. Shape, not content.
                values = [None] * len(df)
            out[col] = pd.Categorical(values, dtype=dtype)
        return out

    def _numeric_block(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.numeric_features_:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").astype("float64")
            else:
                values = pd.Series(np.nan, index=df.index, dtype="float64")
            # Frozen at fit time, so a prediction is dated by the training data
            # rather than by whenever the code happens to run.
            if col in _FILL_WITH_TRAINING_MAX:
                values = values.fillna(self.numeric_fill_.get(col, np.nan))
            out[col] = values
        return out

    # ── convenience ────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Only ever valid on the training split. Named to make misuse visible."""
        return self.fit(df).transform(df)

    def describe(self) -> str:
        """A short human summary — the kind of thing to paste into the findings log."""
        if not self.feature_names_:
            return "FeatureBuilder(unfitted)"
        prev = "included" if self.include_prev_salary else "EXCLUDED (see design.md §4.1)"
        return "\n".join(
            [
                f"FeatureBuilder — {len(self.feature_names_)} features",
                f"  categorical   {len(self.categorical_features_):>3}  "
                f"{', '.join(self.categorical_features_)}",
                f"  skill flags   {len(self.skills.vocabulary_):>3}  "
                f"{', '.join(self.skills.vocabulary_[:10])}...",
                f"  prev_salary   {prev}",
            ]
        )
