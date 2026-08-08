"""Experience features — because "years" is not a straight line to rupees.

## The one idea in this file

Look at the median Indian developer salaries by experience (Stack Overflow 2025,
India, INR):

```
     0–1 yrs   ₹5.4L        ┐
     2–3 yrs   ₹7.0L        │  steep: ~₹1.6L per extra year
     4–5 yrs   ₹12.0L       │
     6–8 yrs   ₹22.0L       ┘
     9–12 yrs  ₹30.0L       ┐
    13–20 yrs  ₹40.0L       │  flattening: ~₹1.0L per extra year, then ~₹0
      20+ yrs  ₹40.0L       ┘
```

Between year 3 and year 7 a developer's pay roughly triples. Between year 15 and
year 25 it barely moves. **The same "+1 year" is worth wildly different money
depending on where you already are.**

### Why a raw `years_experience` column alone would mislead the model

A linear model fits one number: rupees per year. It has to pick a single slope
that compromises between the steep start and the flat end — say ₹1.7L/year. Then
it predicts ₹5L for a fresher (roughly right by luck), ₹22L at year 10 (too low),
and ₹42L at year 25 (too high, and it keeps climbing forever — extrapolating a
40-year veteran to ₹68L, which is not how the market works).

Gradient-boosted trees are not linear, so they *can* learn the curve — but only
by spending many splits discovering it, and only where they have enough rows.
Above 15 years the survey is thin, so the tree spends its scarce data relearning
a shape we already know.

**So we hand the shape to the model instead of making it rediscover it:**

* `experience_log` — `log1p(years)`. On a log scale the early steep years get
  stretched out and the late flat years get squashed, which is exactly the shape
  above. One linear slope on `log1p` means "each year is worth a *percentage*",
  which matches how raises actually work.
* `experience_bucket` — the seven bands above, as a category. A tree can isolate
  "6–8 years" in a single split rather than finding two thresholds.
* `experience_sqrt` — a gentler curve than log, kept because which one wins is an
  empirical question, not something to argue about. Let the model choose.

We keep the raw years column too. Trees are happy to ignore what they don't need,
and dropping a column to *prove a point* is a worse habit than keeping a
redundant one.

## Fit / transform, and why `fit` does nothing here

Every transformer in this package has `fit` and `transform`, even when — as here
— `fit` learns nothing. That is not ceremony, it is *information*: `log1p(7)` is
7 whether it came from the train set or the test set, so nothing about the
training data is baked in, so **nothing can leak**. Compare `skills.py`, where
`fit` genuinely learns a vocabulary from the training rows and therefore must
never see the test rows. Keeping the same interface makes that difference the
thing you notice, instead of hiding it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Nobody has 60 years of professional software experience, and negative years
#: are impossible. Values outside this range are data entry errors (someone typed
#: their birth year, or a "-1" placeholder for "prefer not to say"). We turn them
#: into NaN — "we don't know" — rather than clipping them to 50, because clipping
#: invents a confident claim out of a typo.
MAX_PLAUSIBLE_YEARS = 50.0

#: Bucket edges, left-closed: [0,2) [2,4) [4,6) [6,9) [9,13) [13,21) [21,∞).
#: These follow the median table at the top of the file — they are cut where the
#: slope changes, not at round numbers for the sake of round numbers.
BUCKET_EDGES: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 9.0, 13.0, 21.0, np.inf)
BUCKET_LABELS: tuple[str, ...] = ("0-1", "2-3", "4-5", "6-8", "9-12", "13-20", "20+")

#: The dtype for the bucket column. Declaring the categories up front (rather
#: than letting pandas infer them from whatever happens to be in the frame) means
#: a test batch of three juniors still produces all seven categories, so the
#: columns line up with what the model was trained on.
BUCKET_DTYPE = pd.CategoricalDtype(categories=BUCKET_LABELS, ordered=True)


def clean_years(values: pd.Series | list[float]) -> pd.Series:
    """Coerce to numbers and blank out impossible values.

    Returns a float Series where anything non-numeric, negative, or above
    :data:`MAX_PLAUSIBLE_YEARS` has become NaN.

    Why NaN and not 0: a 0 means "this person is a fresher", which is a *claim*.
    If someone typed "-3", we have no idea what they meant, and saying "fresher"
    on their behalf would put a wrong row into the model with full confidence.
    NaN says "unknown", and LightGBM knows what to do with unknown.
    """
    years = pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")
    return years.where((years >= 0) & (years <= MAX_PLAUSIBLE_YEARS))


def log_years(years: pd.Series) -> pd.Series:
    """`log1p(years)` — the workhorse transform.

    `log1p(x)` is `log(1 + x)`, used instead of plain `log` for one reason:
    freshers have 0 years, and `log(0)` is negative infinity. `log1p(0) == 0`.
    NaN stays NaN throughout — unknown in, unknown out.
    """
    return np.log1p(clean_years(years))


def sqrt_years(years: pd.Series) -> pd.Series:
    """A gentler alternative to the log: still concave, but less aggressive.

    Whether log or sqrt fits the market better is an empirical question. Ship
    both, let feature importance settle it, and write the answer in the findings
    log rather than guessing now.
    """
    return np.sqrt(clean_years(years))


def bucket_years(years: pd.Series) -> pd.Series:
    """Cut years into the seven experience bands, as an ordered category.

    `right=False` makes the bands left-closed, so exactly 2.0 years lands in
    "2-3" and not "0-1" — the boundary behaviour you'd expect from the labels.
    """
    cleaned = clean_years(years)
    cut = pd.cut(cleaned, bins=list(BUCKET_EDGES), labels=list(BUCKET_LABELS), right=False)
    # Re-assert the full category list: pd.cut already uses these labels, but
    # astype makes the guarantee explicit and survives future edits.
    return cut.astype(BUCKET_DTYPE)


class ExperienceFeatures:
    """Turn one `years_experience` column into the family of shapes above.

    Usage mirrors scikit-learn::

        exp = ExperienceFeatures().fit(train_df)
        X_train = exp.transform(train_df)
        X_test = exp.transform(test_df)     # same columns, always
    """

    def __init__(self, column: str = "years_experience", *, include_sqrt: bool = True) -> None:
        self.column = column
        self.include_sqrt = include_sqrt
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> ExperienceFeatures:
        """Learn nothing — `df` is deliberately unused.

        See the module docstring for why that is worth saying out loud.

        We still take `df`, and still set `feature_names_`, so this composes with
        the transformers that *do* learn.
        """
        self.feature_names_ = [self.column, "experience_log", "experience_bucket"]
        if self.include_sqrt:
            self.feature_names_.insert(2, "experience_sqrt")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the experience feature block, indexed like the input."""
        if not self.feature_names_:
            raise RuntimeError("call fit() before transform()")

        # A frame that genuinely lacks the column gets an all-NaN column rather
        # than an exception: the *shape* the model expects must not depend on
        # which source the rows came from.
        raw = df[self.column] if self.column in df.columns else pd.Series(np.nan, index=df.index)
        years = clean_years(raw)
        years.index = df.index

        out = pd.DataFrame(index=df.index)
        out[self.column] = years
        out["experience_log"] = np.log1p(years)
        if self.include_sqrt:
            out["experience_sqrt"] = np.sqrt(years)
        out["experience_bucket"] = bucket_years(years).set_axis(df.index)
        return out[self.feature_names_]
