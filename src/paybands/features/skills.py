"""Skill features — and the clearest example of leakage in this project.

The survey gives skills as one semicolon-separated string per person::

    "Python;SQL;JavaScript;Docker"

We turn that into binary flags: `skill_python = 1`, `skill_rust = 0`, and so on.
There are hundreds of distinct skills and most appear a handful of times, so we
keep only the **top N most common** and ignore the long tail.

## Why "the top N" is a leakage question

"Which skills are most common?" is a fact *learned from data*. That makes it the
kind of thing that can leak.

**Leakage** means information from the test set reaching the model during
training. It is the most common serious mistake in applied machine learning, and
it is nasty precisely because it doesn't look like a bug: your validation score
goes *up*, you feel clever, and the model then fails in production — where the
future genuinely is unknown.

Concretely, suppose you compute the top-25 skills over the whole dataset before
splitting. If `rust` only clears the cut-off because of six rust developers who
all landed in the *test* set, then the training columns were chosen with
knowledge of the test rows. Your test score is no longer an estimate of
performance on unseen data, because those rows already influenced the model's
input space.

The size of the cheat here is small. **The habit is what matters.** The same
mistake with target encoding, scaling parameters, or median imputation is
catastrophic, and it is the identical mistake. So we build the discipline where
the stakes are low:

* `fit(train)` counts skills over the training rows only and stores
  `vocabulary_`.
* `transform(anything)` only ever emits columns from that stored vocabulary.

## What happens to a skill the model has never seen

A test-set person who lists `zig` gets no `skill_zig` column. That is **correct,
not a bug**: the model was never trained on such a column, so there is no weight
to apply to it. Silently adding one at prediction time would change the input
shape and either crash the model or (worse, with some libraries) shift every
column by one. In production the same thing happens constantly — new frameworks
appear every year — and the honest answer is "this model doesn't know about zig
yet; retrain it." Retraining is a decision a human makes, not something a
transform should quietly do for you.

The unseen skill still leaves a trace, though: `n_skills` counts *everything* the
person listed, in-vocabulary or not. That count is computed row by row with no
fitted state, so it cannot leak.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

#: 25 is a judgement call, not a fact. Rationale: in the Indian survey rows the
#: 25th most common language still appears in a few hundred CVs, which is enough
#: rows for a tree to split on; below that you are mostly modelling noise. Raise
#: it when company data arrives with more rows and its own vocabulary.
DEFAULT_TOP_N = 25

#: LightGBM refuses feature names containing the JSON-special characters
#: `" , : [ ] { }` (it serialises the model to JSON), so those get replaced.
#: Spaces are legal but replaced too, purely so names survive being written to a
#: CSV or read off a plot axis. Note what is *not* in this set: `+`, `#`, `/` and
#: `.` are all fine, so `c++` stays recognisably `skill_c++` rather than turning
#: into an unreadable `skill_c__`.
_UNSAFE_CHARS = set('[]{}":, ')


def parse_skills(value: object) -> list[str]:
    """Split one survey cell into a clean list of lowercase skill names.

    Handles the three things real data does: missing values (NaN, not a string),
    empty strings, and stray whitespace around the semicolons. Lowercasing means
    "Python" and "python" are the same skill, which they obviously are but which
    a naive `str.split` would treat as two.
    """
    if not isinstance(value, str):
        return []
    return [part.strip().lower() for part in value.split(";") if part.strip()]


def _safe_name(skill: str) -> str:
    """`skill_` prefix, with characters LightGBM chokes on replaced."""
    cleaned = "".join("_" if ch in _UNSAFE_CHARS else ch for ch in skill)
    return f"skill_{cleaned}"


class SkillFeatures:
    """Learn a skill vocabulary from training rows, then emit binary flags.

    Fitted attributes (the trailing underscore is the scikit-learn convention for
    "this was learned from data, and therefore this is what could leak"):

    * `vocabulary_` — the chosen skills, most common first.
    * `skill_counts_` — how often each of them appeared in the training rows.
      Kept for inspection: when a feature turns out to be useless, the first
      question is always "how many rows even had it?"
    * `feature_names_` — the columns `transform` will produce.
    """

    def __init__(self, column: str = "skills", *, top_n: int = DEFAULT_TOP_N) -> None:
        self.column = column
        self.top_n = top_n
        self.vocabulary_: list[str] = []
        self.skill_counts_: dict[str, int] = {}
        self.feature_names_: list[str] = []

    def _column_or_blank(self, df: pd.DataFrame) -> pd.Series:
        if self.column in df.columns:
            return df[self.column]
        return pd.Series([None] * len(df), index=df.index, dtype=object)

    def fit(self, df: pd.DataFrame) -> SkillFeatures:
        """Count skills across **these rows only** and keep the most common.

        Pass the training split here. Never the full frame, never the test split
        — see the module docstring.
        """
        counts: Counter[str] = Counter()
        for cell in self._column_or_blank(df):
            counts.update(set(parse_skills(cell)))  # set(): one vote per person

        # Sort by count descending, then alphabetically. The tiebreak matters:
        # without it, two skills with identical counts could swap places between
        # runs and silently reorder the model's columns.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_n]

        self.vocabulary_ = [skill for skill, _ in ranked]
        self.skill_counts_ = dict(ranked)
        self.feature_names_ = [_safe_name(s) for s in self.vocabulary_] + ["n_skills"]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary flags for the fitted vocabulary, plus `n_skills`.

        Skills outside `vocabulary_` are dropped here — deliberately. They still
        count towards `n_skills`.
        """
        if not self.feature_names_:
            raise RuntimeError("call fit() before transform()")

        parsed = [set(parse_skills(cell)) for cell in self._column_or_blank(df)]

        out = pd.DataFrame(index=df.index)
        for skill in self.vocabulary_:
            # int8, not bool: LightGBM wants numbers, and 1 byte per cell keeps
            # 25 skill columns cheap even on a laptop.
            out[_safe_name(skill)] = np.fromiter(
                (skill in row for row in parsed), dtype=np.int8, count=len(parsed)
            )

        # Counts every skill listed, including ones the vocabulary ignores. It is
        # a rough proxy for breadth, and it is the only place an unseen skill can
        # still influence the prediction.
        out["n_skills"] = np.fromiter(
            (len(row) for row in parsed), dtype=np.int16, count=len(parsed)
        )
        return out[self.feature_names_]
