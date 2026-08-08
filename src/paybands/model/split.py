"""Cutting the data into a training half and a testing half.

The whole point of a test set is to be a **stand-in for the future**. We hide
some rows from the model, then ask "how wrong were you about rows you'd never
seen?" That number is our only honest estimate of how the model will behave on a
candidate who walks in next month.

That framing decides *how* you cut. There are two ways here, and picking the
wrong one is the most common serious mistake in applied ML:

``random_split``
    Shuffle the rows, take 20% aside. Correct when the rows have no time order —
    like a survey where everybody answered in the same fortnight.

``temporal_split``
    Train on the past, test on the future. Mandatory the moment your rows have
    dates. See the loud warning on that function.

Both return a :class:`Split`, which carries a plain-English description of how
the cut was made — so that in week 4, when a number surprises you, you can find
out which split produced it instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Used everywhere unless a caller overrides it. A fixed seed means "shuffle
#: randomly, but shuffle the SAME random way every time" — so a number you
#: report today can be reproduced tomorrow.
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Split:
    """A train/test pair plus a record of how it was made.

    The ``description`` field is not decoration. Six weeks from now you will
    have a results table with numbers you cannot account for, and the first
    question will be "was that split random or temporal, and what cutoff?".
    Carrying the answer alongside the data means you never have to reconstruct
    it from memory.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    strategy: str  # "random" or "temporal"
    description: str

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def __str__(self) -> str:
        total = self.n_train + self.n_test
        share = self.n_test / total if total else 0.0
        return (
            f"{self.strategy} split — {self.n_train:,} train / {self.n_test:,} test "
            f"({share:.0%} held out)\n  {self.description}"
        )


def random_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> Split:
    """Shuffle the rows and hold out ``test_size`` of them.

    Args:
        df: Any table of rows. The index is preserved, so you can always trace a
            test row back to the row it came from in the full dataset.
        test_size: Fraction held out, between 0 and 1.
        seed: Fixes the shuffle. Same seed + same data = same split, forever.

    **When this is the right choice.** The Stack Overflow survey is one
    snapshot: every respondent answered within the same few weeks. There is no
    "past" and "future" inside it, so shuffling loses nothing.

    **When it is the wrong choice.** The instant rows carry dates — company
    hires, offers, increments — shuffling mixes 2026 rows into the training set
    while 2024 rows sit in the test set. The model then gets to peek at how the
    market moved before being asked to predict it, and its score comes out
    flatteringly, uselessly high. Use :func:`temporal_split` there.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}")
    if len(df) < 2:
        raise ValueError(f"need at least 2 rows to split, got {len(df)}")

    # A Generator seeded explicitly, rather than numpy's global random state.
    # The global state is shared by every library in the process, so anything
    # else drawing a random number would silently change your "reproducible"
    # split. This one belongs to this call and nothing else can touch it.
    rng = np.random.default_rng(seed)
    shuffled_positions = rng.permutation(len(df))

    n_test = max(1, int(round(len(df) * test_size)))
    n_test = min(n_test, len(df) - 1)  # never leave the training set empty

    test_positions = shuffled_positions[:n_test]
    train_positions = shuffled_positions[n_test:]

    # .iloc takes POSITIONS (0, 1, 2...), not index labels. That distinction
    # matters here: we shuffled positions, and using .loc would try to look
    # those numbers up as labels and quietly return the wrong rows.
    return Split(
        train=df.iloc[train_positions].copy(),
        test=df.iloc[test_positions].copy(),
        strategy="random",
        description=(
            f"rows shuffled with seed={seed}, {test_size:.0%} held out; "
            "valid because these rows have no time ordering"
        ),
    )


def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    cutoff: str | pd.Timestamp,
) -> Split:
    """Train on everything before ``cutoff``, test on everything from it onward.

    Args:
        df: Table containing ``date_col``.
        date_col: Column of dates. In the common schema this is ``event_date``.
        cutoff: The dividing line. Rows strictly before it train; rows on or
            after it test.

    ────────────────────────────────────────────────────────────────────────
    **Why this is not optional once your data has dates.**

    Say your company's data covers 2023–2026, and salaries rose 15% over that
    stretch. You shuffle it randomly. Now the training set contains 2026 hires,
    and the model learns "salaries around here are high". Then you score it on
    test rows that are also from 2026, and it looks excellent.

    Deploy it, and it's asked about 2027 — a year it has seen nothing from. The
    excellent score never described that situation. It described a situation
    where the answer had already been shown to the model.

    That is **leakage**: information from the test period reaching the training
    process. It doesn't announce itself. It shows up as a model that scores
    well and then disappoints in production, and by then the split is three
    months behind you.

    A temporal split reproduces the real task exactly: you only ever know the
    past, and you are always asked about the future. The score it gives you will
    be *worse* than the random-split score. That lower number is the true one.
    ────────────────────────────────────────────────────────────────────────

    This function is unused today — the survey data has no dates. It exists now
    so that when company data arrives there is no moment where a random split is
    the convenient thing lying around.
    """
    if date_col not in df.columns:
        raise ValueError(
            f"column {date_col!r} not found. A temporal split needs dates; "
            f"available columns: {list(df.columns)}"
        )

    cutoff_ts = pd.Timestamp(cutoff)
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # Rows with an unparseable or missing date can't be placed on either side of
    # the line. Dropping them silently would change what the dataset represents,
    # so we count them and say so in the description.
    undated = int(dates.isna().sum())
    dated = df[dates.notna()]
    dates = dates[dates.notna()]

    train = dated[dates < cutoff_ts].copy()
    test = dated[dates >= cutoff_ts].copy()

    # An empty side is nearly always a wrong cutoff (or dates in a format pandas
    # read differently than you expected). Failing loudly beats training on
    # nothing and reporting a meaningless score.
    if train.empty or test.empty:
        raise ValueError(
            f"cutoff {cutoff_ts.date()} leaves {len(train)} train / {len(test)} test rows. "
            f"Dates in {date_col!r} run {dates.min()} to {dates.max()}."
        )

    note = f", {undated:,} rows dropped for missing/unparseable dates" if undated else ""
    return Split(
        train=train,
        test=test,
        strategy="temporal",
        description=(
            f"trained on {date_col} < {cutoff_ts.date()}, tested on {date_col} >= "
            f"{cutoff_ts.date()}{note}; no future row can influence training"
        ),
    )
