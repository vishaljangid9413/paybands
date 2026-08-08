"""Tests for train/test splitting.

Splitting looks trivial, which is exactly why it deserves tests. A split bug
does not crash — it produces a model that scores beautifully and fails in
production, and you find out months later. These tests check the two properties
that actually matter: the split is reproducible, and no test row ever reaches
training.
"""

from __future__ import annotations

import pandas as pd
import pytest

from paybands.model.split import random_split, temporal_split


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"salary_annual": range(100), "role": ["Backend"] * 100})


# ─────────────────────────────────────────── random_split


def test_same_seed_gives_the_same_split(frame):
    """Reproducibility is the entire reason we pass a seed.

    Without this, a number in your results table can never be re-derived, and
    "the model improved" and "the split moved" become indistinguishable.
    """
    a = random_split(frame, seed=42)
    b = random_split(frame, seed=42)
    assert list(a.test.index) == list(b.test.index)


def test_different_seeds_give_different_splits(frame):
    """Guards against the seed being accidentally ignored — a bug that would
    make the test above pass for the wrong reason."""
    a = random_split(frame, seed=1)
    b = random_split(frame, seed=2)
    assert list(a.test.index) != list(b.test.index)


def test_train_and_test_never_overlap(frame):
    """The one property that makes a test score mean anything.

    A single row appearing on both sides is a row the model has memorised and
    is then congratulated for recalling.
    """
    split = random_split(frame, test_size=0.2)
    assert set(split.train.index) & set(split.test.index) == set()


def test_every_row_lands_somewhere(frame):
    """No row may be silently dropped — that would quietly shrink the dataset."""
    split = random_split(frame, test_size=0.3)
    assert split.n_train + split.n_test == len(frame)
    assert set(split.train.index) | set(split.test.index) == set(frame.index)


def test_test_size_is_respected(frame):
    split = random_split(frame, test_size=0.25)
    assert split.n_test == 25


def test_split_describes_itself(frame):
    """The description is how a surprising number gets traced back six weeks
    later, so it has to actually contain the settings used."""
    split = random_split(frame, test_size=0.2, seed=7)
    assert "seed=7" in split.description
    assert split.strategy == "random"
    assert "20%" in str(split)


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5])
def test_impossible_test_sizes_rejected(frame, bad):
    with pytest.raises(ValueError, match="test_size"):
        random_split(frame, test_size=bad)


# ─────────────────────────────────────────── temporal_split


@pytest.fixture
def dated_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": pd.date_range("2024-01-01", periods=36, freq="ME"),
            "salary_annual": range(36),
        }
    )


def test_no_future_row_reaches_training(dated_frame):
    """The whole point. If any training row is dated on or after the cutoff,
    the model has seen the future and its score is fiction."""
    cutoff = pd.Timestamp("2025-07-01")
    split = temporal_split(dated_frame, "event_date", cutoff)

    assert split.train["event_date"].max() < cutoff
    assert split.test["event_date"].min() >= cutoff
    assert split.n_train + split.n_test == len(dated_frame)


def test_cutoff_boundary_row_goes_to_test(dated_frame):
    """A row dated exactly on the cutoff is tested, not trained on.

    Arbitrary but it must be decided somewhere; leaving it ambiguous is how an
    off-by-one leak gets in.
    """
    on_cutoff = dated_frame["event_date"].iloc[10]
    split = temporal_split(dated_frame, "event_date", on_cutoff)
    assert on_cutoff in set(split.test["event_date"])
    assert on_cutoff not in set(split.train["event_date"])


def test_undated_rows_are_dropped_and_reported(dated_frame):
    """Rows without a date can't be placed on either side of the line. Dropping
    them is fine; dropping them silently is not."""
    with_gap = dated_frame.copy()
    with_gap.loc[0, "event_date"] = pd.NaT

    split = temporal_split(with_gap, "event_date", "2025-07-01")
    assert split.n_train + split.n_test == len(with_gap) - 1
    assert "1 rows dropped" in split.description


def test_cutoff_outside_the_data_fails_loudly(dated_frame):
    """A cutoff before all the data would train on nothing and still report a
    score. Better to refuse."""
    with pytest.raises(ValueError, match="leaves 0 train"):
        temporal_split(dated_frame, "event_date", "2000-01-01")


def test_missing_date_column_names_the_alternatives(dated_frame):
    with pytest.raises(ValueError, match="not found"):
        temporal_split(dated_frame, "hire_date", "2025-07-01")
