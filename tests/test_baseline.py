"""Tests for the two baselines.

Most of these check the fallback cascade, because that's where a lookup table
either behaves sensibly or quietly starts inventing numbers. The last one is the
one that matters most: on the real survey data, does the `groupby` actually beat
predicting one number for everybody? If it didn't, either the baseline or our
understanding of the data would be wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paybands.data import stackoverflow
from paybands.model.baseline import (
    GlobalMedianBaseline,
    GroupMedianBaseline,
    experience_bucket,
)
from paybands.model.metrics import evaluate
from paybands.model.split import random_split

SURVEY_PATH = Path("data/public/so_2025_raw.csv")


# ─────────────────────────────────────────── experience buckets


@pytest.mark.parametrize(
    ("years", "expected"),
    [
        (0, "0-1"),
        (1, "0-1"),
        (2, "2-3"),
        (3, "2-3"),
        (5, "4-5"),
        (8, "6-8"),
        (12, "9-12"),
        (20, "13-20"),
        (21, "20+"),
        (54, "20+"),
    ],
)
def test_bucket_boundaries(years, expected):
    """The edges, pinned down. Buckets are inclusive of their upper bound, so a
    person with exactly 20 years is "13-20" and 21 years starts "20+"."""
    assert experience_bucket(years) == expected


def test_missing_experience_gets_its_own_bucket():
    """Not dropped, not filled with zero. "We don't know" is a real state, and
    treating it as a fresh graduate would drag that bucket's median down."""
    assert experience_bucket(None) == "unknown"
    assert experience_bucket(float("nan")) == "unknown"


# ─────────────────────────────────────────── the global median baseline


def test_global_baseline_predicts_the_median_for_everyone():
    y = [5_00_000, 10_00_000, 15_00_000, 20_00_000, 1_00_00_000]
    X = pd.DataFrame({"role": ["Backend"] * 5})

    model = GlobalMedianBaseline().fit(X, y)
    predictions = model.predict(X)

    assert set(predictions) == {15_00_000}


def test_global_baseline_uses_median_not_mean():
    """One ₹2Cr founder in a room of ₹10L engineers moves the mean by ₹19L and
    the median not at all. That is the entire argument for the median."""
    y = [10_00_000] * 9 + [2_00_00_000]
    X = pd.DataFrame({"role": ["Backend"] * 10})

    model = GlobalMedianBaseline().fit(X, y)
    assert model.median_ == 10_00_000
    assert np.mean(y) > 2 * model.median_


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        GlobalMedianBaseline().predict(pd.DataFrame({"role": ["Backend"]}))


# ─────────────────────────────────────────── the fallback cascade


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """A small, deliberately lopsided dataset.

    * Backend / 6-8 years  — 12 people. Above the minimum, so trusted.
    * Frontend / 6-8 years —  3 people, all paid absurdly. Below the minimum,
      so their median must be ignored.
    * Data / 2-3 years     — 12 people, to give the "2-3" bucket a fallback.
    """
    rows = []
    salaries = []

    for _ in range(12):
        rows.append({"role": "Backend", "years_experience": 7})
        salaries.append(20_00_000)
    for _ in range(3):
        rows.append({"role": "Frontend", "years_experience": 7})
        salaries.append(9_00_00_000)  # nonsense, and only 3 people say it
    for _ in range(12):
        rows.append({"role": "Data", "years_experience": 3})
        salaries.append(8_00_000)

    return pd.DataFrame(rows), np.array(salaries, dtype=float)


def test_trusted_group_is_used_directly(training_data):
    X, y = training_data
    model = GroupMedianBaseline(min_group_size=10).fit(X, y)

    candidate = pd.DataFrame({"role": ["Backend"], "years_experience": [7]})
    assert model.predict(candidate)[0] == 20_00_000
    assert model.fallback_counts_["role+experience"] == 1


def test_unseen_role_falls_back_to_the_experience_bucket(training_data):
    """A Security engineer with 7 years has no cell in the table at all. The
    answer must come from everyone else with 7 years, not from nowhere."""
    X, y = training_data
    model = GroupMedianBaseline(min_group_size=10).fit(X, y)

    candidate = pd.DataFrame({"role": ["Security"], "years_experience": [7]})
    prediction = model.predict(candidate)[0]

    assert model.fallback_counts_["experience only"] == 1
    # The 6-8 bucket holds 12 Backend at ₹20L and 3 Frontend at ₹9Cr, so its
    # median is still ₹20L — the outliers are outnumbered, which is precisely
    # what a median is for.
    assert prediction == 20_00_000


def test_group_below_minimum_size_is_not_trusted(training_data):
    """Three Frontend engineers claiming ₹9Cr must NOT set the Frontend rate.

    This is the rule that stops one joke response, or one founder writing down
    their equity, from becoming every future candidate's quote. The prediction
    falls back to the bucket median instead — less specific, far better evidenced.
    """
    X, y = training_data
    model = GroupMedianBaseline(min_group_size=10).fit(X, y)

    candidate = pd.DataFrame({"role": ["Frontend"], "years_experience": [7]})
    prediction = model.predict(candidate)[0]

    assert ("Frontend", "6-8") not in model.group_medians_
    assert model.fallback_counts_["experience only"] == 1
    assert prediction == 20_00_000  # the 6-8 bucket, not the ₹9Cr fantasy


def test_lowering_the_minimum_lets_the_noisy_group_through(training_data):
    """The mirror image of the test above — proof the threshold is what's doing
    the work, not some other accident of the fixture."""
    X, y = training_data
    model = GroupMedianBaseline(min_group_size=2).fit(X, y)

    candidate = pd.DataFrame({"role": ["Frontend"], "years_experience": [7]})
    assert model.predict(candidate)[0] == 9_00_00_000


def test_unseen_bucket_falls_all_the_way_to_the_global_median(training_data):
    """A 30-year veteran: no role match, and no "20+" bucket was ever seen. The
    last rung of the cascade has to catch it, because a lookup table that
    returns NaN is a lookup table that crashes the API."""
    X, y = training_data
    model = GroupMedianBaseline(min_group_size=10).fit(X, y)

    candidate = pd.DataFrame({"role": ["Security"], "years_experience": [30]})
    prediction = model.predict(candidate)[0]

    assert model.fallback_counts_["global median"] == 1
    assert prediction == model.global_median_


def test_predictions_are_never_nan_for_any_input(training_data):
    """The cascade's guarantee, stated as a test: whatever walks in, a number
    comes out."""
    X, y = training_data
    model = GroupMedianBaseline().fit(X, y)

    strangers = pd.DataFrame(
        {
            "role": ["Backend", "Astronaut", None, "Data"],
            "years_experience": [7, 99, None, 3],
        }
    )
    assert not np.isnan(model.predict(strangers)).any()


def test_explain_names_the_rung_that_answered(training_data):
    """A number a recruiter can't argue with is a number they shouldn't be
    given. The explanation has to say what evidence produced it."""
    X, y = training_data
    model = GroupMedianBaseline(min_group_size=10).fit(X, y)

    assert "role+experience" in model.explain("Backend", 7)
    assert "experience only" in model.explain("Security", 7)
    assert "₹" in model.explain("Backend", 7)


def test_fallback_report_adds_up(training_data):
    X, y = training_data
    model = GroupMedianBaseline(min_group_size=10).fit(X, y)
    model.predict(X)

    assert sum(model.fallback_counts_.values()) == len(X)
    assert "%" in model.coverage_report()


def test_medians_survive_the_log_transform(training_data):
    """Fitting on log(salary) and exponentiating gives the same answer as
    fitting on rupees, because the median of the logs is the log of the median.

    Worth pinning: it means the baseline can sit in a log-space pipeline
    unchanged, and it is a property the MEAN does not have.
    """
    X, y = training_data

    on_rupees = GroupMedianBaseline().fit(X, y).predict(X)
    on_logs = np.exp(GroupMedianBaseline().fit(X, np.log(y)).predict(X))

    assert on_logs == pytest.approx(on_rupees)


def test_misaligned_lengths_are_refused():
    X = pd.DataFrame({"role": ["Backend"] * 3, "years_experience": [1, 2, 3]})
    with pytest.raises(ValueError, match="3 rows but y has 2"):
        GroupMedianBaseline().fit(X, [1.0, 2.0])


def test_missing_columns_named_in_the_error():
    X = pd.DataFrame({"job": ["Backend"], "years_experience": [3]})
    with pytest.raises(ValueError, match="'role' not found"):
        GroupMedianBaseline().fit(X, [10.0])


# ─────────────────────────────────────────── against the real data


@pytest.mark.skipif(not SURVEY_PATH.exists(), reason="survey CSV not downloaded")
def test_group_median_beats_global_median_on_real_data():
    """The claim the whole baseline exercise exists to make.

    Knowing someone's role and rough experience must predict their salary better
    than knowing nothing about them. If this ever fails, either the baseline is
    broken or the dataset has stopped containing signal — and both are things
    you want to hear about immediately, not in week 4.

    Note this is measured on the TEST half. Measured on the training data the
    lookup table would win trivially, because it memorised those very rows.
    """
    df, _ = stackoverflow.load(SURVEY_PATH, verbose=False)
    split = random_split(df, test_size=0.2, seed=42)

    y_train = split.train["salary_annual"].to_numpy()
    y_test = split.test["salary_annual"].to_numpy()

    global_model = GlobalMedianBaseline().fit(split.train, y_train)
    group_model = GroupMedianBaseline().fit(split.train, y_train)

    global_score = evaluate(y_test, global_model.predict(split.test))
    group_score = evaluate(y_test, group_model.predict(split.test))

    assert group_score.mae < global_score.mae
    assert group_score.median_absolute_error < global_score.median_absolute_error
