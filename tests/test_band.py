"""Tests for the three-quantile band model.

The one that matters most is at the bottom: on the real survey data, does
gradient boosting actually beat a four-line `groupby`? If it doesn't, the model
has not earned its complexity and the honest call is to ship the lookup table.
It is checked across several seeds, because a single split is an anecdote.

Everything above it guards the properties a band has to have before its numbers
mean anything — that it comes out the right way up, that it is not secretly a
fixed ± error bar, and that the same seed gives the same answer twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paybands.data import stackoverflow, synthetic
from paybands.model.band import (
    COMPA_ABOVE_BAND,
    COMPA_BELOW_BAND,
    SalaryBandModel,
    band_report,
    compa_label,
)
from paybands.model.baseline import GroupMedianBaseline
from paybands.model.metrics import evaluate
from paybands.model.split import random_split

SURVEY_PATH = Path("data/public/so_2025_raw.csv")


@pytest.fixture(scope="module")
def synthetic_data() -> pd.DataFrame:
    """A clean, known world — big enough that the model has something to learn."""
    df, _ = synthetic.generate(n=1500, seed=11)
    return df


@pytest.fixture(scope="module")
def fitted(synthetic_data: pd.DataFrame) -> tuple[SalaryBandModel, pd.DataFrame]:
    """Fit once and reuse: three LightGBM models per fit isn't free."""
    split = random_split(synthetic_data, test_size=0.25, seed=3)
    model = SalaryBandModel(seed=3).fit(split.train, split.train["salary_annual"])
    return model, split.test


# ─────────────────────────────────────────── the band comes out the right way up


def test_lower_median_upper_are_ordered_on_every_row(fitted):
    """The property everything else depends on.

    The three models are fitted independently, so nothing *makes* this true —
    the sort inside `predict_quantiles_log` does. If this ever fails, the repair
    has been removed and some recruiter is looking at an inside-out band.
    """
    model, test = fitted
    band = model.predict_band(test)

    assert np.all(band.lower <= band.median)
    assert np.all(band.median <= band.upper)


def test_crossing_is_detected_and_repaired():
    """Force a crossing and check both halves of the response: it gets sorted,
    and it gets *counted*.

    We swap the fitted LightGBM models for stubs that return known, deliberately
    inverted numbers. That's the only way to test this reliably — real crossings
    happen on maybe 1% of rows and we can't choose which.
    """

    class _Fixed:
        """Stands in for a fitted LightGBM model, returning a preset column."""

        def __init__(self, values: list[float]) -> None:
            self.values = np.asarray(values, dtype=float)

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return self.values

    df, _ = synthetic.generate(n=120, seed=5)
    model = SalaryBandModel(seed=5).fit(df, df["salary_annual"])

    rows = df.head(3)
    #        row 0: fine     row 1: low > high    row 2: median above high
    model.models_ = [
        _Fixed([13.0, 15.0, 13.0]),  # the "0.1" model
        _Fixed([14.0, 14.0, 15.5]),  # the "0.5" model
        _Fixed([15.0, 13.5, 15.0]),  # the "0.9" model
    ]

    q = model.predict_quantiles_log(rows)

    assert model.n_crossed_ == 2  # rows 1 and 2, not row 0
    assert model.crossing_rate_ == pytest.approx(2 / 3)
    assert np.all(np.diff(q, axis=1) >= 0)  # every row now ascending
    # Sorting rearranges, it never invents: row 1's three numbers are the same
    # three numbers, in order.
    assert sorted(q[1]) == pytest.approx(sorted([15.0, 14.0, 13.5]))


def test_a_clean_model_reports_a_low_crossing_rate(fitted):
    """A diagnostic, not a pass/fail — but if a well-behaved synthetic dataset
    starts crossing on 20% of rows, the three models have stopped agreeing about
    the shape of the world and the band edges are being set by noise."""
    model, test = fitted
    model.predict_band(test)
    assert model.crossing_rate_ < 0.05


# ─────────────────────────────────────────── the band is not a fixed error bar


def test_band_width_varies_with_the_candidate(fitted):
    """The whole argument for three models instead of one plus an error bar.

    A fixed ± band is equally wide for a fresher and for a CTO, which is too
    vague for the first and too confident for the second. Three quantile models
    are free to learn a different width per row — so if every width came out the
    same, we would have paid for three models and got an error bar.
    """
    model, test = fitted
    width = model.predict_band(test).width

    assert width.std() > 0
    # Not just numerically different — meaningfully different. The widest band
    # should be several times the narrowest.
    assert np.percentile(width, 95) > 2 * np.percentile(width, 5)


def test_senior_bands_are_wider_than_junior_bands(fitted):
    """The direction of that variation, checked against how pay actually works.

    Senior salaries genuinely spread out — two people fifteen years in can be a
    factor of three apart — while the market for a one-year backend engineer is
    narrow. In *rupees* the senior band must therefore be wider.
    """
    model, test = fitted
    band = model.predict_band(test)
    years = test["years_experience"].to_numpy()

    junior = band.width[years <= 2]
    senior = band.width[years >= 12]
    assert junior.size and senior.size
    assert np.median(senior) > np.median(junior)


# ─────────────────────────────────────────── log space


def test_predictions_are_positive_and_plausible(fitted):
    """A consequence of training on logs: `exp` cannot return a negative number,
    so the model physically cannot quote someone a negative salary. A model
    trained on rupees can, and does, for the low end of a wide band."""
    model, test = fitted
    band = model.predict_band(test)
    assert np.all(band.lower > 0)


def test_a_non_positive_salary_is_refused():
    """log(0) is undefined, and a salary of 0 in a training file is a data bug,
    not a data point. Fail loudly at fit time rather than produce -inf."""
    df, _ = synthetic.generate(n=60, seed=1)
    y = df["salary_annual"].to_numpy().copy()
    y[0] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        SalaryBandModel(seed=1).fit(df, y)


# ─────────────────────────────────────────── reproducibility


def test_same_seed_gives_identical_predictions(synthetic_data):
    """Same seed + same data = same numbers, or none of the reported results
    can be reproduced tomorrow."""
    split = random_split(synthetic_data, test_size=0.25, seed=3)
    y = split.train["salary_annual"]

    a = SalaryBandModel(seed=99).fit(split.train, y).predict_band(split.test)
    b = SalaryBandModel(seed=99).fit(split.train, y).predict_band(split.test)

    assert np.array_equal(a.lower, b.lower)
    assert np.array_equal(a.median, b.median)
    assert np.array_equal(a.upper, b.upper)


def test_a_different_seed_gives_different_predictions(synthetic_data):
    """The mirror image — proof the seed is actually wired in and we aren't
    just observing a deterministic algorithm."""
    split = random_split(synthetic_data, test_size=0.25, seed=3)
    y = split.train["salary_annual"]

    a = SalaryBandModel(seed=1).fit(split.train, y).predict_band(split.test)
    b = SalaryBandModel(seed=2).fit(split.train, y).predict_band(split.test)

    assert not np.allclose(a.median, b.median)


# ─────────────────────────────────────────── compa-ratio


def test_compa_ratio_is_actual_over_predicted_median(fitted):
    """One division, and the number both use cases run on. Worth pinning to the
    rupee, because everything downstream — the pay-equity list, the increment
    recommendation — is a comparison against 0.90 and 1.10."""
    model, test = fitted
    actual = test["salary_annual"].to_numpy()

    ratio = model.compa_ratio(actual, test)
    expected = actual / model.predict_band(test).median

    assert ratio == pytest.approx(expected)


def test_compa_ratio_of_someone_paid_exactly_the_midpoint_is_one(fitted):
    model, test = fitted
    median = model.predict_band(test).median
    assert model.compa_ratio(median, test) == pytest.approx(np.ones(len(test)))


def test_compa_ratio_scales_the_way_a_ratio_should(fitted):
    """Pay someone twice the midpoint and the ratio doubles. Trivial arithmetic,
    but it is the arithmetic HR will act on."""
    model, test = fitted
    median = model.predict_band(test).median
    assert model.compa_ratio(2 * median, test) == pytest.approx(2 * np.ones(len(test)))


def test_compa_ratio_rejects_a_length_mismatch(fitted):
    model, test = fitted
    with pytest.raises(ValueError, match="shape mismatch"):
        model.compa_ratio([10_00_000.0, 20_00_000.0], test)


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.80, "below band"),
        (COMPA_BELOW_BAND - 0.001, "below band"),
        (COMPA_BELOW_BAND, "in band"),
        (1.00, "in band"),
        (COMPA_ABOVE_BAND, "in band"),
        (COMPA_ABOVE_BAND + 0.001, "above band"),
    ],
)
def test_compa_labels_at_the_boundaries(ratio, expected):
    """The thresholds are policy, so the edges are pinned: exactly 0.90 is in
    band, not below it."""
    assert compa_label(ratio) == expected


# ─────────────────────────────────────────── plumbing


def test_the_target_never_reaches_the_feature_matrix(fitted):
    """ "The target leaked into the features" produces a suspiciously perfect
    model and is a mistake people genuinely ship. Same for the columns kept only
    for the fairness audit."""
    model, _ = fitted
    for forbidden in ("salary_annual", "gender", "source"):
        assert forbidden not in model.feature_names_


def test_predicting_before_fitting_is_an_error():
    df, _ = synthetic.generate(n=20, seed=1)
    with pytest.raises(RuntimeError, match="fit"):
        SalaryBandModel().predict_band(df)


def test_misaligned_lengths_are_refused():
    df, _ = synthetic.generate(n=20, seed=1)
    with pytest.raises(ValueError, match="20 rows but y has 2"):
        SalaryBandModel().fit(df, [1.0, 2.0])


def test_quantiles_must_be_ordered():
    with pytest.raises(ValueError, match="strictly increasing"):
        SalaryBandModel(quantiles=(0.9, 0.5, 0.1))


def test_band_report_reads_coverage_and_width_together(fitted):
    """Coverage says the band is honest, width says it is useful. A report that
    gave you one without the other would be half an answer."""
    model, test = fitted
    report = band_report(test["salary_annual"], model.predict_band(test), label="synthetic")

    assert 0.0 <= report.coverage <= 1.0
    assert report.target_coverage == pytest.approx(0.8)
    assert report.mean_width > 0
    assert "coverage" in str(report) and "width" in str(report)


def test_tiny_datasets_switch_early_stopping_off():
    """On 30 rows the validation slice is ~6 rows, and "did the validation score
    improve?" is then answered by noise. Fewer trees chosen blindly beats a
    stopping point chosen by coin flip — and the model must say which it did."""
    df, _ = synthetic.generate(n=30, seed=1)
    model = SalaryBandModel(seed=1).fit(df, df["salary_annual"])
    assert "early stopping OFF" in model.describe()


# ─────────────────────────────────────────── against the real data


@pytest.mark.skipif(not SURVEY_PATH.exists(), reason="survey CSV not downloaded")
@pytest.mark.parametrize("seed", [0, 42, 2024])
def test_beats_the_group_median_baseline_on_real_data(seed):
    """The claim this entire phase exists to make.

    Gradient boosting has to beat a `groupby` on data it has never seen. If it
    only ties, the model has not earned its complexity and the right engineering
    decision is to ship the lookup table — it trains instantly and any HR person
    can read it.

    Run across three seeds on purpose. A single split is an anecdote: the gap
    between the two models is a few lakh and the seed-to-seed wobble on 204 test
    rows is comparable, so one favourable split proves nothing.
    """
    df, _ = stackoverflow.load(SURVEY_PATH, verbose=False)
    split = random_split(df, test_size=0.2, seed=seed)
    y_train = split.train["salary_annual"]
    y_test = split.test["salary_annual"].to_numpy()

    model = SalaryBandModel(seed=seed).fit(split.train, y_train)
    baseline = GroupMedianBaseline().fit(split.train, y_train)

    model_score = evaluate(y_test, model.predict(split.test))
    baseline_score = evaluate(y_test, baseline.predict(split.test))

    assert model_score.mae < baseline_score.mae
    assert model_score.median_absolute_error < baseline_score.median_absolute_error
