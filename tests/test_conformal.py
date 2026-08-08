"""Tests for conformalised quantile regression.

Two kinds of test here, and the split is deliberate.

The arithmetic tests use a **stub** quantile model that returns a fixed band, so
``q̂`` can be worked out by hand on paper and compared. Testing the conformal
maths through a trained LightGBM model would mean the assertion depended on
LightGBM's numbers, which tells you nothing about whether the ⌈(n+1)(1−α)⌉ rank
is right.

The behaviour tests use the real model on synthetic data, where the sample can
be made large and clean enough that "coverage lands near nominal" is a testable
claim rather than a coin flip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paybands.data import synthetic
from paybands.model.band import SalaryBandModel, band_report
from paybands.model.conformal import (
    ConformalBand,
    ThreeWaySplit,
    three_way_split,
)

# The stub's band, in log space: ₹10L wide by a factor of e. Chosen so the
# conformity scores below come out as round numbers.
LOW = float(np.log(1_000_000))
MID = LOW + 0.5
HIGH = LOW + 1.0


class FixedBand:
    """A quantile model that always predicts the same band.

    It satisfies the `QuantileModel` protocol and nothing else — no training, no
    LightGBM, no randomness. That is the point: with the band held constant, the
    conformity score of a row is a known function of its salary, so ``q̂`` can be
    predicted on paper before the code runs.
    """

    nominal_coverage = 0.8

    def predict_quantiles_log(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([LOW, MID, HIGH], (len(X), 1))


def salaries_at(offsets: list[float]) -> np.ndarray:
    """Salaries whose log sits ``offset`` above the band's lower edge.

    With the stub band, a row at offset ``t`` has conformity score
    ``max(LOW − y, y − HIGH)`` = ``max(−t, t − 1)``. So t = 1.3 scores +0.3, and
    t = 0.5 scores −0.5 (inside the band, with room to spare).
    """
    return np.exp(LOW + np.asarray(offsets, dtype=float))


def rows(n: int) -> pd.DataFrame:
    """A frame the stub only ever measures the length of."""
    return pd.DataFrame({"years_experience": np.arange(n, dtype=float)})


# ─────────────────────────────────────────── the arithmetic


def test_qhat_is_the_ceil_n_plus_one_rank():
    """The heart of CQR, checked by hand.

    Nine calibration rows with scores 0.1 … 0.9. At α = 0.2 the rank is
    ⌈(9+1) × 0.8⌉ = 8, so ``q̂`` is the 8th smallest score: 0.8.

    Note it is *not* the 80th percentile of the scores (that would be 0.74 by
    numpy's default interpolation). The ``+1`` counts the future row we are
    about to make a promise about, and it is the difference between a guarantee
    and an approximation.
    """
    offsets = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]  # → scores 0.1 … 0.9
    band = ConformalBand(FixedBand(), alpha=0.2).calibrate(rows(9), salaries_at(offsets))

    assert band.qhat_log_ == pytest.approx(0.8)
    assert band.n_calibration_ == 9
    assert band.scores_ == pytest.approx(np.arange(1, 10) / 10)


def test_alpha_is_read_off_the_model_by_default():
    """A model built on the 0.1 and 0.9 quantiles promises 80%, so α is 0.2.
    Leaving the caller to type it again is leaving them a way to make the two
    disagree silently."""
    band = ConformalBand(FixedBand())
    assert band.alpha == pytest.approx(0.2)
    assert band.target_coverage == pytest.approx(0.8)


def test_the_widening_is_multiplicative_in_rupees():
    """Why the calibration happens in log space.

    ``q̂`` is one constant, but adding it in log space multiplies in rupees — so
    every band grows by the same *percentage*, not by the same ₹. Adding a flat
    ₹3L to both the fresher's band and the CTO's would be the fixed-width
    mistake sneaking back in through the calibration step.
    """
    offsets = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    cal = ConformalBand(FixedBand(), alpha=0.2).calibrate(rows(9), salaries_at(offsets))

    band = cal.predict_band(rows(2))
    raw_lower, raw_upper = np.exp(LOW), np.exp(HIGH)

    # Each edge moved by exp(q̂) — the same ratio at both ends and on every row.
    assert band.lower[0] == pytest.approx(raw_lower / np.exp(0.8))
    assert band.upper[0] == pytest.approx(raw_upper * np.exp(0.8))
    assert cal.widening_factor == pytest.approx(np.exp(0.8))


def test_the_midpoint_is_never_moved():
    """CQR adjusts the edges. It says nothing about whether the median model is
    any good — that is what MAE is for — so it must leave it alone."""
    cal = ConformalBand(FixedBand(), alpha=0.2).calibrate(rows(9), salaries_at([1.5] * 9))
    assert cal.predict_band(rows(3)).median == pytest.approx(np.exp(MID))


def test_an_over_cautious_band_gets_narrowed():
    """``q̂`` can be negative, and then CQR *tightens* the band.

    If the quantile models were covering 95% when 80% was asked for, the honest
    correction is a narrower band, and the guarantee still holds. Conformal
    prediction calibrates in both directions — it is not a safety margin bolted
    on the outside.
    """
    # Every calibration salary lands comfortably inside the band, so every score
    # is negative.
    offsets = list(np.linspace(0.3, 0.7, 9))
    cal = ConformalBand(FixedBand(), alpha=0.2).calibrate(rows(9), salaries_at(offsets))

    assert cal.qhat_log_ < 0
    band = cal.predict_band(rows(1))
    assert band.upper[0] - band.lower[0] < np.exp(HIGH) - np.exp(LOW)
    assert band.lower[0] <= band.median[0] <= band.upper[0]


def test_narrowing_can_never_invert_the_band():
    """The edge case of the test above. A band that shrank past its own midpoint
    would be nonsense to show anybody, so the midpoint is the floor."""
    cal = ConformalBand(FixedBand(), alpha=0.2).calibrate(rows(9), salaries_at([0.5] * 9))

    assert cal.qhat_log_ == pytest.approx(-0.5)  # exactly half the band width
    band = cal.predict_band(rows(4))
    assert np.all(band.lower <= band.median)
    assert np.all(band.median <= band.upper)


def test_too_few_calibration_rows_is_refused():
    """With 3 rows, ⌈(3+1) × 0.8⌉ = 4 — a rank that doesn't exist among 3
    scores. The honest ``q̂`` is infinity. Refusing beats quietly returning the
    largest score and calling it a guarantee."""
    with pytest.raises(ValueError, match="cannot certify"):
        ConformalBand(FixedBand(), alpha=0.2).calibrate(rows(3), salaries_at([1.5] * 3))


def test_predicting_before_calibrating_is_an_error():
    with pytest.raises(RuntimeError, match="calibrate"):
        ConformalBand(FixedBand()).predict_band(rows(2))


def test_calibration_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="9 rows but y_cal has 2"):
        ConformalBand(FixedBand()).calibrate(rows(9), [1.0, 2.0])


# ─────────────────────────────────────────── the three-way split


def test_three_way_split_is_disjoint_and_covers_everything():
    df, _ = synthetic.generate(n=500, seed=4)
    split = three_way_split(df, calibration_size=0.2, test_size=0.2, seed=4)

    train, cal, test = (set(f.index) for f in (split.train, split.calibration, split.test))
    assert train & cal == set()
    assert train & test == set()
    assert cal & test == set()
    assert train | cal | test == set(df.index)


def test_split_fractions_are_of_the_whole_dataset():
    """Ask for 20% and 20% and you get 60/20/20 — not 20% of what survived the
    first cut, which is what naive nesting gives you and nobody means."""
    df, _ = synthetic.generate(n=1000, seed=4)
    n_train, n_cal, n_test = three_way_split(df, seed=4).sizes

    assert n_test == pytest.approx(200, abs=2)
    assert n_cal == pytest.approx(200, abs=2)
    assert n_train == pytest.approx(600, abs=4)


def test_three_way_split_is_reproducible():
    df, _ = synthetic.generate(n=400, seed=4)
    a = three_way_split(df, seed=7)
    b = three_way_split(df, seed=7)
    c = three_way_split(df, seed=8)

    assert list(a.calibration.index) == list(b.calibration.index)
    assert list(a.calibration.index) != list(c.calibration.index)


def test_overlapping_sets_are_rejected_at_construction():
    """The guarantee needs three disjoint sets. If something upstream reindexed
    the frames, this is where it must stop — not three steps later in a coverage
    number that looks fine."""
    df, _ = synthetic.generate(n=100, seed=4)
    with pytest.raises(ValueError, match="disjoint"):
        ThreeWaySplit(train=df, calibration=df, test=df.head(10), description="broken")


def test_split_refuses_to_leave_nothing_to_train_on():
    df, _ = synthetic.generate(n=100, seed=4)
    with pytest.raises(ValueError, match="nothing to train on"):
        three_way_split(df, calibration_size=0.5, test_size=0.5)


# ─────────────────────────────────────────── calibration data is never training data


def test_calibrating_on_training_rows_is_refused():
    """The mistake this whole module is defending against.

    The model already fits its training rows far better than it fits new ones,
    so the conformity scores come out too small, ``q̂`` comes out too small, and
    you publish a coverage guarantee that is simply false. Nothing about the
    output would tell you — the numbers look *better* than the honest ones.
    """
    df, _ = synthetic.generate(n=400, seed=6)
    split = three_way_split(df, seed=6)
    model = SalaryBandModel(seed=6).fit(split.train, split.train["salary_annual"])

    with pytest.raises(ValueError, match="also used to train"):
        ConformalBand(model).calibrate(split.train, split.train["salary_annual"])


def test_calibrating_on_even_one_shared_row_is_refused():
    """Not "mostly disjoint". One shared row and the check fires, because there
    is no principled place to draw a line short of zero."""
    df, _ = synthetic.generate(n=400, seed=6)
    split = three_way_split(df, seed=6)
    model = SalaryBandModel(seed=6).fit(split.train, split.train["salary_annual"])

    contaminated = pd.concat([split.calibration, split.train.head(1)])
    with pytest.raises(ValueError, match="1 calibration rows"):
        ConformalBand(model).calibrate(contaminated, contaminated["salary_annual"])


def test_the_model_records_which_rows_it_trained_on():
    """What makes the check above possible. `random_split` preserves the index
    precisely so a row's identity survives being moved around."""
    df, _ = synthetic.generate(n=200, seed=6)
    split = three_way_split(df, seed=6)
    model = SalaryBandModel(seed=6).fit(split.train, split.train["salary_annual"])

    assert model.train_index_ == frozenset(split.train.index)
    assert not (model.train_index_ & frozenset(split.calibration.index))


# ─────────────────────────────────────────── does it actually work?


@pytest.fixture(scope="module")
def calibrated_on_synthetic():
    """A large, clean synthetic sample — the only place "coverage lands near
    80%" is a testable claim rather than a coin flip.

    On 1,200 test rows the sampling noise on a coverage estimate is about 1.2
    percentage points, and the calibration set contributes about as much again.
    So the tolerance below is ±5 points: wide enough not to be flaky, narrow
    enough that a genuinely broken ``q̂`` fails it.
    """
    df, _ = synthetic.generate(n=6000, seed=0)
    split = three_way_split(df, seed=0)
    model = SalaryBandModel(seed=0).fit(split.train, split.train["salary_annual"])
    conformal = ConformalBand(model).calibrate(
        split.calibration, split.calibration["salary_annual"]
    )
    return model, conformal, split


def test_conformal_coverage_lands_near_nominal(calibrated_on_synthetic):
    _, conformal, split = calibrated_on_synthetic
    report = conformal.evaluate(split.test, split.test["salary_annual"])

    assert report.coverage == pytest.approx(0.8, abs=0.05)


def test_conformal_fixes_an_overconfident_raw_band(calibrated_on_synthetic):
    """The finding this module exists to produce, as a test.

    Quantile regression *aims* for 80% and lands short of it — the raw band here
    covers roughly 75%. That gap is not a bug in LightGBM; it is what happens
    when a finite training set meets rows it has never seen. Conformal
    calibration closes it, and the assertion is that it closes it rather than
    that it merely changes something.
    """
    model, conformal, split = calibrated_on_synthetic
    y_test = split.test["salary_annual"]

    raw = band_report(y_test, model.predict_band(split.test), label="raw")
    calibrated = conformal.evaluate(split.test, y_test)

    assert raw.coverage < 0.8  # the honest starting point
    assert abs(calibrated.coverage - 0.8) < abs(raw.coverage - 0.8)


def test_wider_is_the_price_of_honest_coverage(calibrated_on_synthetic):
    """Narrower is better ONLY if coverage holds.

    The calibrated band is wider than the raw one, and that is not a regression
    — it is the raw band's overconfidence being paid for. A band that shrank and
    lost coverage would not have improved; it would have started lying more
    confidently.
    """
    model, conformal, split = calibrated_on_synthetic
    y_test = split.test["salary_annual"]

    raw = band_report(y_test, model.predict_band(split.test), label="raw")
    calibrated = conformal.evaluate(split.test, y_test)

    assert conformal.qhat_log_ > 0
    assert calibrated.mean_width > raw.mean_width
    assert calibrated.coverage > raw.coverage


def test_calibrated_bands_are_still_ordered(calibrated_on_synthetic):
    _, conformal, split = calibrated_on_synthetic
    band = conformal.predict_band(split.test)

    assert np.all(band.lower <= band.median)
    assert np.all(band.median <= band.upper)


def test_describe_names_the_calibration_set_size(calibrated_on_synthetic):
    """A ``q̂`` computed on 12 rows and one computed on 1,200 deserve different
    amounts of trust, so the number has to appear in the report."""
    _, conformal, _ = calibrated_on_synthetic
    text = conformal.describe()

    assert "1,200 rows" in text
    assert "80%" in text
