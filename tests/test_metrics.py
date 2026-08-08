"""Tests for the evaluation metrics.

Metrics are the instruments we judge everything else by, so they get checked
against numbers worked out by hand. An instrument nobody calibrated is an
instrument that agrees with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from paybands.model.metrics import (
    coverage,
    evaluate,
    evaluate_log_predictions,
    format_rupees,
    from_log,
    mae,
    mape,
    median_absolute_error,
    pinball_loss,
    r2,
    to_log,
)

# ─────────────────────────────────────────── the headline scores


def test_mae_is_the_average_miss():
    """Hand-computed: misses of 1L, 2L, 3L → average 2L."""
    truth = [10_00_000, 20_00_000, 30_00_000]
    pred = [11_00_000, 22_00_000, 27_00_000]
    assert mae(truth, pred) == pytest.approx(2_00_000)


def test_median_ignores_the_one_catastrophic_miss():
    """The reason both are reported.

    Four small misses and one enormous one: the mean is dragged far above what
    a typical candidate would experience, the median is not. Seeing the two
    numbers diverge tells you the model is usually fine and occasionally very
    wrong — a different problem, needing a different fix, from being uniformly
    mediocre.
    """
    truth = np.array([10, 10, 10, 10, 10]) * 1_00_000
    pred = np.array([11, 11, 11, 11, 60]) * 1_00_000

    assert median_absolute_error(truth, pred) == pytest.approx(1_00_000)
    assert mae(truth, pred) > 10 * median_absolute_error(truth, pred)


def test_mape_scales_the_error_by_salary():
    """Being ₹5L wrong about ₹10L is 50%; ₹5L wrong about ₹50L is 10%."""
    assert mape([10_00_000], [15_00_000]) == pytest.approx(50.0)
    assert mape([50_00_000], [45_00_000]) == pytest.approx(10.0)


def test_r2_is_zero_when_you_predict_the_mean():
    """R² answers "how much better than always guessing the mean?". Guessing the
    mean must therefore score exactly zero — the definition, pinned down."""
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    assert r2(truth, np.full(4, truth.mean())) == pytest.approx(0.0)


def test_r2_can_go_negative():
    """Worse than guessing the mean is possible and does happen. A metric that
    silently clipped at zero would hide a broken model."""
    assert r2([10.0, 20.0, 30.0], [100.0, 200.0, 300.0]) < 0


# ─────────────────────────────────────────── the log transform


def test_back_transform_round_trips():
    """exp(log(x)) must return exactly the salaries we started with.

    Everything downstream depends on this: the model predicts logs, and the
    report must be in rupees. If the round trip drifted, every number we publish
    would be quietly wrong.
    """
    salaries = np.array([1_00_000, 8_50_000, 15_00_000, 2_00_00_000], dtype=float)
    assert from_log(to_log(salaries)) == pytest.approx(salaries)


def test_exp_of_mean_log_is_the_geometric_mean():
    """The subtlety documented in from_log(), pinned as a test.

    ₹10L, ₹20L, ₹40L: the arithmetic mean is ₹23.33L, but averaging the logs and
    exponentiating gives ₹20L — the geometric mean, and always the lower of the
    two. A log-space model that you exponentiate is estimating the typical
    salary, not the average one.
    """
    salaries = np.array([10, 20, 40], dtype=float) * 1_00_000

    arithmetic = salaries.mean()
    geometric = float(from_log(to_log(salaries).mean()))

    assert arithmetic == pytest.approx(23_33_333, rel=1e-4)
    assert geometric == pytest.approx(20_00_000)
    assert geometric < arithmetic


def test_log_of_zero_salary_is_refused():
    with pytest.raises(ValueError, match="non-positive"):
        to_log([10_00_000, 0])


def test_log_predictions_are_scored_in_rupees():
    """The workflow the real model uses: truth in rupees, predictions in logs,
    error reported in rupees."""
    truth = np.array([10_00_000, 20_00_000], dtype=float)
    pred_log = to_log([11_00_000, 18_00_000])

    result = evaluate_log_predictions(truth, pred_log)
    assert result.mae == pytest.approx(1_50_000)


# ─────────────────────────────────────────── scoring a band


def test_coverage_is_exactly_eighty_percent_when_eight_of_ten_are_inside():
    """Constructed so the answer is known: ten people, bands of ₹10L–₹20L, and
    exactly two of them earn outside that range.

    This is the check that keeps Phase 3 honest. If we promise an 80% band, this
    function is what proves — or disproves — that it contains 80%.
    """
    truth = np.array([12, 13, 14, 15, 16, 17, 18, 19, 5, 40], dtype=float) * 1_00_000
    lower = np.full(10, 10_00_000.0)
    upper = np.full(10, 20_00_000.0)

    assert coverage(truth, lower, upper) == 0.8


def test_coverage_includes_the_boundaries():
    """A salary landing exactly on a band edge counts as covered. Arbitrary, but
    it has to be decided or the tests go flaky on round numbers."""
    assert coverage([10.0, 20.0], [10.0, 10.0], [20.0, 20.0]) == 1.0


def test_a_perfectly_wide_band_covers_everyone():
    """Coverage alone cannot tell you a band is useful — ₹0 to ₹5Cr scores 100%
    and helps nobody. Report width alongside it, always."""
    truth = np.array([1, 50, 500], dtype=float) * 1_00_000
    assert coverage(truth, np.zeros(3), np.full(3, 5_00_00_000.0)) == 1.0


def test_inverted_band_is_rejected():
    with pytest.raises(ValueError, match="inverted"):
        coverage([15.0], [20.0], [10.0])


def test_pinball_at_the_median_is_half_of_mae():
    """The 0.5 quantile charges both directions equally, so it collapses to
    MAE/2. A useful sanity anchor for a metric that's otherwise hard to eyeball.
    """
    truth = [10.0, 20.0, 30.0]
    pred = [12.0, 17.0, 35.0]
    assert pinball_loss(truth, pred, 0.5) == pytest.approx(mae(truth, pred) / 2)


def test_high_quantile_punishes_under_prediction_harder():
    """The asymmetry that makes quantile regression work.

    For a 90th-percentile prediction, landing too LOW is the real failure —
    the band's ceiling would sit under people it was supposed to contain. So
    the same size of error must cost more when it's an under-prediction.
    """
    too_low = pinball_loss([20.0], [10.0], quantile=0.9)
    too_high = pinball_loss([20.0], [30.0], quantile=0.9)
    assert too_low > too_high
    assert too_low == pytest.approx(9.0)  # 0.9 × 10
    assert too_high == pytest.approx(1.0)  # 0.1 × 10


@pytest.mark.parametrize("bad", [0, 1, -0.5, 1.2])
def test_impossible_quantiles_rejected(bad):
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss([1.0], [1.0], bad)


# ─────────────────────────────────────────── reporting


def test_rupees_use_indian_digit_grouping():
    """₹21,34,500 — not ₹2,134,500. Lakh grouping, not thousands."""
    assert format_rupees(2_134_500) == "₹21,34,500"
    assert format_rupees(1_00_000) == "₹1,00,000"
    assert format_rupees(1_00_00_000) == "₹1,00,00,000"
    assert format_rupees(999) == "₹999"
    assert format_rupees(-5_00_000) == "-₹5,00,000"


def test_metricset_prints_rupees_not_raw_floats():
    """The report has to be readable by a recruiter, which means ₹ signs and no
    scientific notation."""
    text = str(evaluate([10_00_000, 20_00_000], [12_00_000, 19_00_000], label="baseline"))
    assert "baseline" in text
    assert "₹" in text
    assert "MAE" in text
    assert "e+0" not in text


# ─────────────────────────────────────────── guardrails


def test_mismatched_lengths_are_refused():
    """Almost always a split gone wrong — predictions from one set scored
    against another's labels. numpy would broadcast some of these into a
    plausible-looking wrong answer."""
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate([1.0, 2.0, 3.0], [1.0, 2.0])


def test_nan_predictions_are_refused():
    """A NaN means the model failed on that row. Averaging it away would report
    a score for a model that didn't answer."""
    with pytest.raises(ValueError, match="NaN"):
        evaluate([1.0, 2.0], [1.0, np.nan])
