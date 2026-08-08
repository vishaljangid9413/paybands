"""Tests for the plots.

**Plots are hard to test, so test the things that can actually be wrong.**

Nobody can assert "this chart is readable" in pytest. But a surprising amount of
what goes wrong with plotting code is not aesthetic at all, and all of it is
checkable:

* it **crashes** on the real data — a category the sample data didn't have, an
  empty group, a NaN;
* it **ignores the ``ax`` you passed** and opens its own figure, which quietly
  breaks every multi-panel layout downstream;
* it **shows the wrong number** — the bar says 80% and the data says 62%;
* it **writes files** you didn't ask for, or leaks figures until the suite runs
  out of memory.

So that is what is tested here, on real survey data *and* on synthetic data,
because the two have genuinely different shapes and a plot that only works on one
of them is a plot that will break the week company data arrives.

What is deliberately not tested: colours, fonts, positions. Those are judgement,
and pinning them down in assertions would mean every visual improvement breaks
the build.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from paybands.data import stackoverflow
from paybands.data.synthetic import generate
from paybands.model.metrics import coverage as metrics_coverage
from paybands.plots import (
    band_width_plot,
    bootstrap_median_ci,
    coverage_by_quantile,
    coverage_plot,
    empirical_coverage,
    gap_comparison,
    residual_by_group,
    salary_by_experience,
    salary_distribution,
    salary_distribution_comparison,
    save,
    skewness,
)
from paybands.plots.style import rupee_formatter, rupee_label

SURVEY_PATH = Path("data/public/so_2025_raw.csv")


# ─────────────────────────────────────────── housekeeping


@pytest.fixture(autouse=True)
def _close_figures():
    """Shut every figure after each test.

    matplotlib keeps a global registry of open figures — that is how
    ``plt.savefig()`` knows what "the current figure" means. Nothing here ever
    closes them on its own, so a suite that makes a few hundred charts holds a
    few hundred rendered canvases in memory and eventually falls over with a
    warning about too many open figures. One autouse fixture, and the problem
    never exists.
    """
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    """A synthetic frame — always available, no download required."""
    df, _ = generate(n=1200, seed=7)
    return df


@pytest.fixture(scope="module")
def survey() -> pd.DataFrame:
    """The real Stack Overflow India extract, skipped if not downloaded."""
    if not SURVEY_PATH.exists():
        pytest.skip("survey CSV not downloaded")
    df, _ = stackoverflow.load(SURVEY_PATH, verbose=False)
    return df


def _fake_bands(salaries: np.ndarray, *, width: float = 0.35, seed: int = 3):
    """A band per person, built without a model.

    The plots must not care where a band came from, so the tests deliberately do
    not use one. This is a point prediction with noise, plus a multiplicative
    band around it — enough structure to exercise every code path, and
    reproducible.
    """
    rng = np.random.default_rng(seed)
    prediction = salaries * np.exp(rng.normal(0.0, 0.2, salaries.size))
    return prediction, prediction * (1 - width), prediction * (1 + width)


# ─────────────────────────────────────────── the rupee formatter


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (15_00_000, "₹15L"),
        (1_20_00_000, "₹1.2Cr"),
        (7_50_000, "₹7.5L"),
        (1_00_000, "₹1L"),
        (1_00_00_000, "₹1Cr"),
        (2_00_00_000, "₹2Cr"),
        (48_000, "₹48,000"),
        (0, "₹0"),
        (-15_00_000, "-₹15L"),
        (12_34_000, "₹12.3L"),
    ],
)
def test_rupee_label_uses_lakhs_and_crores(amount, expected):
    """₹15L, not ₹1.5M.

    A chart axis written in millions is a chart an Indian reader has to translate
    before they can use it — and one they will misread at a glance, because
    ₹1.5M *looks* like it should be somewhere near ₹1.5L. The whole point of the
    formatter is that no translation happens.
    """
    assert rupee_label(amount) == expected


def test_rupee_formatter_is_the_same_function_on_an_axis():
    """The formatter and the plain function must never drift apart.

    ``FuncFormatter`` passes a tick position as a second argument, which is why
    ``rupee_label`` accepts one and ignores it. If that signature ever changed,
    the axis would silently stop matching the labels we write by hand.
    """
    formatter = rupee_formatter()
    assert formatter(15_00_000, 0) == "₹15L"
    assert formatter(1_20_00_000, 3) == "₹1.2Cr"


# ─────────────────────────────────────────── coverage arithmetic


def test_coverage_matches_a_hand_calculation():
    """Ten people, a band of ₹10L–₹20L each, two of them outside it → 0.8.

    The same constructed case as ``test_metrics.py`` uses, on purpose. This is
    the number ``coverage_plot`` draws, and if the arithmetic behind the bar is
    wrong then the chart is confidently reporting a lie — which is worse than
    having no chart.
    """
    truth = np.array([12, 13, 14, 15, 16, 17, 18, 19, 5, 40], dtype=float) * 1_00_000
    lower = np.full(10, 10_00_000.0)
    upper = np.full(10, 20_00_000.0)

    assert empirical_coverage(truth, lower, upper) == 0.8


def test_plot_coverage_agrees_with_the_model_metric():
    """Two implementations of coverage exist — they must never disagree.

    ``plots.calibration`` reimplements it so the plots stay usable on any pair of
    arrays without dragging the model package along. That freedom is only worth
    having if the two definitions stay identical, boundary behaviour included.
    """
    rng = np.random.default_rng(11)
    truth = rng.lognormal(14.0, 0.5, 400)
    lower, upper = truth * 0.7, truth * 1.1
    assert empirical_coverage(truth, lower, upper) == metrics_coverage(truth, lower, upper)

    # And on the boundary, where the two could most easily drift apart.
    assert empirical_coverage([10.0, 20.0], [10.0, 10.0], [20.0, 20.0]) == 1.0


def test_coverage_plot_draws_the_number_it_computed():
    """The bar length must equal the measured coverage, not something near it.

    Constructed so 3 of 4 land inside: coverage is 0.75 against a promised 0.8.
    The chart claims over-confidence, so the chart had better be drawing 0.75.
    """
    truth = np.array([15.0, 15.0, 15.0, 50.0]) * 1_00_000
    lower = np.full(4, 10_00_000.0)
    upper = np.full(4, 20_00_000.0)

    fig, ax = coverage_plot(truth, lower, upper, nominal=0.8)

    widths = sorted(patch.get_width() for patch in ax.patches)
    assert widths == pytest.approx([0.75, 0.8])
    assert isinstance(fig, Figure)


def test_coverage_plot_rejects_an_impossible_confidence():
    with pytest.raises(ValueError, match="nominal"):
        coverage_plot([1.0], [0.0], [2.0], nominal=1.0)


def test_inverted_bands_are_refused():
    """Lower above upper is a sign flip somewhere upstream, not a plot to draw."""
    with pytest.raises(ValueError, match="inverted"):
        empirical_coverage([15.0], [20.0], [10.0])


def test_coverage_by_quantile_needs_at_least_one_level():
    with pytest.raises(ValueError, match="empty"):
        coverage_by_quantile([1.0, 2.0], {})


# ─────────────────────────────────────────── the statistics helpers


def test_skew_is_positive_for_salaries_and_near_zero_for_their_logs():
    """The whole argument for the log transform, as an assertion.

    Real numbers from the survey extract: +2.87 raw, −0.23 logged. Here we only
    pin the *direction*, because the exact figure depends on the data file and a
    test that breaks when the survey is updated is a test nobody keeps.
    """
    rng = np.random.default_rng(5)
    salaries = rng.lognormal(mean=14.0, sigma=0.7, size=5000)

    assert skewness(salaries) > 1.0
    assert abs(skewness(np.log(salaries))) < 0.2


def test_bootstrap_band_is_wider_for_a_smaller_sample():
    """The property the experience plot depends on.

    Same distribution, 30 people vs 3,000. The band on the small sample must be
    visibly wider — that is what makes the ``20+`` bucket's uncertainty show up
    on the chart instead of hiding behind a confident-looking dot.
    """
    rng = np.random.default_rng(1)
    population = rng.lognormal(14.0, 0.6, 100_000)

    small = bootstrap_median_ci(rng.choice(population, 30), seed=0)
    large = bootstrap_median_ci(rng.choice(population, 3000), seed=0)

    assert (small[1] - small[0]) > 5 * (large[1] - large[0])


def test_bootstrap_is_reproducible():
    """A chart that redraws differently every run is a chart nobody can review."""
    values = np.array([5.0, 9.0, 11.0, 14.0, 20.0, 33.0])
    assert bootstrap_median_ci(values, seed=4) == bootstrap_median_ci(values, seed=4)


# ─────────────────────────────────────────── every plot, on both datasets


def _all_plots(df: pd.DataFrame, group_col: str) -> list[tuple[Figure, object]]:
    """Build one of every chart from a common-schema frame."""
    salaries = df["salary_annual"].to_numpy(dtype=float)
    prediction, lower, upper = _fake_bands(salaries)
    bands = {
        level: (prediction * (1 - level / 2), prediction * (1 + level / 2))
        for level in (0.5, 0.7, 0.8, 0.95)
    }

    return [
        salary_distribution(salaries),
        salary_distribution(salaries, log=True),
        salary_distribution_comparison(salaries),
        salary_by_experience(df),
        coverage_plot(salaries, lower, upper, nominal=0.8),
        coverage_by_quantile(salaries, bands),
        band_width_plot(prediction, lower, upper),
        band_width_plot(prediction, lower, upper, relative=True),
        residual_by_group(salaries, prediction, df[group_col].to_numpy()),
        residual_by_group(salaries, prediction, df[group_col].to_numpy(), relative=True),
        gap_comparison(-0.14, -0.06, (-0.10, -0.02), raw_ci=(-0.19, -0.09), known_gap=-0.08),
    ]


def test_every_plot_works_on_synthetic_data(synthetic):
    """Synthetic data first: it is always present, so this runs everywhere."""
    results = _all_plots(synthetic, group_col="gender")

    assert len(results) == 11
    for fig, _ in results:
        assert isinstance(fig, Figure)


@pytest.mark.skipif(not SURVEY_PATH.exists(), reason="survey CSV not downloaded")
def test_every_plot_works_on_the_real_survey(survey):
    """And again on the real thing.

    The two sources differ in ways that break plotting code specifically:
    the survey has missing experience (so a bucket can be empty), string age
    bands rather than a clean two-value group, and a much heavier right tail. A
    plot that only survives the data we invented is a plot that has not been
    tested.
    """
    results = _all_plots(survey, group_col="age_band")

    assert len(results) == 11
    for fig, _ in results:
        assert isinstance(fig, Figure)


def test_experience_plot_shows_the_sample_size(survey):
    """``n`` is on the chart, not in a caption someone will crop off.

    This is the assertion that protects the finding in the README: the ``20+``
    bucket's flat median comes from about 36 people, and a reader who cannot see
    that will read it as "senior pay plateaus".
    """
    _, ax = salary_by_experience(survey)
    tick_text = [label.get_text() for label in ax.get_xticklabels()]

    assert all("n=" in text for text in tick_text)
    assert any(text.startswith("20+") for text in tick_text)


def test_experience_plot_needs_a_salary_column():
    with pytest.raises(ValueError, match="salary_annual"):
        salary_by_experience(pd.DataFrame({"years_experience": [1, 2, 3]}))


def test_experience_plot_needs_experience_somehow():
    with pytest.raises(ValueError, match="experience_bucket"):
        salary_by_experience(pd.DataFrame({"salary_annual": [1e6, 2e6]}))


def test_distribution_refuses_data_it_cannot_log():
    """Zero and negative salaries are dropped; a frame of nothing but those is an
    error, not an empty chart. An empty chart looks like a finding."""
    with pytest.raises(ValueError, match="no positive"):
        salary_distribution([0.0, -5.0, np.nan])


def test_gap_comparison_refuses_an_estimate_outside_its_own_interval():
    """Almost always a units mix-up — a percentage estimate against a rupee
    interval, or a one-sided bound passed as two-sided."""
    with pytest.raises(ValueError, match="outside its own interval"):
        gap_comparison(-0.14, -0.06, (0.01, 0.05))


# ─────────────────────────────────────────── composition


@pytest.mark.parametrize(
    "draw",
    [
        lambda ax: salary_distribution([5e5, 9e5, 15e5, 40e5, 90e5], ax=ax),
        lambda ax: coverage_plot([15e5, 15e5, 50e5], [10e5] * 3, [20e5] * 3, ax=ax),
        lambda ax: coverage_by_quantile([15e5, 16e5, 50e5], {0.8: ([10e5] * 3, [20e5] * 3)}, ax=ax),
        lambda ax: band_width_plot(
            [10e5, 20e5, 40e5], [8e5, 15e5, 30e5], [12e5, 25e5, 50e5], ax=ax
        ),
        lambda ax: residual_by_group(
            [10e5, 20e5, 30e5, 40e5], [11e5, 19e5, 33e5, 38e5], ["a", "b", "a", "b"], ax=ax
        ),
        lambda ax: gap_comparison(-0.1, -0.04, (-0.08, 0.0), ax=ax),
    ],
)
def test_passing_an_ax_draws_into_it(draw):
    """The rule that makes multi-panel figures possible.

    Given an ``ax``, a plotting function must draw into *that* axes and open no
    figure of its own. If it quietly called ``plt.subplots()`` instead, the
    caller's panel would come out blank and a stray figure would leak — the kind
    of bug that only shows up when you try to assemble the report.
    """
    fig, axes = plt.subplots(1, 2)
    before = set(plt.get_fignums())

    returned_fig, returned_ax = draw(axes[0])

    assert returned_fig is fig
    assert returned_ax is axes[0]
    assert set(plt.get_fignums()) == before, "a new figure was created despite ax= being passed"
    # Something was actually drawn — an axes with no artists and no title means
    # the function returned early rather than plotting.
    assert axes[0].has_data() or axes[0].get_title()
    assert not axes[1].has_data(), "drew into the wrong axes"


def test_the_comparison_figure_is_two_composed_panels():
    """``salary_distribution_comparison`` is the payoff of the ``ax`` rule: it
    builds a figure and calls the single-panel function twice."""
    fig, (raw_ax, log_ax) = salary_distribution_comparison([5e5, 9e5, 15e5, 40e5, 90e5])

    assert isinstance(fig, Figure)
    assert isinstance(raw_ax, Axes) and isinstance(log_ax, Axes)
    assert len(fig.axes) == 2
    assert raw_ax.get_xscale() == "linear"
    assert log_ax.get_xscale() == "log"


# ─────────────────────────────────────────── files on disk


def test_save_writes_a_real_file(tmp_path):
    """``save`` is the only function in the package allowed to touch the disk."""
    fig, _ = salary_distribution([5e5, 9e5, 15e5, 40e5, 90e5])

    written = save(fig, tmp_path / "distribution.png")

    assert written == tmp_path / "distribution.png"
    assert written.exists()
    assert written.stat().st_size > 1000, "a valid PNG of a real chart is not tiny"


def test_save_creates_missing_directories(tmp_path):
    """So a caller never has to remember to mkdir the report folder first."""
    fig, _ = salary_distribution([5e5, 9e5, 15e5])

    written = save(fig, tmp_path / "nested" / "deeper" / "chart.png")

    assert written.exists()


def test_no_plot_writes_anything_by_itself(tmp_path, monkeypatch, synthetic):
    """**The side-effect test.**

    Drawing and saving are separate decisions, and a plotting function that
    writes a file as a side effect is how a repo ends up full of stale PNGs
    nobody can match to a model version. Run every chart in an empty directory
    and require that the directory is still empty.
    """
    monkeypatch.chdir(tmp_path)

    _all_plots(synthetic, group_col="gender")

    assert list(tmp_path.iterdir()) == [], "a plot wrote a file without being asked"
