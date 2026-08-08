"""Tests for the synthetic data generator.

The headline test here is `test_regression_recovers_the_injected_gap`. Everything
else checks that the generator is internally consistent; that one checks that it
*does what it claims* — we inject an 8% gap, measure it back with an ordinary
regression, and demand ≈8%.

That matters more than it looks. This dataset is the measuring stick for the
whole fairness phase. If the stick itself is wrong, every fairness number in the
project is wrong too, and nothing downstream would ever tell us.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from paybands.data.schema import CORE_COLUMNS, TARGET, Source
from paybands.data.synthetic import (
    INSTITUTE_TIER_MULTIPLIERS,
    GroundTruth,
    experience_multiplier,
    generate,
)

# Big enough that sampling noise is smaller than the effects we are testing.
# With n=10,000 the standard error on the gender coefficient is about 0.005 in
# log points, so a 0.01 tolerance is roughly two standard errors — tight enough
# to catch a broken generator, loose enough not to fail on luck.
N = 10_000
TOLERANCE = 0.01

#: Everything the audit is allowed to control for. Note what is *missing*:
#: `career_gap_months`. It is added separately, because whether you control for
#: a proxy changes the answer, and that is a lesson in itself.
CONTROLS = [
    "years_experience",
    "role",
    "education",
    "org_size",
    "remote",
    "employment_type",
    "location_tier",
    "institute_tier",
    "prev_company_type",
]


def measured_gap(df: pd.DataFrame, *, control_for_career_gap: bool = True) -> float:
    """Measure the pay gap the way an audit would, and return it as a fraction.

    This is ordinary least squares on `log(salary)`, with one indicator for the
    disadvantaged group and dummy variables for every control. The coefficient
    on that indicator is "the pay difference between two people who are the same
    in every other listed respect".

    Two details that are easy to get wrong and worth stating:

    * **Log, not rupees.** The gap is multiplicative, so it only shows up as a
      single number after taking logs. In rupees it would be a different number
      for every salary level.
    * **`1 - exp(coefficient)`, not `-coefficient`.** A coefficient of −0.0834
      is an 8.0% cut, not an 8.3% one. At small gaps the two nearly agree, which
      is exactly why the mistake survives so long unnoticed.

    Experience is deliberately entered as *dummies*, one per year, rather than
    as the power curve the generator actually used. That way this measurement
    does not assume it already knows the answer — it would still recover the gap
    if the experience curve were some other shape entirely.
    """
    design = pd.get_dummies(df[CONTROLS].astype(str), drop_first=True, dtype=float)
    if control_for_career_gap:
        design["career_gap_years"] = df["career_gap_months"].to_numpy(float) / 12.0
    design["is_disadvantaged"] = (df["gender"] == "female").astype(float)
    design.insert(0, "intercept", 1.0)

    log_salary = np.log(df[TARGET].to_numpy(float))
    coefficients, *_ = np.linalg.lstsq(design.to_numpy(float), log_salary, rcond=None)
    gender_coefficient = dict(zip(design.columns, coefficients, strict=True))["is_disadvantaged"]
    return float(1.0 - np.exp(gender_coefficient))


@pytest.fixture(scope="module")
def biased() -> tuple[pd.DataFrame, GroundTruth]:
    """The default world: an 8% gap, and a proxy that partly encodes it."""
    return generate(N, seed=42)


@pytest.fixture(scope="module")
def fair() -> tuple[pd.DataFrame, GroundTruth]:
    """The control world: no direct gender effect at all."""
    return generate(N, seed=42, pay_gap=0.0)


# ─────────────────────────────────────────── the headline test


def test_regression_recovers_the_injected_gap(biased):
    """Inject 8%, measure it back, get 8%.

    If this fails, the generator's pay rules and its `GroundTruth` disagree —
    which means every fairness number validated against this dataset is
    measuring the wrong target.
    """
    df, truth = biased
    assert measured_gap(df) == pytest.approx(truth.pay_gap, abs=TOLERANCE)


@pytest.mark.parametrize("injected", [0.03, 0.08, 0.15, 0.30])
def test_the_gap_is_recovered_at_any_size(injected):
    """Not just at the default. A generator that only works at 8% would be a
    coincidence, and would quietly mislead anyone who changed the parameter."""
    df, truth = generate(N, seed=7, pay_gap=injected)
    assert measured_gap(df) == pytest.approx(truth.pay_gap, abs=TOLERANCE)


def test_zero_pay_gap_leaves_nothing_to_find(fair):
    """The control case. `pay_gap=0.0` must produce genuinely unbiased pay.

    "Unbiased" means precisely this: two people alike in every recorded respect
    are paid the same, on average. It does *not* mean the two groups earn the
    same on average overall — they still don't, because one group takes more
    career breaks and breaks cost money. That residual difference is explained
    by something in the data, and separating "explained" from "unexplained" is
    the entire job of the audit. See the proxy tests below.
    """
    df, truth = fair
    assert truth.pay_gap == 0.0
    assert measured_gap(df) == pytest.approx(0.0, abs=TOLERANCE)


# ─────────────────────────────────────────── the proxy variable


def test_career_gaps_encode_gender(biased):
    """The proxy has to be real, or the proxy demonstration proves nothing.

    Career breaks are not a neutral fact about a person here: they carry
    information about which group that person is in. That is what makes them a
    proxy, and it is why deleting the `gender` column does not delete the bias.
    """
    df, _ = biased
    by_group = df.groupby("gender")["career_gap_months"]

    assert by_group.mean()["female"] > 5 * by_group.mean()["male"]

    is_disadvantaged = (df["gender"] == "female").astype(float)
    correlation = np.corrcoef(is_disadvantaged, df["career_gap_months"])[0, 1]
    assert correlation > 0.25  # strong enough that any model will find it


def test_the_proxy_carries_part_of_the_gap(biased):
    """Stop controlling for career gaps and the measured gap gets *bigger*.

    This is the proxy problem in one assertion. The extra few percent is the
    part of the unfairness that flows through career breaks rather than through
    gender directly — so a model that never sees the gender column can still
    reproduce it. That is the result Phase 4 has to demonstrate, and this test
    proves the data is capable of demonstrating it.
    """
    df, truth = biased
    controlled = measured_gap(df, control_for_career_gap=True)
    uncontrolled = measured_gap(df, control_for_career_gap=False)

    assert controlled == pytest.approx(truth.pay_gap, abs=TOLERANCE)
    assert uncontrolled > controlled + 0.02


def test_the_proxy_can_be_switched_off():
    """Equal career-break rates for both groups → no proxy channel left.

    Useful when you want the injected gap to be the *only* difference between
    groups: it isolates one mechanism at a time, which is how you debug an audit
    that is reporting something strange.
    """
    df, truth = generate(N, seed=3, career_gap_prob_disadvantaged=0.06)
    assert not truth.has_proxy

    controlled = measured_gap(df, control_for_career_gap=True)
    uncontrolled = measured_gap(df, control_for_career_gap=False)
    assert uncontrolled == pytest.approx(controlled, abs=TOLERANCE)


def test_career_breaks_never_exceed_half_a_career(biased):
    """Someone with two years of experience has not taken a five-year break.

    Nonsense rows like that are the kind of thing that makes a reviewer stop
    trusting synthetic data entirely.
    """
    df, _ = biased
    assert (df["career_gap_months"] <= df["years_experience"] * 6).all()
    assert (df.loc[df["years_experience"] == 0, "career_gap_months"] == 0).all()


# ─────────────────────────────────────────── reproducibility


def test_same_seed_is_byte_for_byte_identical():
    """Not "statistically similar" — identical. Hash the CSV and compare.

    Reproducibility is what lets you answer "which settings produced this
    number?" in week 6. Without it, a fairness result you cannot regenerate is a
    fairness result you cannot defend.
    """
    first, _ = generate(2_000, seed=99)
    second, _ = generate(2_000, seed=99)

    def digest(df: pd.DataFrame) -> str:
        return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()

    assert digest(first) == digest(second)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_data():
    """The seed has to actually do something."""
    first, _ = generate(2_000, seed=1)
    second, _ = generate(2_000, seed=2)
    assert not first[TARGET].equals(second[TARGET])


def test_global_numpy_state_cannot_change_the_output():
    """The generator uses its own `Generator`, never `np.random.seed`.

    Global random state is shared by every library in the process. If this file
    used it, then importing something that happens to draw a random number would
    silently change your dataset — and "same seed, same data" would quietly stop
    being true without any error anywhere.
    """
    np.random.seed(0)
    first, _ = generate(1_000, seed=5)
    np.random.seed(12345)
    np.random.random(50)  # deliberately disturb the global state
    second, _ = generate(1_000, seed=5)
    pd.testing.assert_frame_equal(first, second)


# ─────────────────────────────────────────── the experience curve


def test_experience_curve_increases_but_decelerates():
    """More experience always pays more — but each extra year pays less.

    A straight line would fail the second half of that sentence, and a straight
    line is what you get by default if nobody thinks about it.
    """
    years = np.arange(0, 31)
    curve = np.asarray(experience_multiplier(years))

    steps = np.diff(curve)
    assert (steps > 0).all(), "more experience must never pay less"
    assert (np.diff(steps) < 0).all(), "each extra year must be worth less than the last"

    # Concretely: the first five years of a career are worth several times what
    # years 15-20 are worth. Compare the *raises* (130% vs 23%), not the
    # multipliers themselves — every multiplier is above 1, so comparing those
    # directly would flatter a curve that had barely risen at all.
    early_raise = curve[5] / curve[0] - 1
    late_raise = curve[20] / curve[15] - 1
    assert early_raise > 4 * late_raise


def test_median_salary_flattens_with_experience(biased):
    """The same shape, measured on the generated data rather than the formula.

    Percentage growth per year must shrink as careers progress. Note it is
    *percentage* growth: in rupees the later gains are larger, because a smaller
    percentage of a much larger salary is still a lot of money. Confusing those
    two is how people conclude that experience pays off linearly.
    """
    df, _ = biased
    buckets = pd.cut(df["years_experience"], [-1, 2, 5, 8, 12, 20, 40])
    medians = df.groupby(buckets, observed=True)[TARGET].median()
    midpoints = np.array([1, 3.5, 6.5, 10, 16, 27])

    assert medians.is_monotonic_increasing

    growth_per_year = np.diff(np.log(medians.to_numpy())) / np.diff(midpoints)
    assert (np.diff(growth_per_year) < 0).all(), "growth per year must slow down"


def test_elite_college_premium_fades(biased):
    """An IIT degree is worth ~20% to a fresher and little to a veteran.

    Measured here the way the analysis will measure it: compare tier-1 to other
    institutes among juniors, then among seniors, and check the advantage
    shrinks. It is a satisfying finding precisely because it is checkable.
    """
    df, _ = biased
    assert INSTITUTE_TIER_MULTIPLIERS["tier1"] > INSTITUTE_TIER_MULTIPLIERS["other"]

    def premium(subset: pd.DataFrame) -> float:
        by_tier = subset.groupby("institute_tier")[TARGET].median()
        return by_tier["tier1"] / by_tier["other"]

    juniors = premium(df[df["years_experience"] <= 3])
    veterans = premium(df[df["years_experience"] >= 15])
    assert juniors > veterans
    assert veterans < 1.06  # all but gone


# ─────────────────────────────────────────── shape and sanity


def test_produces_the_common_schema(biased):
    """Same columns as every other loader, so everything downstream just works."""
    df, _ = biased
    assert set(CORE_COLUMNS) <= set(df.columns)
    for column in ("location_tier", "institute_tier", "prev_company_type", "gender", "level"):
        assert column in df.columns
    assert (df["source"] == Source.SYNTHETIC.value).all()
    assert len(df) == N
    assert not df.isna().to_numpy().any()


def test_salaries_are_plausible_indian_salaries(biased):
    """Roughly ₹3L to ₹80L, positive, and not piled up against the clip bounds.

    The clip is a safety net, not a modelling step. If it ever starts binding,
    the multipliers have drifted somewhere unrealistic — so the test asserts it
    does nothing at the default settings rather than assuming it.
    """
    df, _ = biased
    salary = df[TARGET]

    assert (salary > 0).all()
    assert salary.quantile(0.001) > 300_000
    assert salary.quantile(0.999) < 8_000_000
    assert 1_000_000 < salary.median() < 2_500_000

    from paybands.data.schema import MAX_PLAUSIBLE_SALARY, MIN_PLAUSIBLE_SALARY

    assert (salary > MIN_PLAUSIBLE_SALARY).all()
    assert (salary < MAX_PLAUSIBLE_SALARY).all()


def test_ground_truth_reports_what_was_used(biased):
    """The audit is scored against these numbers, so they must be the real ones."""
    df, truth = biased
    assert truth.n == N
    assert truth.seed == 42
    assert truth.pay_gap == 0.08
    assert truth.log_pay_gap == pytest.approx(np.log(0.92))
    assert truth.has_proxy
    assert truth.role_multipliers["QA"] < truth.role_multipliers["Data Science"]
    assert truth.location_tier_multipliers[1] > truth.location_tier_multipliers[2]
    assert truth.prev_company_multipliers["product"] > truth.prev_company_multipliers["services"]
    assert truth.noise_sigma_log > 0

    # It prints as a readable summary, because a ground truth nobody reads is a
    # ground truth nobody checks.
    assert "8.0%" in str(truth)
    assert "female" in str(truth)


def test_ground_truth_is_immutable(biased):
    """Nothing downstream may edit the answer sheet after the fact."""
    _, truth = biased
    with pytest.raises(Exception):  # noqa: B017 — dataclasses raise FrozenInstanceError
        truth.pay_gap = 0.5


@pytest.mark.parametrize("bad_gap", [-0.01, 1.0, 1.5])
def test_impossible_pay_gaps_rejected(bad_gap):
    with pytest.raises(ValueError, match="pay_gap"):
        generate(100, pay_gap=bad_gap)


@pytest.mark.parametrize("bad_n", [0, -1])
def test_impossible_sizes_rejected(bad_n):
    with pytest.raises(ValueError, match="n must be positive"):
        generate(bad_n)
