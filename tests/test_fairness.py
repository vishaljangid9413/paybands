"""Tests for the fairness audit — and, mostly, tests *of* the fairness audit.

There is a difference, and it is the whole point of this file.

Most of these tests do not check that a function returns the right type or
handles an empty list. They check that a **measuring instrument reads correctly
against a known standard**. We generate data in which one group is paid exactly
8% less because we wrote that rule ourselves, we run the audit, and we demand
8% back. If it says 2% or 15%, the audit is broken — and an audit that is
broken in the confident direction gets somebody accused of discrimination they
did not commit, while one broken the other way tells a company everything is
fine when it is not.

You cannot do this on real data. Nobody knows the true pay gap at a real
company, so a wrong answer and a right answer look identical. The synthetic
generator exists so that here, in this file, they do not.

The headline is :func:`test_recovers_the_injected_gap`. Everything else supports
it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paybands.data.schema import TARGET
from paybands.data.synthetic import generate
from paybands.fairness.audit import (
    CONTESTED_CONTROLS,
    LEGITIMATE_CONTROLS,
    FairnessReport,
    adjusted_gap,
    audit,
    raw_gap,
    residual_analysis,
    validate_on_synthetic,
)
from paybands.fairness.proxy import (
    LeastSquares,
    compare_with_without,
    proxy_correlation,
)

# Large enough that sampling noise is smaller than the effects being tested. At
# n=10,000 the standard error on the gap coefficient is about 0.005 in log
# points, so a 0.01 tolerance is roughly two standard errors: tight enough to
# fail a broken audit, loose enough not to fail on luck.
N = 10_000
TOLERANCE = 0.01

#: Everything we accept as a legitimate reason to be paid differently, PLUS the
#: career-break column. Including that one is a choice, not a default — see
#: `test_the_proxy_inflates_the_gap_when_left_out` for what changes without it.
ALL_CONTROLS = [*LEGITIMATE_CONTROLS, *CONTESTED_CONTROLS]


def gap_of(df: pd.DataFrame, truth, *, controls: list[str] | None = None) -> float:
    """Run the audit on a generated frame and return the adjusted gap."""
    return adjusted_gap(
        df[TARGET],
        df["gender"],
        df[controls if controls is not None else ALL_CONTROLS],
        # Read the answer sheet instead of letting the audit auto-detect which
        # group is worse off. Auto-detection picks the lower-paid group, which
        # would make the gap positive by construction — and would turn the
        # zero-gap control test into a test that cannot fail.
        disadvantaged=truth.disadvantaged_group,
    ).gap


@pytest.fixture(scope="module")
def biased() -> tuple[pd.DataFrame, object]:
    """The default world: an 8% gap, plus a proxy that partly encodes it."""
    return generate(N, seed=42)


@pytest.fixture(scope="module")
def fair() -> tuple[pd.DataFrame, object]:
    """The control world: no direct pay penalty for the group at all."""
    return generate(N, seed=42, pay_gap=0.0)


# ═══════════════════════════════════════════ THE HEADLINE: does the audit read true?


@pytest.mark.parametrize("injected", [0.0, 0.03, 0.08, 0.15, 0.25])
@pytest.mark.parametrize("seed", [42, 7])
def test_recovers_the_injected_gap(injected: float, seed: int):
    """Inject a gap we chose; the audit must measure it back.

    This is the test that licenses every other fairness number in the project.
    Ten combinations of size and seed, because an audit that only works at 8%
    on seed 42 is a coincidence dressed as a measurement.
    """
    df, truth = generate(N, seed=seed, pay_gap=injected)
    assert gap_of(df, truth) == pytest.approx(injected, abs=TOLERANCE)


def test_zero_gap_stays_zero(fair):
    """The control case, and the one that matters most for not doing harm.

    A bias detector that finds bias in fair data is worse than no detector: it
    is confidently wrong in the direction that gets someone accused.

    Note what "fair" does and does not mean here. `pay_gap=0.0` means two people
    alike in every recorded respect are paid the same. The two groups still do
    *not* earn the same on average, because one takes more career breaks and
    breaks cost money. Separating "explained" from "unexplained" is the entire
    job of the audit, and this test is where that separation is checked.
    """
    df, truth = fair
    assert truth.pay_gap == 0.0
    assert gap_of(df, truth) == pytest.approx(0.0, abs=TOLERANCE)


def test_the_confidence_interval_contains_the_truth():
    """A 95% interval should contain the true value about 95% of the time.

    Ten seeds is far too few to measure that rate properly, but it is plenty to
    catch an interval that is wrong by a factor — a missing square root, a
    standard error computed on the wrong degrees of freedom. Those bugs fail
    everywhere, not occasionally.
    """
    misses = 0
    for seed in range(10):
        df, truth = generate(3_000, seed=seed, pay_gap=0.08)
        measured = adjusted_gap(
            df[TARGET], df["gender"], df[ALL_CONTROLS], disadvantaged=truth.disadvantaged_group
        )
        misses += not (measured.ci_low <= truth.pay_gap <= measured.ci_high)
    assert misses <= 2


def test_validate_on_synthetic_reports_a_clean_calibration_table():
    """The shipped self-check must pass on the values it ships with.

    `validate_on_synthetic` is what a reader runs first. If it were allowed to
    report a failure, nobody downstream would ever be told.
    """
    table = validate_on_synthetic(n=N, seed=42)

    assert list(table["injected"]) == [0.0, 0.03, 0.08, 0.15, 0.25]
    assert np.allclose(table["recovered"], table["injected"], atol=TOLERANCE)
    assert table["truth_inside_ci"].all()
    # The errors should look like noise, not like a systematic offset that
    # happens to be small. In practice the audit reads about 0.2-0.3 percentage
    # points low across the board, so this bound is deliberately generous — but
    # it would still catch a drift into "always overstates the gap".
    assert table["error"].abs().max() < 0.005


# ═══════════════════════════════════════════ raw vs adjusted


def experience_gap_data(n: int = 4_000, seed: int = 0) -> pd.DataFrame:
    """A hand-built world with NO unfair pay and a large raw gap.

    Everyone is paid by exactly the same rule — ₹6L times a clean experience
    curve — but one group averages four years of experience and the other nine.
    A fair employer, and a big difference in average pay.

    The two ranges deliberately *overlap*. If they did not, there would be no
    pair of people alike in experience but different in group, nothing to
    compare like for like, and `adjusted_gap` would (correctly) refuse to
    answer — see `test_the_audit_refuses_when_the_groups_never_overlap`.

    Built by hand rather than taken from the generator because the generator
    deliberately gives both groups the same experience distribution. To show
    that the raw gap and the adjusted gap can disagree completely, we need a
    world where they do.
    """
    rng = np.random.default_rng(seed)
    group = np.where(rng.random(n) < 0.5, "female", "male")
    years = np.where(group == "female", rng.integers(0, 9, n), rng.integers(4, 14, n))
    salary = 600_000 * (1 + years / 3) ** 0.85 * np.exp(rng.normal(0, 0.15, n))
    return pd.DataFrame({TARGET: salary, "gender": group, "years_experience": years})


def test_raw_and_adjusted_gaps_disagree_when_the_groups_differ():
    """The single most important distinction in the module, as an assertion.

    The raw gap is enormous — well over 30% — because one group has three times
    the experience of the other. The adjusted gap is zero, because at equal
    experience everybody here is paid identically. Reporting the raw number as
    evidence of discrimination would be a false accusation about a demonstrably
    fair employer.
    """
    df = experience_gap_data()

    raw = raw_gap(df[TARGET], df["gender"], disadvantaged="female")
    adjusted = adjusted_gap(
        df[TARGET], df["gender"], df[["years_experience"]], disadvantaged="female"
    )

    assert raw.mean_gap > 0.25
    assert adjusted.gap == pytest.approx(0.0, abs=0.02)
    assert not adjusted.is_significant
    assert raw.mean_gap - adjusted.gap > 0.25  # all of it explained by experience


def test_the_raw_gap_reports_both_mean_and_median(biased):
    """Two averages, because the mean of a skewed distribution misleads.

    Salaries have a long right tail. A handful of very large ones drag the mean
    around, and if they are unevenly spread across groups the mean gap moves
    with them. The median is unmoved. Report both; if they disagree sharply,
    that disagreement is itself the story.
    """
    df, truth = biased
    raw = raw_gap(df[TARGET], df["gender"], disadvantaged=truth.disadvantaged_group)

    assert raw.n_disadvantaged + raw.n_reference == N
    assert raw.mean_gap > 0.05
    assert raw.median_gap > 0.05
    assert abs(raw.mean_gap - raw.median_gap) < 0.05


def test_the_raw_gap_is_bigger_than_the_adjusted_gap(biased):
    """The controls should explain *some* of the difference, never all of it.

    Here the extra is career breaks: one group takes more of them, breaks cost
    money, and that part of the difference has a recorded explanation. What
    survives the controls is the injected unfairness.
    """
    df, truth = biased
    raw = raw_gap(df[TARGET], df["gender"], disadvantaged=truth.disadvantaged_group)
    assert raw.mean_gap > gap_of(df, truth) + 0.01


# ═══════════════════════════════════════════ confidence intervals


def test_intervals_widen_on_smaller_samples():
    """Less data, less certainty — and the audit has to say so out loud.

    A point estimate hides sample size completely: "6.2%" reads the same off 200
    people as off 20,000. The interval is the only part of the output that
    refuses to. It should shrink roughly like 1/sqrt(n), so quadrupling the
    sample should roughly halve the width.
    """
    widths = []
    for n in (400, 1_600, 6_400):
        df, truth = generate(n, seed=11, pay_gap=0.08)
        measured = adjusted_gap(
            df[TARGET], df["gender"], df[ALL_CONTROLS], disadvantaged=truth.disadvantaged_group
        )
        widths.append(measured.ci_width)

    assert widths[0] > widths[1] > widths[2]
    # Each step quadruples n, so each width should fall by roughly half. Loose
    # bounds: this is checking the shape of the relationship, not its constant.
    assert 1.4 < widths[0] / widths[1] < 3.0
    assert 1.4 < widths[1] / widths[2] < 3.0


def test_a_small_sample_cannot_pin_down_a_real_gap():
    """On 150 people, a genuine 8% gap reads anywhere between about 0% and 25%.

    Same world, same injected gap, ten different samples of 150 employees. The
    point estimates swing across a range wider than the gap itself, and some
    samples cannot even rule out zero.

    This is the honest, uncomfortable result, and it is why the interval is not
    optional here. A department of 150 people simply cannot answer this
    question, and the right conclusion is "we cannot tell" — a different
    sentence from "there is no gap". A tool printing only a point estimate lets
    a reader confuse the two, and that confusion runs in whichever direction
    the reader already preferred.
    """
    estimates, unconfirmed = [], 0
    for seed in range(10):
        df, truth = generate(150, seed=seed, pay_gap=0.08)
        measured = adjusted_gap(
            df[TARGET],
            df["gender"],
            # Fewer controls: 150 rows cannot support the full set.
            df[["years_experience", "role", "location_tier"]],
            disadvantaged=truth.disadvantaged_group,
        )
        estimates.append(measured.gap)
        unconfirmed += not measured.is_significant
        assert measured.ci_width > 0.10

    assert max(estimates) - min(estimates) > 0.10
    assert unconfirmed >= 5


def test_a_wider_confidence_level_gives_a_wider_interval(biased):
    df, truth = biased
    narrow = adjusted_gap(
        df[TARGET],
        df["gender"],
        df[ALL_CONTROLS],
        disadvantaged=truth.disadvantaged_group,
        confidence=0.80,
    )
    wide = adjusted_gap(
        df[TARGET],
        df["gender"],
        df[ALL_CONTROLS],
        disadvantaged=truth.disadvantaged_group,
        confidence=0.99,
    )
    assert narrow.gap == pytest.approx(wide.gap)
    assert wide.ci_width > narrow.ci_width


# ═══════════════════════════════════════════ the model's own bias


def test_residuals_are_balanced_when_the_model_is_even_handed(biased):
    """A model that misses randomly must not be reported as biased.

    The false-positive case. Noise is not bias, and a residual test that cannot
    tell them apart is useless — every model has residuals.
    """
    df, _ = biased
    rng = np.random.default_rng(0)
    salary = df[TARGET].to_numpy(float)
    predicted = salary * np.exp(rng.normal(0, 0.2, len(df)))

    check = residual_analysis(salary, predicted, df["gender"], disadvantaged="female")
    assert abs(check.difference) < 0.02
    assert not check.is_significant


def test_residuals_catch_a_model_that_lowballs_one_group(biased):
    """The true-positive case: a model deliberately built to under-predict.

    We take the actual salaries and shave 10% off the predictions for one group
    only. The residual check must find it, get the size roughly right, and get
    the *direction* right — a bias detector that reports the wrong group is
    worse than no detector.
    """
    df, _ = biased
    salary = df[TARGET].to_numpy(float)
    rng = np.random.default_rng(1)
    noise = np.exp(rng.normal(0, 0.15, len(df)))
    predicted = salary * noise * np.where(df["gender"] == "female", 0.90, 1.0)

    check = residual_analysis(salary, predicted, df["gender"], disadvantaged="female")

    # actual/predicted ≈ 1/0.90 → the model reads about 11% low for that group.
    assert check.difference == pytest.approx(1 / 0.9 - 1, abs=0.03)
    assert check.is_significant
    assert check.disadvantaged.group == "female"
    assert check.disadvantaged.n + check.reference.n == N


def test_a_model_that_learned_the_bias_perfectly_passes_the_residual_test(biased):
    """The trap this test exists to document, stated as code.

    A model trained faithfully on unfair data predicts the unfair salaries
    accurately. Its residuals are beautifully balanced across groups — and it
    still recommends 8% less for one of them, every single time.

    So a clean residual check is NOT a clean bill of health. It is why
    `adjusted_gap` and `residual_analysis` are separate functions and why the
    report prints both.
    """
    df, truth = biased
    salary = df[TARGET].to_numpy(float)
    rng = np.random.default_rng(2)
    # A "perfect" model: it reproduces the salaries, unfairness included.
    predicted = salary * np.exp(rng.normal(0, 0.05, len(df)))

    check = residual_analysis(salary, predicted, df["gender"], disadvantaged="female")
    assert not check.is_significant  # the model looks blameless...

    # ...and the gap it will hand every future candidate is still there.
    assert gap_of(df, truth) == pytest.approx(truth.pay_gap, abs=TOLERANCE)


# ═══════════════════════════════════════════ the proxy problem


def test_the_proxy_scan_finds_the_planted_proxy(biased):
    """`career_gap_months` must come out on top, and the honest columns must not.

    The generator plants exactly one proxy. If the scan cannot find a proxy we
    put there on purpose, it will not find the ones a real dataset hides.
    """
    df, _ = biased
    scan = proxy_correlation(df, "gender", [*LEGITIMATE_CONTROLS, *CONTESTED_CONTROLS])

    assert scan.table.iloc[0]["feature"] == "career_gap_months"
    assert scan.table.iloc[0]["association"] > 0.25
    assert scan.leaking() == ["career_gap_months"]
    # Every legitimate control is independent of gender in this world, so none
    # of them should register.
    assert (scan.table.iloc[1:]["association"] < 0.05).all()


def test_the_proxy_inflates_the_gap_when_it_is_left_out(biased):
    """Stop controlling for career breaks and the measured gap grows.

    That extra few points is the share of the unfairness travelling through the
    career-break channel rather than through gender directly. It is the same
    few points a model can still reach after the gender column is deleted,
    which is exactly what makes the next test work.

    Neither number is wrong. They answer different questions, and which one you
    quote is a judgement about whether time out of the workforce is a legitimate
    reason to pay less. State the choice; do not bury it.
    """
    df, truth = biased
    controlled = gap_of(df, truth)
    uncontrolled = gap_of(df, truth, controls=list(LEGITIMATE_CONTROLS))

    assert controlled == pytest.approx(truth.pay_gap, abs=TOLERANCE)
    assert uncontrolled > controlled + 0.02


def test_the_gap_survives_deleting_the_protected_column(biased):
    """**The proxy demonstration.** Remove gender; the disadvantage remains.

    Two models, identical but for one column. The one that never saw gender
    still predicts systematically less for that group, because it rebuilt the
    pattern out of career breaks. "We removed the gender column, so the model
    is fair" is not a cautious claim; it is a false one.
    """
    df, truth = biased
    result = compare_with_without(df, "gender", disadvantaged=truth.disadvantaged_group)

    assert result.gap_with_protected > 0.09
    # The headline. Not "smaller" — SURVIVING. A third or more of the gap comes
    # straight back through a column nobody would think to question.
    assert result.gap_without_protected > 0.03
    assert result.surviving_share > 0.30


def test_deleting_the_protected_column_costs_almost_no_accuracy(biased):
    """And the removal bought nothing, which makes it worse rather than better.

    If blinding the model wrecked its accuracy, at least there would be a
    trade-off to argue about. There isn't one: the proxies carry the same
    information, so the model barely notices — it keeps the predictive power
    *and* the disadvantage.
    """
    df, truth = biased
    result = compare_with_without(df, "gender", disadvantaged=truth.disadvantaged_group)
    assert result.accuracy_cost < 0.05


def test_the_proxy_result_is_not_an_artefact_of_a_linear_model(biased):
    """Same demonstration, gradient boosting. Same conclusion.

    Worth spending the seconds on. A reader can reasonably suspect that a result
    from a linear model is an artefact of linear algebra — that some clever
    model would not do this. Gradient boosting is what this project actually
    ships, and it reconstructs the gap just as happily, because the information
    is in the data and not in the algorithm.
    """
    lightgbm = pytest.importorskip("lightgbm")
    df, truth = biased

    result = compare_with_without(
        df,
        "gender",
        disadvantaged=truth.disadvantaged_group,
        model_factory=lambda: lightgbm.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, verbose=-1, random_state=0
        ),
    )
    assert result.gap_with_protected > 0.08
    assert result.gap_without_protected > 0.03


def test_switching_the_proxy_off_removes_the_survival():
    """The negative control for the proxy demonstration.

    Give both groups the same chance of a career break and the channel is gone.
    A model with no gender column then has nothing to rebuild the gap from, and
    the surviving gap must collapse to roughly nothing.

    Without this test, `test_the_gap_survives_deleting_the_protected_column`
    could be passing for some unrelated reason and we would never know.
    """
    df, truth = generate(N, seed=4, pay_gap=0.08, career_gap_prob_disadvantaged=0.06)
    assert not truth.has_proxy

    result = compare_with_without(df, "gender", disadvantaged=truth.disadvantaged_group)
    assert result.gap_with_protected > 0.06
    assert abs(result.gap_without_protected) < 0.02


# ═══════════════════════════════════════════ the report


def test_the_report_says_what_a_reader_needs_to_know(biased):
    """The printed report must carry its own caveats.

    A fairness number that travels without them gets repeated without them, and
    the raw gap in particular is quoted as proof of discrimination constantly.
    So the string itself has to push back — which makes it testable.
    """
    df, truth = biased
    predicted = df[TARGET].to_numpy(float) * 1.01

    report = audit(
        df[TARGET],
        df["gender"],
        df[ALL_CONTROLS],
        y_pred=predicted,
        disadvantaged=truth.disadvantaged_group,
        notes=["Career breaks were treated as a legitimate control."],
    )
    text = str(report)

    assert isinstance(report, FairnessReport)
    assert "HEADLINE PAY GAP" in text and "LIKE-FOR-LIKE" in text
    assert "proves nothing" in text  # the raw-gap caveat
    assert "not a finding of" in text  # the causation caveat
    assert "years_experience" in text  # the control list, stated out loud
    assert "Career breaks were treated as a legitimate control." in text
    assert "THE MODEL'S OWN ERRORS" in text
    assert report.explained_by_controls > 0
    assert max(len(line) for line in text.splitlines()) <= 100


def test_the_report_omits_the_model_section_without_predictions(biased):
    """No predictions, no claims about a model. Silence beats a placeholder."""
    df, truth = biased
    report = audit(
        df[TARGET], df["gender"], df[ALL_CONTROLS], disadvantaged=truth.disadvantaged_group
    )
    assert report.residual is None
    assert "THE MODEL'S OWN ERRORS" not in str(report)


# ═══════════════════════════════════════════ guard rails


def test_the_audit_refuses_to_control_for_the_protected_attribute(biased):
    """Controlling for gender while measuring the gender gap measures nothing.

    It sounds absurd written down, and it is one column-name typo away in any
    pipeline that builds its control list programmatically. The failure is
    silent otherwise: the audit would report a clean zero.
    """
    df, _ = biased
    controls = df[["years_experience", "role"]].copy()
    controls["gender_copy"] = df["gender"]

    with pytest.raises(ValueError, match="identical to the group labels"):
        adjusted_gap(df[TARGET], df["gender"], controls, disadvantaged="female")


def test_the_audit_refuses_when_the_groups_never_overlap():
    """No comparable pairs, no answer. The audit must say so, not invent one.

    Here one group has 0–5 years of experience and the other 6–12. There is not
    a single pair of people with the same experience and different genders, so
    "same experience, different pay" is a question with no examples in the data.

    Left unchecked this is a silent, dangerous bug: the linear algebra still
    returns a number — it picks one of infinitely many equally good answers —
    and on this data that number came out at −76%, which someone would have
    reported.
    """
    rng = np.random.default_rng(0)
    group = np.where(rng.random(2_000) < 0.5, "female", "male")
    years = np.where(group == "female", rng.integers(0, 6, 2_000), rng.integers(6, 13, 2_000))
    salary = 600_000 * (1 + years / 3) ** 0.85

    with pytest.raises(ValueError, match="already determine group membership"):
        adjusted_gap(
            salary,
            group,
            pd.DataFrame({"years_experience": years}),
            disadvantaged="female",
        )


def test_missing_control_values_are_refused_rather_than_guessed(biased):
    """A blank is not a category, and silently dropping rows changes the answer."""
    df, _ = biased
    controls = df[["years_experience", "role"]].copy()
    controls.loc[0:10, "role"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        adjusted_gap(df[TARGET], df["gender"], controls, disadvantaged="female")


def test_auto_detection_picks_the_lower_paid_group(biased):
    """Convenience, with a documented catch.

    Auto-detection makes the reported gap positive whatever the data says, so
    it can tell you a gap's *size* but never that a gap *exists*. It is here
    because naming the group on every call is friction; the docstring says so
    loudly and this test pins the behaviour down.
    """
    df, _ = biased
    detected = raw_gap(df[TARGET], df["gender"])
    assert detected.disadvantaged == "female"
    assert detected.mean_gap > 0

    # Even in a world with no injected gap it still returns a positive number,
    # which is precisely why the validation code names the group explicitly.
    fair_df, _ = generate(2_000, seed=8, pay_gap=0.0)
    assert raw_gap(fair_df[TARGET], fair_df["gender"]).mean_gap > 0


def test_more_than_two_groups_must_be_named(biased):
    """Three groups is an ambiguous request, not a computable one."""
    df, _ = biased
    three = df["gender"].to_numpy(object).copy()
    three[:100] = "non-binary"

    with pytest.raises(ValueError, match="compares exactly two"):
        raw_gap(df[TARGET], three)

    # Named explicitly, it works — and quietly ignores the third group rather
    # than folding it into one of the two.
    measured = adjusted_gap(
        df[TARGET], three, df[ALL_CONTROLS], disadvantaged="female", reference="male"
    )
    assert measured.n == N - 100


def test_bad_arguments_fail_loudly():
    salary = np.array([100_000.0, 200_000.0, 300_000.0, 400_000.0])
    group = np.array(["f", "m", "f", "m"])
    controls = pd.DataFrame({"years_experience": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="positive"):
        raw_gap(np.array([0.0, 1.0, 2.0, 3.0]), group)
    with pytest.raises(ValueError, match="shape"):
        raw_gap(salary, np.array(["f", "m"]))
    with pytest.raises(ValueError, match="confidence"):
        adjusted_gap(salary, group, controls, confidence=1.5)
    with pytest.raises(ValueError, match="rows"):
        adjusted_gap(salary, group, controls.iloc[:2])
    with pytest.raises(ValueError, match="degrees of freedom"):
        # Four rows, four independent columns: the fit passes exactly through
        # every point, leaves no residual to measure noise with, and would
        # report a confidence interval of width zero around a made-up number.
        adjusted_gap(salary, group, pd.DataFrame({"role": ["a", "b", "c", "a"]}))


def test_the_least_squares_default_refuses_to_predict_before_fitting():
    with pytest.raises(RuntimeError, match="fit the model"):
        LeastSquares().predict(np.ones((2, 2)))
