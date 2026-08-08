"""Tests for the increment and pay-equity rules.

There is no model here, so there is no "close enough". Every number below is
hand-computed from the policy file, in the same way an HR partner would check
it with a calculator — which is the whole claim this layer makes about itself.

The tests that matter most are the ones about *ordering*: a below-band employee
must beat an identical above-band one, equity must survive a cap that merit
does not, and no squeeze of the budget may ever produce a negative increment.
Those are the properties a compensation committee would actually ask about.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from paybands.policy import (
    IncrementRules,
    apply_budget_constraint,
    band_label,
    equity_review,
    recommend,
    recommend_batch,
)

RULES_PATH = "configs/policy/increment_fy_2026_27.yaml"

MIDPOINT = 20_00_000  # a round band midpoint, so every figure below is checkable


@pytest.fixture
def rules() -> IncrementRules:
    return IncrementRules.from_yaml(RULES_PATH)


# ─────────────────────────────────────────── the config loads and validates


def test_config_loads_with_provenance(rules):
    assert rules.financial_year == "2026-27"
    assert rules.merit.by_rating[3] == 0.08
    assert rules.compa.below_band == 0.90
    assert rules.compa.above_band == 1.10
    assert rules.source, "a policy with no stated source is a policy nobody can defend"


def test_compa_thresholds_must_be_ordered(rules):
    """below_band < above_band <= hold_above. A config that gets this backwards
    would silently label everyone 'above band' — better to refuse to load."""
    payload = {**rules.model_dump(), "compa": {**rules.compa.model_dump(), "below_band": 1.5}}
    with pytest.raises(ValueError, match="below_band < above_band"):
        IncrementRules.model_validate(payload)


def test_equity_target_must_reach_the_band(rules):
    """Correcting people to a target that still counts as underpaid is busywork:
    they stay on the list forever, at rising cost."""
    payload = {**rules.model_dump(), "equity": {**rules.equity.model_dump()}}
    payload["equity"]["target_compa_ratio"] = 0.80
    with pytest.raises(ValueError, match="still counts as underpaid"):
        IncrementRules.model_validate(payload)


def test_unknown_config_key_is_rejected(rules):
    """extra='forbid'. A typo in a policy file must be an error at load time,
    not a line that is silently ignored and costs somebody their raise."""
    with pytest.raises(ValueError):
        IncrementRules.model_validate({**rules.model_dump(), "budgett": {"pct_of_payroll": 0.1}})


# ─────────────────────────────────────────── compa-ratio, the central number


def test_compa_ratio_is_salary_over_midpoint(rules):
    rec = recommend(16_00_000, MIDPOINT, 3, rules)
    assert rec.compa_ratio_before == pytest.approx(0.80)
    # After = new salary over the SAME midpoint. The band did not move; the
    # person did.
    assert rec.compa_ratio_after == pytest.approx(rec.new_salary / MIDPOINT)
    assert rec.compa_ratio_after > rec.compa_ratio_before


def test_band_labels(rules):
    assert band_label(0.85, rules) == "below band"
    assert band_label(1.00, rules) == "in band"
    assert band_label(1.20, rules) == "above band"


@pytest.mark.parametrize(
    ("compa", "expected"),
    [(0.90, "in band"), (1.10, "in band")],  # exactly ON a threshold is inside it
)
def test_thresholds_are_inclusive_of_the_band(rules, compa, expected):
    """A boundary has to fall on one side or the other, and the choice must be
    deliberate: at exactly 0.90 you are in band, so no equity correction."""
    rec = recommend(compa * MIDPOINT, MIDPOINT, 3, rules)
    assert rec.band_label == expected
    assert rec.equity_amount == 0


# ─────────────────────────────────────────── merit and equity combine


def test_below_band_beats_above_band_at_the_same_rating(rules):
    """The headline property. Two people, same rating, same band — the one paid
    below it must get more, in both rupees and percent."""
    low = recommend(0.80 * MIDPOINT, MIDPOINT, 4, rules)
    high = recommend(1.20 * MIDPOINT, MIDPOINT, 4, rules)

    assert low.recommended_pct > high.recommended_pct
    assert low.recommended_amount > high.recommended_amount
    assert low.equity_amount > 0
    assert high.equity_amount == 0


def test_merit_rises_with_rating(rules):
    """Same compa-ratio, different ratings — money must follow the rating, or
    the rating table is theatre."""
    amounts = [recommend(MIDPOINT, MIDPOINT, r, rules).recommended_amount for r in (1, 2, 3, 4, 5)]
    assert amounts == sorted(amounts)
    assert amounts[0] == 0  # a rating of 1 earns no merit, by design


def test_equity_closes_the_gap_in_slices(rules):
    """At 0.80 of a ₹20L band: salary ₹16L, target 0.95 × ₹20L = ₹19L, so a ₹3L
    gap = 18.75% of salary. Over 2 years that is 9.375% this year."""
    rec = recommend(16_00_000, MIDPOINT, 3, rules)
    assert rec.equity_amount / rec.current_salary == pytest.approx(0.09375, abs=1e-4)
    assert rec.merit_amount == pytest.approx(0.08 * 16_00_000, abs=100)


def test_single_year_correction_cap_binds(rules):
    """At 0.50 of band the arithmetic wants (0.95/0.50 − 1) / 2 = 45% this year.
    Nobody approves that in one line, so the 15% ceiling holds instead."""
    rec = recommend(0.50 * MIDPOINT, MIDPOINT, 1, rules)  # rating 1 → merit is 0
    assert rec.capped_by_correction_limit
    assert rec.equity_amount / rec.current_salary == pytest.approx(0.15)
    assert rec.compa_ratio_after < rules.equity.target_compa_ratio  # gap remains, honestly


def test_total_cap_trims_merit_and_protects_equity(rules):
    """5 rating (18%) + capped equity (15%) + promotion (12%) = 45%, over the
    35% ceiling. Equity must survive intact; merit is what gives way."""
    rec = recommend(0.50 * MIDPOINT, MIDPOINT, 5, rules, is_promotion=True)

    assert rec.capped_by_total_limit
    assert rec.recommended_pct == pytest.approx(0.35, abs=0.001)
    assert rec.equity_amount / rec.current_salary == pytest.approx(0.15)
    assert rec.promotion_amount / rec.current_salary == pytest.approx(0.12)
    assert rec.merit_amount / rec.current_salary == pytest.approx(0.08, abs=0.001)


def test_above_band_merit_is_damped_not_zeroed(rules):
    """A strong performer high in the band has still done the work. Half, not
    nothing — zeroing them is how you lose them."""
    inside = recommend(1.00 * MIDPOINT, MIDPOINT, 5, rules)
    above = recommend(1.20 * MIDPOINT, MIDPOINT, 5, rules)

    assert above.merit_damped
    assert above.recommended_pct == pytest.approx(0.09)  # 18% × 0.5
    assert 0 < above.recommended_pct < inside.recommended_pct


def test_well_above_band_is_held(rules):
    rec = recommend(1.40 * MIDPOINT, MIDPOINT, 5, rules)
    assert rec.held
    assert rec.recommended_amount == 0
    assert rec.compa_ratio_after == rec.compa_ratio_before


def test_promotion_adds_on_top(rules):
    plain = recommend(MIDPOINT, MIDPOINT, 3, rules)
    promoted = recommend(MIDPOINT, MIDPOINT, 3, rules, is_promotion=True)
    assert promoted.recommended_pct - plain.recommended_pct == pytest.approx(0.12)


# ─────────────────────────────────────────── the reason string


def test_reason_names_every_driver(rules):
    """The output a recruiter reads. If a driver moved the number, the sentence
    has to say so — otherwise nobody can disagree with it."""
    text = recommend(16_00_000, MIDPOINT, 4, rules, is_promotion=True).reason

    assert "merit" in text
    assert "rating of 4" in text
    assert "equity correction" in text
    assert "promotion uplift" in text
    assert "0.80 of band midpoint" in text  # the compa-ratio, in words
    assert "below band" in text
    assert "₹" in text and "%" in text  # the amount and the percentage


def test_reason_flags_an_assumed_rating(rules):
    text = recommend(MIDPOINT, MIDPOINT, None, rules).reason
    assert "assumed" in text
    assert "no rating on file" in text


def test_reason_explains_a_hold(rules):
    text = recommend(1.40 * MIDPOINT, MIDPOINT, 5, rules).reason
    assert "no increase this year" in text
    assert "held" in text


def test_reason_rewrites_itself_when_the_budget_bites(rules):
    """The reason is a property of the components, not a stored string — so it
    cannot drift out of step with a number that was trimmed later."""
    recs = [recommend(MIDPOINT, MIDPOINT, 5, rules)]
    before = recs[0].reason
    after = apply_budget_constraint(recs, 1_00_000).recommendations[0]

    assert "Trimmed to fit the increment budget" not in before
    assert "Trimmed to fit the increment budget" in after.reason
    assert "18%" in before and "18%" not in after.reason


# ─────────────────────────────────────────── edge cases


@pytest.mark.parametrize("bad", [0, -1, -5_00_000])
def test_non_positive_salary_rejected(rules, bad):
    with pytest.raises(ValueError, match="must be positive"):
        recommend(bad, MIDPOINT, 3, rules)


def test_non_positive_midpoint_rejected(rules):
    with pytest.raises(ValueError, match="midpoint must be positive"):
        recommend(10_00_000, 0, 3, rules)


def test_missing_rating_assumes_the_default(rules):
    rec = recommend(MIDPOINT, MIDPOINT, None, rules)
    assert rec.rating is None
    assert rec.rating_used == rules.merit.default_rating
    assert rec.rating_assumed
    assert rec.recommended_pct == pytest.approx(0.08)


def test_missing_rating_can_be_configured_to_pay_no_merit(rules):
    strict = rules.model_copy(
        update={"merit": rules.merit.model_copy(update={"on_missing_rating": "no_merit"})}
    )
    # No merit — but the equity correction still applies. Being underpaid is a
    # fact about the band, not about whether a manager submitted a form.
    rec = recommend(16_00_000, MIDPOINT, None, strict)
    assert rec.merit_amount == 0
    assert rec.equity_amount > 0


def test_unknown_rating_is_an_error_not_a_guess(rules):
    """A rating that is present but off the scale is a data bug. Guessing at it
    would silently pay someone the wrong amount."""
    with pytest.raises(ValueError, match="not in the policy's merit table"):
        recommend(MIDPOINT, MIDPOINT, 7, rules)


def test_rounding_is_always_down(rules):
    """Rounding up can push a set of recommendations past a budget they had just
    fitted inside."""
    # In band (compa 1.00), so merit is the only component. 4% of ₹10,10,101 is
    # ₹40,404.04, which must land on ₹40,400 and never ₹40,500.
    rec = recommend(10_10_101, 10_10_101, 2, rules)
    assert rec.merit_amount == 40_400
    assert rec.recommended_amount <= 0.04 * 10_10_101


def test_band_object_is_accepted(rules):
    """A `BandPrediction` carries arrays; only the midpoint is used here."""
    band = SimpleNamespace(median=[float(MIDPOINT)])
    assert recommend(MIDPOINT, band, 3, rules).band_midpoint == MIDPOINT

    with pytest.raises(ValueError, match="one employee at a time"):
        recommend(MIDPOINT, SimpleNamespace(median=[1.0, 2.0]), 3, rules)


# ─────────────────────────────────────────── pay equity review


@pytest.fixture
def population() -> pd.DataFrame:
    """Five people against a ₹20L band. Compa-ratios: 0.60, 0.85, 0.90, 1.00, 1.30."""
    return pd.DataFrame(
        {
            "employee": ["kiran", "meera", "arjun", "divya", "rohit"],
            "salary_annual": [12_00_000, 17_00_000, 18_00_000, 20_00_000, 26_00_000],
            "performance_rating": [3, 4, 3, 5, 3],
            "gender": ["F", "F", "M", "F", "M"],
        }
    )


@pytest.fixture
def midpoints() -> list[float]:
    return [MIDPOINT] * 5


def test_equity_review_lists_only_the_underpaid(rules, population, midpoints):
    review = equity_review(population, midpoints, rules, id_col="employee")

    assert review.n_reviewed == 5
    assert review.n_below == 2  # 0.60 and 0.85; 0.90 is exactly in band
    assert list(review.frame["employee_id"]) == ["kiran", "meera"]  # worst first
    assert review.share_below == pytest.approx(0.4)
    assert review.worst_compa_ratio == pytest.approx(0.60)


def test_equity_review_totals_are_correct(rules, population, midpoints):
    """Hand-computed against a target of 0.95 × ₹20L = ₹19L.

        kiran  ₹12L → ₹19L = ₹7,00,000 to fix
        meera  ₹17L → ₹19L = ₹2,00,000 to fix
                              ─────────
                              ₹9,00,000

    This year: kiran's gap is 58.3% of salary, halved to 29.2%, capped at 15%
    → ₹1,80,000. Meera's is 11.8%, halved to 5.88% → ₹1,00,000.
    """
    review = equity_review(population, midpoints, rules, id_col="employee")

    assert review.total_cost_to_target == pytest.approx(9_00_000)
    assert review.frame.loc[0, "cost_to_target"] == pytest.approx(7_00_000)
    assert review.frame.loc[1, "cost_to_target"] == pytest.approx(2_00_000)

    assert review.frame.loc[0, "cost_this_year"] == pytest.approx(1_80_000)
    assert review.frame.loc[1, "cost_this_year"] == pytest.approx(1_00_000)
    assert review.total_cost_this_year == pytest.approx(2_80_000)

    # And the shortfall to the *threshold* is a smaller number than the cost to
    # the target — they answer different questions, so they are separate columns.
    assert review.frame.loc[0, "shortfall_to_band"] == pytest.approx(6_00_000)


def test_equity_review_agrees_with_recommend(rules, population, midpoints):
    """The review's this-year cost must equal the equity component `recommend`
    would produce. Two numbers that are meant to be the same number should be
    computed the same way, or they will disagree in front of a customer."""
    review = equity_review(population, midpoints, rules, id_col="employee")
    recs = {
        r.employee_id: r for r in recommend_batch(population, midpoints, rules, id_col="employee")
    }
    for _, row in review.frame.iterrows():
        assert recs[row["employee_id"]].equity_amount == pytest.approx(row["cost_this_year"])


def test_equity_review_carries_extra_columns(rules, population, midpoints):
    """ "Who is underpaid" is always followed by "is it the same group every
    time" — so the audit columns have to survive into the output."""
    review = equity_review(population, midpoints, rules, id_col="employee", keep_cols=["gender"])
    assert list(review.frame["gender"]) == ["F", "F"]


def test_equity_review_with_nobody_underpaid(rules, population):
    review = equity_review(population, [10_00_000] * 5, rules, id_col="employee")
    assert review.n_below == 0
    assert review.total_cost_to_target == 0
    assert "Nothing to correct" in review.summary()


def test_equity_review_summary_is_readable(rules, population, midpoints):
    text = equity_review(population, midpoints, rules, id_col="employee").summary()
    assert "0.90" in text and "cost this year" in text


def test_equity_review_rejects_a_length_mismatch(rules, population):
    with pytest.raises(ValueError, match="midpoints for"):
        equity_review(population, [MIDPOINT] * 3, rules)


# ─────────────────────────────────────────── budget constraint


@pytest.fixture
def recs(rules, population, midpoints):
    return recommend_batch(population, midpoints, rules, id_col="employee")


def test_ample_budget_changes_nothing(rules, recs):
    total = sum(r.recommended_amount for r in recs)
    allocation = apply_budget_constraint(recs, total * 2)

    assert allocation.allocated == pytest.approx(total)
    assert allocation.shortfall == 0
    assert allocation.n_reduced == 0


def test_budget_constraint_never_exceeds_the_pool(rules, recs):
    total = sum(r.recommended_amount for r in recs)
    for fraction in (0.0, 0.1, 0.5, 0.9, 1.0):
        allocation = apply_budget_constraint(recs, total * fraction)
        assert allocation.allocated <= total * fraction + 1e-6
        assert allocation.allocated <= total


def test_budget_constraint_never_produces_a_negative(rules, recs):
    for fraction in (0.0, 0.05, 0.3, 0.75):
        allocation = apply_budget_constraint(
            recs, sum(r.recommended_amount for r in recs) * fraction
        )
        for rec in allocation.recommendations:
            assert rec.merit_amount >= 0
            assert rec.equity_amount >= 0
            assert rec.promotion_amount >= 0
            assert rec.recommended_amount >= 0
            assert rec.new_salary >= rec.current_salary  # nobody's pay ever goes down


def test_equity_first_funds_corrections_before_merit(rules, recs):
    """The documented default. With a pool just big enough for the equity
    corrections, they are paid in full and merit gets what is left — nothing."""
    equity_total = sum(r.equity_amount for r in recs)
    allocation = apply_budget_constraint(recs, equity_total)

    assert sum(r.equity_amount for r in allocation.recommendations) == pytest.approx(equity_total)
    assert sum(r.merit_amount for r in allocation.recommendations) == 0
    assert allocation.priority == "equity_first"


def test_equity_first_fixes_the_worst_cases_completely(rules, recs):
    """When even the equity pool is short, the worst compa-ratio is corrected in
    full rather than everyone being half-corrected. Half-correcting leaves
    everybody still below band and still on the list next year."""
    worst = min(recs, key=lambda r: r.compa_ratio_before)
    allocation = apply_budget_constraint(recs, worst.equity_amount)

    funded = {r.employee_id: r for r in allocation.recommendations}
    assert funded[worst.employee_id].equity_amount == pytest.approx(worst.equity_amount)
    assert sum(r.recommended_amount for r in allocation.recommendations) == pytest.approx(
        worst.equity_amount
    )


def test_merit_first_is_the_mirror_image(rules, recs):
    merit_total = sum(r.merit_amount + r.promotion_amount for r in recs)
    allocation = apply_budget_constraint(recs, merit_total, priority="merit_first")

    assert sum(r.merit_amount for r in allocation.recommendations) == pytest.approx(merit_total)
    assert sum(r.equity_amount for r in allocation.recommendations) == 0


def test_pro_rata_scales_everyone(rules, recs):
    total = sum(r.recommended_amount for r in recs)
    allocation = apply_budget_constraint(recs, total / 2, priority="pro_rata")

    # Everyone who was getting something still gets something — that is the
    # appeal of pro rata, and also its problem.
    for before, after in zip(recs, allocation.recommendations, strict=True):
        if before.recommended_amount > 1000:
            assert after.recommended_amount > 0
            assert after.recommended_amount < before.recommended_amount


def test_budget_from_payroll(rules, population):
    payroll = float(population["salary_annual"].sum())
    assert rules.budget_from_payroll(payroll) == pytest.approx(0.10 * payroll)


def test_budget_summary_reports_the_shortfall(rules, recs):
    total = sum(r.recommended_amount for r in recs)
    allocation = apply_budget_constraint(recs, total * 0.5)
    text = allocation.summary()

    assert allocation.shortfall > 0
    assert "unfunded" in text
    assert "equity first" in text


def test_empty_population_is_not_a_crash(rules):
    allocation = apply_budget_constraint([], 10_00_000)
    assert allocation.recommendations == ()
    assert allocation.allocated == 0
