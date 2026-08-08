"""Tests for the payroll calculator.

The most important test here is `test_against_real_payslip_year_1`. Every
other test checks the logic is internally consistent; that one checks it against
**an actual payslip PDF**.

A calculator that agrees with itself is worthless. A calculator that agrees with
payroll is trustworthy.

Worth knowing how this file got here. The first version of it was built on
figures reported from memory ("PF is about 3,600, insurance about 200"). The
config was then tuned until the tests passed. Everything was green, and three
separate things were wrong: the basic fraction, the PF ceiling rule, and the
insurance premium. A test and a config derived from the same mistaken belief
agree with each other perfectly and tell you nothing.

Only a document could settle it.
"""

from __future__ import annotations

import pytest

from paybands.payroll import PayrollRules, compute_payslip

RULES_PATH = "configs/payroll/fy_2025_26.yaml"


@pytest.fixture
def rules() -> PayrollRules:
    return PayrollRules.from_yaml(RULES_PATH)


def rules_fixture() -> PayrollRules:
    """Same rules, callable outside a fixture-injected test."""
    return PayrollRules.from_yaml(RULES_PATH)


# ─────────────────────────────────────────── the ground-truth test


def test_against_real_payslip_year_1():
    """GROUND TRUTH — an actual payslip PDF, not a remembered figure.

        BASIC    26,000
        HRA      13,000   (= 50% of basic)
        SPECIAL  11,000
        gross    50,000
        − PF      1,800   (= 12% of the 15,000 ceiling, NOT of the 26,000 basic)
        − Medical   300
        net      47,900
        CTC      51,800   (= gross + employer PF 1,800)

    This REPLACED an earlier test built on verbally reported numbers, which had
    the basic fraction, the PF ceiling and the insurance premium all wrong and
    still passed — because the config had been bent to fit them. A test and a
    config derived from the same mistaken belief agree with each other perfectly
    and tell you nothing.
    """
    slip = compute_payslip(50_000 * 12, rules_fixture())

    assert slip.monthly_gross == pytest.approx(50_000)
    assert slip.annual_basic / 12 == pytest.approx(26_000, abs=2)
    assert slip.monthly_pf == pytest.approx(1_800, abs=1)
    assert slip.monthly_insurance == pytest.approx(300)
    assert slip.annual_income_tax == pytest.approx(0)
    assert slip.monthly_net == pytest.approx(47_900, abs=2)
    assert slip.monthly_ctc == pytest.approx(51_800, abs=2)


def test_payslip_own_tax_projection_reproduces():
    """The payslip's TDS block does its own annual projection: gross 6,00,000,
    standard deduction 75,000, taxable 5,25,000. Reproducing that is a SECOND
    independent check — of the tax path rather than the deduction path."""
    slip = compute_payslip(6_00_000, rules_fixture())
    assert slip.taxable_income == pytest.approx(5_25_000)
    assert slip.annual_income_tax == pytest.approx(0)


def test_pf_is_capped_by_the_ceiling_not_by_basic():
    """The single most informative number on the payslip.

    12% of the actual basic (26,000) would be 3,120. The payslip says 1,800,
    which is 12% of the 15,000 statutory ceiling exactly. So this employer applies
    the ceiling — and PF therefore does NOT rise with salary.
    """
    at_current = compute_payslip(50_000 * 12, rules_fixture()).monthly_pf
    at_double = compute_payslip(1_00_000 * 12, rules_fixture()).monthly_pf
    assert at_current == pytest.approx(1_800, abs=1)
    assert at_double == pytest.approx(1_800, abs=1), "capped PF must not grow with salary"


def test_ctc_exceeds_gross_by_employer_pf():
    """CTC, gross and take-home are three different numbers, and confusing them
    is the most common salary-data error in India. Here they are, in order."""
    slip = compute_payslip(50_000 * 12, rules_fixture())
    assert slip.annual_ctc > slip.annual_gross > slip.annual_net
    assert slip.annual_ctc - slip.annual_gross == pytest.approx(slip.annual_employer_pf)
    assert slip.annual_employer_pf == pytest.approx(slip.annual_pf)


# ─────────────────────────────────────────── tax logic


def test_slabs_are_marginal_not_flat(rules):
    """Crossing into a higher bracket must NOT tax the whole salary at that rate.

    This is the most commonly misunderstood thing about income tax. If it were
    flat, a small raise across a boundary would cost you tens of thousands.

    PICKING THE VALUES IS THE WHOLE TEST, and an earlier version of it got this
    wrong. Slabs apply to TAXABLE income, which is gross minus the ₹75,000
    standard deduction — so to straddle the ₹16,00,000 taxable boundary you need
    a gross either side of ₹16,75,000, not either side of ₹16,00,000.

    The original version compared ₹15,99,000 and ₹16,01,000 gross. Those give
    taxable incomes of ₹15,24,000 and ₹15,26,000 — both comfortably inside the
    same 15% band. No boundary was crossed, so a flat-tax implementation would
    have passed too. The test was green and proved nothing.
    """
    # ₹16,70,000 gross → ₹15,95,000 taxable (just below the boundary)
    # ₹16,80,000 gross → ₹16,05,000 taxable (just above it)
    just_below = compute_payslip(16_70_000, rules).annual_income_tax
    just_above = compute_payslip(16_80_000, rules).annual_income_tax

    # Marginal: only the ₹10,000 that crossed is taxed at the higher rate, so the
    # extra tax is about ₹2,000 plus cess.
    # Flat would tax the WHOLE ₹16,05,000 at 20% instead of 15%, a jump of ~₹81,750.
    # ₹5,000 sits between the two by a factor of sixteen either way.
    assert just_above > just_below, "more income must not mean less tax"
    assert just_above - just_below < 5_000, (
        f"tax jumped ₹{just_above - just_below:,.0f} for ₹10,000 more income — "
        "that is flat-rate behaviour, not marginal"
    )


def test_high_earner_tax_working(rules):
    """₹30L, computed by hand slab by slab.

    4-8L    @  5%  =   20,000
    8-12L   @ 10%  =   40,000
    12-16L  @ 15%  =   60,000
    16-20L  @ 20%  =   80,000
    20-24L  @ 25%  = 1,00,000
    24-29.25L @30% = 1,57,500
                     ─────────
                     4,57,500  + 4% cess (18,300) = 4,75,800
    """
    slip = compute_payslip(30_00_000, rules)

    assert slip.taxable_income == pytest.approx(29_25_000)
    assert slip.income_tax_before_rebate == pytest.approx(4_57_500)
    assert slip.rebate_applied == 0  # well above the rebate threshold
    assert slip.cess == pytest.approx(18_300)
    assert slip.annual_income_tax == pytest.approx(4_75_800)


def test_rebate_wipes_out_tax_below_threshold(rules):
    """Below the 87A threshold the tax goes to zero — and so does the cess,
    because cess is charged on tax, not on income."""
    slip = compute_payslip(11_00_000, rules)
    assert slip.income_tax_before_rebate > 0  # tax was computed...
    assert slip.rebate_applied > 0  # ...then rebated away
    assert slip.annual_income_tax == pytest.approx(0)
    assert slip.cess == pytest.approx(0)


def test_rebate_cliff_is_real(rules):
    """Just above the threshold, tax appears suddenly. This is a genuine cliff in
    Indian tax law, not a bug — worth a test so nobody 'fixes' it later."""
    below = compute_payslip(12_74_000, rules).annual_income_tax
    above = compute_payslip(12_80_000, rules).annual_income_tax
    assert below == pytest.approx(0)
    assert above > 0


# ─────────────────────────────────────────── PF rules


def test_statutory_ceiling_caps_pf(rules):
    """With the ceiling ON — which the payslip shows this employer uses — PF is
    capped at 12% of ₹15,000 = ₹1,800/month no matter how high the salary."""
    slip = compute_payslip(50_00_000, rules)
    assert slip.monthly_pf == pytest.approx(1_800)


def test_pf_scales_with_salary_when_uncapped(rules):
    """Other employers apply PF to full basic instead. Then PF grows with pay —
    doubling salary doubles PF. Kept because the config supports both and a
    company using the generous rule would need this branch to be right."""
    uncapped = rules.model_copy(
        update={
            "provident_fund": rules.provident_fund.model_copy(
                update={"apply_statutory_ceiling": False}
            )
        }
    )
    a = compute_payslip(10_00_000, uncapped).annual_pf
    b = compute_payslip(20_00_000, uncapped).annual_pf
    assert b == pytest.approx(2 * a)


# ─────────────────────────────────────────── sanity


def test_deductions_never_exceed_gross(rules):
    """For any realistic salary, take-home is positive and below gross."""
    for gross in (3_00_000, 10_80_000, 50_00_000, 1_00_00_000):
        slip = compute_payslip(gross, rules)
        assert 0 < slip.annual_net < slip.annual_gross


@pytest.mark.parametrize("bad", [0, -1, -10_00_000])
def test_non_positive_salary_rejected(rules, bad):
    """Insurance is a FLAT premium, so a zero salary would compute a negative
    take-home — arithmetically right, completely meaningless.

    An earlier version of this suite passed `0` in with the realistic salaries
    and failed. The right fix was not to clamp the output but to reject the
    input: a salary band model has no answer for a zero salary.
    """
    with pytest.raises(ValueError, match="must be positive"):
        compute_payslip(bad, rules)


def test_explain_is_readable(rules):
    text = compute_payslip(50_000 * 12, rules).explain()
    assert "Monthly take-home" in text
    assert "47,900" in text
    # CTC must appear too, so a reader cannot mistake gross for cost-to-company.
    assert "51,800" in text


# ─────────────────────────────────────────── surcharge (income above ₹50L)


def test_no_surcharge_below_fifty_lakh(rules):
    """Surcharge starts above ₹50L of TAXABLE income. Below it, nothing."""
    slip = compute_payslip(50_75_000, rules)  # taxable exactly ₹50,00,000
    assert slip.taxable_income == pytest.approx(50_00_000)
    assert slip.surcharge == 0


def test_surcharge_rates_match_the_official_table(rules):
    """New-regime surcharge, AY 2026-27: 10% above ₹50L, 15% above ₹1Cr.

    Checked well clear of the thresholds so marginal relief is not binding and
    the raw rate is what shows.
    """
    ten = compute_payslip(80_00_000, rules)
    assert ten.surcharge == pytest.approx(ten.income_tax_before_rebate * 0.10)

    fifteen = compute_payslip(1_50_00_000, rules)
    assert fifteen.surcharge == pytest.approx(fifteen.income_tax_before_rebate * 0.15)


def test_marginal_relief_kills_the_surcharge_cliff(rules):
    """WITHOUT relief, crossing ₹50L taxable would slap 10% on the WHOLE tax
    bill — about ₹1.08 lakh of extra tax for one more rupee of income.

    The law forbids that: tax + surcharge may not exceed the tax at the
    threshold plus the income earned above it. So the extra tax must be roughly
    the extra income, not a cliff.
    """
    at = compute_payslip(50_75_000, rules)  # taxable ₹50,00,000 exactly
    just_over = compute_payslip(50_80_000, rules)  # taxable ₹50,05,000

    extra_income = 5_000
    extra_tax = just_over.annual_income_tax - at.annual_income_tax

    assert just_over.surcharge > 0, "surcharge should have engaged"
    assert extra_tax < extra_income * 1.5, (
        f"₹{extra_income:,} more income cost ₹{extra_tax:,.0f} more tax — "
        "that is the cliff marginal relief exists to prevent"
    )


def test_marginal_relief_can_be_switched_off(rules):
    """The relief is config-driven, and turning it off must produce the cliff —
    otherwise the flag is decorative and we would not know the relief is what is
    doing the work."""
    no_relief = rules.model_copy(
        update={"income_tax": rules.income_tax.model_copy(update={"marginal_relief": False})}
    )
    with_relief = compute_payslip(50_80_000, rules).surcharge
    without = compute_payslip(50_80_000, no_relief).surcharge
    assert without > with_relief * 10, "relief should be doing real work here"


def test_cess_is_charged_on_tax_plus_surcharge(rules):
    """Cess is 4% of (tax + surcharge), not 4% of tax alone and never of income."""
    slip = compute_payslip(80_00_000, rules)
    base = slip.income_tax_before_rebate - slip.rebate_applied + slip.surcharge
    assert slip.cess == pytest.approx(base * 0.04)


def test_ordinary_salaries_are_untouched_by_surcharge(rules):
    """The whole surcharge path must stay dormant for normal salaries — this is
    a regression guard on the range the model actually predicts."""
    for gross in (6_00_000, 12_00_000, 30_00_000):
        assert compute_payslip(gross, rules).surcharge == 0


# ─────────────────────────────────────────── two financial years side by side


CURRENT_RULES_PATH = "configs/payroll/fy_2026_27.yaml"


@pytest.fixture
def rules_2026_27() -> PayrollRules:
    return PayrollRules.from_yaml(CURRENT_RULES_PATH)


def test_against_real_payslip_year_2(rules_2026_27):
    """GROUND TRUTH for the current year — an actual payslip PDF.

        BASIC   51,000
        HRA     25,500   (= 50% of basic)
        SPECIAL 23,500
        gross  1,00,000
        − PF     1,800   (still 12% of the 15,000 ceiling)
        − Medical  350
        net     97,850
        CTC    1,01,800   (= gross + employer PF 1,800)

    NOTE THE INPUT: gross, not CTC. CTC is ₹1,01,800/month here, and feeding
    that in where gross belongs would overstate take-home by ₹1,800/month. The
    two figures on a payslip are ~2% apart and nothing labels which is which.
    """
    slip = compute_payslip(1_00_000 * 12, rules_2026_27)

    assert slip.monthly_gross == pytest.approx(1_00_000)
    assert slip.annual_basic / 12 == pytest.approx(51_000, abs=5)
    assert slip.monthly_pf == pytest.approx(1_800, abs=1)
    assert slip.monthly_insurance == pytest.approx(350)
    assert slip.annual_income_tax == pytest.approx(0)
    assert slip.monthly_net == pytest.approx(97_850, abs=2)
    assert slip.monthly_ctc == pytest.approx(1_01_800, abs=2)


def test_current_payslip_own_tax_projection_reproduces(rules_2026_27):
    """The year 2 TDS block: gross 12,00,000 → taxable 11,25,000 → nil."""
    slip = compute_payslip(12_00_000, rules_2026_27)
    assert slip.taxable_income == pytest.approx(11_25_000)
    assert slip.annual_income_tax == pytest.approx(0)


def test_pf_deducted_is_half_of_what_lands_in_the_pf_account(rules_2026_27):
    """₹1,800 leaves the salary; ₹3,600 arrives in the PF account.

    The payslip prints "PF 1,800" under Deductions and "PFEmpr 1,800"
    separately. Both figures are real and they answer different questions:
    ₹1,800 determines take-home, ₹3,600 is what the retirement balance grows by.

    Reading the pair as a single ₹3,600 deduction is an easy mistake and it was
    made in this project — it briefly drove the ceiling in this config to
    ₹30,000 to accommodate it.
    """
    slip = compute_payslip(1_00_000 * 12, rules_2026_27)
    assert slip.monthly_pf == pytest.approx(1_800, abs=1)
    assert slip.annual_employer_pf == pytest.approx(slip.annual_pf)
    total_into_pf = slip.annual_pf + slip.annual_employer_pf
    assert total_into_pf / 12 == pytest.approx(3_600, abs=2)
    # And the deduction must not be the combined figure.
    assert slip.monthly_gross - slip.monthly_net < 2_500


def test_the_same_salary_gives_different_pf_in_different_years(rules, rules_2026_27):
    """THE POINT OF ONE-FILE-PER-YEAR, demonstrated.

    Identical gross salary, two config files, two different correct answers —
    here because the medical premium rose from ₹300 to ₹350 between them. Had
    the 2025-26 file simply been edited, there would be no honest answer to
    "what was my take-home last year?".
    """
    gross = 1_00_000 * 12
    last_year = compute_payslip(gross, rules)
    this_year = compute_payslip(gross, rules_2026_27)

    # PF is identical — the ceiling binds in both years.
    assert last_year.monthly_pf == pytest.approx(this_year.monthly_pf, abs=1)
    # The medical premium rose from 300 to 350, so take-home differs.
    assert this_year.monthly_insurance > last_year.monthly_insurance
    assert this_year.monthly_net < last_year.monthly_net


def test_last_years_payslip_still_reconciles_against_last_years_config(rules):
    """A REGRESSION GUARD ON HISTORY.

    The year 1 payslip must keep reconciling forever. If someone ever edits
    fy_2025_26.yaml to match a newer rule, this test fails and says why.
    """
    slip = compute_payslip(50_000 * 12, rules)
    assert slip.monthly_pf == pytest.approx(1_800, abs=1)
    assert slip.monthly_net == pytest.approx(47_900, abs=2)
