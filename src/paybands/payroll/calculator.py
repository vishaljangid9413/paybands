"""Layer 2 — the payroll calculator.

**There is no machine learning in this file, and that is the point.**

PF, insurance and income tax are arithmetic defined by rules — a percentage, a
flat premium, a published slab table. Making a model guess at them would be less
accurate than computing them, and would break every time the Budget changes.

So: the model predicts base salary. This file turns base salary into take-home.

Everything here is driven by ``configs/payroll/fy_YYYY_YY.yaml``. No rate, slab,
or threshold is written into the code — because rates change yearly and code
shouldn't have to.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────── config models


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProvidentFundRules(Frozen):
    employee_rate: float = Field(ge=0, le=1)
    basic_as_fraction_of_gross: float = Field(gt=0, le=1)
    apply_statutory_ceiling: bool
    statutory_ceiling_monthly: float = Field(ge=0)

    #: The employer contributes the same rate again. It is NOT deducted from the
    #: employee, so it never touches take-home — but it IS part of CTC, which is
    #: why a payslip's CTC exceeds its gross by exactly this amount.
    #:
    #: Confusing the two is the single most common salary-data error in India:
    #: "I earn 45k" and "I earn 43.2k" can be the same person, and a dataset
    #: mixing CTC with gross has a silent few-percent error running through it.
    employer_matches: bool = True


class InsuranceRules(Frozen):
    monthly_premium: float = Field(ge=0)


class ProfessionalTaxRules(Frozen):
    enabled: bool = False
    monthly_amount: float = Field(default=0, ge=0)


class TaxSlab(Frozen):
    upto: float | None  # None = the top slab, no upper limit
    rate: float = Field(ge=0, le=1)


class Rebate87A(Frozen):
    taxable_income_threshold: float = Field(ge=0)
    max_rebate: float = Field(ge=0)


class SurchargeBand(Frozen):
    """An extra percentage charged ON THE TAX once income passes ``above``."""

    above: float = Field(ge=0)
    rate: float = Field(ge=0, le=1)


class IncomeTaxRules(Frozen):
    standard_deduction: float = Field(ge=0)
    slabs: tuple[TaxSlab, ...]
    rebate_87a: Rebate87A
    cess_rate: float = Field(ge=0, le=1)
    surcharge: tuple[SurchargeBand, ...] = ()
    marginal_relief: bool = True


class PayrollRules(Frozen):
    financial_year: str
    regime: str
    provident_fund: ProvidentFundRules
    insurance: InsuranceRules
    professional_tax: ProfessionalTaxRules
    income_tax: IncomeTaxRules
    # Provenance. Not used in any calculation, but a config that silently
    # forgets where its numbers came from is a config nobody can audit.
    source: str = ""
    verified_on: str | None = None
    verified_against: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> PayrollRules:
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))


# ─────────────────────────────────────────────────────── results


class SlabCharge(Frozen):
    """One row of the tax working — kept so the result can be explained.

    A take-home number nobody can check is a number nobody should trust.
    """

    band_from: float
    band_to: float | None
    rate: float
    taxable_in_band: float
    tax: float


class Payslip(Frozen):
    """A full breakdown, annual and monthly."""

    annual_gross: float
    annual_basic: float

    annual_pf: float
    annual_employer_pf: float
    annual_insurance: float
    annual_professional_tax: float

    taxable_income: float
    income_tax_before_rebate: float
    rebate_applied: float
    surcharge: float
    cess: float
    annual_income_tax: float

    annual_net: float
    slab_working: tuple[SlabCharge, ...]

    # -- monthly views (what people actually recognise on a payslip) ----

    @property
    def monthly_gross(self) -> float:
        return self.annual_gross / 12

    @property
    def monthly_pf(self) -> float:
        return self.annual_pf / 12

    @property
    def monthly_insurance(self) -> float:
        return self.annual_insurance / 12

    @property
    def monthly_income_tax(self) -> float:
        return self.annual_income_tax / 12

    @property
    def monthly_net(self) -> float:
        return self.annual_net / 12

    @property
    def annual_ctc(self) -> float:
        """Cost to company: gross plus the employer's own PF contribution.

        Always LARGER than gross, and larger still than take-home. When a salary
        dataset says "salary", the first job is finding out which of these three
        it means — they can differ by 10% or more, and mixing them silently
        poisons every model trained on the result.
        """
        return self.annual_gross + self.annual_employer_pf

    @property
    def monthly_ctc(self) -> float:
        return self.annual_ctc / 12

    @property
    def total_annual_deductions(self) -> float:
        return self.annual_gross - self.annual_net

    def explain(self) -> str:
        """Human-readable breakdown, for a payslip check or an API response."""
        lines = [
            f"Monthly CTC            {self.monthly_ctc:>12,.0f}   (incl. employer PF)",
            f"Monthly gross          {self.monthly_gross:>12,.0f}",
            f"  - Provident Fund     {self.monthly_pf:>12,.0f}",
            f"  - Insurance          {self.monthly_insurance:>12,.0f}",
        ]
        if self.annual_professional_tax:
            lines.append(f"  - Professional tax   {self.annual_professional_tax / 12:>12,.0f}")
        lines += [
            f"  - Income tax         {self.monthly_income_tax:>12,.0f}",
            f"{'':─<36}",
            f"Monthly take-home      {self.monthly_net:>12,.0f}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────── the calculation


def _slab_tax(taxable: float, rules: IncomeTaxRules) -> float:
    """Just the marginal slab arithmetic, no rebate, surcharge or cess.

    Split out because marginal relief needs to ask "what would the tax have been
    exactly at the surcharge threshold?" — and answering that by re-running the
    whole pipeline would recurse.
    """
    tax, lower = 0.0, 0.0
    for slab in rules.slabs:
        upper = slab.upto if slab.upto is not None else float("inf")
        tax += max(0.0, min(taxable, upper) - lower) * slab.rate
        lower = upper
        if taxable <= upper:
            break
    return tax


def _surcharge(taxable: float, tax: float, rules: IncomeTaxRules) -> float:
    """Surcharge on the TAX (not the income), with marginal relief.

    Surcharge is a cliff. At ₹50,00,001 of taxable income a 10% surcharge lands
    on the *whole* tax bill, so one extra rupee of income could cost well over a
    lakh. The law forbids that outcome via **marginal relief**: tax + surcharge
    may not exceed the tax at the threshold plus the income earned above it.

    Without the relief this function would overstate tax for anyone just above a
    boundary — and "just above" is exactly where a salary negotiation lands.
    """
    band = None
    for candidate in rules.surcharge:
        if taxable > candidate.above:
            band = candidate  # bands are ascending; the last match wins
    if band is None or tax <= 0:
        return 0.0

    surcharge = tax * band.rate
    if not rules.marginal_relief:
        return surcharge

    # Cap: what someone exactly at the threshold pays, plus every rupee above it.
    tax_at_threshold = _slab_tax(band.above, rules)
    cap = tax_at_threshold + (taxable - band.above)
    if tax + surcharge > cap:
        surcharge = max(0.0, cap - tax)
    return surcharge


def _income_tax(
    taxable: float, rules: IncomeTaxRules
) -> tuple[float, float, float, float, list[SlabCharge]]:
    """Marginal slab tax, then rebate, then cess.

    **Slabs are marginal.** You pay each rate only on the portion of income that
    falls inside that band — not on your whole salary. Crossing into the 30%
    bracket does not suddenly tax everything at 30%. This is the single most
    common misunderstanding about income tax, and getting it wrong here would
    massively overstate the tax.
    """
    tax = 0.0
    working: list[SlabCharge] = []
    lower = 0.0

    for slab in rules.slabs:
        upper = slab.upto if slab.upto is not None else float("inf")
        # How much of the income sits inside THIS band?
        in_band = max(0.0, min(taxable, upper) - lower)
        if in_band > 0:
            charge = in_band * slab.rate
            tax += charge
            working.append(
                SlabCharge(
                    band_from=lower,
                    band_to=slab.upto,
                    rate=slab.rate,
                    taxable_in_band=in_band,
                    tax=charge,
                )
            )
        lower = upper
        if taxable <= upper:
            break

    # Section 87A: below the threshold, the tax is wiped out (up to a cap).
    # This is why someone on ₹10.8L pays no income tax at all.
    rebate = 0.0
    if taxable <= rules.rebate_87a.taxable_income_threshold:
        rebate = min(tax, rules.rebate_87a.max_rebate)
    tax_after_rebate = max(0.0, tax - rebate)

    # Surcharge lands on the tax, above ₹50L of income. Nil for most salaries.
    surcharge = _surcharge(taxable, tax_after_rebate, rules)

    # Cess is charged on tax PLUS surcharge, not on income — and only if tax
    # remains, which is why a rebated-to-zero bill carries no cess either.
    cess = (tax_after_rebate + surcharge) * rules.cess_rate

    return tax, rebate, surcharge, cess, working


def compute_payslip(annual_gross: float, rules: PayrollRules) -> Payslip:
    """Turn an annual gross (base) salary into a full take-home breakdown.

    Args:
        annual_gross: The annual base salary. This is what the band model
            predicts, and the only input that varies per person.
        rules: Loaded from the financial year's config file.
    """
    # Why reject zero and not just negatives:
    #
    # Insurance is a FLAT premium (₹3,600/year at the configured ₹300/month),
    # not a percentage. So a gross of ₹0 gives a NEGATIVE take-home — correct
    # arithmetic, and complete
    # nonsense. The same is true for any salary below the fixed deductions.
    #
    # A test caught this. The fix isn't to clamp the number (that would hide the
    # problem); it's to say plainly that a salary band model has no meaningful
    # answer for a zero salary. Guard the input, document the assumption.
    if annual_gross <= 0:
        raise ValueError(
            f"annual_gross must be positive, got {annual_gross}. Fixed deductions "
            "(insurance) would exceed pay and produce a negative take-home."
        )

    pf_rules = rules.provident_fund

    # 1. Basic pay — PF is a percentage of basic, not of gross.
    monthly_basic = (annual_gross / 12) * pf_rules.basic_as_fraction_of_gross
    if pf_rules.apply_statutory_ceiling:
        monthly_basic_for_pf = min(monthly_basic, pf_rules.statutory_ceiling_monthly)
    else:
        monthly_basic_for_pf = monthly_basic

    annual_pf = monthly_basic_for_pf * pf_rules.employee_rate * 12
    # Same rate again from the employer. Not a deduction — a company cost.
    annual_employer_pf = annual_pf if pf_rules.employer_matches else 0.0
    annual_insurance = rules.insurance.monthly_premium * 12
    annual_ptax = (
        rules.professional_tax.monthly_amount * 12 if rules.professional_tax.enabled else 0.0
    )

    # 2. Taxable income. Under the new regime the standard deduction applies;
    #    PF is NOT deductible there (it is under the old regime via 80C — one
    #    reason the two regimes are kept in separate config files).
    taxable = max(0.0, annual_gross - rules.income_tax.standard_deduction)

    tax_before, rebate, surcharge, cess, working = _income_tax(taxable, rules.income_tax)
    annual_income_tax = max(0.0, tax_before - rebate) + surcharge + cess

    annual_net = annual_gross - annual_pf - annual_insurance - annual_ptax - annual_income_tax

    return Payslip(
        annual_gross=annual_gross,
        annual_basic=monthly_basic * 12,
        annual_pf=annual_pf,
        annual_employer_pf=annual_employer_pf,
        annual_insurance=annual_insurance,
        annual_professional_tax=annual_ptax,
        taxable_income=taxable,
        income_tax_before_rebate=tax_before,
        rebate_applied=rebate,
        surcharge=surcharge,
        cess=cess,
        annual_income_tax=annual_income_tax,
        annual_net=annual_net,
        slab_working=tuple(working),
    )
