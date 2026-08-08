"""Layer 3 — increment and pay-equity rules.

**There is no machine learning in this file, and that is the point.**

It is tempting to build a second model for increments: feed it last year's
raises, let it learn what a raise looks like. Don't. An increment is not a
pattern hiding in data — it is a *decision*. How big is the pool? What is a
rating of 4 worth in rupees? Do we fix someone's underpayment this year or over
three? Those are choices a company makes, and a model trained on last year's
choices would do nothing but repeat them, including the ones that were wrong.

Worse, it would repeat them *unarguably*. A rule says "8% because your rating
was 3, plus 5% because you sit at 0.86 of band". A model says "13%". Only one of
those can be challenged by the person it is about, and being challengeable is
the whole job here.

**Knowing when not to use machine learning is part of being good at machine
learning.** Layer 1 (`model/band.py`) predicts the band, because what the market
pays genuinely is a pattern in data. Layer 2 (`payroll/calculator.py`) computes
take-home, because tax is arithmetic. This layer decides what to *do* about the
band, because that is policy — and policy belongs in a file HR can read.

---

## The one number everything runs on

    compa-ratio = actual salary ÷ midpoint of the predicted band

A **compa-ratio** (short for "comparative ratio", standard compensation
vocabulary) of 0.86 means you are paid 86% of the going rate for your job. One
division, and it drives both use cases in this project:

| compa-ratio | meaning | what this module does |
|---|---|---|
| below 0.90 | paid below band, a genuine flight risk | equity correction on top of merit |
| 0.90 – 1.10 | in band | merit only |
| above 1.10 | paid above band | merit damped |
| above 1.25 | well above band | held — no increase this year |

**Pay equity** is `equity_review`: everyone below 0.90, worst first, with a
price tag. **Increments** are `recommend`: merit from the rating, plus equity
from the compa-ratio, capped, then squeezed into the budget by
`apply_budget_constraint`.

Every threshold, percentage and cap in that table comes from
``configs/policy/increment_fy_YYYY_YY.yaml``. Nothing is written into the code,
because a policy hard-coded into Python is a policy that needs an engineer to
change — and the people who own compensation policy are not engineers.

---

## What the output has to be

Not a number. A number, a percentage, the compa-ratio before and after, and a
sentence in plain English:

    ₹1.6L (13.4%): 8% merit for a rating of 3, plus 5.4% equity correction —
    at 0.86 of band midpoint (below band); this brings pay up to the 0.95
    target.

If a recruiter cannot read that sentence and say "no, hang on, her rating was
4", the tool has failed even when the arithmetic is perfect.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ─────────────────────────────────────────────────────── config models
#
# Same shape as payroll/calculator.py: frozen pydantic models that mirror the
# YAML exactly, with `extra="forbid"` so a typo in the config file is a loud
# error at load time rather than a silently ignored line that costs somebody
# their raise.


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


#: How scarcity is allocated when recommendations cost more than the pool.
#: Spelled out in the config file, at length — it is the most contested line.
BudgetPriority = Literal["equity_first", "merit_first", "pro_rata"]


class BudgetRules(Frozen):
    pct_of_payroll: float = Field(ge=0, le=1)
    priority: BudgetPriority = "equity_first"


class MeritRules(Frozen):
    by_rating: dict[int, float]
    on_missing_rating: Literal["assume_default", "no_merit"] = "assume_default"
    default_rating: int = 3

    @model_validator(mode="after")
    def _check_table(self) -> MeritRules:
        if not self.by_rating:
            raise ValueError("merit.by_rating must not be empty")
        if any(pct < 0 for pct in self.by_rating.values()):
            raise ValueError("merit.by_rating percentages must not be negative")
        if self.on_missing_rating == "assume_default" and self.default_rating not in self.by_rating:
            raise ValueError(
                f"merit.default_rating {self.default_rating} is not in by_rating "
                f"{sorted(self.by_rating)}"
            )
        return self


class CompaRules(Frozen):
    below_band: float = Field(gt=0)
    above_band: float = Field(gt=0)
    hold_above: float = Field(gt=0)
    above_band_merit_multiplier: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check_order(self) -> CompaRules:
        if not self.below_band < self.above_band <= self.hold_above:
            raise ValueError(
                "compa thresholds must satisfy below_band < above_band <= hold_above, got "
                f"{self.below_band}, {self.above_band}, {self.hold_above}"
            )
        return self


class EquityRules(Frozen):
    target_compa_ratio: float = Field(gt=0)
    close_gap_over_years: int = Field(ge=1)
    max_single_year_correction_pct: float = Field(ge=0)


class PromotionRules(Frozen):
    uplift_pct: float = Field(ge=0)


class LimitRules(Frozen):
    max_total_increase_pct: float = Field(gt=0)
    round_down_to_nearest: float = Field(default=100, ge=0)


class IncrementRules(Frozen):
    """The whole policy, loaded from one financial year's config file."""

    financial_year: str
    budget: BudgetRules
    merit: MeritRules
    compa: CompaRules
    equity: EquityRules
    promotion: PromotionRules
    limits: LimitRules

    # Provenance. Never used in a calculation. A policy that has forgotten who
    # approved it is a policy nobody can defend when it is questioned.
    source: str = ""
    approved_by: str | None = None
    approved_on: str | None = None

    @model_validator(mode="after")
    def _check_target_is_in_band(self) -> IncrementRules:
        # Correcting someone to a target that is still below the band would be
        # busywork: they stay on the equity list forever, at increasing cost.
        if self.equity.target_compa_ratio < self.compa.below_band:
            raise ValueError(
                f"equity.target_compa_ratio {self.equity.target_compa_ratio} is below "
                f"compa.below_band {self.compa.below_band} — correcting people to a "
                "target that still counts as underpaid achieves nothing"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> IncrementRules:
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def budget_from_payroll(self, total_annual_payroll: float) -> float:
        """The increment pool in rupees, from the payroll it is a percentage of.

        The config stores a percentage rather than an amount so the number
        survives headcount changes — 10% of payroll is still 10% of payroll
        after twenty people join.
        """
        if total_annual_payroll < 0:
            raise ValueError(f"payroll must not be negative, got {total_annual_payroll}")
        return total_annual_payroll * self.budget.pct_of_payroll


# ─────────────────────────────────────────────────────── small helpers


def _lakhs(amount: float) -> str:
    """Rupees in the unit Indians actually speak: ₹1.6L, ₹1.24Cr, ₹48,000.

    Deliberately *not* `model.metrics.format_rupees`, which spells the number
    out in full (₹1,60,000). That is right for a table; this is right for a
    sentence someone reads at speed. Note that below one lakh, Indian and
    Western digit grouping agree (₹48,000), so plain formatting is safe there.
    """
    sign = "-" if amount < 0 else ""
    a = abs(amount)
    if a >= 1_00_00_000:
        return f"{sign}₹{a / 1_00_00_000:.2f}Cr"
    if a >= 1_00_000:
        return f"{sign}₹{a / 1_00_000:.1f}L"
    return f"{sign}₹{a:,.0f}"


def _pct(fraction: float) -> str:
    """0.08 → "8%", 0.054 → "5.4%". Drops the decimal when it is a clean whole
    number, because "8.0% merit" reads like a machine wrote it."""
    hundredths = fraction * 100
    if abs(hundredths - round(hundredths)) < 0.05:
        return f"{round(hundredths):.0f}%"
    return f"{hundredths:.1f}%"


def _floor_to(amount: float, step: float) -> float:
    """Round DOWN to a multiple of ``step``. See `limits.round_down_to_nearest`
    in the config for why down and never nearest."""
    if step <= 0:
        return float(amount)
    return math.floor(amount / step) * step


def _band_midpoint(band: Any) -> float:
    """Pull one midpoint out of whatever the caller had to hand.

    Accepts a plain number, or anything with a ``.median`` — which is what
    `model.band.BandPrediction` carries. Duck-typed on purpose: this module
    must not import Layer 1, both because the dependency runs the wrong way
    (policy should not need LightGBM installed) and because a spreadsheet of
    midpoints from a consultant is a perfectly valid input here.
    """
    value: Any = getattr(band, "median", band)
    # A plain float, a numpy scalar and a one-element array all arrive here.
    # Only the last has a length — checking for that is what separates them.
    if hasattr(value, "__len__"):
        seq = list(value)
        if len(seq) != 1:
            raise ValueError(
                f"recommend() handles one employee at a time; got {len(seq)} band midpoints. "
                "Use recommend_batch() for a whole population."
            )
        value = seq[0]
    midpoint = float(value)
    if midpoint <= 0:
        raise ValueError(f"band midpoint must be positive, got {midpoint}")
    return midpoint


def _midpoints(bands: Any, n: int) -> list[float]:
    """The same idea for a whole population."""
    values: Any = getattr(bands, "median", bands)
    seq = [float(v) for v in values]
    if len(seq) != n:
        raise ValueError(f"got {len(seq)} band midpoints for {n} employees")
    if any(m <= 0 for m in seq):
        raise ValueError("every band midpoint must be positive")
    return seq


# ─────────────────────────────────────────────────────── the recommendation


@dataclass(frozen=True)
class IncrementRecommendation:
    """One person's recommended increment, and why.

    Frozen, and every derived figure is a *property* rather than a stored
    field. That is not tidiness — it means the recommended amount, the
    compa-ratio after, and the English reason can never drift apart from the
    three component amounts they describe. When `apply_budget_constraint`
    trims the merit component, the reason string rewrites itself to match. A
    recommendation whose explanation no longer matches its number is the exact
    failure mode this whole layer exists to prevent.
    """

    current_salary: float
    band_midpoint: float

    # The three components, in rupees. Everything else is derived from these.
    merit_amount: float
    equity_amount: float
    promotion_amount: float

    rating: int | None  # as supplied — None means none was on file
    rating_used: int | None  # after the missing-rating policy was applied

    rules: IncrementRules = field(repr=False)

    employee_id: object | None = None

    # Flags, purely so the reason string can explain itself.
    rating_assumed: bool = False
    merit_damped: bool = False
    held: bool = False
    capped_by_total_limit: bool = False
    capped_by_correction_limit: bool = False
    budget_reduced: bool = False

    # ── derived numbers ────────────────────────────────────────────────

    @property
    def recommended_amount(self) -> float:
        return self.merit_amount + self.equity_amount + self.promotion_amount

    @property
    def recommended_pct(self) -> float:
        """As a fraction of current salary — computed after every cap and
        rounding, so it is what the person actually gets, not what a rule
        proposed at some intermediate step."""
        return self.recommended_amount / self.current_salary

    @property
    def new_salary(self) -> float:
        return self.current_salary + self.recommended_amount

    @property
    def compa_ratio_before(self) -> float:
        return self.current_salary / self.band_midpoint

    @property
    def compa_ratio_after(self) -> float:
        return self.new_salary / self.band_midpoint

    @property
    def band_label(self) -> str:
        """The word HR would use: below band, in band, above band."""
        return band_label(self.compa_ratio_before, self.rules)

    # ── the part a human reads ─────────────────────────────────────────

    @property
    def reason(self) -> str:
        """Plain English, naming every driver and every number.

        Written to be *disagreed with*. If the rating is wrong, or the band is
        wrong, or the policy is wrong, this sentence is where a human notices.
        """
        clauses: list[str] = []

        if self.merit_amount > 0:
            merit_pct = self.merit_amount / self.current_salary
            clause = f"{_pct(merit_pct)} merit for a rating of {self.rating_used}"
            if self.rating_assumed:
                clause += " (assumed — no rating on file)"
            if self.merit_damped:
                mult = self.rules.compa.above_band_merit_multiplier
                clause += f", cut to {_pct(mult)} of normal because pay is above band"
            # When the total cap bit, merit is what absorbed it. Say so, or the
            # sentence reads as though a rating of 5 is worth 8%.
            if self.capped_by_total_limit and self.rating_used is not None:
                full = self.rules.merit.by_rating[self.rating_used]
                if full > merit_pct + 0.0005:
                    clause += f" (trimmed from {_pct(full)} by the single-year cap)"
            clauses.append(clause)
        elif self.held:
            clauses.append("no increase this year")
        elif self.rating_used is not None and self.rules.merit.by_rating.get(self.rating_used) == 0:
            clauses.append(f"no merit increase at a rating of {self.rating_used}")
        elif self.rating is None and self.rules.merit.on_missing_rating == "no_merit":
            clauses.append("no merit increase — no rating on file")

        if self.equity_amount > 0:
            clause = f"{_pct(self.equity_amount / self.current_salary)} equity correction"
            if self.capped_by_correction_limit:
                cap = self.rules.equity.max_single_year_correction_pct
                clause += f" (at the {_pct(cap)} single-year ceiling)"
            clauses.append(clause)

        if self.promotion_amount > 0:
            clauses.append(f"{_pct(self.promotion_amount / self.current_salary)} promotion uplift")

        body = ", plus ".join(clauses) if clauses else "no increase"

        before = self.compa_ratio_before
        tail = f" — at {before:.2f} of band midpoint ({self.band_label})"

        if before < self.rules.compa.below_band:
            target = self.rules.equity.target_compa_ratio
            if self.compa_ratio_after >= target - 1e-9:
                tail += f"; this brings pay up to the {target:.2f} target"
            else:
                tail += f"; the remaining gap closes in later reviews (target {target:.2f})"
        elif self.held:
            tail += f"; pay is held above the {self.rules.compa.hold_above:.2f} line"

        notes = ""
        if self.capped_by_total_limit:
            notes += (
                f" Capped at the {_pct(self.rules.limits.max_total_increase_pct)} "
                "single-year maximum."
            )
        if self.budget_reduced:
            notes += " Trimmed to fit the increment budget."

        lead = f"{_lakhs(self.recommended_amount)} ({_pct(self.recommended_pct)})"
        return f"{lead}: {body}{tail}.{notes}"

    def as_row(self) -> dict[str, object]:
        """A flat dict, for building a DataFrame or an API response."""
        return {
            "employee_id": self.employee_id,
            "current_salary": self.current_salary,
            "band_midpoint": self.band_midpoint,
            "compa_ratio_before": self.compa_ratio_before,
            "compa_ratio_after": self.compa_ratio_after,
            "rating": self.rating,
            "merit_amount": self.merit_amount,
            "equity_amount": self.equity_amount,
            "promotion_amount": self.promotion_amount,
            "recommended_amount": self.recommended_amount,
            "recommended_pct": self.recommended_pct,
            "new_salary": self.new_salary,
            "reason": self.reason,
        }

    def __str__(self) -> str:
        who = f"{self.employee_id}: " if self.employee_id is not None else ""
        return who + self.reason


def band_label(compa_ratio: float, rules: IncrementRules) -> str:
    """Turn a compa-ratio into the word HR would use for it.

    `model.band.compa_label` says the same thing against module constants; this
    one reads the thresholds from the policy config, and the config is the
    authority for anything that spends money.
    """
    if compa_ratio < rules.compa.below_band:
        return "below band"
    if compa_ratio > rules.compa.above_band:
        return "above band"
    return "in band"


# ─────────────────────────────────────────────────────── the rule itself


def _resolve_rating(rating: int | None, rules: MeritRules) -> tuple[int | None, float, bool]:
    """Turn a (possibly missing) rating into a merit percentage.

    Returns ``(rating_used, merit_pct, was_assumed)``.
    """
    if rating is None:
        if rules.on_missing_rating == "no_merit":
            return None, 0.0, False
        return rules.default_rating, rules.by_rating[rules.default_rating], True

    # A rating that is present but not in the table is a *data* error, not a
    # missing value, and guessing at it would silently pay someone the wrong
    # amount. Fail loudly instead.
    if rating not in rules.by_rating:
        raise ValueError(
            f"performance rating {rating!r} is not in the policy's merit table "
            f"{sorted(rules.by_rating)}. Fix the rating or the config — do not guess."
        )
    return rating, rules.by_rating[rating], False


def recommend(
    current_salary: float,
    band: Any,
    performance_rating: int | None,
    rules: IncrementRules,
    *,
    is_promotion: bool = False,
    employee_id: object | None = None,
) -> IncrementRecommendation:
    """Recommend one person's increment, with the reasoning attached.

    Args:
        current_salary: Annual base salary today, in rupees.
        band: The predicted band for this person's job — a
            `model.band.BandPrediction` for a single row, or just the midpoint
            as a number. Only the midpoint is used; that is what compa-ratio
            divides by.
        performance_rating: The rating from the appraisal cycle, or ``None`` if
            there isn't one. See `merit.on_missing_rating` in the config.
        rules: Loaded from that financial year's policy file.
        is_promotion: Whether this cycle also moves the person up a level.
        employee_id: Carried through to the output, never used in the maths.

    The order of operations, which is itself policy:

    1. **Merit** from the rating, damped (or zeroed) if pay is already above
       the band. Rewarding performance is fine; paying 18% more to someone
       already 25% over the market rate is how a band stops meaning anything.
    2. **Equity** if pay is below the band: the distance to the target
       compa-ratio, divided into equal annual slices, then capped.
    3. **Promotion** uplift if applicable.
    4. **Cap** the total. When the cap bites, merit is trimmed first and the
       equity correction is protected last — same reasoning as the budget
       priority: a mistake the company made outranks a reward it is choosing
       to give.
    5. **Round down** every component, so the total can never creep upward.
    """
    if current_salary <= 0:
        raise ValueError(f"current_salary must be positive, got {current_salary}")

    midpoint = _band_midpoint(band)
    compa = current_salary / midpoint

    # ── 1. merit ───────────────────────────────────────────────────────
    rating_used, merit_pct, assumed = _resolve_rating(performance_rating, rules.merit)

    held = compa > rules.compa.hold_above
    damped = False
    if held:
        merit_pct = 0.0
    elif compa > rules.compa.above_band:
        merit_pct *= rules.compa.above_band_merit_multiplier
        damped = merit_pct > 0

    # ── 2. equity ──────────────────────────────────────────────────────
    equity_pct = 0.0
    correction_capped = False
    if compa < rules.compa.below_band:
        target_salary = rules.equity.target_compa_ratio * midpoint
        full_gap_pct = (target_salary - current_salary) / current_salary
        equity_pct = full_gap_pct / rules.equity.close_gap_over_years
        if equity_pct > rules.equity.max_single_year_correction_pct:
            equity_pct = rules.equity.max_single_year_correction_pct
            correction_capped = True

    # ── 3. promotion ───────────────────────────────────────────────────
    promotion_pct = rules.promotion.uplift_pct if is_promotion else 0.0

    # ── 4. the total cap, absorbed by merit first, then promotion ──────
    total_pct = merit_pct + equity_pct + promotion_pct
    total_capped = total_pct > rules.limits.max_total_increase_pct
    if total_capped:
        room = rules.limits.max_total_increase_pct
        # Equity is protected: it comes out of the room first.
        equity_pct = min(equity_pct, room)
        room -= equity_pct
        promotion_pct = min(promotion_pct, room)
        room -= promotion_pct
        merit_pct = min(merit_pct, room)

    # ── 5. rupees, rounded down ────────────────────────────────────────
    step = rules.limits.round_down_to_nearest
    merit_amount = _floor_to(current_salary * merit_pct, step)
    equity_amount = _floor_to(current_salary * equity_pct, step)
    promotion_amount = _floor_to(current_salary * promotion_pct, step)

    return IncrementRecommendation(
        current_salary=float(current_salary),
        band_midpoint=midpoint,
        merit_amount=merit_amount,
        equity_amount=equity_amount,
        promotion_amount=promotion_amount,
        rating=performance_rating,
        rating_used=rating_used,
        rules=rules,
        employee_id=employee_id,
        rating_assumed=assumed,
        merit_damped=damped,
        held=held,
        capped_by_total_limit=total_capped,
        capped_by_correction_limit=correction_capped,
    )


def recommend_batch(
    employees: pd.DataFrame,
    bands: Any,
    rules: IncrementRules,
    *,
    salary_col: str = "salary_annual",
    rating_col: str | None = "performance_rating",
    promotion_col: str | None = None,
    id_col: str | None = None,
) -> list[IncrementRecommendation]:
    """`recommend` over a whole population — the input `apply_budget_constraint`
    expects.

    ``bands`` is anything with one midpoint per row: a `BandPrediction`, a
    Series, an array, a list.
    """
    midpoints = _midpoints(bands, len(employees))
    ids = employees[id_col].to_numpy() if id_col else employees.index.to_numpy()

    out: list[IncrementRecommendation] = []
    for position, (_, row) in enumerate(employees.iterrows()):
        rating = None
        if rating_col and rating_col in employees.columns:
            raw = row[rating_col]
            rating = None if pd.isna(raw) else int(raw)
        promoted = bool(row[promotion_col]) if promotion_col else False
        out.append(
            recommend(
                float(row[salary_col]),
                midpoints[position],
                rating,
                rules,
                is_promotion=promoted,
                employee_id=ids[position],
            )
        )
    return out


# ─────────────────────────────────────────────────────── pay equity


@dataclass(frozen=True)
class EquityReview:
    """Everyone paid below their band, worst first, with the bill attached.

    This is the output that makes a company want the tool. A model that
    predicts bands is interesting; a list that says "these eleven people are
    underpaid, here is who is worst, and it costs ₹34.2L to fix" is a thing a
    Head of People takes into a budget meeting.
    """

    frame: pd.DataFrame  # only the people below the threshold, worst first
    n_reviewed: int
    threshold: float
    target: float

    #: What it would cost to bring everyone to the target compa-ratio, all at
    #: once. The honest total — the size of the problem.
    total_cost_to_target: float

    #: What this year's slice costs under the phasing and cap rules. The number
    #: that has to fit in an actual budget.
    total_cost_this_year: float

    @property
    def n_below(self) -> int:
        return len(self.frame)

    @property
    def share_below(self) -> float:
        return self.n_below / self.n_reviewed if self.n_reviewed else 0.0

    @property
    def worst_compa_ratio(self) -> float | None:
        return None if self.frame.empty else float(self.frame["compa_ratio"].iloc[0])

    def summary(self) -> str:
        """The paragraph that goes at the top of the slide."""
        if self.frame.empty:
            return (
                f"Pay equity review — {self.n_reviewed:,} employees, "
                f"none below a compa-ratio of {self.threshold:.2f}. Nothing to correct."
            )
        lines = [
            f"Pay equity review — {self.n_reviewed:,} employees",
            f"  below {self.threshold:.2f} compa-ratio  "
            f"{self.n_below:>8,}  ({self.share_below:.1%} of the population)",
            f"  worst case               {self.worst_compa_ratio:>8.2f}  of the band midpoint",
            f"  cost to reach {self.target:.2f}       {_lakhs(self.total_cost_to_target):>8}  "
            "in full, all at once",
            f"  cost this year           {_lakhs(self.total_cost_this_year):>8}  "
            "under the phasing and cap rules",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def equity_review(
    employees: pd.DataFrame,
    bands: Any,
    rules: IncrementRules,
    *,
    salary_col: str = "salary_annual",
    id_col: str | None = None,
    keep_cols: Iterable[str] = (),
) -> EquityReview:
    """The pay-equity list: everyone below the band, sorted by severity.

    Args:
        employees: One row per person, with a salary column.
        bands: One band midpoint per row — a `BandPrediction`, a Series, or an
            array. Predicted from each person's *job*, never from their pay.
        rules: The policy, which owns the threshold and the target.
        salary_col: Which column holds annual base salary.
        id_col: Which column identifies the person. Defaults to the index.
        keep_cols: Extra columns to carry into the output — ``["gender",
            "location"]`` is the usual choice, because the second question
            after "who is underpaid" is always "is it the same group every
            time".

    **Sorted by compa-ratio, not by rupees.** Someone at 0.78 is more wronged
    than someone at 0.88, whatever their salaries are — the ratio is how unfair
    it is *to that person*. The rupee costs are in the frame too, so re-sort by
    ``cost_to_target`` when the question is "what can we afford", which is a
    different question and deserves to be asked separately.
    """
    midpoints = _midpoints(bands, len(employees))
    salaries = employees[salary_col].astype(float).to_numpy()
    if (salaries <= 0).any():
        raise ValueError(f"every {salary_col} must be positive")

    ids = employees[id_col].to_numpy() if id_col else employees.index.to_numpy()

    threshold = rules.compa.below_band
    target = rules.equity.target_compa_ratio
    step = rules.limits.round_down_to_nearest

    rows: list[dict[str, object]] = []
    for position, salary in enumerate(salaries):
        midpoint = midpoints[position]
        compa = salary / midpoint
        if compa >= threshold:
            continue

        cost_to_target = target * midpoint - salary
        # This year's slice: the same arithmetic `recommend` uses for the
        # equity component, so the two can never disagree about the bill.
        this_year_pct = min(
            (cost_to_target / salary) / rules.equity.close_gap_over_years,
            rules.equity.max_single_year_correction_pct,
        )
        this_year = _floor_to(salary * this_year_pct, step)

        row: dict[str, object] = {
            "employee_id": ids[position],
            "current_salary": salary,
            "band_midpoint": midpoint,
            "compa_ratio": compa,
            "shortfall_to_band": threshold * midpoint - salary,
            "cost_to_target": cost_to_target,
            "cost_this_year": this_year,
        }
        for col in keep_cols:
            row[col] = employees[col].to_numpy()[position]
        rows.append(row)

    columns = [
        "employee_id",
        "current_salary",
        "band_midpoint",
        "compa_ratio",
        "shortfall_to_band",
        "cost_to_target",
        "cost_this_year",
        *keep_cols,
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values("compa_ratio").reset_index(drop=True)

    return EquityReview(
        frame=frame,
        n_reviewed=len(employees),
        threshold=threshold,
        target=target,
        total_cost_to_target=float(frame["cost_to_target"].sum()) if not frame.empty else 0.0,
        total_cost_this_year=float(frame["cost_this_year"].sum()) if not frame.empty else 0.0,
    )


# ─────────────────────────────────────────────────────── the budget


@dataclass(frozen=True)
class BudgetAllocation:
    """What the pool could actually pay for, and what it could not."""

    recommendations: tuple[IncrementRecommendation, ...]
    budget: float
    requested: float
    allocated: float
    priority: BudgetPriority

    @property
    def shortfall(self) -> float:
        """How much more the recommendations wanted than the pool held."""
        return max(0.0, self.requested - self.allocated)

    @property
    def n_reduced(self) -> int:
        return sum(1 for r in self.recommendations if r.budget_reduced)

    @property
    def utilisation(self) -> float:
        return self.allocated / self.budget if self.budget else 0.0

    def summary(self) -> str:
        lines = [
            f"Increment budget — {len(self.recommendations):,} employees, "
            f"allocated {self.priority.replace('_', ' ')}",
            f"  pool          {_lakhs(self.budget):>10}",
            f"  recommended   {_lakhs(self.requested):>10}",
            f"  allocated     {_lakhs(self.allocated):>10}   ({self.utilisation:.0%} of pool)",
        ]
        if self.shortfall > 0:
            lines.append(
                f"  unfunded      {_lakhs(self.shortfall):>10}   "
                f"({self.n_reduced:,} people reduced)"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def apply_budget_constraint(
    recommendations: Sequence[IncrementRecommendation],
    budget: float,
    *,
    priority: BudgetPriority | None = None,
) -> BudgetAllocation:
    """Squeeze a set of recommendations into the pool that actually exists.

    The realistic case: the rules recommend ₹1.4Cr and the CFO has approved
    ₹1.1Cr. Something has to give, and *what* gives is a policy choice — so it
    is a named, switchable one rather than whatever the code happened to do.

    ``equity_first`` (the config default) funds every equity correction, worst
    compa-ratio first, then shares whatever is left across merit and promotion
    uplifts pro rata. The argument: underpayment is a mistake the company
    already made, merit is a reward it is choosing to give, and fixing a
    mistake outranks giving a reward. It is also the cheapest order over time,
    because the people below band are the ones most likely to resign.

    ``merit_first`` is the mirror image and is genuinely defensible — it
    protects the top performers you can least afford to lose, and some
    companies correct underpayment off-cycle rather than out of the annual
    pool. ``pro_rata`` scales everyone down equally, which feels fair and
    quietly treats an unfairness the company caused as an equal claim to a
    bonus it chose to award.

    Nothing here can produce a negative increment: every component is clamped
    at zero and rounded down, so the worst outcome for anyone is ₹0.
    """
    recs = list(recommendations)
    if budget < 0:
        raise ValueError(f"budget must not be negative, got {budget}")

    requested = sum(r.recommended_amount for r in recs)
    if not recs:
        return BudgetAllocation((), float(budget), 0.0, 0.0, priority or "equity_first")

    if priority is None:
        priority = recs[0].rules.budget.priority
    if requested <= budget:
        # Nothing to do — and say so by returning the originals untouched,
        # rather than re-deriving numbers that would only pick up rounding.
        return BudgetAllocation(tuple(recs), float(budget), requested, requested, priority)

    step = min(r.rules.limits.round_down_to_nearest for r in recs)
    n = len(recs)
    merit = [0.0] * n
    equity = [0.0] * n
    promo = [0.0] * n

    def fund_equity(remaining: float) -> float:
        """Worst compa-ratio first, in full, until the money runs out.

        Deliberately *not* pro rata across the underpaid. Half-correcting
        everybody leaves everybody still below band and still on the list next
        year; fully correcting the worst cases takes real people off it. When
        you cannot fix everything, fix something completely.
        """
        for i in sorted(range(n), key=lambda j: recs[j].compa_ratio_before):
            take = _floor_to(min(recs[i].equity_amount, remaining), step)
            equity[i] = max(0.0, take)
            remaining -= equity[i]
        return remaining

    def fund_merit(remaining: float) -> float:
        """Merit and promotion together, scaled by one common factor.

        Pro rata rather than best-ratings-first: cutting the bottom half of the
        performance range to zero to protect the top would make the rating
        table mean something different at budget time than it does on paper.
        """
        pool = sum(r.merit_amount + r.promotion_amount for r in recs)
        scale = min(1.0, remaining / pool) if pool > 0 else 0.0
        for i, rec in enumerate(recs):
            merit[i] = _floor_to(rec.merit_amount * scale, step)
            promo[i] = _floor_to(rec.promotion_amount * scale, step)
            remaining -= merit[i] + promo[i]
        return remaining

    if priority == "equity_first":
        fund_merit(fund_equity(float(budget)))
    elif priority == "merit_first":
        fund_equity(fund_merit(float(budget)))
    else:  # pro_rata
        scale = budget / requested
        for i, rec in enumerate(recs):
            merit[i] = _floor_to(rec.merit_amount * scale, step)
            equity[i] = _floor_to(rec.equity_amount * scale, step)
            promo[i] = _floor_to(rec.promotion_amount * scale, step)

    trimmed = tuple(
        replace(
            rec,
            merit_amount=merit[i],
            equity_amount=equity[i],
            promotion_amount=promo[i],
            # Only claim "trimmed by the budget" when money was actually taken
            # away. The half-rupee tolerance keeps float noise out of a flag
            # that changes what the reason string says.
            budget_reduced=(merit[i] + equity[i] + promo[i]) < rec.recommended_amount - 0.5,
        )
        for i, rec in enumerate(recs)
    )
    allocated = sum(r.recommended_amount for r in trimmed)
    return BudgetAllocation(trimmed, float(budget), requested, allocated, priority)
