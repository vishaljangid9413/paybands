"""A salary dataset where *we* decide the truth.

**Why invent data when we already have real data?**

We are going to build a fairness audit — code that answers "does this model
underpay one group?". On real data you cannot check whether that audit works.
Nobody knows the true size of the real pay gap, so if the audit reports 3% you
have no way of telling whether the gap is 3%, or the gap is 9% and the audit is
broken.

So we build a world we control. We decide that one group is paid exactly 8%
less, generate people accordingly, and then run the audit. If it comes back with
≈8%, the audit works. If it says 2% or 15%, the audit is wrong and we fix it
*before* pointing it at real people.

This is calibrating a thermometer in boiling water before trusting it on a
patient. It is the whole reason this file exists.

**How salary is built here.** Everything is multiplicative, in log space:

    log(salary) = log(base)
                + log(experience curve)
                + log(role multiplier)
                + log(location tier multiplier)
                + log(institute premium, faded by experience)
                + log(previous company type multiplier)
                + log(education multiplier)
                + log(org size multiplier)
                + log(remote multiplier)
                + (career gap months / 12) * log(career gap penalty)
                + [is disadvantaged group] * log(1 - pay_gap)
                + noise ~ Normal(0, sigma)

Multiplicative because that is how people actually talk about pay: "a 30%
raise", never "a ₹2.4 lakh raise". A 30% raise means something to a fresher and
to a principal engineer; ₹2.4L means completely different things to the two.
Adding in log space *is* multiplying in rupees, so a sum of log terms is the
honest model.

Read `GroundTruth` below: it hands the audit every number this file used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import (
    CORE_COLUMNS,
    MAX_PLAUSIBLE_SALARY,
    MIN_PLAUSIBLE_SALARY,
    TARGET,
    Source,
)

# ─────────────────────────────────────────────── the generating parameters
#
# These are the "physics" of our invented world. They live at module level, in
# plain sight, as *multipliers* rather than log coefficients — 1.35 reads as
# "35% more" to anyone; 0.3001 reads as nothing to anybody.
#
# Every one of them is a judgement call informed by the Indian market, not a
# measured fact. They are here to be argued with, which is why they are in one
# visible block instead of scattered through the code.

#: What a fresher with every baseline attribute earns: tier-1 city, services
#: background, non-premium college, backend, mid-size company, in person.
BASE_SALARY = 600_000.0

#: The experience curve, as a power law:  (1 + years/knee) ** growth
#:
#: Why not a straight line: real medians (see ROADMAP) climb steeply for the
#: first several years and then flatten hard. Years 0→5 are worth far more than
#: years 15→20. A straight line would overpay veterans and underpay juniors,
#: and would make the "experience is non-linear" finding impossible to show.
EXPERIENCE_GROWTH = 0.85
EXPERIENCE_KNEE_YEARS = 3.0

#: How fast the elite-college premium decays. After this many years the premium
#: has shrunk to ~37% of its day-one size; after twice as long, ~14%.
#: A satisfying, checkable finding: where you studied stops mattering.
INSTITUTE_FADE_YEARS = 8.0

ROLE_MULTIPLIERS: dict[str, float] = {
    "Backend": 1.00,  # the reference role
    "Fullstack": 0.98,
    "Frontend": 0.92,
    "Mobile": 0.95,
    "Data Science": 1.18,
    "Data Engineering": 1.12,
    "DevOps": 1.08,
    "QA": 0.78,
    "Management": 1.30,
}

#: Tier 1 = Bangalore, Mumbai, Delhi-NCR, Hyderabad, Pune, Chennai. The gap to
#: tier 2 is large and real; part of it is cost of living, part is that the
#: high-paying employers are simply not present elsewhere.
LOCATION_TIER_MULTIPLIERS: dict[int, float] = {1: 1.00, 2: 0.78, 3: 0.64}

#: IIT / NIT / BITS / IIIT and similar. This is the *day-one* premium; it fades
#: with experience (see INSTITUTE_FADE_YEARS).
INSTITUTE_TIER_MULTIPLIERS: dict[str, float] = {"tier1": 1.20, "tier2": 1.08, "other": 1.00}

#: Product vs services is one of the largest pay gaps in Indian tech — two
#: people with identical experience can be 35% apart on this alone. A model
#: that ignores it will be badly wrong, so the generator must contain it.
PREV_COMPANY_MULTIPLIERS: dict[str, float] = {
    "services": 1.00,
    "startup": 1.12,
    "gcc": 1.25,  # global capability centre — the India arm of a foreign firm
    "product": 1.35,
}

#: Deliberately weak. Education usually matters far less than people expect
#: once experience is in the model, and the generator should say so.
EDUCATION_MULTIPLIERS: dict[str, float] = {
    "Below bachelor's": 0.90,
    "Bachelor's degree": 1.00,
    "Master's degree": 1.08,
    "Doctoral degree": 1.15,
}

ORG_SIZE_MULTIPLIERS: dict[str, float] = {
    "1-20": 0.88,
    "21-100": 0.95,
    "101-1000": 1.00,
    "1001-5000": 1.08,
    "5000+": 1.15,
}

REMOTE_MULTIPLIERS: dict[str, float] = {"In-person": 1.00, "Hybrid": 1.03, "Remote": 1.06}

#: Contract work is generated but given no pay effect of its own. Real contract
#: rates differ, but modelling that properly needs a day-rate/annualised
#: distinction we do not have — so we leave it flat rather than invent it.
EMPLOYMENT_TYPES: dict[str, float] = {"Employed, full-time": 1.00, "Contractor": 1.00}

# How common each value is. Kept next to the multipliers so the shape of the
# population is as visible as the pay rules.
_ROLE_SHARES = (0.20, 0.14, 0.12, 0.08, 0.10, 0.08, 0.10, 0.11, 0.07)
_LOCATION_SHARES = (0.62, 0.28, 0.10)
_INSTITUTE_SHARES = (0.12, 0.23, 0.65)
_PREV_COMPANY_SHARES = (0.42, 0.16, 0.18, 0.24)
_EDUCATION_SHARES = (0.05, 0.66, 0.26, 0.03)
_ORG_SIZE_SHARES = (0.14, 0.22, 0.30, 0.19, 0.15)
_REMOTE_SHARES = (0.46, 0.38, 0.16)
_EMPLOYMENT_SHARES = (0.93, 0.07)

# Experience is drawn from a right-skewed gamma: lots of people early in their
# careers, a thin tail of veterans. Mean ≈ 6.6 years.
_EXPERIENCE_SHAPE = 2.2
_EXPERIENCE_SCALE = 3.0
_MAX_EXPERIENCE_YEARS = 30

# Career break length, in months, for people who have one. Mean ≈ 14 months.
_CAREER_GAP_SHAPE = 2.0
_CAREER_GAP_SCALE = 7.0
_MAX_CAREER_GAP_MONTHS = 60

#: Level is a *label* for experience, not an independent pay driver. Giving it
#: its own multiplier would double-count experience — the person would be paid
#: twice for the same years. Included because company data will have it and the
#: pipeline must handle the column.
_LEVEL_BANDS = ((2, "Junior"), (5, "Mid"), (9, "Senior"), (14, "Lead"), (20, "Principal"))
_TOP_LEVEL = "Director"


def experience_multiplier(
    years: np.ndarray | float,
    *,
    growth: float = EXPERIENCE_GROWTH,
    knee: float = EXPERIENCE_KNEE_YEARS,
) -> np.ndarray | float:
    """How much experience alone multiplies pay: ``(1 + years/knee) ** growth``.

    Two properties this curve must have, and a straight line does not:

    * **Increasing** — more experience never pays less.
    * **Decelerating** — each extra year is worth less than the one before.
      With the defaults, years 0→5 add **130%**; years 15→20 add **23%**.
      (Verified by calling this function, not remembered. An earlier version of
      this docstring said 90% and 17%, which had been true of different constants
      and silently stopped being true when they changed. A comment that quotes
      numbers is a comment that can rot — check it against the code before
      trusting it.)

    It is a power law, so it is easy to state in one sentence: every doubling of
    ``1 + years/3`` multiplies pay by ``2 ** 0.85`` ≈ 1.8.
    """
    return (1.0 + np.asarray(years) / knee) ** growth


def institute_multiplier(
    years: np.ndarray | float,
    day_one_premium: np.ndarray | float,
    *,
    fade_years: float = INSTITUTE_FADE_YEARS,
) -> np.ndarray | float:
    """The elite-college premium, shrinking with experience.

    The interaction is written out explicitly rather than left implicit, because
    it is one of the findings we want the analysis to *recover*: an IIT degree
    is worth ~20% to a fresher and almost nothing to someone with 20 years of
    track record. By then, the track record is the evidence.
    """
    decay = np.exp(-np.asarray(years) / fade_years)
    return 1.0 + (np.asarray(day_one_premium) - 1.0) * decay


# ─────────────────────────────────────────────────────── the ground truth


@dataclass(frozen=True)
class GroundTruth:
    """Every number the generator used — the answer sheet for the audit.

    The fairness audit is *scored against this*. If the audit reports 0.03 when
    ``pay_gap`` here says 0.08, the audit is broken. That comparison is the only
    reason we generate data at all, so the truth has to be returned alongside
    it, not left implicit in the code.
    """

    n: int
    seed: int

    #: The injected unfairness. 0.08 means the disadvantaged group is paid 8%
    #: less than an otherwise identical person. Multiplicative, so it is the
    #: same 8% at ₹6L and at ₹60L.
    pay_gap: float
    disadvantaged_group: str
    share_disadvantaged: float

    #: Spread of the lognormal noise, in log points. 0.22 means a typical person
    #: sits about ±22% away from what the rules alone would pay them — which is
    #: roughly how much two equally good engineers really differ.
    noise_sigma_log: float

    base_salary: float = BASE_SALARY
    experience_growth: float = EXPERIENCE_GROWTH
    experience_knee_years: float = EXPERIENCE_KNEE_YEARS
    institute_fade_years: float = INSTITUTE_FADE_YEARS

    role_multipliers: dict[str, float] = field(default_factory=lambda: dict(ROLE_MULTIPLIERS))
    location_tier_multipliers: dict[int, float] = field(
        default_factory=lambda: dict(LOCATION_TIER_MULTIPLIERS)
    )
    institute_tier_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(INSTITUTE_TIER_MULTIPLIERS)
    )
    prev_company_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(PREV_COMPANY_MULTIPLIERS)
    )
    education_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(EDUCATION_MULTIPLIERS)
    )
    org_size_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(ORG_SIZE_MULTIPLIERS)
    )
    remote_multipliers: dict[str, float] = field(default_factory=lambda: dict(REMOTE_MULTIPLIERS))

    #: The proxy channel. Pay multiplier per *year* of career break, and how
    #: likely each group is to have had one.
    career_gap_penalty_per_year: float = 0.93
    career_gap_prob_disadvantaged: float = 0.45
    career_gap_prob_other: float = 0.06

    @property
    def log_pay_gap(self) -> float:
        """The gap as a log-space coefficient — what a regression will report.

        A regression of log(salary) on a group indicator recovers
        ``log(1 - pay_gap)``, not ``-pay_gap``. For 8% those are −0.0834 and
        −0.08 — close enough to confuse, far enough apart to fail a tight test.
        """
        return float(np.log1p(-self.pay_gap))

    @property
    def has_proxy(self) -> bool:
        """Whether career breaks encode group membership at all.

        Set the two probabilities equal and the proxy disappears — useful when
        you want the *only* difference between groups to be the injected gap.
        """
        return self.career_gap_prob_disadvantaged != self.career_gap_prob_other

    def __str__(self) -> str:
        lines = [
            f"Synthetic salary data — n={self.n:,}, seed={self.seed}",
            f"  injected pay gap           {self.pay_gap:>8.1%}  "
            f"against '{self.disadvantaged_group}' "
            f"({self.share_disadvantaged:.0%} of rows)",
            f"  as a log coefficient       {self.log_pay_gap:>8.4f}  "
            "← what a regression should recover",
            f"  noise (log sd)             {self.noise_sigma_log:>8.2f}",
            f"  base salary                ₹{self.base_salary:>10,.0f}  (fresher, all baselines)",
            f"  experience curve           (1 + yrs/{self.experience_knee_years:g})"
            f" ** {self.experience_growth:g}",
            f"  institute premium fade     {self.institute_fade_years:g} years",
            "  proxy — career breaks      "
            f"{self.career_gap_prob_disadvantaged:.0%} vs "
            f"{self.career_gap_prob_other:.0%} by group, "
            f"×{self.career_gap_penalty_per_year:g} pay per year of break",
        ]
        if not self.has_proxy:
            lines.append("    (proxy disabled — both groups equally likely to have breaks)")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────── the generator


def _pick(rng: np.random.Generator, values: tuple, shares: tuple[float, ...], n: int) -> np.ndarray:
    """Draw n values from `values` with the given probabilities."""
    return rng.choice(np.array(values, dtype=object), size=n, p=shares)


def _multipliers(values: np.ndarray, table: dict) -> np.ndarray:
    """Look up each row's multiplier. Raises KeyError on an unknown value.

    Deliberately not `.get(v, 1.0)`: silently paying a 1.0 multiplier for a
    category nobody defined would hide a typo as a plausible-looking salary.
    """
    return np.array([table[v] for v in values], dtype=float)


def _level_for(years: np.ndarray) -> np.ndarray:
    """Map experience onto a career ladder label."""
    level = np.full(len(years), _TOP_LEVEL, dtype=object)
    # Walk the bands from the top down so the first (lowest) band wins.
    for cutoff, name in reversed(_LEVEL_BANDS):
        level[years < cutoff] = name
    return level


def generate(
    n: int = 5000,
    seed: int = 42,
    pay_gap: float = 0.08,
    *,
    disadvantaged_group: str = "female",
    share_disadvantaged: float = 0.28,
    noise_sigma_log: float = 0.22,
    career_gap_penalty_per_year: float = 0.93,
    career_gap_prob_disadvantaged: float = 0.45,
    career_gap_prob_other: float = 0.06,
) -> tuple[pd.DataFrame, GroundTruth]:
    """Invent `n` employees and pay them by rules we wrote down.

    Args:
        n: How many people to invent.
        seed: Same seed → identical data, every time, on every machine.
        pay_gap: The unfairness to inject. 0.08 pays the disadvantaged group 8%
            less than an otherwise identical person. **0.0 is the control case**
            — no direct gender effect at all — and there is a test for it.
        disadvantaged_group: Which value of `gender` takes the cut.
        share_disadvantaged: That group's share of the population.
        noise_sigma_log: Spread of the lognormal noise, in log points.
        career_gap_penalty_per_year: What one year of career break does to pay.
            0.93 = a 7% cut per year out. This is the *proxy channel*.
        career_gap_prob_disadvantaged: Chance of having had a career break.
        career_gap_prob_other: Same, for the other group. Set the two equal to
            switch the proxy off.

    Returns:
        `(DataFrame in the common schema, GroundTruth)`.

    **On the proxy.** `career_gap_months` is correlated with gender *and* it
    depresses pay on its own. That is deliberate and it is the point: later we
    delete the gender column, re-run the audit, and watch the gap survive —
    because the model rebuilds it out of career gaps. Everyone's first instinct
    ("just drop the sensitive column") is wrong, and this dataset is how we
    prove it rather than assert it.

    Nothing *else* correlates with gender here. Real life is messier — women are
    unevenly distributed across roles and cities too — but a single clean proxy
    channel makes the demonstration readable. Adding more channels later is easy
    and makes for a good follow-up.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 <= pay_gap < 1.0:
        raise ValueError(
            f"pay_gap must be in [0, 1), got {pay_gap}. It is a fractional cut: "
            "0.08 means 'paid 8% less', and 1.0 would mean 'paid nothing'."
        )

    # A LOCAL generator, never np.random.seed(). Global random state is shared
    # by every library in the process, so anything else that draws a number —
    # a sklearn split, another module's import-time shuffle — silently changes
    # your data. A local Generator cannot be disturbed from outside, which is
    # what makes "same seed, same bytes" actually true.
    rng = np.random.default_rng(seed)

    # ── who these people are ──────────────────────────────────────────
    years = np.minimum(
        rng.gamma(_EXPERIENCE_SHAPE, _EXPERIENCE_SCALE, n).round(), _MAX_EXPERIENCE_YEARS
    )
    role = _pick(rng, tuple(ROLE_MULTIPLIERS), _ROLE_SHARES, n)
    location_tier = _pick(rng, tuple(LOCATION_TIER_MULTIPLIERS), _LOCATION_SHARES, n)
    institute_tier = _pick(rng, tuple(INSTITUTE_TIER_MULTIPLIERS), _INSTITUTE_SHARES, n)
    prev_company = _pick(rng, tuple(PREV_COMPANY_MULTIPLIERS), _PREV_COMPANY_SHARES, n)
    education = _pick(rng, tuple(EDUCATION_MULTIPLIERS), _EDUCATION_SHARES, n)
    org_size = _pick(rng, tuple(ORG_SIZE_MULTIPLIERS), _ORG_SIZE_SHARES, n)
    remote = _pick(rng, tuple(REMOTE_MULTIPLIERS), _REMOTE_SHARES, n)
    employment = _pick(rng, tuple(EMPLOYMENT_TYPES), _EMPLOYMENT_SHARES, n)

    other_group = "male" if disadvantaged_group == "female" else "female"
    is_disadvantaged = rng.random(n) < share_disadvantaged
    gender = np.where(is_disadvantaged, disadvantaged_group, other_group)

    # ── the proxy: career breaks, correlated with group ───────────────
    gap_prob = np.where(is_disadvantaged, career_gap_prob_disadvantaged, career_gap_prob_other)
    has_gap = rng.random(n) < gap_prob
    gap_length = rng.gamma(_CAREER_GAP_SHAPE, _CAREER_GAP_SCALE, n).round()
    # A break cannot be longer than half a career — someone with 2 years of
    # experience has not taken a 5-year break. Freshers therefore have none.
    max_gap = np.minimum(years * 6.0, _MAX_CAREER_GAP_MONTHS)
    career_gap_months = np.where(has_gap, np.clip(gap_length, 1.0, max_gap), 0.0)
    career_gap_months = np.where(max_gap < 1.0, 0.0, career_gap_months)

    # ── what they get paid ────────────────────────────────────────────
    #
    # Built term by term in log space. Each line is one reason a salary is what
    # it is, and every line is a multiplier you can read off in percent.
    log_salary = (
        np.log(BASE_SALARY)
        + np.log(experience_multiplier(years))
        + np.log(_multipliers(role, ROLE_MULTIPLIERS))
        + np.log(_multipliers(location_tier, LOCATION_TIER_MULTIPLIERS))
        + np.log(
            institute_multiplier(years, _multipliers(institute_tier, INSTITUTE_TIER_MULTIPLIERS))
        )
        + np.log(_multipliers(prev_company, PREV_COMPANY_MULTIPLIERS))
        + np.log(_multipliers(education, EDUCATION_MULTIPLIERS))
        + np.log(_multipliers(org_size, ORG_SIZE_MULTIPLIERS))
        + np.log(_multipliers(remote, REMOTE_MULTIPLIERS))
        # The proxy's own effect on pay. Log-linear in months, so two years out
        # costs 0.93 ** 2, not 2 × 7%.
        + (career_gap_months / 12.0) * np.log(career_gap_penalty_per_year)
        # THE INJECTED UNFAIRNESS. One line. Nothing about this person's work
        # differs — this is the term the fairness audit has to find.
        + is_disadvantaged * np.log1p(-pay_gap)
        # Normal noise in log space = lognormal noise in rupees: a long right
        # tail, no negatives possible, and the spread scales with the salary.
        # A ±₹2L wobble is huge at ₹6L and trivial at ₹60L; ±20% is neither.
        + rng.normal(0.0, noise_sigma_log, n)
    )

    # Real offers come out in round numbers, so we round too. At ₹1,000 the
    # rounding is under 0.2% of even the smallest salary here — visible in the
    # data, invisible to any measurement made on it.
    salary = np.round(np.exp(log_salary), -3)

    # A safety net, not a modelling step: the schema's plausibility bounds are
    # the same ones used to clean the real survey. At the default settings this
    # should bind for essentially nobody — if it starts binding often, the
    # parameters above have drifted somewhere unrealistic and you want to know.
    salary = np.clip(salary, MIN_PLAUSIBLE_SALARY, MAX_PLAUSIBLE_SALARY)

    out = pd.DataFrame(
        {
            TARGET: salary,
            "years_experience": years.astype(int),
            "role": role,
            "education": education,
            "org_size": org_size,
            "remote": remote,
            "employment_type": employment,
            "source": Source.SYNTHETIC.value,
            # Optional schema columns — the ones the fairness work needs.
            "location_tier": location_tier.astype(int),
            "institute_tier": institute_tier,
            "prev_company_type": prev_company,
            "gender": gender,
            "career_gap_months": career_gap_months.astype(int),
            "level": _level_for(years),
        }
    )
    # Core columns first, in schema order, so every loader's output lines up.
    out = out[list(CORE_COLUMNS) + [c for c in out.columns if c not in CORE_COLUMNS]]

    truth = GroundTruth(
        n=n,
        seed=seed,
        pay_gap=pay_gap,
        disadvantaged_group=disadvantaged_group,
        share_disadvantaged=share_disadvantaged,
        noise_sigma_log=noise_sigma_log,
        career_gap_penalty_per_year=career_gap_penalty_per_year,
        career_gap_prob_disadvantaged=career_gap_prob_disadvantaged,
        career_gap_prob_other=career_gap_prob_other,
    )
    return out, truth
