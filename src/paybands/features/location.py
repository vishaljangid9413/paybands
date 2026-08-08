"""City → location tier, for the Indian market.

## Why tiers instead of city names

Raw city is a high-cardinality column: hundreds of values, most appearing once or
twice. A model given "Bhilai" with three rows learns noise, and at prediction
time it meets cities it has never seen at all. What actually drives pay is not
the city's identity but its *tier* — the size of its tech market and its cost of
living. Bangalore, Hyderabad and Pune behave alike; Indore and Coimbatore behave
alike; and a tier is three values instead of three hundred.

This is the same reasoning `stackoverflow.py` applies to job titles: group the
long tail into a handful of families you can actually learn from.

## Important: this does not apply to the Stack Overflow data

**The survey has no city column.** It records country, and we already filter to
India. So this module contributes nothing to Phase 1.

It is written now anyway, and tested now, because of the pipeline promise in
`docs/design.md`: all three sources produce the same table shape, so when company
data finally arrives — with a real `city` column — the feature that consumes it
already exists and already has tests. The alternative is writing brand-new
untested code on the same day you first touch real salary data, which is the
worst possible day to be debugging.

Synthetic data (Phase 2) will emit a `city`, so this gets exercised well before
then.

## Nothing here is learned, so nothing here can leak

The tier lists below are **hand-written domain knowledge**, not statistics. They
do not change when the training split changes, so `fit` is a no-op and no
information can cross from test to train. Contrast `skills.py`, where the
vocabulary genuinely is learned. Knowing which of your transformers learn and
which don't is exactly the audit you want to be able to do at a glance.

A consequence worth stating: the lists encode *2026 opinions*. If Jaipur's tech
market grows, this file needs editing — it will not update itself from data. That
is a trade: hand-written rules are auditable and stable, but they go stale
silently. Revisit when company data shows tier-2 salaries closing the gap.
"""

from __future__ import annotations

import re

import pandas as pd

#: Tier 1 — the metros with deep tech markets and the highest pay.
TIER_1_CITIES: frozenset[str] = frozenset(
    {
        "bangalore",
        "mumbai",
        "delhi",
        "gurgaon",
        "noida",
        "hyderabad",
        "pune",
        "chennai",
    }
)

#: Tier 2 — real tech presence, noticeably lower pay and cost of living.
TIER_2_CITIES: frozenset[str] = frozenset(
    {
        "ahmedabad",
        "kolkata",
        "jaipur",
        "indore",
        "kochi",
        "coimbatore",
        "chandigarh",
        "bhubaneswar",
        "nagpur",
        "lucknow",
        "thiruvananthapuram",
        "mysore",
        "visakhapatnam",
        "vadodara",
        "surat",
        "nashik",
        "mangalore",
        "madurai",
        "bhopal",
        "dehradun",
        "guwahati",
        "patna",
        "raipur",
        "ranchi",
        "goa",
    }
)

#: Everything else. Not a value judgement about the city — just "we have no
#: reason to think its tech salaries track the metros".
DEFAULT_TIER = 3

#: Spelling variants, renamings, colloquialisms and airport codes, all mapped to
#: the canonical name. This map is the whole reason the module exists: real
#: HR systems contain "Bengaluru", "BLR", "bangalore " and "Bangalore, KA" in the
#: same column, and every one of them is the same city. Left unhandled, three of
#: those four silently become tier 3 and the model learns that Bangalore is cheap.
CITY_ALIASES: dict[str, str] = {
    # Bangalore
    "bengaluru": "bangalore",
    "blr": "bangalore",
    "banglore": "bangalore",  # extremely common misspelling
    "bangaluru": "bangalore",
    # Mumbai
    "bombay": "mumbai",
    "navi mumbai": "mumbai",
    "thane": "mumbai",
    "bom": "mumbai",
    # Delhi-NCR — one labour market, four municipalities.
    "new delhi": "delhi",
    "delhi ncr": "delhi",
    "ncr": "delhi",
    "national capital region": "delhi",
    "del": "delhi",
    "gurugram": "gurgaon",
    "ggn": "gurgaon",
    "greater noida": "noida",
    "faridabad": "delhi",
    "ghaziabad": "delhi",
    # Hyderabad
    "hyd": "hyderabad",
    "secunderabad": "hyderabad",
    "cyberabad": "hyderabad",
    # Chennai
    "madras": "chennai",
    "maa": "chennai",
    # Pune
    "poona": "pune",
    "pimpri chinchwad": "pune",
    # Kolkata
    "calcutta": "kolkata",
    "ccu": "kolkata",
    # Kochi
    "cochin": "kochi",
    "ernakulam": "kochi",
    # Others
    "trivandrum": "thiruvananthapuram",
    "mysuru": "mysore",
    "vizag": "visakhapatnam",
    "baroda": "vadodara",
    "bhubaneshwar": "bhubaneswar",
    "mangaluru": "mangalore",
    "pondicherry": "puducherry",
    "gandhinagar": "ahmedabad",
    "mohali": "chandigarh",
    "panchkula": "chandigarh",
}

# Strips anything that isn't a letter or a space: punctuation, digits, the
# stray "-" in "Delhi-NCR". Run after lowercasing.
_NON_LETTERS = re.compile(r"[^a-z ]+")


def normalise_city(value: object) -> str | None:
    """Reduce a free-text city to a canonical lowercase name.

    Handles case, surrounding whitespace, punctuation, "City, State" forms, and
    the alias table. Returns `None` for missing or unusable input — see
    :func:`city_tier` for why `None` is not the same as tier 3.

        >>> normalise_city("  Bengaluru ")
        'bangalore'
        >>> normalise_city("Delhi-NCR")
        'delhi'
        >>> normalise_city("Gurugram, Haryana")
        'gurgaon'
    """
    if not isinstance(value, str):
        return None

    # "Gurugram, Haryana" and "Pune, India" — keep the part before the comma,
    # which is the city in every form we've seen.
    head = value.split(",")[0]
    cleaned = _NON_LETTERS.sub(" ", head.lower())
    cleaned = " ".join(cleaned.split())  # collapse runs of whitespace
    if not cleaned:
        return None

    return CITY_ALIASES.get(cleaned, cleaned)


def city_tier(value: object) -> float:
    """Map a city name to tier 1, 2, or 3. Returns NaN when the city is missing.

    The distinction between "missing" and "tier 3" is deliberate:

    * `"Bhilai"` → **3**. We recognise it as a real place that isn't a tech
      metro. That's a genuine claim about the city.
    * `None` / `""` / NaN → **NaN**. We were not told where this person works.
      Calling that tier 3 would invent a fact — and specifically it would tell
      the model "this person is in a low-paying city", which would drag the
      predicted band down for everyone whose record was simply incomplete.

    NaN is not a problem to be fixed here. LightGBM routes NaN down whichever
    branch fits the data best, which is a better answer than any guess we could
    make in this function.
    """
    canonical = normalise_city(value)
    if canonical is None:
        return float("nan")
    if canonical in TIER_1_CITIES:
        return 1.0
    if canonical in TIER_2_CITIES:
        return 2.0
    return float(DEFAULT_TIER)


class LocationFeatures:
    """Derive `location_tier` from a `city` column.

    If the frame already carries a `location_tier` (a company export might, or a
    synthetic generator that works in tiers directly), that column is used as-is
    and no city mapping happens. Same output column either way, which is the
    point: everything downstream stops caring where the tier came from.
    """

    def __init__(self, city_column: str = "city", tier_column: str = "location_tier") -> None:
        self.city_column = city_column
        self.tier_column = tier_column
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> LocationFeatures:
        """No-op, on purpose — the tier lists are hand-written, not learned."""
        del df  # nothing is learned; the argument exists for interface symmetry
        self.feature_names_ = [self.tier_column]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a one-column frame of tiers as floats, NaN where unknown."""
        if not self.feature_names_:
            raise RuntimeError("call fit() before transform()")

        out = pd.DataFrame(index=df.index)
        if self.city_column in df.columns:
            out[self.tier_column] = [city_tier(v) for v in df[self.city_column]]
        elif self.tier_column in df.columns:
            out[self.tier_column] = pd.to_numeric(df[self.tier_column], errors="coerce")
        else:
            # Neither column present — the Stack Overflow case. An all-NaN column
            # keeps the feature matrix the same width for every source, so a
            # model trained on synthetic data can still score survey rows.
            out[self.tier_column] = float("nan")
        return out.astype({self.tier_column: "float64"})
