"""The common table every data source is converted into.

Public survey data, synthetic data, and company data all look different on the
outside. Rather than teach the model three shapes, each loader converts its
source into *this* shape.

Why that matters: when company data finally arrives, you drop in one loader and
everything downstream — features, model, fairness audit, API — works unchanged,
because it was already tested against two other sources producing the same
columns. You will have tested the whole pipeline before the real data exists.
"""

from __future__ import annotations

from enum import StrEnum

# The target. Annual gross base salary in rupees — never take-home, never CTC.
TARGET = "salary_annual"


class Source(StrEnum):
    STACKOVERFLOW = "stackoverflow"
    SYNTHETIC = "synthetic"
    COMPANY = "company"


#: Columns every loader must produce. Missing values are allowed (as NaN) —
#: a source that genuinely lacks a field should say "unknown", not invent one.
CORE_COLUMNS: tuple[str, ...] = (
    TARGET,
    "years_experience",  # professional years, not years-since-first-line-of-code
    "role",  # job family: Backend, Data, DevOps...
    "education",
    "org_size",
    "remote",
    "employment_type",
    "source",  # which loader produced this row
)

#: Columns only some sources have. The model uses them when present.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "location_tier",  # 1 = Bangalore/Mumbai/Delhi-NCR/Hyderabad/Pune/Chennai
    "skills",
    "level",
    "institute_tier",
    "prev_company_type",  # product vs services — a large pay gap in Indian tech
    "prev_salary",
    "performance_rating",
    "event_date",
    "gender",  # fairness audit only
    "age_band",  # fairness audit only
)


# ── Cleaning thresholds ───────────────────────────────────────────────
#
# These are JUDGEMENT CALLS, not facts, so they live in one visible place
# rather than scattered through the code. Anyone reviewing the project can
# disagree with them here, in one line, instead of hunting.
#
# Justification, in Indian annual gross terms:
#   Below ₹1L/year is ₹8,300/month. Not a plausible full-time developer
#   salary — these rows are students, unemployed respondents, people who
#   entered monthly figures, or blanks typed as 0.
#   Above ₹2Cr/year is possible but vanishingly rare, and in this dataset
#   those rows are mostly joke entries and CTC confusion.
#
# Every row these rules drop gets COUNTED and REPORTED. Silent filtering is
# how a dataset quietly becomes a different dataset than you think it is.

MIN_PLAUSIBLE_SALARY = 100_000
MAX_PLAUSIBLE_SALARY = 20_000_000
