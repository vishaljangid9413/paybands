"""Loader for the Stack Overflow Developer Survey.

Filters to India, converts to the common schema, and — importantly — reports
exactly how many rows it dropped and why.

**Why the drop report matters.** Every cleaning rule quietly changes what your
dataset represents. Drop the zeros and you've excluded the unemployed. Drop the
top 0.5% and you've excluded the highest earners. Neither is wrong, but if you
don't count them you can't tell whether you cleaned a dataset or replaced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .schema import MAX_PLAUSIBLE_SALARY, MIN_PLAUSIBLE_SALARY, TARGET, Source

# Only the columns we need. The full file is 172 columns and 134MB — reading it
# all would use over a gigabyte of RAM for no reason.
_RAW_COLUMNS = [
    "Country",
    "Currency",
    "CompTotal",
    "WorkExp",
    "YearsCode",
    "DevType",
    "EdLevel",
    "OrgSize",
    "RemoteWork",
    "Employment",
    "Age",
    "LanguageHaveWorkedWith",
]


@dataclass
class LoadReport:
    """What the loader kept, what it dropped, and why.

    Printed after every load. If these numbers surprise you, the data changed
    or a rule is wrong — either way you want to know immediately.
    """

    total_responses: int = 0
    india_rows: int = 0
    inr_rows: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    final_rows: int = 0

    def __str__(self) -> str:
        lines = [
            "Stack Overflow 2025 → India",
            f"  survey responses           {self.total_responses:>7,}",
            f"  from India                 {self.india_rows:>7,}",
            f"  reporting salary in INR    {self.inr_rows:>7,}",
        ]
        for reason, n in self.dropped.items():
            share = n / self.inr_rows if self.inr_rows else 0
            lines.append(f"    − {reason:<23} {n:>7,}  ({share:.1%})")
        kept = self.final_rows / self.inr_rows if self.inr_rows else 0
        lines.append(
            f"  usable rows                {self.final_rows:>7,}  ({kept:.0%} of INR rows)"
        )
        return "\n".join(lines)


def _map_role(devtype: object) -> str:
    """Collapse free-text DevType into a small set of job families.

    The raw field has dozens of values and lets people pick several. A model
    given hundreds of rare categories learns noise, so we group. This mapping is
    a judgement call and deliberately crude — it's a starting point to improve,
    not a truth.
    """
    if not isinstance(devtype, str):
        return "unknown"
    d = devtype.lower()
    # Order matters: the first match wins, so more specific rules go first.
    for keyword, family in [
        ("data scientist", "Data Science"),
        ("machine learning", "Data Science"),
        ("data engineer", "Data Engineering"),
        ("data or business analyst", "Analytics"),
        ("devops", "DevOps"),
        ("site reliability", "DevOps"),
        ("cloud infrastructure", "DevOps"),
        ("security", "Security"),
        ("mobile", "Mobile"),
        ("full-stack", "Fullstack"),
        ("back-end", "Backend"),
        ("front-end", "Frontend"),
        ("embedded", "Embedded"),
        ("qa or test", "QA"),
        ("engineering manager", "Management"),
        ("student", "Student"),
        ("designer", "Design"),
    ]:
        if keyword in d:
            return family
    return "Other"


def load(path: str | Path, *, verbose: bool = True) -> tuple[pd.DataFrame, LoadReport]:
    """Load the survey, filter to Indian INR salaries, return the common schema."""
    report = LoadReport()

    raw = pd.read_csv(path, usecols=_RAW_COLUMNS, low_memory=False)
    report.total_responses = len(raw)

    india = raw[raw["Country"] == "India"]
    report.india_rows = len(india)

    # Why filter on currency too: ~70 Indian respondents report in USD. Those are
    # usually remote workers paid by foreign employers — a genuinely different
    # market. Mixing them in would inflate the band for local roles.
    inr = india[india["Currency"].astype(str).str.startswith("INR")].copy()
    report.inr_rows = len(inr)

    before = len(inr)
    inr = inr[inr["CompTotal"].notna()]
    report.dropped["no salary reported"] = before - len(inr)

    before = len(inr)
    inr = inr[inr["CompTotal"] >= MIN_PLAUSIBLE_SALARY]
    report.dropped[f"below ₹{MIN_PLAUSIBLE_SALARY:,}"] = before - len(inr)

    before = len(inr)
    inr = inr[inr["CompTotal"] <= MAX_PLAUSIBLE_SALARY]
    report.dropped[f"above ₹{MAX_PLAUSIBLE_SALARY:,}"] = before - len(inr)

    out = pd.DataFrame(
        {
            TARGET: inr["CompTotal"].astype(float),
            # WorkExp is professional experience. YearsCode counts from the first
            # line of code ever written, including school — a weaker signal, so
            # we prefer WorkExp and fall back only when it is missing.
            "years_experience": pd.to_numeric(inr["WorkExp"], errors="coerce").fillna(
                pd.to_numeric(inr["YearsCode"], errors="coerce")
            ),
            "role": inr["DevType"].map(_map_role),
            "education": inr["EdLevel"].fillna("unknown"),
            "org_size": inr["OrgSize"].fillna("unknown"),
            "remote": inr["RemoteWork"].fillna("unknown"),
            "employment_type": inr["Employment"].fillna("unknown"),
            "skills": inr["LanguageHaveWorkedWith"].fillna(""),
            "age_band": inr["Age"].fillna("unknown"),
            "source": Source.STACKOVERFLOW.value,
        }
    ).reset_index(drop=True)

    report.final_rows = len(out)
    if verbose:
        print(report)
    return out, report
