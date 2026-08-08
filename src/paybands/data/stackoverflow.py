"""Loader for the Stack Overflow Developer Survey.

Filters to India, converts to the common schema, and — importantly — reports
exactly how many rows it dropped and why.

**Why the drop report matters.** Every cleaning rule quietly changes what your
dataset represents. Drop the zeros and you've excluded the unemployed. Drop the
top 0.5% and you've excluded the highest earners. Neither is wrong, but if you
don't count them you can't tell whether you cleaned a dataset or replaced it.
"""

from __future__ import annotations

from collections.abc import Iterable
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


# ── Column names across survey years ──────────────────────────────────
#
# Stack Overflow renamed things repeatedly between 2019 and 2025. Every mapping
# below was verified by reading the header row of each downloaded CSV, not taken
# from documentation — the published schema files and the actual files disagree
# in places.
#
# The first name in each tuple that exists in the file wins; None means the year
# genuinely has no such column and the field becomes "unknown".
#
#                   2019/20         2021+
#   currency        CurrencySymbol  Currency
#   languages       LanguageWorkedWith
#                                   LanguageHaveWorkedWith
#   remote          WorkRemote      (absent in 2020 and 2021) RemoteWork from 2022
#   experience      YearsCodePro    ... through 2024; 2025 dropped it for WorkExp
#
# CompFreq exists 2019-2022, where CompTotal may be a weekly or monthly figure.
# From 2023 CompTotal is already annual. Getting this wrong would read a monthly
# salary as an annual one — a 12x error that looks like a plausible senior salary
# and would poison the band silently.
_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "currency": ("Currency", "CurrencySymbol"),
    "skills": ("LanguageHaveWorkedWith", "LanguageWorkedWith"),
    "remote": ("RemoteWork", "WorkRemote"),
    # YearsCodePro is asked as professional experience and is the better field.
    # WorkExp only exists 2022+ and is the ONLY option in 2025. Checked against
    # each other on 2022 and 2023 India rows, where both are present:
    # correlation 0.90, identical median (5.0 years), WorkExp running +0.40 years
    # higher on average, exactly equal in ~70% of rows. Close enough to pool,
    # and the offset is small next to the spread this model is trying to explain.
    "experience": ("YearsCodePro", "WorkExp", "YearsCode"),
}

#: How to annualise CompTotal when CompFreq says it is not yearly.
_FREQ_MULTIPLIER = {"yearly": 1.0, "monthly": 12.0, "weekly": 52.0}


def _pick(available: set[str], field_name: str) -> str | None:
    """First candidate column for `field_name` that this file actually has."""
    for name in _COLUMN_CANDIDATES[field_name]:
        if name in available:
            return name
    return None


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
    survey_year: int | None = None
    #: How many rows were reported per month/week and had to be annualised.
    #: Published because a silent 12x is the worst bug this loader could have.
    annualised: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Stack Overflow {self.survey_year or '?'} → India",
            f"  survey responses           {self.total_responses:>7,}",
            f"  from India                 {self.india_rows:>7,}",
            f"  reporting salary in INR    {self.inr_rows:>7,}",
        ]
        for reason, n in self.dropped.items():
            share = n / self.inr_rows if self.inr_rows else 0
            lines.append(f"    − {reason:<23} {n:>7,}  ({share:.1%})")
        for period, n in self.annualised.items():
            if n:
                lines.append(f"    ↑ {n:>7,} reported {period}, multiplied up to annual")
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
    """Load one survey year, filter to Indian INR salaries, return the common schema.

    Works on any year from 2019 to 2025. The column names differ across years —
    see `_COLUMN_CANDIDATES` — so the file's own header decides what gets read
    rather than a fixed list. A year missing a field entirely (2020 and 2021 have
    no remote-work question at all) yields "unknown" for it, which is the honest
    encoding: the model has a learned response to missing, and inventing a value
    would be worse than admitting the question was never asked.
    """
    path = Path(path)
    report = LoadReport()
    report.survey_year = _year_from_filename(path)

    header = set(pd.read_csv(path, nrows=0).columns)
    currency_col = _pick(header, "currency")
    skills_col = _pick(header, "skills")
    remote_col = _pick(header, "remote")
    experience_col = _pick(header, "experience")
    if currency_col is None or experience_col is None:
        raise ValueError(f"{path.name}: no currency or experience column — not a survey file?")

    wanted = [
        c
        for c in (
            "Country",
            currency_col,
            "CompTotal",
            "CompFreq",
            experience_col,
            "DevType",
            "EdLevel",
            "OrgSize",
            remote_col,
            "Employment",
            "Age",
            skills_col,
        )
        if c and c in header
    ]
    raw = pd.read_csv(path, usecols=wanted, low_memory=False)
    report.total_responses = len(raw)

    india = raw[raw["Country"] == "India"]
    report.india_rows = len(india)

    # Why filter on currency too: ~70 Indian respondents report in USD. Those are
    # usually remote workers paid by foreign employers — a genuinely different
    # market. Mixing them in would inflate the band for local roles.
    inr = india[india[currency_col].astype(str).str.contains("INR", na=False)].copy()
    report.inr_rows = len(inr)

    before = len(inr)
    inr = inr[inr["CompTotal"].notna()]
    report.dropped["no salary reported"] = before - len(inr)

    # ANNUALISE BEFORE FILTERING, not after. 2019-2022 let respondents answer
    # weekly or monthly; a ₹90,000 monthly salary is ₹10.8L a year, but read as
    # an annual figure it looks like a low-but-plausible salary and sails past
    # every range check. That is a 12x error that never announces itself.
    salary = inr["CompTotal"].astype(float)
    if "CompFreq" in inr.columns:
        freq = inr["CompFreq"].astype(str).str.strip().str.lower()
        multiplier = freq.map(_FREQ_MULTIPLIER)
        report.dropped["unreadable pay frequency"] = int(multiplier.isna().sum())
        salary = salary * multiplier
        report.annualised = {
            period: int((freq == period).sum()) for period in ("monthly", "weekly")
        }
    inr = inr.assign(_annual=salary)
    inr = inr[inr["_annual"].notna()]

    before = len(inr)
    inr = inr[inr["_annual"] >= MIN_PLAUSIBLE_SALARY]
    report.dropped[f"below ₹{MIN_PLAUSIBLE_SALARY:,}"] = before - len(inr)

    before = len(inr)
    inr = inr[inr["_annual"] <= MAX_PLAUSIBLE_SALARY]
    report.dropped[f"above ₹{MAX_PLAUSIBLE_SALARY:,}"] = before - len(inr)

    missing = pd.Series("unknown", index=inr.index)
    out = pd.DataFrame(
        {
            TARGET: inr["_annual"],
            # YearsCodePro where the year has it. 2025 dropped it, leaving WorkExp
            # — checked against each other on the years carrying both: r = 0.90,
            # same median, WorkExp +0.40 years on average. See _COLUMN_CANDIDATES.
            "years_experience": pd.to_numeric(inr[experience_col], errors="coerce"),
            "role": inr["DevType"].map(_map_role),
            "education": inr["EdLevel"].fillna("unknown"),
            "org_size": inr["OrgSize"].fillna("unknown"),
            "remote": (inr[remote_col] if remote_col else missing).fillna("unknown"),
            "employment_type": inr["Employment"].fillna("unknown"),
            "skills": (inr[skills_col] if skills_col else "").fillna(""),
            "age_band": inr["Age"].fillna("unknown"),
            # Not decoration. Indian tech pay in this survey roughly doubled
            # between 2019 and 2024, so pooling years without letting the model
            # see which year a row came from folds real wage growth into the
            # spread and makes the band WIDER. See docs/findings.md.
            "survey_year": report.survey_year,
            "source": Source.STACKOVERFLOW.value,
        }
    ).reset_index(drop=True)

    report.final_rows = len(out)
    if verbose:
        print(report)
    return out, report


def _year_from_filename(path: Path) -> int:
    """`so_2023_raw.csv` -> 2023. Raises rather than guessing."""
    for token in path.stem.replace("-", "_").split("_"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    raise ValueError(
        f"cannot tell which survey year {path.name} is — expected e.g. so_2023_raw.csv"
    )


def load_years(
    paths: Iterable[str | Path], *, verbose: bool = True
) -> tuple[pd.DataFrame, list[LoadReport]]:
    """Load several survey years and stack them into one frame.

    Files are read one at a time and reduced to their India rows before the next
    is opened. The seven raw CSVs total 938MB; holding them all in memory at once
    would be pointless when the part we keep is a few thousand rows.

    **This does not adjust for inflation.** `survey_year` is carried as a column
    so the model can learn the trend itself, which is preferable to applying a
    deflator this project cannot verify. Anyone pooling these years without
    handling that column will get wider bands, not narrower ones.
    """
    frames, reports = [], []
    for path in paths:
        frame, report = load(path, verbose=verbose)
        frames.append(frame)
        reports.append(report)
        if verbose:
            print()

    pooled = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"Pooled {len(reports)} survey years → {len(pooled):,} usable Indian rows")
        by_year = pooled.groupby("survey_year")[TARGET].agg(["size", "median"])
        for year, row in by_year.iterrows():
            print(f"  {year}  {int(row['size']):>6,} rows   median ₹{row['median']:>12,.0f}")
    return pooled, reports
