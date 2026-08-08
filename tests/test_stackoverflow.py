"""Tests for the Stack Overflow loader, across seven survey years.

Every test builds a tiny CSV by hand rather than reading `data/public/`. Those
files are 940MB, gitignored, and absent in CI — a test that needed them would
pass on a laptop and fail everywhere else. Hand-made frames also let a single
row carry exactly the property under test, which a real survey row never does.

The annualisation tests are the important ones. Between 2019 and 2022 the survey
let people answer weekly or monthly, so `CompTotal` is not always a yearly
figure. Reading a ₹90,000 monthly salary as annual is a 12x error that produces
a low-but-entirely-plausible number, passes every range check, and quietly drags
the whole band down. Nothing else in this file could catch that.
"""

from __future__ import annotations

import pandas as pd
import pytest

from paybands.data.schema import TARGET
from paybands.data.stackoverflow import load, load_years

#: Columns a 2022-era file has: CompFreq present, `Currency`, `RemoteWork`,
#: `YearsCodePro` and `LanguageHaveWorkedWith`.
_MODERN = {
    "Country": "India",
    "Currency": "INR\tIndian rupee",
    "CompTotal": 1_500_000,
    "CompFreq": "Yearly",
    "YearsCodePro": 6,
    "DevType": "Developer, back-end",
    "EdLevel": "Bachelor's degree",
    "OrgSize": "100 to 499 employees",
    "RemoteWork": "Remote",
    "Employment": "Employed, full-time",
    "Age": "25-34 years old",
    "LanguageHaveWorkedWith": "Python;SQL",
}

#: A 2019-era file: `CurrencySymbol`, `WorkRemote`, `LanguageWorkedWith`.
_LEGACY = {
    "Country": "India",
    "CurrencySymbol": "INR",
    "CompTotal": 1_500_000,
    "CompFreq": "Yearly",
    "YearsCodePro": 6,
    "DevType": "Developer, back-end",
    "EdLevel": "Bachelor's degree",
    "OrgSize": "100 to 499 employees",
    "WorkRemote": "Never",
    "Employment": "Employed full-time",
    "Age": "25-34 years old",
    "LanguageWorkedWith": "Python;SQL",
}


def write(tmp_path, year: int, rows: list[dict]):
    path = tmp_path / f"so_{year}_raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ── the 12x bug ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("freq", "reported", "expected"),
    [
        ("Yearly", 1_500_000, 1_500_000),
        ("Monthly", 90_000, 1_080_000),  # the payslip case: 90k/month is 10.8L/year
        ("Weekly", 20_000, 1_040_000),
    ],
)
def test_pay_frequency_is_annualised(tmp_path, freq, reported, expected):
    path = write(tmp_path, 2022, [_MODERN | {"CompFreq": freq, "CompTotal": reported}])
    df, _ = load(path, verbose=False)
    assert df[TARGET].iloc[0] == pytest.approx(expected)


def test_monthly_salaries_are_annualised_before_the_range_filter(tmp_path):
    """Order of operations, and it is not a detail.

    ₹9,000 a month is ₹1,08,000 a year — above the ₹1,00,000 floor and a real
    junior salary. Filter first and it is discarded as implausible; annualise
    first and it is kept. Doing these in the wrong order silently deletes the
    low-paid, which is a bias with a direction, not a rounding error.
    """
    path = write(tmp_path, 2022, [_MODERN | {"CompFreq": "Monthly", "CompTotal": 9_000}])
    df, _ = load(path, verbose=False)
    assert len(df) == 1
    assert df[TARGET].iloc[0] == pytest.approx(108_000)


def test_unreadable_pay_frequency_is_dropped_and_counted(tmp_path):
    """A frequency we cannot parse must not be assumed yearly.

    Assuming would turn one unreadable row into a wrong number rather than a
    missing one, and wrong numbers do not announce themselves.
    """
    path = write(tmp_path, 2022, [_MODERN, _MODERN | {"CompFreq": "Fortnightly"}])
    df, report = load(path, verbose=False)
    assert len(df) == 1
    assert report.dropped["unreadable pay frequency"] == 1


def test_years_without_compfreq_are_left_alone(tmp_path):
    """From 2023 CompTotal is already annual and there is no CompFreq column.

    Multiplying here would be the same 12x bug in the other direction.
    """
    row = {k: v for k, v in _MODERN.items() if k != "CompFreq"}
    path = write(tmp_path, 2024, [row])
    df, report = load(path, verbose=False)
    assert df[TARGET].iloc[0] == pytest.approx(1_500_000)
    assert report.annualised == {}


# ── column names that moved between years ─────────────────────────────


def test_legacy_column_names_are_resolved(tmp_path):
    """2019 spells three columns differently. The file's header decides."""
    df, _ = load(write(tmp_path, 2019, [_LEGACY]), verbose=False)
    assert len(df) == 1
    assert df["skills"].iloc[0] == "Python;SQL"  # LanguageWorkedWith
    assert df["remote"].iloc[0] == "Never"  # WorkRemote
    assert df[TARGET].iloc[0] == pytest.approx(1_500_000)  # CurrencySymbol matched


def test_a_year_missing_remote_entirely_yields_unknown(tmp_path):
    """2020 and 2021 never asked about remote work.

    "unknown" is the honest encoding. Inventing a value — defaulting to
    "In-person", say — would be a claim the survey never made.
    """
    row = {k: v for k, v in _MODERN.items() if k != "RemoteWork"}
    df, _ = load(write(tmp_path, 2021, [row]), verbose=False)
    assert df["remote"].iloc[0] == "unknown"


def test_workexp_is_used_when_yearscodepro_is_absent(tmp_path):
    """2025 dropped YearsCodePro. WorkExp is the only professional-experience
    field left, and the loader must fall back rather than silently emit NaN."""
    row = {k: v for k, v in _MODERN.items() if k != "YearsCodePro"} | {"WorkExp": 7}
    df, _ = load(write(tmp_path, 2025, [row]), verbose=False)
    assert df["years_experience"].iloc[0] == 7


def test_yearscodepro_wins_when_both_are_present(tmp_path):
    """2022-2024 carry both. YearsCodePro is the question actually about
    professional experience, so it takes precedence."""
    df, _ = load(write(tmp_path, 2023, [_MODERN | {"WorkExp": 99}]), verbose=False)
    assert df["years_experience"].iloc[0] == 6


# ── filtering and provenance ──────────────────────────────────────────


def test_non_india_and_non_inr_rows_are_excluded(tmp_path):
    rows = [
        _MODERN,
        _MODERN | {"Country": "Germany"},
        _MODERN | {"Currency": "USD\tUnited States dollar"},
    ]
    df, report = load(write(tmp_path, 2022, rows), verbose=False)
    assert len(df) == 1
    assert report.india_rows == 2
    assert report.inr_rows == 1


def test_survey_year_comes_from_the_filename(tmp_path):
    df, report = load(write(tmp_path, 2020, [_LEGACY]), verbose=False)
    assert report.survey_year == 2020
    assert df["survey_year"].iloc[0] == 2020


def test_an_unnamed_year_raises_rather_than_guessing(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame([_MODERN]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="which survey year"):
        load(path, verbose=False)


def test_load_years_stacks_and_labels_every_row(tmp_path):
    """Pooling is only safe if each row still says which market it came from.

    Indian salaries in this survey roughly doubled over these years. A pooled
    frame that forgot the year would present 2019 pay as current.
    """
    paths = [
        write(tmp_path, 2019, [_LEGACY, _LEGACY]),
        write(tmp_path, 2024, [{k: v for k, v in _MODERN.items() if k != "CompFreq"}]),
    ]
    pooled, reports = load_years(paths, verbose=False)
    assert len(pooled) == 3
    assert len(reports) == 2
    assert sorted(pooled["survey_year"].unique()) == [2019, 2024]
    assert pooled["survey_year"].notna().all()
