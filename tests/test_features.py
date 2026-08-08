"""Tests for feature engineering.

The most important test in this file is `test_fit_on_train_does_not_see_test_skills`.
Everything else checks the transforms behave sensibly; that one checks the
**discipline** — that fitting on the training split really does keep test-set
information out of the model's inputs.

If that test ever starts failing, do not relax it. It is guarding the mistake
that makes a model look excellent in validation and fail in production.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paybands.data.schema import TARGET
from paybands.features import (
    ExperienceFeatures,
    FeatureBuilder,
    LocationFeatures,
    SkillFeatures,
    bucket_years,
    city_tier,
    clean_years,
    log_years,
    normalise_city,
    parse_skills,
)

SURVEY_CSV = Path("data/public/so_2025_raw.csv")


@pytest.fixture
def train_df() -> pd.DataFrame:
    """A small, hand-written training split."""
    return pd.DataFrame(
        {
            TARGET: [600_000, 1_200_000, 2_200_000, 3_000_000, 4_000_000, 900_000],
            "years_experience": [1, 4, 7, 11, 18, 2],
            "role": ["Backend", "Backend", "Data Science", "Backend", "Management", "QA"],
            "education": ["Bachelors", "Bachelors", "Masters", "Bachelors", "Masters", "Bachelors"],
            "org_size": ["100 to 499", "500 to 999", "100 to 499", "10,000+", "10,000+", "2 to 9"],
            "remote": ["Remote", "Hybrid", "Hybrid", "In-person", "Hybrid", "Remote"],
            "employment_type": ["Employed, full-time"] * 6,
            "skills": [
                "Python;SQL",
                "Python;JavaScript;SQL",
                "Python;SQL;Go",
                "Java;SQL",
                "Java;Python",
                "JavaScript",
            ],
            "city": ["Bengaluru", "Pune", "Hyderabad", "Indore", "Mumbai", "Bhilai"],
            "gender": ["M", "F", "F", "M", "M", "F"],
            "source": ["synthetic"] * 6,
        }
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    """A test split containing things the training split never had: a new skill
    (`zig`), a new role (`Security`), a new city, and a missing experience."""
    return pd.DataFrame(
        {
            TARGET: [1_500_000, 2_500_000, 800_000],
            "years_experience": [5, np.nan, 3],
            "role": ["Backend", "Security", "Data Science"],
            "education": ["Bachelors", "Doctorate", "Masters"],
            "org_size": ["100 to 499", "10,000+", "2 to 9"],
            "remote": ["Hybrid", "Remote", "Remote"],
            "employment_type": ["Employed, full-time"] * 3,
            "skills": ["Python;Zig", "Zig;Rust;Elixir", "SQL"],
            "city": ["Bangalore ", "Kota", None],
            "gender": ["F", "M", "F"],
            "source": ["synthetic"] * 3,
        }
    )


# ══════════════════════════════════════════════ leakage — the point of the file


def test_fit_on_train_does_not_see_test_skills(train_df, test_df):
    """Fit on train, transform test: no test-only skill may enter the vocabulary.

    `zig`, `rust` and `elixir` appear *only* in the test split. If any of them
    turned up in `vocabulary_`, it would mean the fitted state was influenced by
    rows the model is supposed to be evaluated on — and the evaluation would be
    measuring memory rather than generalisation.
    """
    skills = SkillFeatures(top_n=10).fit(train_df)

    test_only = {"zig", "rust", "elixir"}
    assert test_only.isdisjoint(skills.vocabulary_)

    # Transforming the test set must not retroactively grow the vocabulary.
    before = list(skills.vocabulary_)
    out = skills.transform(test_df)
    assert skills.vocabulary_ == before
    assert "skill_zig" not in out.columns


def test_builder_fit_on_train_freezes_categories(train_df, test_df):
    """The same guarantee at the builder level, for category levels."""
    builder = FeatureBuilder().fit(train_df)

    # "Security" and "Doctorate" exist only in the test split.
    assert "Security" not in builder.category_levels_["role"]
    assert "Doctorate" not in builder.category_levels_["education"]

    x_test = builder.transform(test_df)
    assert list(x_test["role"].cat.categories) == builder.category_levels_["role"]


def test_fitting_on_everything_would_leak(train_df, test_df):
    """A demonstration, not just an assertion.

    Fit on the concatenation of train and test — the tempting shortcut — and
    test-only skills appear in the vocabulary. This test *asserts the bug exists*
    when you do it that way, so the contrast with the test above is explicit.
    """
    leaky = SkillFeatures(top_n=30).fit(pd.concat([train_df, test_df], ignore_index=True))
    assert "zig" in leaky.vocabulary_  # ← information from the test set, in the model's inputs


def test_target_never_becomes_a_feature(train_df):
    """If the target leaks into X, the model scores near-perfectly and is useless."""
    x = FeatureBuilder().fit_transform(train_df)
    assert TARGET not in x.columns


def test_fairness_columns_never_become_features(train_df):
    """`gender` is carried for the audit, never fed to the model."""
    x = FeatureBuilder().fit_transform(train_df)
    assert "gender" not in x.columns
    assert "source" not in x.columns


def test_prev_salary_excluded_by_default_and_opt_in(train_df):
    """The previous-salary trap (design.md §4.1) is off by default, and the
    Phase-4 comparison is a one-flag change."""
    df = train_df.assign(prev_salary=[5e5, 9e5, 1.8e6, 2.5e6, 3.2e6, 7e5])

    assert "prev_salary" not in FeatureBuilder().fit_transform(df).columns
    assert "prev_salary" in FeatureBuilder(include_prev_salary=True).fit_transform(df).columns


# ══════════════════════════════════════════════ unseen values must not crash


def test_unseen_category_becomes_nan_not_crash(train_df, test_df):
    """A role the model never trained on is NaN — LightGBM's "unknown" — and the
    other rows in the same frame are unaffected."""
    builder = FeatureBuilder().fit(train_df)
    x = builder.transform(test_df)

    assert pd.isna(x["role"].iloc[1])  # "Security", unseen
    assert x["role"].iloc[0] == "Backend"  # seen, still mapped correctly


def test_unseen_skill_is_ignored_but_still_counted(train_df, test_df):
    """No column for `zig`, but `n_skills` still knows the person listed three."""
    skills = SkillFeatures(top_n=10).fit(train_df)
    out = skills.transform(test_df)

    assert out["n_skills"].tolist() == [2, 3, 1]
    assert out["skill_python"].tolist() == [1, 0, 0]


def test_transform_works_on_a_frame_missing_columns_entirely(train_df):
    """Stack Overflow rows have no `city`. The matrix must stay the same width so
    a model trained on one source can score another."""
    builder = FeatureBuilder().fit(train_df)
    x = builder.transform(train_df.drop(columns=["city", "skills"]))

    assert list(x.columns) == builder.feature_names_
    assert x["location_tier"].isna().all()
    assert (x["n_skills"] == 0).all()


def test_transform_before_fit_raises(train_df):
    unfitted = (FeatureBuilder(), SkillFeatures(), ExperienceFeatures(), LocationFeatures())
    for transformer in unfitted:
        with pytest.raises(RuntimeError, match="fit"):
            transformer.transform(train_df)


# ══════════════════════════════════════════════ experience


def test_log_is_monotonic_in_years():
    """More experience must never produce a smaller feature value — otherwise the
    transform has scrambled the ordering the model depends on."""
    years = pd.Series([0, 1, 2, 3, 5, 8, 13, 21, 40])
    values = log_years(years).tolist()
    assert values == sorted(values)
    # Strictly increasing, not merely non-decreasing: ties would mean two
    # different levels of experience became indistinguishable to the model.
    pairs = list(zip(values, values[1:], strict=False))
    assert all(later > earlier for earlier, later in pairs)


def test_log1p_handles_the_fresher_case():
    """`log(0)` is −inf and would poison the column. `log1p(0)` is 0."""
    assert log_years(pd.Series([0.0])).iloc[0] == 0.0
    assert np.isfinite(log_years(pd.Series([0.0]))).all()


def test_log_compresses_late_years_more_than_early_ones():
    """The whole reason for the transform: year 1→3 must move the feature more
    than year 18→20 does, matching the salary curve in the roadmap."""
    v = log_years(pd.Series([1.0, 3.0, 18.0, 20.0])).tolist()
    early_gain = v[1] - v[0]
    late_gain = v[3] - v[2]
    assert early_gain > 5 * late_gain


@pytest.mark.parametrize("absurd", [-1, -30, 51, 120, 2005])
def test_absurd_experience_becomes_nan_not_clipped(absurd):
    """A typo is unknown data, not a confident claim about a 120-year career."""
    assert math.isnan(clean_years(pd.Series([absurd])).iloc[0])


def test_missing_experience_stays_missing():
    """NaN in, NaN out — through every derived column. No silent zero-filling."""
    exp = ExperienceFeatures().fit(pd.DataFrame({"years_experience": [1.0]}))
    out = exp.transform(pd.DataFrame({"years_experience": [np.nan, "not a number", 6.0]}))

    assert out["years_experience"].isna().tolist() == [True, True, False]
    assert out["experience_log"].isna().tolist() == [True, True, False]
    assert out["experience_sqrt"].isna().tolist() == [True, True, False]
    assert out["experience_bucket"].isna().tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("years", "expected"),
    [
        (0, "0-1"),
        (1.5, "0-1"),
        (2, "2-3"),  # left-closed: exactly 2 belongs to the second band
        (5, "4-5"),
        (8, "6-8"),
        (12, "9-12"),
        (20, "13-20"),
        (21, "20+"),
        (45, "20+"),
    ],
)
def test_buckets_follow_the_roadmap_bands(years, expected):
    assert bucket_years(pd.Series([years])).iloc[0] == expected


def test_buckets_are_ordered_and_complete_even_on_a_tiny_frame():
    """Three juniors must still produce all seven categories, or the test matrix
    would not line up with the training matrix."""
    out = bucket_years(pd.Series([0.0, 1.0, 1.5]))
    assert list(out.cat.categories) == ["0-1", "2-3", "4-5", "6-8", "9-12", "13-20", "20+"]
    assert out.cat.ordered


# ══════════════════════════════════════════════ skills parsing


def test_parse_skills_handles_the_mess_real_data_contains():
    assert parse_skills("Python;SQL") == ["python", "sql"]
    assert parse_skills(" Python ; ; SQL ") == ["python", "sql"]
    assert parse_skills("") == []
    assert parse_skills(np.nan) == []
    assert parse_skills(None) == []


def test_vocabulary_is_ordered_by_frequency_then_deterministic(train_df):
    skills = SkillFeatures(top_n=3).fit(train_df)
    # python 4, sql 4, javascript 2, java 2, go 1 → ties broken alphabetically.
    assert skills.vocabulary_ == ["python", "sql", "java"]
    assert skills.skill_counts_["python"] == 4


def test_skill_flags_are_binary_and_named_safely(train_df):
    df = train_df.assign(skills=["C++;C#" for _ in range(len(train_df))])
    out = SkillFeatures(top_n=5).fit(df).transform(df)
    assert "skill_c++" in out.columns  # `+` and `#` are safe for LightGBM
    assert set(out["skill_c++"].unique()) == {1}


# ══════════════════════════════════════════════ location


@pytest.mark.parametrize(
    "variant",
    ["Bangalore", "bengaluru", "Bangalore ", "  BLR", "BENGALURU", "Banglore", "Bangalore, KA"],
)
def test_bangalore_variants_all_map_to_tier_1(variant):
    """Real HR exports contain every one of these spellings in one column. Miss
    them and the model learns that Bangalore is a cheap city."""
    assert city_tier(variant) == 1


@pytest.mark.parametrize(
    ("city", "tier"),
    [
        ("Gurugram", 1),
        ("gurgaon", 1),
        ("Delhi-NCR", 1),
        ("New Delhi", 1),
        ("Navi Mumbai", 1),
        ("Bombay", 1),
        ("HYD", 1),
        ("Madras", 1),
        ("Indore", 2),
        ("Calcutta", 2),
        ("Cochin", 2),
        ("bhubaneshwar", 2),
        ("Vizag", 2),
        ("Bhilai", 3),
        ("Some Village", 3),
    ],
)
def test_city_tiers(city, tier):
    assert city_tier(city) == tier


def test_missing_city_is_nan_not_tier_3():
    """ "We weren't told" is different from "we know it's a small town". Calling
    the first one tier 3 would push predicted bands down for everyone whose
    record was merely incomplete."""
    assert math.isnan(city_tier(None))
    assert math.isnan(city_tier(""))
    assert math.isnan(city_tier(np.nan))
    assert math.isnan(city_tier("   "))


def test_normalise_city_strips_state_suffix_and_punctuation():
    assert normalise_city("Gurugram, Haryana") == "gurgaon"
    assert normalise_city("Pune - 411001") == "pune"


def test_location_uses_existing_tier_column_when_there_is_no_city():
    """Company exports may already store a tier. Same output either way."""
    df = pd.DataFrame({"location_tier": [1, 2, 3]})
    out = LocationFeatures().fit(df).transform(df)
    assert out["location_tier"].tolist() == [1.0, 2.0, 3.0]


# ══════════════════════════════════════════════ builder contract


def test_row_count_and_index_are_preserved(train_df, test_df):
    """A transform that drops rows silently desynchronises X from y."""
    builder = FeatureBuilder().fit(train_df)
    for df in (train_df, test_df, train_df.iloc[2:4]):
        out = builder.transform(df)
        assert len(out) == len(df)
        assert out.index.equals(df.index)


def test_feature_names_matches_transform_output(train_df, test_df):
    builder = FeatureBuilder().fit(train_df)
    assert list(builder.transform(train_df).columns) == builder.feature_names_
    assert list(builder.transform(test_df).columns) == builder.feature_names_


def test_missing_values_are_not_filled_with_zero(train_df, test_df):
    """LightGBM handles NaN natively. A zero would be a claim that the value IS
    zero — "fresher", "tier 1 city" — which is a lie about the data."""
    builder = FeatureBuilder().fit(train_df)
    x = builder.transform(test_df)

    # Row 1 has no experience recorded and an unknown city.
    assert x["years_experience"].isna().iloc[1]
    assert x["experience_log"].isna().iloc[1]
    assert pd.isna(x["experience_bucket"].iloc[1])
    assert x["location_tier"].isna().iloc[2]  # city was None

    assert (x["years_experience"] == 0).sum() == 0


def test_categoricals_stay_category_dtype_not_one_hot(train_df):
    """One-hot on high-cardinality columns makes trees split badly — see
    builder.py Decision 1. So the dtype must survive to the model."""
    builder = FeatureBuilder().fit(train_df)
    x = builder.transform(train_df)

    for col in builder.categorical_features_:
        assert isinstance(x[col].dtype, pd.CategoricalDtype), col

    # And no dummy columns crept in.
    assert not any(c.startswith("role_") for c in x.columns)


def test_output_is_numeric_or_categorical_only(train_df):
    """LightGBM cannot consume object/string columns. Catch that here rather than
    in a stack trace during training."""
    x = FeatureBuilder().fit_transform(train_df)
    for col in x.columns:
        dtype = x[col].dtype
        assert isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_numeric_dtype(dtype), (
            f"{col} has unusable dtype {dtype}"
        )


def test_expected_feature_block_is_present(train_df):
    builder = FeatureBuilder(top_skills=5).fit(train_df)
    names = builder.feature_names_

    for expected in (
        "years_experience",
        "experience_log",
        "experience_sqrt",
        "experience_bucket",
        "location_tier",
        "role",
        "n_skills",
        "skill_python",
    ):
        assert expected in names, expected
    assert len(names) == len(set(names)), "duplicate feature names"


def test_describe_is_readable(train_df):
    text = FeatureBuilder().fit(train_df).describe()
    assert "EXCLUDED" in text  # prev_salary warning is visible by default


# ══════════════════════════════════════════════ the real data


@pytest.mark.skipif(
    not SURVEY_CSV.exists(),
    reason="survey CSV not downloaded — see data/public/README.md",
)
def test_round_trip_on_the_real_survey():
    """End-to-end on the actual Stack Overflow rows: load → split → fit → transform.

    Slow (the CSV is 140MB) but worth it. Hand-written fixtures agree with your
    assumptions by construction; only real data disagrees with you.
    """
    from paybands.data import stackoverflow

    df, _ = stackoverflow.load(SURVEY_CSV, verbose=False)

    # A deterministic split. Shuffle first: survey rows are not in random order.
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    cut = int(len(shuffled) * 0.8)
    train, test = shuffled.iloc[:cut], shuffled.iloc[cut:]

    builder = FeatureBuilder().fit(train)
    x_train = builder.transform(train)
    x_test = builder.transform(test)

    assert len(x_train) == len(train)
    assert len(x_test) == len(test)
    assert list(x_train.columns) == list(x_test.columns) == builder.feature_names_
    assert TARGET not in x_train.columns

    # The survey has no city column, so every tier is NaN — stated as a fact of
    # this source, not a defect (see location.py).
    assert x_train["location_tier"].isna().all()

    # The vocabulary should have found the languages you'd expect in India.
    assert {"python", "javascript", "sql"} <= set(builder.skills.vocabulary_)

    # Experience is mostly present and the log transform is finite where it is.
    known = x_train["years_experience"].notna()
    assert known.mean() > 0.5
    assert np.isfinite(x_train.loc[known, "experience_log"]).all()
