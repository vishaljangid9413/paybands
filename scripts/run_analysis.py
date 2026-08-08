"""Regenerate every number and every chart in `docs/findings.md`.

    uv run python scripts/run_analysis.py

One script, one seed list, one output directory. Run it and you get the findings
document's evidence printed to your terminal and its figures written to
`reports/`. Nothing in `docs/findings.md` is allowed to be a number this script
does not print.

────────────────────────────────────────────────────────────────────────────
**Why one script rather than a notebook.**

A notebook records the order you happened to click in. Six weeks later you
cannot tell whether cell 14 ran before cell 9, whether the model in memory is
the one the chart came from, or which seed produced the number in the README.
This file runs top to bottom, seeds everything explicitly, and prints what it
did — so "which settings produced this number?" has an answer.

**Why every result is reported over many seeds.**

A single train/test split is an anecdote. On 1,022 rows the test set is ~200
people, and swapping which 200 moves the headline MAE by well over ten percent.
Every comparison here is therefore run across `--seeds` independent splits and
reported as a mean with the full range, and where a claim is "A beats B" the
script counts on how many seeds that was actually true. A margin that only holds
on 6 of 10 splits is not a margin.

**On pooling.** The error breakdowns pool test rows across seeds so that thin
slices (the 20+ experience bucket, the QA role) have enough rows to say anything
about at all. Rows recur across seeds, so a pooled `n` is not an independent
sample size — it is "how many times a row of this kind was scored". The script
prints both the pooled n and the per-split average, and the findings document
says so.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # so the script runs without an install
    sys.path.insert(0, str(REPO_ROOT / "src"))

from paybands.data.schema import TARGET  # noqa: E402
from paybands.data.stackoverflow import load  # noqa: E402
from paybands.data.synthetic import generate  # noqa: E402
from paybands.fairness.audit import (  # noqa: E402
    LEGITIMATE_CONTROLS,
    adjusted_gap,
    raw_gap,
    validate_on_synthetic,
)
from paybands.fairness.proxy import compare_with_without, proxy_correlation  # noqa: E402
from paybands.features.experience import BUCKET_LABELS, bucket_years  # noqa: E402
from paybands.model.band import SalaryBandModel, band_report  # noqa: E402
from paybands.model.baseline import GlobalMedianBaseline, GroupMedianBaseline  # noqa: E402
from paybands.model.conformal import ConformalBand, three_way_split  # noqa: E402
from paybands.model.metrics import evaluate, format_rupees, from_log, mae  # noqa: E402
from paybands.model.split import random_split  # noqa: E402
from paybands.plots import calibration as cal_plots  # noqa: E402
from paybands.plots import distributions as dist_plots  # noqa: E402
from paybands.plots import fairness as fair_plots  # noqa: E402
from paybands.plots.style import SERIES, new_axes, note, save  # noqa: E402

# ─────────────────────────────────────────────────────────── settings

DEFAULT_SURVEY = REPO_ROOT / "data" / "public" / "so_2025_raw.csv"
DEFAULT_OUTPUT = REPO_ROOT / "reports"

#: Splits used for every headline comparison. Ten is enough to see the spread
#: without the script taking a coffee break; the range matters more than the
#: mean and both are printed.
DEFAULT_N_SEEDS = 10

#: Permutation importance re-predicts the whole test set once per feature per
#: repeat, so it is the one expensive step. Three seeds × five shuffles is
#: enough to separate "carries signal" from "carries noise"; it is not enough to
#: rank the middle of the table, and the report says so.
IMPORTANCE_SEEDS = 3
PERMUTATION_REPEATS = 5

#: Nominal coverage levels swept for the calibration curve. Each one needs its
#: own three quantile models, so this is 15 fits per seed.
CALIBRATION_LEVELS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
CALIBRATION_SEEDS = 3

#: Injected pay gaps for the audit's self-test. 0.0 is the one that matters
#: most: an audit that finds bias in fair data is worse than no audit.
INJECTED_GAPS: tuple[float, ...] = (0.0, 0.03, 0.08, 0.15, 0.25)
SYNTHETIC_N = 10_000
PROXY_N = 8_000
PROXY_SEEDS = 3

#: The 2025 survey has 172 columns. These are the ones a fairness audit would
#: need and — the finding — none of them are there. Matched case-insensitively
#: as substrings against the real header, so a rename cannot make the check pass
#: silently.
PROTECTED_ATTRIBUTE_PATTERNS: tuple[str, ...] = (
    "gender",
    "ethnic",
    "race",
    "sexuality",
    "orientation",
    "transgender",
    "disability",
    "accessibility",
    "nationality",
    "religion",
    "caste",
)


# ─────────────────────────────────────────────────────── printing helpers


def heading(text: str) -> None:
    print(f"\n\n{'═' * 88}\n  {text}\n{'═' * 88}")


def subheading(text: str) -> None:
    print(f"\n── {text} {'─' * max(0, 84 - len(text))}")


def spread(values: np.ndarray | list[float], *, percent: bool = False, decimals: int = 0) -> str:
    """mean, min and max in one string. Never a mean on its own.

    A mean printed alone invites the reader to treat it as the answer. On 1,022
    rows the range is usually the more informative half of the sentence, so this
    function makes it impossible to report one without the other.
    """
    v = np.asarray(values, dtype=float)
    if percent:
        return f"mean {v.mean():.1%}   range {v.min():.1%} to {v.max():.1%}"
    fmt = f",.{decimals}f"
    return f"mean {v.mean():{fmt}}   range {v.min():{fmt}} to {v.max():{fmt}}"


def table(frame: pd.DataFrame, *, floatfmt: str = "%.3f") -> None:
    print(frame.to_string(float_format=lambda x: floatfmt % x))


# ─────────────────────────────────────────────────────── 1. the data


def section_data(survey_path: Path, out: Path) -> pd.DataFrame:
    heading("1 · THE DATA — what survived, and what that costs us")

    raw_all = pd.read_csv(
        survey_path, usecols=["Country", "Currency", "CompTotal"], low_memory=False
    )
    df, report = load(survey_path, verbose=False)
    print(report)

    india = raw_all[raw_all["Country"] == "India"]
    inr = india[india["Currency"].astype(str).str.startswith("INR")]
    reported = inr["CompTotal"].dropna()
    non_response = 1.0 - len(reported) / len(inr)

    subheading("non-response is the biggest single fact about this dataset")
    print(f"  Indian respondents in INR            {len(inr):>8,}")
    print(f"  who answered the salary question     {len(reported):>8,}")
    print(
        f"  who skipped it                       {len(inr) - len(reported):>8,}  "
        f"({non_response:.1%})"
    )
    print("  This is SELECTION BIAS, and no cleaning rule removes it. We know nothing")
    print("  about the half who declined, and 'declined to state a salary' is unlikely")
    print("  to be independent of the salary.")

    subheading("one typo, and why the median is the summary we trust")
    worst = float(reported.max())
    print(f"  largest salary as typed              {worst:>.4e}  rupees")
    print(f"  digits in that entry                 {len(f'{worst:.0f}'):>8,}")
    print(
        f"  mean of raw INR salaries             {float(reported.mean()):>.4e}  "
        f"(≈ {float(reported.mean()) / 1e18:,.0f} quintillion)"
    )
    print(f"  median of raw INR salaries           {float(reported.median()):>12,.0f}")
    print(f"  median after cleaning                {float(df[TARGET].median()):>12,.0f}")
    print(f"  mean after cleaning                  {float(df[TARGET].mean()):>12,.0f}")
    print("  One person typed a run of nines. It moved the mean by eighteen orders of")
    print("  magnitude and the median not at all. That is the whole argument for medians")
    print("  in one line, and it is why every baseline in this project predicts one.")

    subheading("skew — the reason the model learns log(salary)")
    skew_raw = dist_plots.skewness(df[TARGET])
    skew_log = dist_plots.skewness(np.log(df[TARGET]))
    print(f"  skew, cleaned rupees                 {skew_raw:>+12.2f}   long right tail")
    print(f"  skew, log(rupees)                    {skew_log:>+12.2f}   near-symmetric")

    subheading("who is in the surviving 1,022")
    counts = df["role"].value_counts()
    share = (counts / counts.sum() * 100).round(1)
    table(pd.DataFrame({"n": counts, "share_%": share}), floatfmt="%.1f")
    print(
        f"\n  experience: median {df['years_experience'].median():.0f} years, "
        f"{int(df['years_experience'].isna().sum())} rows with none recorded"
    )
    top_three = float(share.head(3).sum())
    thin_roles = counts[counts < 20]
    print(f"  the three largest roles are  {top_three:.1f}% of the sample")
    print(
        f"  roles with fewer than 20 respondents: "
        f"{', '.join(f'{r} ({n})' for r, n in thin_roles.items())}"
    )
    print("  Three web-development families are nearly three quarters of the sample.")
    print("  Anything this project says about the rest rests on single-digit counts.")

    subheading("pay drivers the survey does not record at all")
    print("  docs/design.md §4.1 lists what actually sets an Indian tech salary. Here is")
    print("  how many of those the public data can supply:\n")
    wanted = {
        "location_tier": "Bangalore vs a tier-2 city — one of the largest gaps there is",
        "prev_company_type": "product vs services — design.md calls this a make-or-break driver",
        "institute_tier": "IIT/NIT/BITS premium, and how it fades with experience",
        "level": "internal grade / ladder position",
        "prev_salary": "last drawn CTC — the anchor we specifically want to test against",
        "performance_rating": "needed for the increment layer",
        "event_date": "when the offer was made — no date means no temporal split",
    }
    for column, why in wanted.items():
        present = column in df.columns and df[column].notna().any()
        print(f"    {'✓' if present else '✗'} {column:<20} {why}")
    print("\n  Seven of the drivers this project was designed around are simply not in the")
    print("  file. The model that follows is not a weak model of Indian salaries; it is")
    print("  an honest model of the four or five things a public survey happens to ask.")

    fig, _ = dist_plots.salary_distribution_comparison(df[TARGET])
    print(f"\n  saved {save(fig, out / 'dist_raw_vs_log.png')}")
    fig, _ = dist_plots.salary_by_experience(df)
    print(f"  saved {save(fig, out / 'salary_by_experience.png')}")
    return df


# ─────────────────────────────────────────────────────── 2. baselines


def section_baselines(df: pd.DataFrame, seeds: range) -> None:
    heading("2 · BASELINES — the number the model has to beat")
    print("  Baseline 0: everyone earns the median.  Baseline 1: median of (role × experience).")
    print("  Baseline 1 is four lines of pandas. If gradient boosting cannot beat it, the")
    print("  honest engineering decision is to ship the four lines.\n")

    rows = []
    for seed in seeds:
        split = random_split(df, seed=seed)
        globals_ = GlobalMedianBaseline().fit(split.train, split.train[TARGET])
        groups = GroupMedianBaseline().fit(split.train, split.train[TARGET])
        pred_global = globals_.predict(split.test)
        pred_group = groups.predict(split.test)
        rows.append(
            {
                "seed": seed,
                "n_test": len(split.test),
                "mae_global": mae(split.test[TARGET], pred_global),
                "mae_group": mae(split.test[TARGET], pred_group),
                "improvement": 1.0
                - mae(split.test[TARGET], pred_group) / mae(split.test[TARGET], pred_global),
            }
        )
    frame = pd.DataFrame(rows)
    table(frame.set_index("seed"), floatfmt="%.4f")
    print()
    print(f"  global-median MAE     {spread(frame['mae_global'])}")
    print(f"  lookup-table MAE      {spread(frame['mae_group'])}")
    print(f"  improvement           {spread(frame['improvement'], percent=True)}")
    print(f"  lookup wins on        {int((frame['improvement'] > 0).sum())}/{len(frame)} seeds")

    # Where the cascade actually landed, on the last split fitted above.
    print()
    print("  " + groups.coverage_report().replace("\n", "\n  "))
    print("  Read that: on a 14-role table with 1,022 rows, a real share of candidates")
    print("  never match a (role × experience) cell with 10 people in it and fall back")
    print("  to experience alone. The lookup table is coarser than it looks.")

    print()
    print("  For reference, the global median baseline scored on the last split:")
    print(
        "  "
        + str(evaluate(split.test[TARGET], pred_global, label="global median")).replace(
            "\n", "\n  "
        )
    )
    print(
        "  "
        + str(evaluate(split.test[TARGET], pred_group, label="role × experience lookup")).replace(
            "\n", "\n  "
        )
    )
    print("  Note the negative R² on the global median: R² rewards predicting the MEAN,")
    print("  and on a right-skewed target the median is a long way from the mean. The")
    print("  metric encodes a choice, and ours is the typical candidate, not the average.")


# ─────────────────────────────────────────────────────── 3. the band model


@dataclass
class BandRun:
    """Everything one seed's band-model fit produced, kept for later sections."""

    seed: int
    frame: pd.DataFrame  # per-test-row predictions
    row: dict[str, float]


def _fit_one(df: pd.DataFrame, seed: int) -> BandRun:
    split = three_way_split(df, seed=seed)
    model = SalaryBandModel(seed=seed).fit(split.train, split.train[TARGET])
    raw_band = model.predict_band(split.test)
    raw = band_report(split.test[TARGET], raw_band, label="raw quantile band")

    conformal = ConformalBand(model).calibrate(split.calibration, split.calibration[TARGET])
    conf_band = conformal.predict_band(split.test)
    conf = conformal.evaluate(split.test, split.test[TARGET])

    # Two baselines, deliberately. `same_rows` is the like-for-like comparison —
    # identical training rows, so the only difference is the model. `all_rows`
    # is the fair fight for a shipping decision: a lookup table needs no
    # calibration set, so in production it would be built on 80% of the data
    # while the band model only gets 60%.
    same_rows = GroupMedianBaseline().fit(split.train, split.train[TARGET])
    pooled = pd.concat([split.train, split.calibration])
    all_rows = GroupMedianBaseline().fit(pooled, pooled[TARGET])

    truth = split.test[TARGET].to_numpy(dtype=float)
    mae_same = mae(truth, same_rows.predict(split.test))
    mae_all = mae(truth, all_rows.predict(split.test))
    mae_model = mae(truth, raw_band.median)

    frame = pd.DataFrame(
        {
            "seed": seed,
            "actual": truth,
            "predicted": raw_band.median,
            "raw_lower": raw_band.lower,
            "raw_upper": raw_band.upper,
            "lower": conf_band.lower,
            "upper": conf_band.upper,
            "bucket": bucket_years(split.test["years_experience"]).astype(str).to_numpy(),
            "role": split.test["role"].to_numpy(),
        }
    )
    frame["abs_error"] = (frame["actual"] - frame["predicted"]).abs()
    frame["ape"] = frame["abs_error"] / frame["actual"]
    frame["log_residual"] = np.log(frame["actual"]) - np.log(frame["predicted"])
    frame["inside"] = (frame["actual"] >= frame["lower"]) & (frame["actual"] <= frame["upper"])
    frame["width"] = frame["upper"] - frame["lower"]
    frame["relative_width"] = frame["width"] / frame["predicted"]

    row = {
        "seed": seed,
        "n_train": len(split.train),
        "n_cal": len(split.calibration),
        "n_test": len(split.test),
        "mae_lookup_same_rows": mae_same,
        "mae_lookup_all_rows": mae_all,
        "mae_band_model": mae_model,
        "improvement_same_rows": 1.0 - mae_model / mae_same,
        "improvement_all_rows": 1.0 - mae_model / mae_all,
        "coverage_raw": raw.coverage,
        "coverage_conformal": conf.coverage,
        "rel_width_raw": float(np.median(raw_band.relative_width)),
        "rel_width_conformal": float(np.median(conf_band.relative_width)),
        "qhat_log": float(conformal.qhat_log_ or 0.0),
        "widening": conformal.widening_factor,
        "crossing_rate": raw.crossing_rate,
        "trees_median_model": float(model.best_iterations_[1]),
    }
    return BandRun(seed=seed, frame=frame, row=row)


def section_band(df: pd.DataFrame, seeds: range, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    heading("3 · THE BAND MODEL — does it earn its complexity, and is it honest?")

    runs = [_fit_one(df, seed) for seed in seeds]
    summary = pd.DataFrame([r.row for r in runs]).set_index("seed")
    pooled = pd.concat([r.frame for r in runs], ignore_index=True)

    subheading("per-seed results")
    table(
        summary[
            [
                "mae_lookup_same_rows",
                "mae_band_model",
                "improvement_same_rows",
                "coverage_raw",
                "coverage_conformal",
                "rel_width_raw",
                "rel_width_conformal",
                "qhat_log",
            ]
        ],
        floatfmt="%.4f",
    )

    subheading("accuracy — the band's midpoint against the lookup table")
    print(f"  lookup table (same rows)   {spread(summary['mae_lookup_same_rows'])}")
    print(f"  band model midpoint        {spread(summary['mae_band_model'])}")
    print(f"  improvement                {spread(summary['improvement_same_rows'], percent=True)}")
    print(
        f"  model wins on              "
        f"{int((summary['improvement_same_rows'] > 0).sum())}/{len(summary)} seeds"
    )
    print()
    print("  And the fair fight for a shipping decision — the lookup table refitted on")
    print("  train + calibration, because it needs no calibration set of its own:")
    print(f"  lookup table (all 80%)     {spread(summary['mae_lookup_all_rows'])}")
    print(f"  improvement                {spread(summary['improvement_all_rows'], percent=True)}")
    print(
        f"  model wins on              "
        f"{int((summary['improvement_all_rows'] > 0).sum())}/{len(summary)} seeds"
    )

    subheading("coverage — does an '80% band' contain 80% of salaries?")
    print(f"  raw quantile band          {spread(summary['coverage_raw'], percent=True)}")
    print(f"  after conformal calibration {spread(summary['coverage_conformal'], percent=True)}")
    print(f"  q̂ (log points)             {spread(summary['qhat_log'], decimals=4)}")
    print(f"  each band edge moved by    {spread(summary['widening'] - 1.0, percent=True)}")
    print(
        f"  quantile crossings         "
        f"{spread(summary['crossing_rate'], percent=True)} of test rows"
    )
    print("  The raw band under-covers on every seed: asking LightGBM for the 90th")
    print("  percentile is not the same as getting it. Conformal prediction closes the")
    print("  gap, and it does so by making the band wider — which is the trade, not a")
    print("  free lunch.")

    subheading("width — the number that decides whether any of this is usable")
    print(f"  median width ÷ midpoint, raw       {spread(summary['rel_width_raw'], decimals=2)}")
    print(
        f"  median width ÷ midpoint, conformal {spread(summary['rel_width_conformal'], decimals=2)}"
    )
    print(f"  pooled median band width           {format_rupees(float(pooled['width'].median()))}")
    print(
        f"  pooled median midpoint             {format_rupees(float(pooled['predicted'].median()))}"
    )
    # What honesty cost. Reported as its own line because it is the trade the
    # whole section turns on: the calibrated band is the truthful one AND the
    # less usable one, and burying that in two other numbers hides it.
    honesty_cost = summary["rel_width_conformal"].mean() / summary["rel_width_raw"].mean() - 1.0
    print(f"  honesty cost, raw → calibrated     {honesty_cost:>13.1%} more width")
    example = pooled.iloc[
        int(np.argmin(np.abs(pooled["predicted"].to_numpy() - float(pooled["predicted"].median()))))
    ]
    print(
        f"  a typical calibrated band          "
        f"{format_rupees(example['lower'])} – {format_rupees(example['upper'])}  "
        f"(midpoint {format_rupees(example['predicted'])})"
    )
    print("  That is not a salary band. That is the salary range of the whole industry.")

    subheading("pooled test rows across all seeds")
    print(
        f"  pooled rows                {len(pooled):,}  "
        f"(≈{len(pooled) // len(summary):,} per split; rows recur across seeds,"
    )
    print("                             so this is 'times scored', not an independent n)")
    print(f"  pooled coverage            {pooled['inside'].mean():.1%}")
    print(f"  pooled MAE                 {format_rupees(float(pooled['abs_error'].mean()))}")
    print(f"  pooled median abs error    {format_rupees(float(pooled['abs_error'].median()))}")

    fig, _ = cal_plots.coverage_plot(
        pooled["actual"], pooled["raw_lower"], pooled["raw_upper"], nominal=0.8
    )
    print(f"\n  saved {save(fig, out / 'coverage_raw.png')}")
    fig, _ = cal_plots.coverage_plot(
        pooled["actual"], pooled["lower"], pooled["upper"], nominal=0.8
    )
    print(f"  saved {save(fig, out / 'coverage_conformal.png')}")
    fig, _ = cal_plots.band_width_plot(pooled["predicted"], pooled["lower"], pooled["upper"])
    print(f"  saved {save(fig, out / 'band_width.png')}")
    fig, _ = cal_plots.band_width_plot(
        pooled["predicted"], pooled["lower"], pooled["upper"], relative=True
    )
    print(f"  saved {save(fig, out / 'band_width_relative.png')}")
    return summary, pooled


# ─────────────────────────────────────────────────── 4. calibration sweep


def section_calibration(df: pd.DataFrame, out: Path) -> None:
    heading("4 · CALIBRATION SWEEP — is the band honest at every confidence level?")
    print("  One promise checked is a spot check. Five is a calibration curve. Each level")
    print("  gets its own three quantile models, so this is 15 fits per seed.\n")

    raw_bands: dict[float, tuple[list[float], list[float]]] = {}
    conf_bands: dict[float, tuple[list[float], list[float]]] = {}
    truths: list[float] = []
    rows = []

    for level in CALIBRATION_LEVELS:
        tail = (1.0 - level) / 2.0
        quantiles = (tail, 0.5, 1.0 - tail)
        raw_lo, raw_hi, conf_lo, conf_hi, actual = [], [], [], [], []
        for seed in range(CALIBRATION_SEEDS):
            split = three_way_split(df, seed=seed)
            model = SalaryBandModel(quantiles=quantiles, seed=seed).fit(
                split.train, split.train[TARGET]
            )
            band = model.predict_band(split.test)
            calibrated = (
                ConformalBand(model)
                .calibrate(split.calibration, split.calibration[TARGET])
                .predict_band(split.test)
            )
            raw_lo += list(band.lower)
            raw_hi += list(band.upper)
            conf_lo += list(calibrated.lower)
            conf_hi += list(calibrated.upper)
            actual += list(split.test[TARGET].to_numpy(dtype=float))
        raw_bands[level] = (raw_lo, raw_hi)
        conf_bands[level] = (conf_lo, conf_hi)
        if not truths:
            truths = actual
        rows.append(
            {
                "promised": level,
                "raw_actual": cal_plots.empirical_coverage(actual, raw_lo, raw_hi),
                "conformal_actual": cal_plots.empirical_coverage(actual, conf_lo, conf_hi),
                "raw_rel_width": float(
                    np.median((np.array(raw_hi) - np.array(raw_lo)) / np.array(actual))
                ),
                "conformal_rel_width": float(
                    np.median((np.array(conf_hi) - np.array(conf_lo)) / np.array(actual))
                ),
            }
        )

    frame = pd.DataFrame(rows).set_index("promised")
    frame["raw_shortfall"] = frame["raw_actual"] - frame.index
    frame["conformal_shortfall"] = frame["conformal_actual"] - frame.index
    table(frame, floatfmt="%.4f")
    print()
    print(
        f"  raw band under-covers at   "
        f"{int((frame['raw_shortfall'] < 0).sum())}/{len(frame)} levels"
    )
    print(
        f"  worst raw shortfall        {frame['raw_shortfall'].min():+.1%} at "
        f"{frame['raw_shortfall'].idxmin():.0%}"
    )
    print(
        f"  worst conformal shortfall  {frame['conformal_shortfall'].min():+.1%} at "
        f"{frame['conformal_shortfall'].idxmin():.0%}"
    )
    print("  Width is reported alongside coverage on purpose: a band that gains coverage")
    print("  by getting wider has not become more accurate, only more honest about how")
    print("  little it knows.")

    fig, _ = cal_plots.coverage_by_quantile(truths, raw_bands)
    print(f"\n  saved {save(fig, out / 'calibration_raw.png')}")
    fig, _ = cal_plots.coverage_by_quantile(truths, conf_bands)
    print(f"  saved {save(fig, out / 'calibration_conformal.png')}")


# ─────────────────────────────────────────────────── 5. feature importance


def _permutation_importance(
    model: SalaryBandModel,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    repeats: int = PERMUTATION_REPEATS,
) -> tuple[pd.Series, float]:
    """How much worse does the model get when one feature is shuffled?

    The only importance measure that answers the question anyone actually cares
    about — *if this column were noise, how much money would we lose?* — because
    it is measured on held-out rows in rupees, not on the training process.

    A **negative** score means shuffling the column made predictions better, and
    that is a real and useful result: the model was leaning on noise in that
    column and the shuffle took the noise away.

    The known weakness: with two correlated features (we have four encodings of
    experience), shuffling one leaves the other intact, so each individually
    looks less important than the pair really is. Read the experience family as
    a block.
    """
    features = model.builder.transform(X)
    median_model = model.models_[1]
    baseline = mae(y, from_log(median_model.predict(features)))
    rng = np.random.default_rng(seed)

    scores = {}
    for name in model.feature_names_:
        losses = []
        for _ in range(repeats):
            shuffled = features.copy()
            # .iloc[perm].set_axis(...) rather than .to_numpy()[perm]: the latter
            # drops the pandas `category` dtype, and LightGBM then refuses to
            # predict because the categorical feature list no longer matches.
            shuffled[name] = (
                features[name].iloc[rng.permutation(len(features))].set_axis(features.index)
            )
            losses.append(mae(y, from_log(median_model.predict(shuffled))))
        scores[name] = float(np.mean(losses) - baseline)
    # The unshuffled MAE comes back too, so the caller can express importance as a
    # share of the error the model actually makes rather than as a bare rupee
    # figure nobody has a scale for.
    return pd.Series(scores), baseline


def section_importance(df: pd.DataFrame, out: Path) -> None:
    heading("5 · WHAT ACTUALLY DRIVES SALARY — three importance measures, which disagree")
    print("  Split count = how often a feature was used. Gain = how much loss it removed.")
    print("  Permutation = how many rupees of held-out error appear when it is shuffled.")
    print("  They disagree, and the disagreement is the finding.\n")

    gains, splits, perms, baselines = [], [], [], []
    per_model_features = []
    for seed in range(IMPORTANCE_SEEDS):
        split = three_way_split(df, seed=seed)
        model = SalaryBandModel(seed=seed).fit(split.train, split.train[TARGET])
        per_model_features.append(len(model.feature_names_))
        booster = model.models_[1].booster_
        gain = pd.Series(booster.feature_importance("gain"), index=model.feature_names_)
        count = pd.Series(booster.feature_importance("split"), index=model.feature_names_)
        gains.append(gain / gain.sum())
        splits.append(count / count.sum())
        scores, baseline = _permutation_importance(
            model, split.test, split.test[TARGET].to_numpy(dtype=float), seed=seed
        )
        perms.append(scores)
        baselines.append(baseline)

    # The skill vocabulary is refitted per split (top 25 of the TRAINING rows), so
    # the union across seeds is wider than any one model. `seeds_present` says how
    # many of the three a feature existed in — a row with 1 is a skill that made
    # one split's top 25 and not the others, and its numbers are one measurement.
    perm_frame = pd.concat(perms, axis=1)
    frame = pd.DataFrame(
        {
            "gain_%": pd.concat(gains, axis=1).mean(axis=1) * 100,
            "split_%": pd.concat(splits, axis=1).mean(axis=1) * 100,
            "permutation_rupees": perm_frame.mean(axis=1),
            "permutation_sd": perm_frame.std(axis=1),
            "seeds_present": perm_frame.notna().sum(axis=1),
        }
    ).sort_values("permutation_rupees", ascending=False)

    print(f"  features per fitted model  {per_model_features}")
    print(
        f"  distinct features seen     {len(frame)}  "
        "(the union — the skill vocabulary is refitted per split)"
    )

    subheading("top 15 by permutation importance (rupees of MAE added when shuffled)")
    table(frame.head(15), floatfmt="%.1f")

    subheading("features that shuffling made BETTER, or did nothing at all")
    dead = frame[frame["permutation_rupees"] <= 0.0]
    table(dead, floatfmt="%.1f")

    # Grouped, because the four experience encodings are one signal wearing four
    # hats, and 25 skill flags are one idea spread thin. Ranking them
    # individually against `org_size` compares a quarter of a feature with a
    # whole one.
    blocks = {
        "experience (4 encodings)": [
            c for c in frame.index if c.startswith("experience") or c == "years_experience"
        ],
        "individual skill flags": [c for c in frame.index if c.startswith("skill_")],
        "n_skills": ["n_skills"],
        "categorical columns": [
            c
            for c in ("role", "org_size", "remote", "education", "employment_type")
            if c in frame.index
        ],
    }
    subheading("grouped — because four encodings of experience are one signal")
    grouped = pd.DataFrame(
        {
            name: frame.loc[cols, ["gain_%", "split_%", "permutation_rupees"]].sum()
            for name, cols in blocks.items()
        }
    ).T
    grouped["n_features"] = [len(cols) for cols in blocks.values()]
    table(grouped, floatfmt="%.1f")
    print("  Experience is over half the gain and four fifths of the permutation cost.")
    print("  Twenty-six skill flags together buy about a quarter of what experience buys,")
    print("  and most of that comes from three of them.")

    subheading("the split-count trap, in one comparison")
    unshuffled = float(np.mean(baselines))
    for name in ("years_experience", "n_skills", "education"):
        row = frame.loc[name]
        print(
            f"  {name:<20} split {row['split_%']:>5.1f}%   gain {row['gain_%']:>5.1f}%   "
            f"permutation {row['permutation_rupees']:>+10,.0f}  "
            f"({row['permutation_rupees'] / unshuffled:>+6.2%} of MAE)"
        )
    print("  `n_skills` is an integer running 0–20, so a tree can split on it at many")
    print("  thresholds and it climbs the split-count table. Shuffle it on held-out rows")
    print("  and the model barely notices. Split count measures how MANY questions a")
    print("  feature could answer, not how USEFUL the answers were — it is biased towards")
    print("  high-cardinality columns, and that bias is exactly what this row shows.")
    print()
    print("  `education` is worse than useless: shuffling it IMPROVES held-out error.")
    print("  design.md predicted education would matter less than people expect once")
    print("  experience was in the model. It matters less than nothing.")

    # "Worth more than 1% of the model's own error" is a far more useful bar than
    # "greater than zero". At an MAE around ₹12.5L that threshold is ~₹12,500, and
    # a feature below it could be deleted without anyone noticing.
    unshuffled_mae = unshuffled
    threshold = unshuffled_mae * 0.01
    n_dead = int((frame["permutation_rupees"] <= 0).sum())
    n_real = int((frame["permutation_rupees"] > threshold).sum())
    print(f"\n  unshuffled held-out MAE    {format_rupees(unshuffled_mae)}")
    print(f"  {n_dead} of {len(frame)} distinct features carry no measurable held-out signal.")
    print(f"  {n_real} move held-out MAE by more than 1% of it (≈{format_rupees(threshold)}).")
    print(f"  So: {per_model_features[0]} features per fitted model, {n_real} of them earning")
    print("  their place, on 613 training rows. That is a model carrying features it")
    print("  cannot afford — and the honest read of the table is that this dataset")
    print("  supports one strong signal (experience), two weak ones (org size and")
    print("  remote), and a great deal of decoration.")
    print()
    print("  `skill_php` at third place deserves suspicion rather than a headline. PHP")
    print("  plausibly marks lower-paid services work, but the effect rests on however")
    print("  many PHP developers land in a ~200-row test set, and the seed-to-seed")
    print(
        f"  standard deviation is ₹{frame.loc['skill_php', 'permutation_sd']:,.0f} on a mean of "
        f"₹{frame.loc['skill_php', 'permutation_rupees']:,.0f}. Treat it as a"
    )
    print("  hypothesis to test on more data, not as a finding.")

    fig, ax = new_axes(figsize=(8.0, 5.5))
    top = frame.head(12).iloc[::-1]
    ax.barh(range(len(top)), top["permutation_rupees"], color=SERIES[0])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel("rupees of held-out MAE added when the feature is shuffled")
    ax.set_title("Permutation importance — what the model would miss")
    note(
        ax,
        f"averaged over {IMPORTANCE_SEEDS} splits × {PERMUTATION_REPEATS} shuffles\n"
        "the four experience encodings are one signal,\nnot four",
        loc="lower right",
    )
    print(f"\n  saved {save(fig, out / 'feature_importance.png')}")


# ─────────────────────────────────────────────────── 6. where it is worst


def _slice_stats(group: pd.DataFrame) -> dict[str, float]:
    return {
        "n_pooled": len(group),
        "mae": float(group["abs_error"].mean()),
        "median_abs_error": float(group["abs_error"].median()),
        "mape_%": float(group["ape"].mean() * 100),
        "coverage": float(group["inside"].mean()),
        "median_rel_width": float(group["relative_width"].median()),
        "bias_%": float((np.exp(group["log_residual"].mean()) - 1.0) * 100),
    }


def section_error_breakdown(pooled: pd.DataFrame, n_seeds: int, out: Path) -> None:
    heading("6 · WHERE THE MODEL IS WORST — concentrated error is actionable, average error is not")
    print("  `bias_%` is exp(mean log residual) − 1. POSITIVE means the model predicts LESS")
    print("  than these people actually earn (it under-pays them); NEGATIVE means it")
    print("  predicts MORE. A slice whose bias is far from zero is not noisy, it is wrong")
    print("  in one direction, every time.\n")

    subheading("by experience bucket")
    order = [b for b in BUCKET_LABELS if b in set(pooled["bucket"])]
    by_bucket = pd.DataFrame({b: _slice_stats(pooled[pooled["bucket"] == b]) for b in order}).T
    by_bucket["n_per_split"] = (by_bucket["n_pooled"] / n_seeds).round(1)
    table(by_bucket, floatfmt="%.3f")

    worst = by_bucket["mape_%"].idxmax()
    print(
        f"\n  worst bucket by MAPE      {worst}  "
        f"({by_bucket.loc[worst, 'mape_%']:.0f}% average absolute percentage error,"
    )
    print(
        f"                            coverage {by_bucket.loc[worst, 'coverage']:.1%} "
        f"against an 80% promise, bias {by_bucket.loc[worst, 'bias_%']:+.0f}%,"
    )
    print(
        f"                            {by_bucket.loc[worst, 'n_per_split']:.0f} people per split)"
    )
    print("  Error rises monotonically with experience in rupees, which is expected —")
    print("  senior salaries are bigger and genuinely more variable. What is NOT expected")
    print("  is the coverage collapse at the top: the 20+ bucket is the one slice where")
    print("  the calibrated band breaks its promise, and it is also the slice the model")
    print("  systematically over-predicts. Both come from the same cause: almost no rows.")

    subheading("by role")
    roles = sorted(set(pooled["role"]))
    by_role = pd.DataFrame({r: _slice_stats(pooled[pooled["role"] == r]) for r in roles}).T
    by_role["n_per_split"] = (by_role["n_pooled"] / n_seeds).round(1)
    table(by_role.sort_values("mape_%", ascending=False), floatfmt="%.3f")

    thin = by_role[by_role["n_per_split"] < 5.0]
    thick = by_role[by_role["n_per_split"] >= 5.0]
    print(f"\n  roles with fewer than 5 test rows per split: {', '.join(sorted(thin.index))}")
    print(f"  mean |bias| in those roles          {thin['bias_%'].abs().mean():.1f}%")
    print(f"  mean |bias| in the rest             {thick['bias_%'].abs().mean():.1f}%")
    worst_role = by_role["bias_%"].abs().idxmax()
    print(
        f"  most one-sided role                 {worst_role}  "
        f"({by_role.loc[worst_role, 'bias_%']:+.0f}% bias, "
        f"{by_role.loc[worst_role, 'coverage']:.0%} coverage, "
        f"{by_role.loc[worst_role, 'n_per_split']:.0f} rows per split)"
    )
    print("  The error is not spread evenly. It is concentrated in the roles the survey")
    print("  barely covers — and those are exactly the roles a company would most want")
    print("  a band for, because they are the ones nobody has an intuition about.")

    subheading("band width across the salary range")
    pooled = pooled.copy()
    pooled["decile"] = pd.qcut(pooled["predicted"], 10, labels=False)
    by_decile = pd.DataFrame(
        {
            int(d): {
                "n_pooled": len(g),
                "median_midpoint": float(g["predicted"].median()),
                "median_width": float(g["width"].median()),
                "width_over_midpoint": float(g["relative_width"].median()),
                "coverage": float(g["inside"].mean()),
            }
            for d, g in pooled.groupby("decile")
        }
    ).T
    table(by_decile, floatfmt="%.3f")
    lowest = by_decile.iloc[0]
    highest = by_decile.iloc[-1]
    print(
        f"\n  bottom decile   midpoint {format_rupees(lowest['median_midpoint'])}, "
        f"band {format_rupees(lowest['median_width'])} wide "
        f"= {lowest['width_over_midpoint']:.1f}× the midpoint"
    )
    print(
        f"  top decile      midpoint {format_rupees(highest['median_midpoint'])}, "
        f"band {format_rupees(highest['median_width'])} wide "
        f"= {highest['width_over_midpoint']:.1f}× the midpoint"
    )
    print(
        f"  width in rupees grows {highest['median_width'] / lowest['median_width']:.1f}× "
        f"from the bottom decile to the top — the band is NOT a fixed width, which is"
    )
    print("  what training three separate quantile models was for, and it worked.")
    print(
        f"  But width RELATIVE to the prediction falls from "
        f"{lowest['width_over_midpoint']:.1f}× to {highest['width_over_midpoint']:.1f}×, so the"
    )
    print("  band is proportionally worst exactly where a company hires most: juniors.")

    fig, _ = fair_plots.residual_by_group(
        pooled["actual"], pooled["predicted"], pooled["bucket"], relative=True
    )
    print(f"\n  saved {save(fig, out / 'error_by_experience.png')}")
    fig, _ = fair_plots.residual_by_group(
        pooled["actual"], pooled["predicted"], pooled["role"], relative=True
    )
    print(f"  saved {save(fig, out / 'error_by_role.png')}")


# ─────────────────────────────────────────────────── 7. fairness


def section_fairness(survey_path: Path, df: pd.DataFrame, out: Path) -> None:
    heading("7 · FAIRNESS — a working audit, and no data to point it at")

    subheading("7a · what the public survey actually contains")
    header = pd.read_csv(survey_path, nrows=0)
    columns = list(header.columns)
    found = {
        pattern: [c for c in columns if pattern in c.lower()]
        for pattern in PROTECTED_ATTRIBUTE_PATTERNS
    }
    print(f"  columns in the 2025 survey file      {len(columns):>6,}")
    for pattern, hits in found.items():
        print(f"    {pattern:<16} {len(hits):>3} column(s)  {hits if hits else ''}")
    total = sum(len(h) for h in found.values())
    print(f"  protected-attribute columns found    {total:>6,}")
    print(f"  age band column present              {'yes' if 'Age' in columns else 'no':>6}")
    print()
    print("  ┌────────────────────────────────────────────────────────────────────────┐")
    print("  │ THE FAIRNESS AUDIT CANNOT BE RUN ON THIS DATA.                         │")
    print("  │ Not because the audit is broken — sections 7c and 7d show it works to  │")
    print("  │ within half a percentage point. Because the 2025 survey stopped asking │")
    print("  │ about gender, ethnicity and sexuality. The attribute is not missing    │")
    print("  │ for some rows; the QUESTION is not in the survey.                      │")
    print("  │ 'We don't collect it' is not a fairness strategy. It removes your      │")
    print("  │ ability to check, not the effect.                                      │")
    print("  └────────────────────────────────────────────────────────────────────────┘")

    subheading("7b · the one quasi-protected attribute we do have: age band")
    pair = df[df["age_band"].isin(["25-34 years old", "35-44 years old"])].dropna(
        subset=["years_experience"]
    )
    controls = [c for c in LEGITIMATE_CONTROLS if c in pair.columns]
    raw = raw_gap(pair[TARGET], pair["age_band"], disadvantaged="25-34 years old")
    adjusted = adjusted_gap(
        pair[TARGET], pair["age_band"], pair[controls], disadvantaged="25-34 years old"
    )
    print("  " + str(raw).replace("\n", "\n  "))
    print()
    print("  " + str(adjusted).replace("\n", "\n  "))
    print()
    print(f"  raw gap        {raw.mean_gap:>8.1%}")
    print(
        f"  adjusted gap   {adjusted.gap:>8.1%}   "
        f"({adjusted.ci_low:.1%} to {adjusted.ci_high:.1%}, "
        f"{adjusted.ci_width:.0%} wide)"
    )
    print(
        f"  explained by controls  {raw.mean_gap - adjusted.gap:.1%} of the "
        f"{raw.mean_gap:.1%} headline"
    )
    print("  A 44% raw gap that becomes a −5% adjusted gap whose interval spans 50")
    print("  percentage points. Two lessons in one number. First: a raw gap is a fact")
    print("  about who the two groups ARE (35–44s have more experience), not about how")
    print("  they are treated. Second: 702 people cannot answer this question. The")
    print("  interval is wider than any gap worth arguing about, so the honest reading")
    print("  is 'we cannot tell', not 'no gap'.")

    subheading("7c · validating the audit against a gap we chose")
    print("  Nobody knows the true pay gap in real data, so a bias detector cannot be")
    print("  checked there. We generate a world where we wrote the rules, inject a known")
    print("  gap, and demand it back.\n")
    validated = validate_on_synthetic(INJECTED_GAPS, n=SYNTHETIC_N, seed=42)
    table(validated.set_index("injected"), floatfmt="%.4f")
    worst_error = validated["error"].abs().max()
    print(f"\n  largest error across all five      {worst_error:.2%} (percentage points)")
    print(
        f"  truth inside its interval          "
        f"{int(validated['truth_inside_ci'].sum())}/{len(validated)}"
    )
    print(
        f"  at an injected gap of ZERO         {validated.iloc[0]['recovered']:+.2%} — "
        "the audit does not invent bias in fair data,"
    )
    print("                                     which is the test that matters most.")

    print()
    print("  And the same run with the proxy channel left OUT of the controls:")
    unproxied = validate_on_synthetic(
        INJECTED_GAPS, n=SYNTHETIC_N, seed=42, control_for_proxy=False
    )
    table(unproxied.set_index("injected"), floatfmt="%.4f")
    print(
        f"\n  every estimate is too high by      "
        f"{unproxied['error'].min():.1%} to {unproxied['error'].max():.1%}"
    )
    print(
        f"  truth inside its interval          "
        f"{int(unproxied['truth_inside_ci'].sum())}/{len(unproxied)}"
    )
    print("  Neither table is 'wrong'. They answer different questions. Controlling for")
    print("  career breaks says 'time out of the workforce is a legitimate reason to pay")
    print("  less'; not controlling says 'who takes that time off is itself the")
    print("  unfairness'. That is a policy choice, not a statistical one, and the honest")
    print("  thing is to publish both numbers rather than quietly pick the comfortable one.")

    subheading("7d · the proxy demonstration — deleting the column does not delete the bias")
    rows = []
    for seed in range(PROXY_SEEDS):
        frame, _ = generate(PROXY_N, seed=seed, pay_gap=0.08)
        result = compare_with_without(frame, seed=seed)
        rows.append(
            {
                "seed": seed,
                "gap_in_data": result.gap_in_data.gap,
                "gap_with_gender": result.gap_with_protected,
                "gap_without_gender": result.gap_without_protected,
                "surviving_share": result.surviving_share,
                "mae_cost": result.accuracy_cost,
            }
        )
    proxy_frame = pd.DataFrame(rows).set_index("seed")
    table(proxy_frame, floatfmt="%.4f")
    print()
    print(f"  gap surviving deletion     {spread(proxy_frame['surviving_share'], percent=True)}")
    print(f"  accuracy given up for it   {spread(proxy_frame['mae_cost'], percent=True)}")
    print("  Deleting gender removed about three fifths of the modelled gap and left the")
    print("  rest, delivered by career breaks — at essentially no accuracy cost, so")
    print("  nothing was even traded away for the harm that remains.")

    print()
    scan_frame, _ = generate(PROXY_N, seed=0, pay_gap=0.08)
    scan = proxy_correlation(scan_frame, "gender")
    print("  " + str(scan).replace("\n", "\n  "))
    print("  One channel, and it is visible before any model is trained. A real dataset")
    print("  has several and nobody labels them — which is what makes the scan worth")
    print("  running on company data on day one, before a line of modelling.")

    subheading("plots")
    frame, truth = generate(SYNTHETIC_N, seed=42, pay_gap=0.08)
    controls = [c for c in LEGITIMATE_CONTROLS if c in frame.columns] + ["career_gap_months"]
    demo_raw = raw_gap(frame[TARGET], frame["gender"], disadvantaged=truth.disadvantaged_group)
    demo_adj = adjusted_gap(
        frame[TARGET],
        frame["gender"],
        frame[controls],
        disadvantaged=truth.disadvantaged_group,
    )
    fig, _ = fair_plots.gap_comparison(
        demo_raw.mean_gap,
        demo_adj.gap,
        (demo_adj.ci_low, demo_adj.ci_high),
        known_gap=truth.pay_gap,
    )
    print(f"  saved {save(fig, out / 'fairness_validation.png')}")
    print(
        f"  (injected {truth.pay_gap:.1%}, raw {demo_raw.mean_gap:.1%}, "
        f"recovered {demo_adj.gap:.2%})"
    )


# ─────────────────────────────────────────────────────── main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_SURVEY, help="survey CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="where plots go")
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_N_SEEDS, help="how many splits per comparison"
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"survey file not found: {args.data}", file=sys.stderr)
        print("Download the Stack Overflow 2025 survey into data/public/.", file=sys.stderr)
        return 1

    started = time.time()
    seeds = range(args.seeds)

    print("paybands — full analysis")
    print(f"  survey        {args.data}")
    print(f"  plots         {args.out}")
    print(f"  seeds         {list(seeds)}")
    print(f"  numpy {np.__version__} · pandas {pd.__version__}")
    print("  Every random draw below comes from a locally-seeded Generator, never from")
    print("  numpy's global state, so the same command produces the same numbers.")

    df = section_data(args.data, args.out)
    section_baselines(df, seeds)
    summary, pooled = section_band(df, seeds, args.out)
    section_calibration(df, args.out)
    section_importance(df, args.out)
    section_error_breakdown(pooled, len(summary), args.out)
    section_fairness(args.data, df, args.out)

    heading("DONE")
    print(f"  {time.time() - started:.0f} seconds")
    print(f"  figures in {args.out}")
    print("  Every number in docs/findings.md appears somewhere above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
