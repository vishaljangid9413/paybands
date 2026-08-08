"""Per-prediction explanations — why *this* candidate got *this* number.

Everything before this file produced a band. This one produces the **argument**
behind it, which is a different and in some ways more important output.

The reason is in `ROADMAP.md` Phase 6 and it is worth stating in full: a
recruiter must be able to **argue with the model**. A salary number with no
reasoning attached cannot be challenged, and an unchallengeable salary number is
exactly the kind that causes harm — it launders a guess into an authority. The
same explanation is also what makes a decision defensible six months later, when
somebody asks why this candidate was offered less than that one.

---

## Idea 1 — SHAP, and why `TreeExplainer` specifically

**SHAP** (SHapley Additive exPlanations) answers one question: starting from what
the model predicts for a *typical* candidate, how much did each feature of *this*
candidate push the answer up or down? The pushes are called **SHAP values**, and
their defining property is that they **add up exactly**::

    prediction = baseline + sum(shap value for every feature)

That is not an approximation and not a fit — it is guaranteed by construction.
It is what separates SHAP from "feature importance", which tells you what matters
*on average* and says nothing at all about the person in front of you.

`shap.TreeExplainer` computes those values **exactly** for tree ensembles, in
milliseconds, by walking the trees. The general-purpose explainers
(`KernelExplainer`, `PermutationExplainer`) *sample* — they are slower by orders
of magnitude and they return a different answer each run unless you pin a seed.
For LightGBM there is no reason to use them, and a sampled explanation of a
salary is a bad thing to hand somebody who wants to argue with it.

We explain the **median** model of the three (see `model/band.py`). The band's
edges are honesty; the midpoint is what people actually act on, so the midpoint
is what needs a defence.

---

## Idea 2 — the log-space trap, which is the whole difficulty here

`model/band.py` trains on ``log(salary)``. So SHAP values come back **in log
units**. A raw SHAP value of ``0.18`` is not a salary, not a percentage, and not
anything a recruiter has ever seen. Handing it over unconverted is the single
easiest way to make an explanation feel rigorous while communicating nothing.

There are exactly two honest conversions, and this module reports both because
they answer different questions.

**(a) Multiplicative — `Contribution.multiplier`.** ``exp(0.18) = 1.20``: "eight
years of experience multiplies the prediction by 1.20×". This is **exact and
complete**::

    prediction = baseline × multiplier₁ × multiplier₂ × ... × multiplierₖ

because multiplying is what adding in log space *is*. Nothing is lost, nothing is
approximated, and it is the scale the model actually learned on. It is the number
to trust.

**(b) Rupee attribution — `Contribution.rupees`.** Recruiters think in rupees, so
we also answer "what is this feature worth, in money?" The convention used here
is a **leave-one-out counterfactual against this prediction**::

    rupees = prediction − prediction ÷ multiplier

Read as: "take this candidate's contribution away and the prediction would fall
from ₹23.6L to ₹20.4L, so it is worth about ₹3.2L." That statement is exactly
true for any one feature on its own.

**These rupee figures do not add up, and this module says so out loud rather than
fixing it.** They cannot: ``exp`` is not linear, so the rupee value of a feature
depends on where the other features have already put you. The same 20% is ₹1L at
₹5L and ₹10L at ₹50L. `Explanation.rupee_residual` reports the leftover in
rupees, `Explanation.approximation_note` says it in words, and the API ships that
note as a required field.

The tempting alternative is to rescale every rupee figure by
``(prediction − baseline) ÷ sum(rupees)`` so the column sums correctly. **Don't.**
That produces numbers that are individually wrong, sum to something true, and
carry no warning — the worst combination available. A number that visibly does
not tie out invites the question "why not?", and the answer to that question is
real information about the model.

---

## Idea 3 — four experience columns is one fact wearing four hats

`docs/findings.md` §4.1: ``years_experience`` alone is **36.4%** of the model's
gain, and the four experience encodings together (`years_experience`,
`experience_log`, `experience_sqrt`, `experience_bucket`) are **53.7%** of gain
and about **four fifths** of permutation importance. They are four encodings of
one number, deliberately handed to the model in `features/experience.py` so it
did not have to rediscover the shape of the experience-to-pay curve.

Shown to a recruiter unmerged, that becomes four separate rows — "experience
added ₹3.2L, experience (log scale) added ₹1.1L, the 6–8 band subtracted ₹0.2L,
experience (square-root scale) added ₹0.1L" — which reads like four independent
findings that partly contradict each other. It is one finding.

So `DEFAULT_GROUPS` folds them into a single ``experience`` row by default, which
is **exact**: SHAP values add, so summing four of them is summing, not
averaging or approximating.

**The trade-off, stated because grouping hides something real.** Merged, you can
no longer see *which* encoding the trees leaned on, and that is a genuine
modelling diagnostic — if `experience_bucket` is doing all the work, the
continuous columns are dead weight. Pass ``groups={}`` to get the raw view back.
Grouping is right for the recruiter-facing output and wrong for model debugging,
so it is a default rather than a rule.

Skill flags are deliberately **not** grouped. "Knowing Go" is something a
candidate can point at and a recruiter can weigh; a merged ``skills`` row is not
actionable by anyone.

---

## Idea 4 — a missing input is an answer, not a blank

If a caller does not send `employment_type`, `FeatureBuilder` emits NaN, LightGBM
routes NaN down whichever branch fits the training data best, and that routing
has a SHAP value like any other. You will see rows such as "employment type not
given — ×1.04". That looks odd and it is correct: the model learned a response to
missingness, and the person reading the explanation deserves to know that a field
they left blank moved the number.

---

## Idea 5 — rupees are rounded to ₹1,000, on purpose

Reporting "experience added ₹3,22,135" claims a precision the band's own width
(currently ~1.9× the midpoint) flatly disproves. `RUPEE_ROUNDING` matches
`api/service.RUPEE_ROUNDING`, so the numbers in an explanation and the numbers in
the band it explains round the same way and cannot disagree. Multipliers and log
contributions are **not** rounded — those are the exact quantities, and rounding
them would break the additivity a reader might want to check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..model.metrics import format_rupees

# ─────────────────────────────────────────────── knobs

#: Rupee rounding for every money figure this module reports. Same value as
#: `api/service.RUPEE_ROUNDING` — see Idea 5.
RUPEE_ROUNDING = 1_000

#: How many contributions a sentence mentions before it stops. Three is the
#: number of reasons a person repeats after hearing them once.
DEFAULT_SENTENCE_TOP_N = 3

#: Below this in absolute log units a contribution is noise, not a reason.
#: ``0.005`` is a half-percent move on the prediction — smaller than the rupee
#: rounding on most bands, so listing it would be reporting a number we have
#: already declared meaningless.
NEGLIGIBLE_LOG_CONTRIBUTION = 0.005

#: The four columns `features/experience.py` builds out of one `years_experience`
#: value. Folded into a single row by default — Idea 3.
EXPERIENCE_FEATURES: tuple[str, ...] = (
    "years_experience",
    "experience_log",
    "experience_sqrt",
    "experience_bucket",
)

#: ``group name -> the raw feature columns it absorbs``. Only experience, and
#: only because those four are provably one signal. Pass ``groups={}`` to
#: `explain_prediction` for the unmerged view.
DEFAULT_GROUPS: dict[str, tuple[str, ...]] = {"experience": EXPERIENCE_FEATURES}


# ─────────────────────────────────────────────── turning a column into English

#: Short human labels, used for the "not given" phrasing and as a fallback. A
#: feature missing from this map still explains fine — it just gets its raw
#: column name, which is the right failure mode for a column somebody added
#: yesterday.
FEATURE_LABELS: dict[str, str] = {
    "experience": "experience",
    "years_experience": "experience",
    "experience_log": "experience (log scale)",
    "experience_sqrt": "experience (square-root scale)",
    "experience_bucket": "experience band",
    "location_tier": "location",
    "role": "role",
    "education": "education",
    "org_size": "company size",
    "remote": "work arrangement",
    "employment_type": "employment type",
    "level": "seniority level",
    "institute_tier": "institute tier",
    "prev_company_type": "previous company type",
    "performance_rating": "performance rating",
    "prev_salary": "previous salary",
    "n_skills": "number of listed skills",
}

#: `features/location.py` maps cities to three tiers and the phrasing has to
#: carry what a tier *means* — "tier 3" alone tells a reader nothing, and tier 3
#: is also the catch-all for a city the alias table did not recognise.
TIER_PHRASES: dict[int, str] = {
    1: "a tier-1 (metro) location",
    2: "a tier-2 city",
    3: "a tier-3 (non-metro) location",
}


def _is_missing(value: object) -> bool:
    """True for NaN / None / NaT. Scalars only — see `_phrase`'s callers."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # arrays and exotic objects
        return value is None


def _pretty_skill(raw: str) -> str:
    """``"typescript"`` → ``"TypeScript"``.

    `features/skills.py` lowercases every skill so that "Python" and "python"
    cannot become two features. That is right for the model and wrong for a
    sentence, so the casing is put back here. Anything not in the map gets
    ``.title()``, which is imperfect ("node.js" → "Node.Js") and still far better
    than shouting "PYTHON" or whispering "python" at a hiring manager.
    """
    known = {
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "html/css": "HTML/CSS",
        "sql": "SQL",
        "c#": "C#",
        "c++": "C++",
        "php": "PHP",
        "aws": "AWS",
        "gcp": "GCP",
        "node.js": "Node.js",
        "postgresql": "PostgreSQL",
        "mysql": "MySQL",
        "mongodb": "MongoDB",
        "ios": "iOS",
    }
    return known.get(raw.lower(), raw.title())


def _phrase(name: str, value: object, *, skill_names: dict[str, str] | None = None) -> str:
    """A noun phrase that slots into "… added ₹6L".

    The target grammar is the roadmap's own example sentence: *"8 years
    experience added ₹6L, Bangalore added ₹3L, coming from a services company
    subtracted ₹2L."* So every phrase here has to read as a **thing**, not as a
    column name — "a services company", never "prev_company_type = services".

    Args:
        name: The feature (or group) name.
        value: Its value for this candidate, straight out of the feature matrix.
        skill_names: ``{"skill_go": "go"}`` — the original vocabulary entry for a
            skill flag, so casing can be restored. `BandExplainer` builds it.
    """
    skill_names = skill_names or {}
    label = FEATURE_LABELS.get(name) or skill_names.get(name) or name

    if name.startswith("skill_"):
        pretty = _pretty_skill(skill_names.get(name, name.removeprefix("skill_")))
        # A skill flag is 1/0, so "missing" and "does not have it" are the same
        # thing to the model and the phrasing should not pretend otherwise.
        has_it = not _is_missing(value) and float(value) > 0
        return f"knowing {pretty}" if has_it else f"no {pretty}"

    if _is_missing(value):
        return f"{label} not given"

    if name in {"experience", "years_experience"}:
        return f"{float(value):g} years of experience"
    if name == "experience_bucket":
        return f"the {value} year experience band"
    if name in {"experience_log", "experience_sqrt"}:
        return label
    if name == "location_tier":
        return TIER_PHRASES.get(int(round(float(value))), f"location tier {value}")
    if name == "role":
        return f"a {value} role"
    if name == "education":
        return f"a {value}"
    if name == "org_size":
        return f"a {value}-person company"
    if name == "remote":
        return f"{str(value).lower()} working"
    if name == "employment_type":
        return f"being {value}"
    if name == "level":
        return f"a {value}-level role"
    if name == "institute_tier":
        return f"a {value} institute"
    if name == "prev_company_type":
        return f"coming from a {value} company"
    if name == "performance_rating":
        return f"a performance rating of {float(value):g}"
    if name == "prev_salary":
        return f"a previous salary of {format_rupees(float(value))}"
    if name == "n_skills":
        return f"{float(value):g} listed skills"

    return f"{label} = {value}"


# ─────────────────────────────────────────────── what one explanation looks like


def _round_rupees(amount: float) -> float:
    """Nearest `RUPEE_ROUNDING`, sign preserved. See Idea 5."""
    return float(np.round(float(amount) / RUPEE_ROUNDING) * RUPEE_ROUNDING)


@dataclass(frozen=True)
class Contribution:
    """One feature's push on one prediction, in three scales at once.

    Frozen, like `BandPrediction` and for the same reason: an explanation you can
    edit after the fact is an explanation nobody can audit.

    Read the fields in this order:

    * `log_contribution` — the raw SHAP value. Exact, additive, and unreadable.
    * `multiplier` — ``exp(log_contribution)``. **Exact**, and the product of
      every multiplier times the baseline is the prediction, to the rupee.
    * `rupees` — approximate, and the one a recruiter will quote. See Idea 2 for
      the convention and for why these do not sum.
    """

    feature: str
    #: Which raw columns were folded in. Length 1 unless this row is a group.
    members: tuple[str, ...]
    #: The candidate's value, as the model saw it. ``nan`` when the field was
    #: not supplied — which is itself a contribution. See Idea 4.
    value: Any
    log_contribution: float
    multiplier: float
    rupees: float
    phrase: str

    @property
    def direction(self) -> str:
        """``"increased"`` / ``"decreased"`` / ``"no effect"``."""
        if abs(self.log_contribution) < NEGLIGIBLE_LOG_CONTRIBUTION:
            return "no effect"
        return "increased" if self.log_contribution > 0 else "decreased"

    @property
    def percent(self) -> float:
        """The push as a fraction: ``0.20`` means "+20% on the prediction"."""
        return self.multiplier - 1.0

    @property
    def verb(self) -> str:
        """The word the sentence generator uses. Money words, not maths words."""
        return {"increased": "added", "decreased": "subtracted", "no effect": "changed"}[
            self.direction
        ]

    def describe(self) -> str:
        """One line, the way it would appear in a table."""
        return (
            f"{self.phrase:<42.42s} ×{self.multiplier:.3f}  "
            f"{self.percent:+7.1%}  ≈ {format_rupees(self.rupees):>14}"
        )


@dataclass(frozen=True)
class Explanation:
    """Every contribution behind one predicted midpoint, ranked by size.

    The invariant worth knowing, and the one `tests/test_explain.py` asserts::

        baseline_log + sum(c.log_contribution for c in contributions) == prediction_log

    exactly, in floating point tolerance. That is SHAP's additivity guarantee,
    and it survives grouping because grouping is addition. It does **not** hold
    for the `rupees` column — see `rupee_residual`.
    """

    contributions: tuple[Contribution, ...]

    #: What the model predicts before it knows anything about this person: the
    #: average prediction over the rows the trees were fitted on, in rupees. The
    #: honest reading is "a typical candidate in the training data", **not** "the
    #: market median" — the training data is whatever we happened to collect.
    baseline: float
    #: The median model's prediction for this candidate, in rupees.
    prediction: float

    #: The same two numbers in log units, unrounded. Additivity is checked here.
    baseline_log: float
    prediction_log: float

    #: Which quantile was explained, so nobody has to guess whether an
    #: explanation belongs to the midpoint or to a band edge.
    quantile: float = 0.5

    @property
    def n_features(self) -> int:
        return len(self.contributions)

    @property
    def rupee_residual(self) -> float:
        """``(prediction − baseline) − sum(rupees)``. **Should not be zero.**

        This is the honesty gauge for Idea 2. The rupee column is a set of
        leave-one-out counterfactuals, each measured against the full prediction,
        so when several features push the same way they each get credit for
        ground the others also covered and the column over-counts. Publishing the
        leftover is the alternative to quietly rescaling it away.
        """
        return (self.prediction - self.baseline) - sum(c.rupees for c in self.contributions)

    @property
    def approximation_note(self) -> str:
        """The sentence that stops the rupee column from being misread.

        Required output, never optional, for the same reason
        `api/service.build_caveat` is: JSON strips uncertainty by default, and a
        column of rupee figures that a UI renders in a neat stack will be read as
        a sum unless something explicitly says it is not one.
        """
        return (
            "Rupee figures are approximate: each is what the prediction would lose if that "
            "one factor were removed, so they overlap and do NOT add up to the total "
            f"(they are out by about {format_rupees(abs(self.rupee_residual))} here). "
            "The multipliers are exact — multiply them all by the baseline and you get the "
            "prediction, because the model works in log space where these effects are "
            "percentages rather than rupees."
        )

    def top(self, n: int = DEFAULT_SENTENCE_TOP_N) -> tuple[Contribution, ...]:
        """The `n` largest contributions, negligible ones dropped."""
        real = [c for c in self.contributions if c.direction != "no effect"]
        return tuple(real[:n])

    def sentence(self, n: int = DEFAULT_SENTENCE_TOP_N, *, include_note: bool = False) -> str:
        """The explanation as one line a person would say out loud.

            Predicted ₹23,64,000 — a 5000+-person company added ₹3,89,000,
            8 years of experience added ₹3,22,000, coming from a services
            company subtracted ₹3,53,000.

        This is the deliverable. Everything else in this module is the machinery
        that lets this sentence be true.

        Args:
            n: How many factors to name. Three by default — a list of twelve
                reasons is not an explanation, it is a data dump.
            include_note: Append `approximation_note`. The API sets this false
                and ships the note as its own field, so a UI can style it.
        """
        headline = f"Predicted {format_rupees(self.prediction)}"
        chosen = self.top(n)
        if not chosen:
            return (
                f"{headline} — no single factor moved this prediction much; it sits close to "
                f"the {format_rupees(self.baseline)} typical for the training data."
            )
        clauses = ", ".join(f"{c.phrase} {c.verb} {format_rupees(abs(c.rupees))}" for c in chosen)
        text = f"{headline} — {clauses}."
        return f"{text} {self.approximation_note}" if include_note else text

    def to_frame(self) -> pd.DataFrame:
        """The contributions as a table, for eyeballing or writing to disk."""
        return pd.DataFrame(
            {
                "feature": [c.feature for c in self.contributions],
                "value": [c.value for c in self.contributions],
                "phrase": [c.phrase for c in self.contributions],
                "log_contribution": [c.log_contribution for c in self.contributions],
                "multiplier": [c.multiplier for c in self.contributions],
                "rupees": [c.rupees for c in self.contributions],
            }
        )

    def describe(self, n: int = 8) -> str:
        """A block of text to paste into the findings log."""
        lines = [
            self.sentence(),
            "",
            f"  baseline (typical candidate)  {format_rupees(self.baseline):>16}",
            f"  prediction (q{self.quantile:g})            {format_rupees(self.prediction):>16}",
            "",
        ]
        lines += [f"  {c.describe()}" for c in self.contributions[:n]]
        if self.n_features > n:
            lines.append(f"  … and {self.n_features - n} smaller factors")
        lines += ["", f"  {self.approximation_note}"]
        return "\n".join(lines)


# ─────────────────────────────────────────────── global patterns


@dataclass(frozen=True)
class BatchExplanation:
    """Explanations for many rows at once — the *global* view, built bottom-up.

    A per-row explanation says why one candidate got their number. Averaging the
    **absolute** log contributions over many rows says which features do the work
    in general, and it is a better global importance measure than LightGBM's
    built-in `gain` or `split` counts for a specific reason `docs/findings.md`
    §4.1 demonstrates: split-count importance rewards high-cardinality columns
    for being *splittable* rather than useful (`n_skills` reaches 12.3% of split
    count while shuffling it costs the model 0.25% of its error).

    **What this still is not.** Every number here is measured on the model, not
    on reality. A feature can dominate SHAP because the model leans on it and be
    worthless on held-out rows — `education` in this project is exactly that. For
    "does this feature earn its place?", permutation importance on held-out rows
    is the measure; for "what is the model doing?", this one is.
    """

    #: One row per explained candidate, one column per (grouped) feature, in log
    #: units. The raw material — everything else here is a summary of it.
    log_contributions: pd.DataFrame
    baseline_log: float
    quantile: float = 0.5

    @property
    def n_rows(self) -> int:
        return len(self.log_contributions)

    @property
    def baseline(self) -> float:
        return float(np.exp(self.baseline_log))

    def to_frame(self) -> pd.DataFrame:
        """Features ranked by mean absolute push, with the direction alongside.

        Columns:

        * ``mean_abs_log`` — the ranking column: typical size of the push,
          regardless of sign.
        * ``mean_swing`` — the same thing as a percentage, which is what a log
          unit *means*: 0.12 is "this feature typically moves a prediction by
          about 12%".
        * ``mean_log`` — the **signed** average. Near zero with a large
          ``mean_abs_log`` is the interesting case: the feature matters a lot and
          pushes both ways, which is a real feature. Large and one-signed means
          the feature mostly shifts everybody the same direction, which a
          constant would have done.
        """
        abs_mean = self.log_contributions.abs().mean()
        signed = self.log_contributions.mean()
        frame = pd.DataFrame(
            {
                "mean_abs_log": abs_mean,
                "mean_swing": np.exp(abs_mean) - 1.0,
                "mean_log": signed,
                "share": abs_mean / abs_mean.sum() if abs_mean.sum() else abs_mean,
            }
        )
        return frame.sort_values("mean_abs_log", ascending=False)

    def describe(self, n: int = 10) -> str:
        ranked = self.to_frame().head(n)
        lines = [
            f"SHAP over {self.n_rows:,} rows (q{self.quantile:g}), "
            f"baseline {format_rupees(self.baseline)}",
            f"  {'feature':<28}{'typical swing':>15}{'share':>9}",
        ]
        lines += [
            f"  {name:<28}{row.mean_swing:>14.1%}{row.share:>9.1%}"
            for name, row in ranked.iterrows()
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────── the explainer


def median_quantile_model(model: object) -> tuple[Any, Any, float]:
    """Dig out the LightGBM regressor for the **median** quantile, and its builder.

    Callers hand us whatever they have — a `ConformalBand`, a `SalaryBandModel`,
    or (in a test) something shaped like one. `ConformalBand` wraps the model it
    calibrates on ``.model``, so we unwrap until we find the object that owns
    three fitted quantile regressors.

    Unwrapping the conformal layer is not a shortcut, it is correct: conformal
    calibration moves the band's **edges** by a constant in log space and leaves
    the median untouched (see `model/conformal.py`). There is nothing about the
    midpoint for it to explain.

    Returns:
        ``(lightgbm regressor, feature builder, the quantile explained)``.
    """
    inner = model
    for _ in range(4):  # a depth cap, so a self-referential object cannot hang us
        if hasattr(inner, "models_"):
            break
        nested = getattr(inner, "model", None)
        if nested is None or nested is inner:
            break
        inner = nested

    fitted = getattr(inner, "models_", None)
    if not fitted:
        raise TypeError(
            f"{type(model).__name__} does not expose fitted quantile models. "
            "Pass a fitted SalaryBandModel, or a ConformalBand wrapping one."
        )
    builder = getattr(inner, "builder", None)
    if builder is None:
        raise TypeError(f"{type(inner).__name__} has no feature builder to transform rows with")

    quantiles = np.asarray(getattr(inner, "quantiles", (0.1, 0.5, 0.9)), dtype=float)
    # Nearest to 0.5 rather than index 1, so a model built on unusual quantiles
    # still gets its most-central model explained instead of whatever sits in the
    # middle slot.
    middle = int(np.argmin(np.abs(quantiles - 0.5)))
    return fitted[middle], builder, float(quantiles[middle])


class BandExplainer:
    """A `shap.TreeExplainer` for the median model, built once and reused.

        explainer = BandExplainer(bundle.band)
        print(explainer.explain(candidate_frame).sentence())

    **Why this is a class and not just a function.** Constructing a
    `TreeExplainer` parses the whole booster; explaining a row afterwards takes
    about a millisecond. Rebuilding it per request would make the cheap part pay
    for the expensive part on every call, so the API holds one of these for the
    process lifetime — the same argument as `PredictionService` holding one
    model.

    `shap` is imported **inside** the constructor rather than at module scope,
    which is deliberate. `shap` pulls in `numba`, which takes seconds to import;
    doing that at module scope would tax every `import paybands.api`, every test
    collection, and every process that never asks for an explanation.
    """

    def __init__(self, model: object) -> None:
        import shap  # noqa: PLC0415 — deliberately lazy; see the class docstring

        self.regressor, self.builder, self.quantile = median_quantile_model(model)
        self.explainer = shap.TreeExplainer(self.regressor)
        self.feature_names: list[str] = list(self.builder.feature_names_)

        # `expected_value` is a scalar for a single-output regressor, but shap
        # has shipped it as a length-1 array in several versions. `ravel` accepts
        # both rather than betting on one.
        self.baseline_log: float = float(np.ravel(self.explainer.expected_value)[0])

        # `skill_typescript` -> `typescript`, so phrasing can restore the casing
        # that `features/skills.py` deliberately threw away. The skill block's
        # own `feature_names_` is the safe-name list in vocabulary order, with
        # `n_skills` appended — so zipping the two lines them up, and
        # ``strict=False`` is what drops that trailing extra column.
        skills = getattr(self.builder, "skills", None)
        self.skill_names: dict[str, str] = dict(
            zip(
                getattr(skills, "feature_names_", []),
                getattr(skills, "vocabulary_", []),
                strict=False,
            )
        )

    @property
    def baseline(self) -> float:
        """The typical training-data prediction, in rupees."""
        return float(np.exp(self.baseline_log))

    # ── the primitive ──────────────────────────────────────────────────

    def _shap_matrix(self, X: pd.DataFrame) -> tuple[NDArray[np.float64], pd.DataFrame]:
        """SHAP values in log units, plus the feature matrix they describe."""
        features = self.builder.transform(X)
        values = np.asarray(self.explainer.shap_values(features), dtype=float)
        if values.ndim == 1:  # a single row can come back flat
            values = values.reshape(1, -1)
        return values, features

    # ── one row ────────────────────────────────────────────────────────

    def explain(
        self,
        X_row: pd.DataFrame,
        *,
        groups: dict[str, tuple[str, ...]] | None = None,
    ) -> Explanation:
        """Explain **one** candidate. See `explain_prediction` for the arguments."""
        if len(X_row) != 1:
            raise ValueError(f"explain() takes exactly one row, got {len(X_row)}")

        values, features = self._shap_matrix(X_row)
        row = features.iloc[0]
        groups = DEFAULT_GROUPS if groups is None else groups

        collected = _group(self.feature_names, values[0], row, groups)
        prediction_log = self.baseline_log + float(values[0].sum())
        prediction = float(np.exp(prediction_log))

        contributions = [
            Contribution(
                feature=name,
                members=members,
                value=value,
                log_contribution=log_value,
                # Exact: exp of a sum of logs is the product of the parts.
                multiplier=float(np.exp(log_value)),
                # Approximate, and knowingly so — Idea 2. "Remove this one
                # factor and the prediction falls by this much."
                rupees=_round_rupees(prediction - prediction / float(np.exp(log_value))),
                phrase=_phrase(name, value, skill_names=self.skill_names),
            )
            for name, members, value, log_value in collected
        ]
        contributions.sort(key=lambda c: abs(c.log_contribution), reverse=True)

        return Explanation(
            contributions=tuple(contributions),
            baseline=_round_rupees(np.exp(self.baseline_log)),
            prediction=_round_rupees(prediction),
            baseline_log=self.baseline_log,
            prediction_log=prediction_log,
            quantile=self.quantile,
        )

    # ── many rows ──────────────────────────────────────────────────────

    def explain_many(
        self,
        X: pd.DataFrame,
        *,
        groups: dict[str, tuple[str, ...]] | None = None,
    ) -> BatchExplanation:
        """Explain a whole frame. See `explain_batch` for the arguments."""
        if len(X) == 0:
            raise ValueError("cannot explain an empty frame")

        values, _ = self._shap_matrix(X)
        groups = DEFAULT_GROUPS if groups is None else groups

        raw = pd.DataFrame(values, columns=self.feature_names, index=X.index)
        member_to_group = {member: name for name, members in groups.items() for member in members}
        # Summing columns is the whole of grouping — SHAP values add, so a group
        # is the sum of its parts with nothing lost. `sort=False` keeps the
        # builder's column order, which is the order the model sees.
        keys = [member_to_group.get(column, column) for column in raw.columns]
        grouped = raw.T.groupby(keys, sort=False).sum().T

        return BatchExplanation(
            log_contributions=grouped,
            baseline_log=self.baseline_log,
            quantile=self.quantile,
        )


def _group(
    names: list[str],
    values: NDArray[np.float64],
    row: pd.Series,
    groups: dict[str, tuple[str, ...]],
) -> list[tuple[str, tuple[str, ...], Any, float]]:
    """Fold grouped columns into one entry each, keeping the builder's order.

    Returns ``(display name, member columns, headline value, summed log value)``.
    The headline value for a group is the value of its **first** listed member —
    for experience that is `years_experience`, the one a human actually said.
    """
    member_to_group = {member: name for name, members in groups.items() for member in members}

    order: list[str] = []
    totals: dict[str, float] = {}
    members: dict[str, list[str]] = {}
    for name, value in zip(names, values, strict=True):
        key = member_to_group.get(name, name)
        if key not in totals:
            order.append(key)
            totals[key] = 0.0
            members[key] = []
        totals[key] += float(value)
        members[key].append(name)

    collected = []
    for key in order:
        headline_column = next(
            (m for m in groups.get(key, (key,)) if m in row.index),
            members[key][0],
        )
        collected.append((key, tuple(members[key]), row.get(headline_column), totals[key]))
    return collected


# ─────────────────────────────────────────────── the two functions to call


def explain_prediction(
    model: object,
    X_row: pd.DataFrame,
    *,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> Explanation:
    """Why did *this* candidate get *this* midpoint? The headline function.

        band = ConformalBand(model).calibrate(cal_df, cal_df["salary_annual"])
        print(explain_prediction(band, one_candidate_frame).sentence())
        # Predicted ₹23,64,000 — a 5000+-person company added ₹3,89,000, …

    Args:
        model: A fitted `SalaryBandModel`, or a `ConformalBand` around one. The
            **median** quantile model is what gets explained: the band's edges
            are the honesty, the midpoint is what people act on, so the midpoint
            is what needs a defence.
        X_row: A **one-row common-schema frame** — the same thing you would pass
            to `predict_band`, not a feature matrix. The model's own
            `FeatureBuilder` does the transform, so an explanation can never
            describe a differently-built matrix from the prediction it explains.
        groups: Columns to merge into one row. Defaults to `DEFAULT_GROUPS`,
            which merges the four experience encodings — Idea 3. Pass ``{}`` for
            the raw per-column view, which is what you want when debugging the
            model rather than talking to a recruiter.

    Returns:
        An `Explanation`, ranked largest push first.

    Builds a fresh `BandExplainer` each call. Fine for a script, wasteful in a
    loop or a server — hold a `BandExplainer` there instead.
    """
    return BandExplainer(model).explain(X_row, groups=groups)


def explain_batch(
    model: object,
    X: pd.DataFrame,
    *,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> BatchExplanation:
    """The same machinery over many rows — what the model relies on *in general*.

    Args:
        model: As `explain_prediction`.
        X: A common-schema frame of any height.
        groups: As `explain_prediction`.

    Returns:
        A `BatchExplanation`. Start with ``.describe()``, then ``.to_frame()``.

    Prefer held-out rows. Run on the training rows this measures what the model
    memorised as much as what it learned, which is a different question and a
    less interesting one.
    """
    return BandExplainer(model).explain_many(X, groups=groups)


def sentence(explanation: Explanation, n: int = DEFAULT_SENTENCE_TOP_N) -> str:
    """Module-level alias for `Explanation.sentence`, for readable pipelines."""
    return explanation.sentence(n)
