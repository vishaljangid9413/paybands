"""Loading the model once, and refusing to hand out a number without its caveat.

`app.py` is HTTP plumbing. This file is where the modelling judgement lives, and
there are four decisions in it worth reading before the code.

---

## Decision 1 — the caveat is a required output, not a nicety

Measured on the real survey data, the calibrated band is **2.40× as wide as its
own midpoint** (median across 10 splits; see docs/findings.md §3). Read that in
rupees: a midpoint of ₹13,95,079 comes with edges at ₹4,91,660 and ₹39,84,224.
Both are real salaries for real engineers, and the band is honest to say so —
82.3% coverage was *measured*, not assumed. But it means the model is describing
a market, not pricing a person.

Note the figure carefully: **1.94× is the RAW band, 2.40× is the calibrated one.**
An earlier version of this docstring quoted 1.94 while describing conformal
results, which is quoting the width of the band that lies about its own
confidence. Honesty costs 23.4% more width, and that cost belongs in the number.

Now consider what an API does with that. It serialises `{"midpoint": 1395079}`,
a UI renders "₹13,95,079", and a recruiter quotes it. Nothing in that chain was
technically false and the outcome is a fabricated number presented as an
estimate. **JSON strips uncertainty by default** — the width has to be carried
as an explicit field and an explicit sentence, or it is gone.

So `build_caveat` always returns a non-empty string, `Confidence.decision_grade`
is a boolean a caller can branch on, and `models.PredictBandResponse.caveat` is
typed `str` rather than `str | None` so it cannot be quietly dropped.

## Decision 2 — where the threshold sits, and why 0.5

`DECISION_GRADE_MAX_RELATIVE_WIDTH = 0.5`. That is a **judgement call, not a
measurement**, so it is one named constant rather than a comparison buried in a
function.

The reasoning is external to this model. A published HR salary band at a large
company typically runs from about 80% to 120% of its midpoint — a relative width
of **0.4**. That is the width compensation teams have decided is narrow enough
to hire against. Allowing 0.5 is already more generous than industry practice;
anything wider is not a band in the sense an HR person means the word.

The current model sits near 2.40. It fails its own test, loudly, on every
request. **That is the correct behaviour for a model that is not ready** — the
alternative is picking a threshold the model happens to pass, which is choosing
the ruler to fit the object.

## Decision 3 — take-home is computed from the midpoint we return, exactly

`compute_payslip` is called on the same rounded midpoint that appears in the
response, so a caller can take `band.midpoint`, the published payroll config,
and reproduce every number under `take_home` to the rupee. Computing it from a
different (unrounded) internal value would make the response internally
inconsistent for a fraction of a percent of accuracy nobody asked for.

And it is applied to the **midpoint only**. Running the calculator on the lower
and upper edges too would imply a "take-home band", which would quietly present
tax arithmetic — the one part of this system with no uncertainty in it — as
though it were uncertain.

## Decision 4 — the service must start without a model

A model artifact is a build output. `models/` is gitignored precisely so a
trained binary never lands in the repository, which means a fresh clone, a fresh
container, and CI all start with no model on disk. If importing `app.py`
unpickled a file at module scope, every one of those would crash on startup and
you could never deploy the service *before* training — the wrong dependency
direction.

So loading is guarded, a failure is recorded rather than raised, `/health`
reports `no_model` with a 503, and prediction endpoints return 503 with the
command to run. `train_bundle` below builds one in a few seconds from synthetic
data.

## Decision 5 — the explanation ships by default, and its failure is not fatal

`ROADMAP.md` Phase 6: a recruiter must be able to argue with the model. So
`/predict-band` computes a SHAP explanation unless asked not to, and the request
flag exists for bulk pay-equity scoring rather than as a performance escape
hatch — `shap.TreeExplainer` is exact and costs about a millisecond per row once
the explainer is built, which is why `_explainer` is built once and cached beside
the model.

But `_explanation` catches everything. An explanation is an *addition* to a band,
and a band the caller can use beats a 500 they cannot. The failure is reported in
`notes` rather than swallowed, because a silently missing explanation is a
degraded product that nobody notices has degraded.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import synthetic
from ..data.schema import TARGET
from ..explain.shapley import BandExplainer, Explanation
from ..features.location import DEFAULT_TIER as DEFAULT_CITY_TIER
from ..features.location import city_tier
from ..model.band import BandPrediction, SalaryBandModel, compa_label
from ..model.conformal import ConformalBand, three_way_split
from ..model.metrics import format_rupees
from ..model.split import DEFAULT_SEED
from ..payroll.calculator import PayrollRules, compute_payslip
from . import models as api_models

# ─────────────────────────────────────────────── where things live

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default artifact location. Under `models/`, which `.gitignore` covers — a
#: pickled LightGBM model is a build output, not source, and committing one
#: means the repository silently carries a model nobody can reproduce.
DEFAULT_MODEL_PATH = Path(os.environ.get("PAYBANDS_MODEL_PATH", _REPO_ROOT / "models/band.pkl"))

PAYROLL_CONFIG_DIR = _REPO_ROOT / "configs/payroll"

# ─────────────────────────────────────────────── the honesty thresholds
#
# Judgement calls, gathered here so anyone reviewing the project can disagree
# with them in one place instead of hunting. See Decision 2 in the module
# docstring for where 0.5 comes from.

#: Above this ``(upper − lower) ÷ midpoint``, the band is not something to quote
#: in an offer. Set against real HR practice (bands run ~80%–120% of midpoint,
#: i.e. 0.4), not against what this model currently achieves.
DECISION_GRADE_MAX_RELATIVE_WIDTH = 0.5

#: Below this, the band is genuinely tight — ±15% around the midpoint.
TIGHT_RELATIVE_WIDTH = 0.3

#: What the CALIBRATED band measured on the Stack Overflow India rows: median
#: width ÷ midpoint across 10 splits. Kept as a constant purely so the number in
#: the docs and the number in the code cannot drift apart. Not used in any
#: calculation.
#:
#: It drifted anyway. This was 1.9 — the RAW band's width — while every
#: surrounding sentence described the calibrated one. The constant existed to
#: prevent exactly that and did not, because nothing checks it against
#: docs/findings.md. If you change the model, re-run scripts/run_analysis.py and
#: update this by hand; a comment holding a number is a comment that can rot.
MEASURED_RELATIVE_WIDTH_ON_SURVEY_DATA = 2.40

#: Rupee rounding for everything the API reports. Real offers are made in round
#: numbers and the model's precision does not remotely justify reporting to the
#: rupee — ₹18,73,412.77 implies an accuracy the band's own width disproves.
#:
#: `explain/shapley.RUPEE_ROUNDING` is the same number, deliberately: the
#: explanation's `prediction` and the band's `midpoint` are the same quantity
#: rounded twice, and two different rounding rules would make a response
#: disagree with itself by a few hundred rupees for no reason anyone could find.
RUPEE_ROUNDING = 1_000

#: How many contributions `/predict-band` returns. Five, because an explanation
#: is a thing a person repeats out loud — and `docs/findings.md` §4.1 found only
#: nine of this model's 37 features move held-out error at all, so a longer list
#: would be mostly noise dressed as reasons.
EXPLANATION_TOP_N = 5


# ─────────────────────────────────────────────── the loaded artifact


@dataclass(frozen=True)
class ModelBundle:
    """Everything a prediction needs, loaded once and never mutated.

    Carries its own provenance. `holdout_coverage` and `holdout_relative_width`
    were measured on the test split at training time, on rows neither the trees
    nor the conformal calibration ever saw — they are the numbers `/health` and
    every response report, and they are the only basis on which anyone should
    decide to trust this service.
    """

    band: ConformalBand
    version: str
    trained_on: str
    n_train: int
    holdout_coverage: float | None = None
    holdout_relative_width: float | None = None

    @property
    def known_categories(self) -> dict[str, list[object]]:
        """The category vocabularies frozen at fit time.

        Used to tell a caller their `role` was never seen in training. Reaching
        through the conformal wrapper to the underlying model is deliberate: the
        wrapper adjusts band *edges* and knows nothing about features.
        """
        builder = getattr(self.band.model, "builder", None)
        return dict(getattr(builder, "category_levels_", {}))

    def to_info(self) -> api_models.ModelInfo:
        return api_models.ModelInfo(
            version=self.version,
            trained_on=self.trained_on,
            n_train=self.n_train,
            holdout_relative_width=self.holdout_relative_width,
        )


# ─────────────────────────────────────────────── training a local artifact


def train_bundle(
    df: pd.DataFrame | None = None,
    *,
    n_synthetic: int = 4_000,
    seed: int = DEFAULT_SEED,
    label: str | None = None,
) -> ModelBundle:
    """Fit, calibrate and score a band model — the whole Phase 4–5 pipeline in one call.

    Args:
        df: A common-schema frame. `None` generates synthetic data, so the
            service is runnable on a clone with no data files present.
        n_synthetic: Rows to invent when `df` is None.
        seed: Same seed, same model.
        label: What to report in `trained_on`.

    **The three-way split is not optional here.** `conformal.py` explains it at
    length: calibrating ``q̂`` on rows the trees were fitted to produces small
    conformity scores, a band that barely widens, and a published coverage
    guarantee that is simply false — silently, with better-looking numbers than
    the honest version. `three_way_split` makes the correct cut and
    `ConformalBand.calibrate` refuses to run if the sets overlap.

    The test slice is spent on measuring coverage and width. That measurement is
    what the caveat is built from, so a bundle without it would be a model that
    cannot say how uncertain it is.
    """
    if df is None:
        df, _ = synthetic.generate(n=n_synthetic, seed=seed)
        label = label or f"synthetic ({n_synthetic:,} rows, seed {seed}) — NOT real market data"
    label = label or "caller-supplied frame"

    split = three_way_split(df, seed=seed)
    model = SalaryBandModel(seed=seed).fit(split.train, split.train[TARGET])
    band = ConformalBand(model).calibrate(split.calibration, split.calibration[TARGET])

    report = band.evaluate(split.test, split.test[TARGET])
    return ModelBundle(
        band=band,
        version=f"band-lgbm-cqr-seed{seed}",
        trained_on=label,
        n_train=len(split.train),
        holdout_coverage=report.coverage,
        holdout_relative_width=report.mean_relative_width,
    )


def save_bundle(bundle: ModelBundle, path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Pickle a bundle to disk. Never commit the result — see `.gitignore`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return path


def load_bundle(path: Path = DEFAULT_MODEL_PATH) -> ModelBundle:
    """Unpickle a bundle.

    ``pickle`` executes whatever it reads, so this must only ever be pointed at
    an artifact you produced yourself. It is the right tool here — a LightGBM
    booster, a fitted `FeatureBuilder` and a calibrated ``q̂`` travel together or
    not at all, and splitting them into a "safe" format invites the failure
    where the model and its calibration go out of sync.
    """
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} does not contain a ModelBundle (got {type(bundle).__name__})")
    return bundle


def load_payroll_rules(path: Path | None = None) -> PayrollRules:
    """Load the payroll config, defaulting to the latest financial year on disk.

    Latest-by-filename, because `configs/payroll/` is versioned by financial
    year and never edited in place (see the header of `fy_2025_26.yaml`). Serving
    the newest file is right for a live quote; historical recomputation passes an
    explicit path.
    """
    if path is None:
        env = os.environ.get("PAYBANDS_PAYROLL_CONFIG")
        if env:
            path = Path(env)
        else:
            candidates = sorted(PAYROLL_CONFIG_DIR.glob("fy_*.yaml"))
            if not candidates:
                raise FileNotFoundError(f"no payroll config found in {PAYROLL_CONFIG_DIR}")
            path = candidates[-1]
    return PayrollRules.from_yaml(path)


# ─────────────────────────────────────────────── candidate → model input


#: Request fields that map straight onto common-schema columns.
_PASSTHROUGH_FIELDS: tuple[str, ...] = (
    "years_experience",
    "role",
    "education",
    "org_size",
    "remote",
    "employment_type",
    "level",
    "institute_tier",
    "prev_company_type",
    "performance_rating",
)

#: Which request fields are checked against the model's learned vocabularies.
_CATEGORICAL_FIELDS: tuple[str, ...] = (
    "role",
    "education",
    "org_size",
    "remote",
    "employment_type",
    "level",
    "institute_tier",
    "prev_company_type",
)


def candidate_to_frame(candidate: api_models.Candidate) -> pd.DataFrame:
    """One request → a one-row common-schema frame.

    Only fields the caller actually sent become columns. Omitted fields are left
    *out* rather than written as NaN, and the result is identical either way —
    `FeatureBuilder.transform` emits an all-NaN column for anything absent, so
    the feature matrix keeps its width regardless. Building it this way makes
    the frame a faithful record of the request, which is what you want when
    debugging why a band came back wide.
    """
    row: dict[str, object] = {}
    for field in _PASSTHROUGH_FIELDS:
        value = getattr(candidate, field)
        if value is not None:
            row[field] = value

    # `LocationFeatures` prefers a `city` column over `location_tier` when both
    # are present, and that precedence is right: the city is the more specific
    # claim and the tier is derived from it. Sending only one is the normal case.
    if candidate.city is not None:
        row["city"] = candidate.city
    elif candidate.location_tier is not None:
        row["location_tier"] = candidate.location_tier

    if candidate.skills:
        # `parse_skills` splits on ";" — the survey's own format, kept so the
        # serving path and the training path parse skills through exactly the
        # same function.
        row["skills"] = ";".join(candidate.skills)

    # NOTE: `prev_salary` is deliberately not carried across. See
    # docs/design.md §4.1 and `models.Candidate.prev_salary`.
    return pd.DataFrame([row], index=[0])


def unrecognised_inputs(candidate: api_models.Candidate, bundle: ModelBundle) -> list[str]:
    """Fields the caller sent whose value the model has never seen.

    Not an error, and specifically not a 500. `FeatureBuilder` Decision 2 maps
    an unseen category to NaN, which is the honest encoding — the model has no
    learned response to "Prompt Engineer" if no such row existed in training, so
    it answers the more general question instead and the band widens to match.

    Reporting it matters because that widening is the *only* trace left. A
    caller who sees a plausible band for a role we have never priced would
    otherwise have no way to know their most important input was discarded.
    """
    unknown: list[str] = []
    vocabularies = bundle.known_categories
    for field in _CATEGORICAL_FIELDS:
        value = getattr(candidate, field)
        if value is None:
            continue
        levels = vocabularies.get(field)
        # `str(v)` mirrors `_categorical_block`, which casts before comparing so
        # that 3 and "3" cannot become two different categories.
        if levels is not None and str(value) not in {str(v) for v in levels}:
            unknown.append(field)
    # A city is only *unrecognised* when it cannot be read at all — "..." or
    # "12345", which `normalise_city` reduces to nothing and `city_tier` returns
    # NaN for. A city it simply hasn't heard of is a different case; see
    # `_notes` below.
    if candidate.city is not None and np.isnan(city_tier(candidate.city)):
        unknown.append("city")
    return unknown


# ─────────────────────────────────────────────── the honesty layer


def confidence_level(relative_width: float, *, degraded: bool = False) -> str:
    """Map band width onto a three-value label.

    Three values, not a score. A `confidence: 0.62` invites the caller to invent
    their own threshold, and the threshold that matters — is this quotable? — is
    already published as `decision_grade`.

    `degraded` caps the label at "moderate" when an input was unrecognised. The
    reason is subtle and worth stating: an unknown role becomes NaN, LightGBM
    routes NaN down whichever branch fits training best, and all three quantile
    models route it the *same* way — so the band can come back deceptively tight
    while resting on a feature the model never actually read.
    """
    if relative_width <= TIGHT_RELATIVE_WIDTH:
        return "moderate" if degraded else "high"
    if relative_width <= DECISION_GRADE_MAX_RELATIVE_WIDTH:
        return "moderate"
    return "low"


def build_caveat(
    band: api_models.Band,
    *,
    unknown: list[str],
    holdout_coverage: float | None,
) -> str:
    """The sentence that stops this API from being misleading. Never empty.

    Structure is fixed so a UI can render it verbatim or split on ". ":

    1. what the band's width means, in rupees a person recognises;
    2. whether it is decision-grade, in the imperative;
    3. which inputs were ignored, if any;
    4. that the number is base salary, not take-home, not CTC.

    Point 4 is on *every* response, including the tightest ones, because "₹19L"
    means at least three different things in an Indian salary conversation and
    the API should never be the ambiguous party.
    """
    parts: list[str] = [
        f"This band spans {band.relative_width:.0%} of its own midpoint "
        f"({format_rupees(band.lower)} to {format_rupees(band.upper)} "
        f"around {format_rupees(band.midpoint)})."
    ]

    if band.relative_width > DECISION_GRADE_MAX_RELATIVE_WIDTH:
        parts.append(
            "NOT DECISION-GRADE: that is wider than the "
            f"{DECISION_GRADE_MAX_RELATIVE_WIDTH:.0%}-of-midpoint limit a band has to meet "
            "to be quotable (published HR bands typically run 80%-120% of midpoint). "
            "Use it as a rough market orientation only — do not put these numbers in "
            "an offer, a budget, or a pay-review decision."
        )
    else:
        parts.append(
            "The band is inside the "
            f"{DECISION_GRADE_MAX_RELATIVE_WIDTH:.0%}-of-midpoint limit, so it is tight "
            "enough to reason about — but it is still a range, and the midpoint on "
            "its own is not an estimate of this person's salary."
        )

    if unknown:
        parts.append(
            f"The model has never seen the value you sent for: {', '.join(unknown)}. "
            "Those inputs were treated as unknown, so this band answers a more "
            "general question than the one you asked."
        )

    if holdout_coverage is not None:
        parts.append(
            f"Coverage was measured at {holdout_coverage:.0%} on held-out rows against a "
            f"promise of {band.coverage:.0%} — the band edges are calibrated, the "
            "midpoint is not a guarantee."
        )

    parts.append(
        "All figures are annual gross BASE salary, not CTC and not take-home; "
        "the take-home block is computed from the midpoint by payroll rules, "
        "not predicted."
    )
    return " ".join(parts)


# ─────────────────────────────────────────────── the service


class ModelNotLoaded(RuntimeError):
    """Raised when a prediction is asked for and no artifact was found.

    A named exception rather than an HTTPException so this module stays free of
    HTTP: `app.py` decides that this maps to 503.
    """


class PredictionService:
    """The model, the payroll rules, and the rules for talking about both.

    Constructed **once**, at startup, and held for the process lifetime. Loading
    per request would re-read a pickle, rebuild three LightGBM boosters and
    re-parse a YAML config on every call — but the real cost is correctness, not
    latency: two requests could then be answered by two different artifacts if
    the file changed underneath, and nothing in either response would say so.
    """

    def __init__(
        self,
        bundle: ModelBundle | None = None,
        rules: PayrollRules | None = None,
        *,
        load_error: str | None = None,
    ) -> None:
        self.bundle = bundle
        self.rules = rules
        self.load_error = load_error

        # Built on first use rather than here, for two reasons. `shap` imports
        # `numba`, which costs seconds — and a service constructed with no model
        # (the fresh-deployment case, Decision 4) has nothing to build an
        # explainer *from*. Lazy makes both work with one line.
        self._explainer: BandExplainer | None = None
        self._explainer_error: str | None = None

    # ── construction ───────────────────────────────────────────────────

    @classmethod
    def from_disk(cls, model_path: Path = DEFAULT_MODEL_PATH) -> PredictionService:
        """Load what is there, record what is not, and never raise.

        A missing or unreadable artifact is a *state* this service reports, not
        an error that stops it existing. See Decision 4 in the module docstring.
        """
        try:
            rules: PayrollRules | None = load_payroll_rules()
        except Exception as exc:  # noqa: BLE001 — any config failure is reportable
            return cls(load_error=f"payroll config could not be loaded: {exc}")

        path = Path(model_path)
        if not path.exists():
            return cls(
                rules=rules,
                load_error=(
                    f"no model artifact at {path}. Train one with: "
                    "uv run python -m paybands.api.service"
                ),
            )
        try:
            return cls(bundle=load_bundle(path), rules=rules)
        except Exception as exc:  # noqa: BLE001 — a bad pickle must not kill startup
            return cls(rules=rules, load_error=f"model at {path} could not be loaded: {exc}")

    @property
    def is_ready(self) -> bool:
        return self.bundle is not None and self.rules is not None

    def _require(self) -> tuple[ModelBundle, PayrollRules]:
        if self.bundle is None or self.rules is None:
            raise ModelNotLoaded(self.load_error or "model not loaded")
        return self.bundle, self.rules

    # ── the prediction path ────────────────────────────────────────────

    def _band_for(self, candidate: api_models.Candidate) -> tuple[BandPrediction, api_models.Band]:
        bundle, _ = self._require()
        frame = candidate_to_frame(candidate)
        raw = bundle.band.predict_band(frame)

        lower, mid, upper = (
            _round_rupees(raw.lower[0]),
            _round_rupees(raw.median[0]),
            _round_rupees(raw.upper[0]),
        )
        # Rounding is monotone, so lower <= mid <= upper survives it — the
        # conformal wrapper already clipped the edges to the median so the band
        # cannot invert.
        band = api_models.Band(
            lower=lower,
            midpoint=mid,
            upper=upper,
            coverage=raw.nominal_coverage,
            width=upper - lower,
            relative_width=(upper - lower) / mid,
        )
        return raw, band

    def _confidence(self, band: api_models.Band, *, unknown: list[str]) -> api_models.Confidence:
        bundle, _ = self._require()
        return api_models.Confidence(
            level=confidence_level(band.relative_width, degraded=bool(unknown)),
            decision_grade=band.relative_width <= DECISION_GRADE_MAX_RELATIVE_WIDTH,
            relative_width=band.relative_width,
            relative_width_threshold=DECISION_GRADE_MAX_RELATIVE_WIDTH,
            nominal_coverage=band.coverage,
            measured_coverage=bundle.holdout_coverage,
        )

    # ── the explanation path ───────────────────────────────────────────

    def explainer(self) -> BandExplainer:
        """The process's one `BandExplainer`, built on first use.

        Constructing it parses the whole LightGBM booster; explaining a row
        afterwards takes about a millisecond. Rebuilding per request would make
        the cheap part pay for the expensive part on every call — the same
        argument as holding one `PredictionService` rather than reloading the
        pickle.

        Cached even on failure (`_explainer_error`), so a model `shap` cannot
        read is diagnosed once instead of on every request.
        """
        bundle, _ = self._require()
        if self._explainer is None:
            if self._explainer_error is not None:
                raise RuntimeError(self._explainer_error)
            self._explainer = BandExplainer(bundle.band)
        return self._explainer

    def _explanation(self, candidate: api_models.Candidate) -> Explanation:
        """The SHAP explanation for one candidate, in log-space-honest form."""
        frame = candidate_to_frame(candidate)
        return self.explainer().explain(frame)

    @staticmethod
    def _to_api_explanation(explanation: Explanation) -> api_models.Explanation:
        """`explain.shapley.Explanation` → the wire shape.

        Truncated to `EXPLANATION_TOP_N`, with `n_factors` carrying the full
        count so a caller can tell a summary from the whole story. Nothing is
        recomputed here: every number was already rounded by `shapley`, using the
        same `RUPEE_ROUNDING` the band uses, so `explanation.prediction` and
        `band.midpoint` are the same rupee.
        """
        top = explanation.contributions[:EXPLANATION_TOP_N]
        return api_models.Explanation(
            summary=explanation.sentence(),
            baseline=explanation.baseline,
            prediction=explanation.prediction,
            contributions=[
                api_models.Contribution(
                    feature=c.feature,
                    phrase=c.phrase,
                    direction=c.direction,  # type: ignore[arg-type]
                    rupees=c.rupees,
                    multiplier=c.multiplier,
                    log_contribution=c.log_contribution,
                )
                for c in top
            ],
            n_factors=explanation.n_features,
            approximation_note=explanation.approximation_note,
        )

    def _take_home(self, midpoint: float) -> api_models.TakeHome:
        """Layer 2, applied to the midpoint we are about to return.

        The same `midpoint` value appears in the response, so a caller with the
        payroll config can reproduce every field here exactly. Only the midpoint
        is run through the calculator — see Decision 3.
        """
        _, rules = self._require()
        slip = compute_payslip(midpoint, rules)
        return api_models.TakeHome(
            annual_gross=slip.annual_gross,
            annual_net=slip.annual_net,
            monthly_gross=slip.monthly_gross,
            monthly_net=slip.monthly_net,
            annual_pf=slip.annual_pf,
            annual_insurance=slip.annual_insurance,
            annual_professional_tax=slip.annual_professional_tax,
            annual_income_tax=slip.annual_income_tax,
            financial_year=rules.financial_year,
            regime=rules.regime,
        )

    @staticmethod
    def _notes(candidate: api_models.Candidate) -> list[str]:
        notes: list[str] = []
        if candidate.prev_salary is not None:
            notes.append(
                "prev_salary was accepted and deliberately NOT used. Anchoring an offer "
                "to last drawn pay makes an early underpayment follow someone for their "
                "whole career; see docs/design.md §4.1."
            )
        if candidate.city is not None and candidate.location_tier is not None:
            notes.append("Both city and location_tier were sent; city took precedence.")
        if candidate.city is not None and city_tier(candidate.city) == DEFAULT_CITY_TIER:
            # `features/location.py` is explicit that tier 3 is a *claim* — "a
            # real place that is not a tech metro" — and not a shrug. It is also
            # the catch-all, so a misspelling the alias table does not cover
            # lands here silently and gets priced as a non-metro. Saying so is
            # the difference between a considered answer and a typo.
            notes.append(
                f"'{candidate.city}' is not in the tier-1 or tier-2 city list, so it was "
                "priced as a tier-3 (non-metro) location. Check the spelling if that is "
                "not what you meant — the metro premium is large."
            )
        return notes

    def predict_band(
        self,
        candidate: api_models.Candidate,
        *,
        explain: bool = True,
    ) -> api_models.PredictBandResponse:
        """Candidate → band, take-home, confidence, caveat, and why.

        Args:
            candidate: The request payload's candidate block.
            explain: Attach the SHAP explanation. Default true — see Decision 5.
                Turn it off for bulk scoring, where nobody reads ten thousand
                sentences and the millisecond each starts to add up.
        """
        bundle, _ = self._require()
        _, band = self._band_for(candidate)
        unknown = unrecognised_inputs(candidate, bundle)
        notes = self._notes(candidate)

        explanation: api_models.Explanation | None = None
        if explain:
            try:
                explanation = self._to_api_explanation(self._explanation(candidate))
            except Exception as exc:  # noqa: BLE001 — see Decision 5
                # The band is still correct and still useful; only the argument
                # for it is missing. Say so rather than returning a 500 or, worse,
                # a quietly explanation-free response that looks complete.
                self._explainer_error = f"{type(exc).__name__}: {exc}"
                notes.append(
                    "The band could not be explained on this request "
                    f"({self._explainer_error}). The prediction itself is unaffected, but "
                    "treat a number you cannot interrogate with more caution, not less."
                )

        return api_models.PredictBandResponse(
            band=band,
            take_home=self._take_home(band.midpoint),
            confidence=self._confidence(band, unknown=unknown),
            explanation=explanation,
            caveat=build_caveat(band, unknown=unknown, holdout_coverage=bundle.holdout_coverage),
            unrecognised_inputs=unknown,
            notes=notes,
            model=bundle.to_info(),
        )

    def compa_ratio(self, request: api_models.CompaRatioRequest) -> api_models.CompaRatioResponse:
        """``actual ÷ midpoint``, with the band that produced the denominator.

        The denominator is the **median**, not the arithmetic middle of lower and
        upper. On a right-skewed distribution those differ, and the median is what
        HR means by "the middle of the band" — the salary half the market is
        below. `SalaryBandModel.compa_ratio` divides by the same thing.

        The band is returned alongside on purpose. A compa-ratio is a single
        number derived from an uncertain one, so "0.84, underpaid" carries all
        the uncertainty of the midpoint it was divided by and none of it is
        visible in the ratio itself.
        """
        bundle, _ = self._require()
        candidate = request.candidate
        _, band = self._band_for(candidate)
        unknown = unrecognised_inputs(candidate, bundle)

        ratio = request.actual_salary / band.midpoint
        return api_models.CompaRatioResponse(
            compa_ratio=ratio,
            position=compa_label(ratio),
            actual_salary=request.actual_salary,
            band=band,
            confidence=self._confidence(band, unknown=unknown),
            caveat=build_caveat(band, unknown=unknown, holdout_coverage=bundle.holdout_coverage),
            unrecognised_inputs=unknown,
            notes=self._notes(candidate),
            model=bundle.to_info(),
        )

    # ── health ─────────────────────────────────────────────────────────

    def health(self) -> api_models.HealthResponse:
        if self.bundle is None or self.rules is None:
            return api_models.HealthResponse(
                status="no_model",
                model_loaded=False,
                payroll_financial_year=self.rules.financial_year if self.rules else None,
                payroll_regime=self.rules.regime if self.rules else None,
                detail=self.load_error or "model not loaded",
            )
        coverage = self.bundle.holdout_coverage
        width = self.bundle.holdout_relative_width
        detail = (
            f"model loaded; held-out coverage "
            f"{'unmeasured' if coverage is None else f'{coverage:.0%}'}, "
            f"mean band width "
            f"{'unmeasured' if width is None else f'{width:.2f}x midpoint'}"
        )
        if width is not None and width > DECISION_GRADE_MAX_RELATIVE_WIDTH:
            detail += " — WIDER than the decision-grade limit, so predictions are advisory only"
        return api_models.HealthResponse(
            status="ok",
            model_loaded=True,
            model=self.bundle.to_info(),
            payroll_financial_year=self.rules.financial_year,
            payroll_regime=self.rules.regime,
            detail=detail,
        )


def _round_rupees(value: float) -> float:
    """Round to `RUPEE_ROUNDING`, and never below ₹1 — `compute_payslip` rejects
    a non-positive gross, and a band edge that rounded to zero would take the
    whole request down with it."""
    return max(float(np.round(float(value) / RUPEE_ROUNDING) * RUPEE_ROUNDING), 1.0)


# ─────────────────────────────────────────────── the process-wide singleton

_service: PredictionService | None = None


def get_service() -> PredictionService:
    """The one `PredictionService` this process will ever build.

    Lazy rather than import-time so that importing `app` — which happens in
    tests, in `--reload` workers, and in any tooling that inspects the OpenAPI
    schema — does not touch the filesystem. `app.py`'s lifespan calls this at
    startup so the first real request never pays for the load.
    """
    global _service
    if _service is None:
        _service = PredictionService.from_disk()
    return _service


def set_service(service: PredictionService | None) -> None:
    """Replace the singleton. For tests and for the training entry point below."""
    global _service
    _service = service


# ─────────────────────────────────────────────── local training entry point


def main() -> None:  # pragma: no cover — a convenience script, not library code
    """`uv run python -m paybands.api.service` — train a model and cache it.

    Deliberately trains on **synthetic** data by default. It is reproducible
    from a fresh clone with no data files, which is what makes the API runnable
    locally, and `trained_on` says loudly that the numbers are invented so
    nobody mistakes a demo band for a market rate. Point it at a real frame in
    Python when you want a real model.
    """
    bundle = train_bundle()
    path = save_bundle(bundle)
    coverage = bundle.holdout_coverage or float("nan")
    width = bundle.holdout_relative_width or float("nan")
    print(f"trained on {bundle.trained_on}")
    print(f"  held-out coverage      {coverage:.1%}")
    print(f"  mean width ÷ midpoint  {width:.2f}")
    if width > DECISION_GRADE_MAX_RELATIVE_WIDTH:
        print(
            f"  ⚠️  wider than the {DECISION_GRADE_MAX_RELATIVE_WIDTH:.0%} decision-grade "
            "limit — every response will carry a not-decision-grade caveat"
        )
    print(f"saved to {path}  (gitignored — do not commit)")


if __name__ == "__main__":  # pragma: no cover
    # Re-import rather than calling the `main` defined above. Run as
    # `python -m paybands.api.service`, this module is executed under the name
    # `__main__`, so `ModelBundle` here is `__main__.ModelBundle` — and pickle
    # records that path. The artifact then fails to load in the API process,
    # where the class is `paybands.api.service.ModelBundle`. Importing the module
    # under its real name first means the bundle is pickled with the name every
    # other process will look it up by.
    from paybands.api.service import main as _main

    _main()
