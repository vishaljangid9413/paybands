"""The wire contract — what a caller sends, and what they get back.

This file is deliberately the *only* place the JSON shape is defined, because a
recruiter-facing UI will be built against it and then nobody will want to change
it. Everything here is a stable field name plus the reason it exists.

Three modelling ideas are baked into the shape, and they are worth stating
before the code:

**1. There is no `salary` field, anywhere, and there never will be.** The
response returns `lower`, `midpoint` and `upper`. A single number is precisely
the fake precision this project was built to avoid — and once a field called
`salary` exists in a JSON payload, some UI will render it alone and the band
will have been for nothing.

**2. `caveat` is a required field, not an optional one.** See
`service.build_caveat`. Making it `str` rather than `str | None` means a client
that renders every field it receives cannot accidentally drop the warning, and a
future edit that forgets to set it fails at serialisation time rather than
shipping a confident-looking number.

**3. The take-home block is arithmetic, not prediction.** `docs/design.md` §1:
the model predicts base salary and nothing else. Every field under `take_home`
comes out of `payroll/calculator.py` applied to the midpoint we just returned —
so a caller can recompute it themselves from the published config and get the
same rupee. Keeping them in one response but under a separate key is the API
saying out loud which half was learned and which half was computed.

**4. The explanation ships by default, not on request.** `Explanation` is the
block that lets a recruiter *argue* with the band, and an explanation a caller
has to opt into is one nobody sees. `PredictRequest.explain` therefore defaults
to `True`; it exists so bulk pay-equity scoring can turn it off, not so a UI can
forget to turn it on.

**Validation philosophy.** Requests use ``extra="forbid"``. A misspelt field
(`"experience"` instead of `"years_experience"`) would otherwise be silently
dropped, the model would treat that input as unknown, and the caller would get a
plausible-looking band computed from less information than they thought they had
sent. That failure is invisible in the response. A 422 is not.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..data.schema import MAX_PLAUSIBLE_SALARY, MIN_PLAUSIBLE_SALARY
from ..features.experience import MAX_PLAUSIBLE_YEARS

#: Free-text categorical inputs (role, city, education...). Stripped, non-empty,
#: and length-capped. The cap is not a security measure so much as an honesty
#: one: a 4,000-character "role" is not a role, and letting it through would put
#: a junk category in front of the model.
CategoryStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


# ─────────────────────────────────────────────── request


class Candidate(BaseModel):
    """One person, in the project's common schema (`data/schema.py`).

    Every field except `years_experience` is optional, and that is a modelling
    decision rather than laziness. `FeatureBuilder` (Decision 3) never fills a
    missing value with a default — LightGBM routes NaN down whichever branch
    fits the data best, which is a better answer than any guess this layer could
    invent. So "I don't know this candidate's institute tier" is a legitimate,
    well-handled request; "institute_tier: other" asserted on their behalf is
    not.

    The consequence a caller should understand: the fewer fields you send, the
    *wider* the band comes back, and the response says so in `caveat`. That is
    the system working, not degrading.
    """

    model_config = ConfigDict(extra="forbid")

    # ── the one field that is genuinely required ──────────────────────
    #
    # Bounded by `features.experience.MAX_PLAUSIBLE_YEARS` rather than by a
    # number typed here, so the API and the training-time cleaning rule can
    # never disagree about what "impossible" means. 200 years is a 422; the
    # feature builder would have turned it into NaN and answered anyway, and
    # answering a nonsense question with a straight face is worse than refusing.
    years_experience: float = Field(ge=0, le=MAX_PLAUSIBLE_YEARS)

    # ── everything the model can use, all optional ────────────────────
    role: CategoryStr | None = None
    education: CategoryStr | None = None
    org_size: CategoryStr | None = None
    remote: CategoryStr | None = None
    employment_type: CategoryStr | None = None
    level: CategoryStr | None = None
    institute_tier: CategoryStr | None = None
    prev_company_type: CategoryStr | None = None

    #: Free text — "Bengaluru", "BLR", "Gurugram, Haryana" all work, via
    #: `features.location.CITY_ALIASES`. If both `city` and `location_tier` are
    #: sent, the city wins: it is the more specific claim, and the tier is
    #: derivable from it but not the other way round.
    city: CategoryStr | None = None
    location_tier: int | None = Field(default=None, ge=1, le=3)

    skills: list[CategoryStr] | None = Field(default=None, max_length=50)

    performance_rating: float | None = Field(default=None, ge=0, le=5)

    #: Accepted, ignored, and that is the point. See `docs/design.md` §4.1 — the
    #: previous-salary trap. Anchoring to last drawn pay makes an early
    #: underpayment follow someone for their whole career, so the default
    #: `FeatureBuilder` is built with `include_prev_salary=False` and this value
    #: never reaches a feature matrix. It is on the request so a caller who has
    #: it can send it and see, in `notes`, that we deliberately declined to use
    #: it.
    prev_salary: float | None = Field(default=None, gt=0, le=MAX_PLAUSIBLE_SALARY)


class CompaRatioRequest(BaseModel):
    """A candidate plus what they are actually paid.

    The salary bounds are `data/schema.py`'s plausibility thresholds, reused
    rather than retyped. Rows outside them were dropped from training, so the
    model has genuinely never seen a ₹50,000/year or ₹5Cr/year engineer, and a
    compa-ratio computed against a midpoint extrapolated that far out would be a
    number with no evidence behind it.
    """

    model_config = ConfigDict(extra="forbid")

    candidate: Candidate
    actual_salary: float = Field(ge=MIN_PLAUSIBLE_SALARY, le=MAX_PLAUSIBLE_SALARY)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: Candidate

    #: Whether to compute the SHAP explanation. **Defaults to true**, and that
    #: default is the point: an explanation a caller has to opt into is an
    #: explanation nobody sees, and a salary number nobody can argue with is the
    #: kind that causes harm (`ROADMAP.md` Phase 6).
    #:
    #: It is a flag rather than a fixed behaviour only because the cost is real
    #: but small — `shap.TreeExplainer` is exact and takes about a millisecond
    #: per row once built, so this exists for bulk scoring (running ten thousand
    #: employees through `/predict-band` for the pay-equity list, where nobody
    #: reads ten thousand sentences), not for shaving a single request.
    explain: bool = True


# ─────────────────────────────────────────────── response pieces


class Band(BaseModel):
    """The band, in rupees. Three numbers, never one."""

    lower: float
    midpoint: float
    upper: float
    currency: Literal["INR"] = "INR"

    #: What fraction of people like this the band claims to contain, *after*
    #: conformal calibration (`model/conformal.py`). 0.8 means "we expect 8 in
    #: 10 real salaries for this profile to land inside these edges" — and that
    #: claim has been measured on held-out rows, not assumed from the quantile
    #: levels we asked for.
    coverage: float

    width: float

    #: ``(upper − lower) ÷ midpoint``. **The number to read first.** Rupee width
    #: is not comparable across salary levels — ₹6L is enormous at ₹8L and tight
    #: at ₹60L — but 0.4 is 0.4 everywhere. Everything in `confidence` is a
    #: function of this.
    relative_width: float


class TakeHome(BaseModel):
    """Layer 2 — computed from the midpoint, never predicted.

    A caller can reproduce every one of these from `configs/payroll/` and the
    midpoint above. That is the test of whether a number was learned or
    calculated, and these are all calculated.
    """

    annual_gross: float
    annual_net: float
    monthly_gross: float
    monthly_net: float

    annual_pf: float
    annual_insurance: float
    annual_professional_tax: float
    annual_income_tax: float

    #: Which config file produced the deductions. Included because take-home
    #: changes when the Union Budget changes while market value does not — if
    #: someone's in-hand moved, this field is the first place to look.
    financial_year: str
    regime: str


class Confidence(BaseModel):
    """Whether this band is tight enough to act on. Read `decision_grade` first."""

    #: Coarse label derived from `relative_width` alone. Deliberately three
    #: values, not a percentage: a "confidence score" of 0.62 invites a caller
    #: to build a threshold on it, and the threshold that matters is already
    #: here as `decision_grade`.
    level: Literal["high", "moderate", "low"]

    #: **The field to branch on.** False means: show the band as a rough market
    #: orientation, do not quote it in an offer letter.
    decision_grade: bool

    relative_width: float
    relative_width_threshold: float

    #: What the band promises for this prediction...
    nominal_coverage: float
    #: ...and what the same model actually delivered on held-out rows it had
    #: never seen. Read together. A band that promises 80% and delivers 70% is
    #: lying to a recruiter; this is where you can catch it doing so.
    measured_coverage: float | None = None


class Contribution(BaseModel):
    """One factor's push on the midpoint, in the three scales `explain/shapley.py`
    reports. Read `phrase` and `rupees`; read `multiplier` when it matters.

    The distinction is the whole difficulty of explaining a log-space model, and
    it is worth carrying into the wire format rather than resolving it silently:

    * `multiplier` is **exact**. Multiply `Explanation.baseline` by every
      contribution's multiplier and you land on `Explanation.prediction`.
    * `rupees` is **approximate**. It answers "remove this one factor and how far
      does the prediction fall?", which is exactly true one factor at a time and
      does not sum across factors, because ``exp`` is not linear. See
      `Explanation.approximation_note` — the field that says so on every
      response.
    """

    #: The feature (or merged group) name — ``"experience"``, ``"org_size"``.
    #: Stable, machine-readable, and the thing to key a UI off.
    feature: str

    #: The same thing as English a recruiter can repeat: ``"8 years of
    #: experience"``, ``"coming from a services company"``.
    phrase: str

    #: What this factor did to the midpoint.
    direction: Literal["increased", "decreased", "no effect"]

    #: Approximate rupee value, signed. Rounded to ₹1,000 for the same reason
    #: the band is: the precision is not there.
    rupees: float

    #: ``exp(log_contribution)``. 1.21 means "multiplied the prediction by 1.21".
    multiplier: float

    #: The raw SHAP value, in log units. Nobody reads this — it is here so the
    #: additivity claim above is checkable from the response alone.
    log_contribution: float


class Explanation(BaseModel):
    """Why this candidate got this midpoint. **The field that makes the band
    arguable.**

    A band with no reasoning attached cannot be challenged, and an
    unchallengeable salary number is exactly the kind that causes harm — it
    launders a model's guess into an authority. This block is what a recruiter
    pushes back on, and what makes the decision defensible six months later.

    The four experience columns the model uses (`features/experience.py` builds
    `years_experience`, `experience_log`, `experience_sqrt` and
    `experience_bucket` out of one number) are **merged into a single
    `experience` contribution**. They are one fact wearing four hats — 53.7% of
    the model's gain between them, per `docs/findings.md` §4.1 — and shown
    separately they read like four independent, partly contradictory findings.
    Merging is exact, because SHAP values add. See `explain/shapley.py` Idea 3
    for what that hides and how to get it back.
    """

    #: The whole thing as one line, ready to render: *"Predicted ₹23,64,000 — 8
    #: years of experience added ₹4,14,000, …"*. If a UI shows one field from
    #: this block, it should be this one.
    summary: str

    #: What the model predicts before it knows anything about this person: the
    #: average prediction over the training rows. Read it as "a typical candidate
    #: in our data", **not** "the market median" — the training data is whatever
    #: we happened to collect, and for the default artifact it is synthetic.
    baseline: float

    #: The midpoint these contributions explain. Equals `Band.midpoint`.
    prediction: float

    #: Largest push first, and only the top few — a list of twelve reasons is a
    #: data dump, not an explanation.
    contributions: list[Contribution]

    #: How many factors the model actually weighed, before truncation to the
    #: list above. Published so nobody reads five rows as the whole story.
    n_factors: int

    #: **Required, never empty**, same rule as `caveat` and for the same reason:
    #: JSON strips uncertainty by default. A stack of rupee figures will be read
    #: as a sum unless something in the payload says it is not one.
    approximation_note: str


class ModelInfo(BaseModel):
    version: str
    trained_on: str
    n_train: int
    #: Mean relative width across the whole held-out set. The per-request
    #: `relative_width` above is one candidate; this is the model's general
    #: sharpness, and it is the number that decides whether the service is worth
    #: deploying at all.
    holdout_relative_width: float | None = None


# ─────────────────────────────────────────────── responses


class PredictBandResponse(BaseModel):
    band: Band
    take_home: TakeHome
    confidence: Confidence

    #: Why the midpoint is what it is. Present unless the caller sent
    #: ``explain: false``, or the explainer could not run — in which case the
    #: reason appears in `notes` and the band is still returned. **A failure to
    #: explain must never cost the caller their prediction**, but it must also
    #: never pass silently, because an unexplained band is a weaker product than
    #: the caller asked for.
    explanation: Explanation | None = None

    #: **Always present, never empty.** The plain-English version of everything
    #: in `confidence`, written for whoever reads the screen rather than the
    #: schema. If the band is too wide to quote, this says so in words.
    caveat: str

    #: Fields the caller sent that the model has never seen — an unknown role, a
    #: city not in the tier map. Not an error: `FeatureBuilder` Decision 2 maps
    #: them to NaN, which is the honest encoding of "no learned response". But
    #: the caller should know their input was not used, because a band that
    #: ignored the role is answering a more general question than the one asked.
    unrecognised_inputs: list[str] = Field(default_factory=list)

    #: Non-fatal remarks about how the request was interpreted — e.g. that
    #: `prev_salary` was accepted and deliberately not used.
    notes: list[str] = Field(default_factory=list)

    model: ModelInfo


class CompaRatioResponse(BaseModel):
    """``actual ÷ predicted midpoint`` — one division, and the centre of the tool.

    Both headline use cases are this number. **Pay equity** is everyone below
    0.90. **Increments** are "more to the low ratios, less to the high ones,
    within budget". No second model is needed for either — see
    `docs/design.md` §4.2.
    """

    compa_ratio: float

    #: `"below band"` / `"in band"` / `"above band"`, from
    #: `model.band.compa_label`. The wording and the 0.90 / 1.10 cut points are
    #: standard HR vocabulary and live in `model/band.py` as named constants —
    #: they are company policy, so they are edited in one place, not
    #: reimplemented here.
    position: str

    actual_salary: float
    band: Band
    confidence: Confidence

    #: Same rule as `/predict-band`, and it matters *more* here. A compa-ratio is
    #: a ratio against the midpoint, so a wide band means the denominator itself
    #: is uncertain — "0.82, underpaid" and "1.05, fine" can be the same person
    #: under two plausible midpoints. Never open a pay-equity conversation on an
    #: undecidable band.
    caveat: str

    unrecognised_inputs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    model: ModelInfo


class HealthResponse(BaseModel):
    """Is this thing safe to send traffic to, and what exactly is loaded?

    Returns `200` when a model is loaded and `503` when one is not — so a load
    balancer gets the answer from the status line, and a human gets the reason
    from the body. The service starts either way: an API that crashes on import
    because no model artifact exists cannot be deployed before the first
    training run, which is the wrong dependency direction.
    """

    status: Literal["ok", "no_model"]
    model_loaded: bool
    model: ModelInfo | None = None

    #: The payroll config in force. Take-home is only reproducible if you know
    #: which financial year's rules produced it.
    payroll_financial_year: str | None = None
    payroll_regime: str | None = None

    detail: str
