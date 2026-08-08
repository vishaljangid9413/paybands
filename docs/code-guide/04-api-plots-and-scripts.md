# Part 04 — API, plots, and scripts

This is the last part of the code reading guide. It covers ten files: the three that turn the model
into an HTTP service, the five that draw every chart in the project, and the two scripts that fetch
the data and regenerate every number in `docs/findings.md`.

**Read guides [01](01-payroll-and-data.md), [02](02-features-and-model.md) and
[03](03-fairness-policy-explain.md) first.** Everything here consumes what those parts built. This
part explains almost no modelling of its own — it explains what you have to *say* about a model
when you hand it to somebody else.

| # | File | Lines | What it is |
|---|---|---|---|
| 1 | `src/paybands/api/models.py` | 414 | The JSON contract |
| 2 | `src/paybands/api/service.py` | 863 | Loading the model, building the caveat |
| 3 | `src/paybands/api/app.py` | 110 | Three FastAPI routes |
| 4 | `src/paybands/plots/style.py` | 325 | One look, and the rupee formatter |
| 5 | `src/paybands/plots/distributions.py` | 385 | Why we log-transform; salary by experience |
| 6 | `src/paybands/plots/calibration.py` | 388 | **The most important plots in the project** |
| 7 | `src/paybands/plots/fairness.py` | 323 | Residuals by group; raw vs adjusted gap |
| 8 | `src/paybands/plots/explain.py` | 213 | The SHAP waterfall, on a log axis |
| 9 | `scripts/fetch_data.py` | 109 | Downloads the survey |
| 10 | `scripts/run_analysis.py` | 1,069 | Regenerates every number and all 12 charts |

You are an experienced backend engineer, so the FastAPI mechanics will take you about four minutes.
The interesting content in the API section is not the routing — it is the three or four places
where a modelling fact had to be forced into the wire format so that it could not be dropped.

---

### `src/paybands/api/models.py`

> The pydantic schemas defining exactly what a caller may send and exactly what comes back.

**Read time:** 15 minutes · **Difficulty:** easy
**Read it when:** you understand what a *band* is (guide 02, `model/band.py`) and what a *compa-ratio*
is (guide 03, `policy/`). You do not need to understand SHAP yet — skim `Contribution` and
`Explanation` and come back after guide 03's `explain/shapley.py` section if it is still fresh.

#### What problem it solves

A model that has been carefully built to admit uncertainty can have all that care erased in the last
ten metres, by serialisation. JSON has no place to put "and I am not very sure about this". If the
response is `{"midpoint": 1900000}`, a UI renders `₹19,00,000` and a recruiter quotes it. Nothing in
that chain was technically false and the outcome is a fabricated number presented as an estimate.

So this file is a set of deliberate structural choices designed to make the uncertainty
*unloseable*:

- **There is no `salary` field, anywhere.** The response returns `lower`, `midpoint` and `upper`.
  Once a field called `salary` exists in a payload, some UI will eventually render it alone.
- **`caveat` is typed `str`, not `str | None`.** A required field cannot be quietly dropped by a
  client that renders everything it receives, and a future edit that forgets to set it fails at
  serialisation time rather than shipping a confident-looking number.
- **The take-home block sits under its own key.** That is the API saying out loud which half of the
  response was *learned* and which half was *computed*.
- **The explanation ships by default.** `PredictRequest.explain` defaults to `True`.

#### The type alias

```python
CategoryStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
```

Every free-text categorical input (role, city, education, …) uses this. The 80-character cap is not
a security measure so much as an honesty one: a 4,000-character "role" is not a role, and letting it
through would put a junk category in front of the model.

#### Classes

##### Request models

**`Candidate`** — one person, in the project's common schema. `model_config = ConfigDict(extra="forbid")`.

| Field | Type | Plain English |
|---|---|---|
| `years_experience` | `float`, `ge=0`, `le=MAX_PLAUSIBLE_YEARS` | **The only required field.** The bound is imported from `features.experience`, not typed here, so the API and the training-time cleaning rule can never disagree about what "impossible" means. `MAX_PLAUSIBLE_YEARS` is 50.0. |
| `role` | `CategoryStr \| None` | Job family — "Backend", "Fullstack". |
| `education` | `CategoryStr \| None` | Highest qualification. |
| `org_size` | `CategoryStr \| None` | Company size band. |
| `remote` | `CategoryStr \| None` | Remote / hybrid / in-person. |
| `employment_type` | `CategoryStr \| None` | Full-time, contract, … |
| `level` | `CategoryStr \| None` | Internal grade / ladder position. |
| `institute_tier` | `CategoryStr \| None` | IIT/NIT/BITS vs other. |
| `prev_company_type` | `CategoryStr \| None` | Product vs services. |
| `city` | `CategoryStr \| None` | Free text. "Bengaluru", "BLR", "Gurugram, Haryana" all work, via `features.location.CITY_ALIASES`. |
| `location_tier` | `int \| None`, `ge=1`, `le=3` | Sent *instead of* `city`. If both arrive, the city wins — it is the more specific claim, and the tier is derivable from it but not the other way round. |
| `skills` | `list[CategoryStr] \| None`, `max_length=50` | |
| `performance_rating` | `float \| None`, `ge=0`, `le=5` | |
| `prev_salary` | `float \| None`, `gt=0`, `le=MAX_PLAUSIBLE_SALARY` | **Accepted, and deliberately ignored.** See the gotchas below. |

Everything except `years_experience` is optional, and that is a modelling decision rather than
laziness. `FeatureBuilder` never fills a missing value with a default — LightGBM routes NaN down
whichever branch fits the data best, which is a better answer than any guess this layer could
invent. So "I don't know this candidate's institute tier" is a legitimate, well-handled request;
`institute_tier: "other"` asserted on their behalf is not.

The consequence a caller should understand: **the fewer fields you send, the wider the band comes
back**, and the response says so in `caveat`. That is the system working, not degrading.

**`CompaRatioRequest`** — `candidate: Candidate` plus `actual_salary: float`, bounded by
`MIN_PLAUSIBLE_SALARY` (₹100,000) and `MAX_PLAUSIBLE_SALARY` (₹20,000,000), both imported from
`data/schema.py`. Rows outside those bounds were dropped from training, so the model has genuinely
never seen a ₹50,000/year engineer, and a compa-ratio computed against a midpoint extrapolated that
far out would be a number with no evidence behind it.

**`PredictRequest`** — `candidate: Candidate` and `explain: bool = True`.

##### Response pieces

**`Band`** — the band, in rupees. Three numbers, never one.

| Field | Type | Plain English |
|---|---|---|
| `lower`, `midpoint`, `upper` | `float` | The band, in rupees, already rounded to ₹1,000. |
| `currency` | `Literal["INR"] = "INR"` | |
| `coverage` | `float` | What fraction of people like this the band *claims* to contain, after conformal calibration. 0.8 means "we expect 8 in 10 real salaries for this profile to land inside these edges". |
| `width` | `float` | `upper − lower`, in rupees. |
| `relative_width` | `float` | `(upper − lower) ÷ midpoint`. **The number to read first.** Rupee width is not comparable across salary levels — ₹6L is enormous at ₹8L and tight at ₹60L — but 0.4 is 0.4 everywhere. |

**`TakeHome`** — layer 2, computed from the midpoint and never predicted. Fields: `annual_gross`,
`annual_net`, `monthly_gross`, `monthly_net`, `annual_pf`, `annual_insurance`,
`annual_professional_tax`, `annual_income_tax`, `financial_year`, `regime`.

`financial_year` and `regime` are there because take-home changes when the Union Budget changes
while market value does not — if someone's in-hand moved, that is the first place to look.

**`Confidence`** — whether this band is tight enough to act on.

| Field | Type | Plain English |
|---|---|---|
| `level` | `Literal["high", "moderate", "low"]` | A coarse label derived from `relative_width` alone. Three values, not a percentage: a "confidence score" of 0.62 invites a caller to invent a threshold, and the threshold that matters is already published. |
| `decision_grade` | `bool` | **The field to branch on.** `False` means: show the band as a rough market orientation, do not quote it in an offer letter. |
| `relative_width` | `float` | Same as `Band.relative_width`. |
| `relative_width_threshold` | `float` | The bar it was judged against (0.5). Published so the caller can disagree with it. |
| `nominal_coverage` | `float` | What the band *promises* for this prediction. |
| `measured_coverage` | `float \| None` | What the same model actually *delivered* on held-out rows. Read the two together: a band that promises 80% and delivers 70% is lying to a recruiter, and this is where you catch it doing so. |

**`Contribution`** — one factor's push on the midpoint. Fields: `feature` (machine-readable name),
`phrase` (English a recruiter can repeat), `direction`
(`Literal["increased", "decreased", "no effect"]`), `rupees`, `multiplier`, `log_contribution`.

The distinction carried in those three numeric fields is the whole difficulty of explaining a
log-space model:

- `multiplier` is **exact**. Multiply `Explanation.baseline` by every contribution's multiplier and
  you land on `Explanation.prediction`.
- `rupees` is **approximate**. It answers "remove this one factor and how far does the prediction
  fall?", which is exactly true one factor at a time and does *not* sum across factors, because
  `exp` is not linear.
- `log_contribution` is the raw SHAP value. Nobody reads it — it is there so the additivity claim is
  checkable from the response alone.

**`Explanation`** — `summary` (the whole thing as one renderable line), `baseline`, `prediction`,
`contributions: list[Contribution]`, `n_factors: int`, `approximation_note: str`.

`n_factors` is the count *before* truncation, published so nobody reads five rows as the whole
story. `approximation_note` is required and never empty, same rule as `caveat` and for the same
reason: a stack of rupee figures will be read as a sum unless something in the payload says it is
not one.

**`ModelInfo`** — `version`, `trained_on`, `n_train`, `holdout_relative_width: float | None`.

`holdout_relative_width` is the mean relative width across the whole held-out set — the model's
general sharpness, as opposed to the per-request `relative_width` for one candidate.

##### Responses

**`PredictBandResponse`** — `band`, `take_home`, `confidence`, `explanation: Explanation | None`,
`caveat: str`, `unrecognised_inputs: list[str]`, `notes: list[str]`, `model: ModelInfo`.

**`CompaRatioResponse`** — `compa_ratio: float`, `position: str` (from `model.band.compa_label`:
`"below band"` / `"in band"` / `"above band"`), `actual_salary`, `band`, `confidence`, `caveat`,
`unrecognised_inputs`, `notes`, `model`.

**`HealthResponse`** — `status: Literal["ok", "no_model"]`, `model_loaded: bool`,
`model: ModelInfo | None`, `payroll_financial_year: str | None`, `payroll_regime: str | None`,
`detail: str`.

#### The one thing to understand here

**`extra="forbid"` is the most important line in this file, and it is easy to walk past.**

Pydantic's default behaviour is to **silently ignore** fields it does not recognise. Consider what
that means for this particular API. During development the author sent:

```json
{"years_experience": 3, "role": "Backend"}
```

when the schema wants those nested under `candidate`. With `extra="forbid"` the API returned **422,
naming all three wrong fields**. Without it — with pydantic's default — every field in that request
would have been discarded as unknown, `candidate` would have been missing… and if the shape had been
slightly different, say a misspelt `"experience"` inside a valid `candidate` block, the request would
have returned **200 OK with a perfectly plausible band computed from no inputs at all.**

That is the failure mode worth internalising. It is not a crash. It is a confident, well-formatted,
professional-looking answer to a question nobody asked, and there is nothing in the response body
that would let you notice.

> A strict schema turns a silent wrong answer into a loud error.

The same rule bit and then paid off elsewhere in the project: `PayrollRules` also uses
`extra="forbid"`, and adding a `verified_against` comment to the payroll YAML broke 13 tests. The fix
was to add `verified_against` as a real field, not to loosen the validation.

#### Surprises and gotchas

- **`prev_salary` is accepted and then thrown away.** This looks like dead code. It is a deliberate
  statement: anchoring an offer to last drawn pay makes an early underpayment follow someone for
  their whole career, so the default `FeatureBuilder` is built with `include_prev_salary=False` and
  the value never reaches a feature matrix. It is on the request so a caller who has it can send it
  and see, in `notes`, that we declined to use it. Silently rejecting it would teach nobody
  anything.
- **The module docstring says "Three modelling ideas are baked into the shape" and then lists
  four.** Harmless, but you will notice it and wonder whether you missed something. You did not.
- **`explain` defaults to `True` and that is not a performance oversight.** `shap.TreeExplainer` is
  exact and costs about a millisecond per row once the explainer is built. The flag exists for bulk
  pay-equity scoring — running ten thousand employees through `/predict-band`, where nobody reads
  ten thousand sentences — not for shaving a single request.
- **`location_tier` is bounded `1..3` but `city` is unbounded free text.** They are not
  interchangeable at the schema level even though they are at the model level; the precedence rule
  lives in `service.candidate_to_frame`, not here.

---

### `src/paybands/api/service.py`

> Loads the model once, turns a `Candidate` into a band, and refuses to hand out a number without
> the sentence that says what it is worth.

**Read time:** 30 minutes · **Difficulty:** medium
**Read it when:** you have read `models.py` above, and you understand `ConformalBand` and
`SalaryBandModel` from guide 02. The five numbered "Decisions" in the module docstring are the
map — read those before the code.

#### What problem it solves

`app.py` is HTTP plumbing. This file is where the modelling judgement lives. Three separate
concerns land here:

1. **Loading.** A model artifact is a build output, not source. `models/` is gitignored, so a fresh
   clone, a fresh container and CI all start with no model on disk. The service must start anyway.
2. **Translation.** A `Candidate` (optional fields, free-text city, a list of skills) has to become
   a one-row pandas frame in the common schema, and the resulting band has to come back rounded to
   something a human would say out loud.
3. **Honesty.** Every number that leaves has to carry the width of the band it came from and a
   plain-English caveat.

#### Constants — the judgement calls, gathered in one place

```python
DEFAULT_MODEL_PATH = Path(os.environ.get("PAYBANDS_MODEL_PATH", _REPO_ROOT / "models/band.pkl"))
PAYROLL_CONFIG_DIR = _REPO_ROOT / "configs/payroll"

DECISION_GRADE_MAX_RELATIVE_WIDTH = 0.5
TIGHT_RELATIVE_WIDTH = 0.3
MEASURED_RELATIVE_WIDTH_ON_SURVEY_DATA = 1.9
RUPEE_ROUNDING = 1_000
EXPLANATION_TOP_N = 5
```

**`DECISION_GRADE_MAX_RELATIVE_WIDTH = 0.5` deserves its own paragraph, and it is the single best
thing in this file.**

That number is a judgement call, not a measurement, which is why it is one named constant rather
than a comparison buried in a function. And crucially, **it comes from outside this model.** A
published HR salary band at a large company typically runs from about 80% to 120% of its midpoint —
a relative width of **0.4**. That is the width compensation teams have already decided is narrow
enough to hire against. Allowing 0.5 is therefore already more generous than industry practice;
anything wider is not a band in the sense an HR person means the word.

The measured calibrated band on the real survey data is **2.40× its own midpoint** (`findings.md`
§3). So the model **fails its own test, loudly, on every single request today.** Every response
carries `decision_grade: false` and a caveat beginning "NOT DECISION-GRADE".

That is the correct behaviour for a model that is not ready. The alternative — looking at 2.40,
shrugging, and setting the threshold at 3.0 — is choosing the ruler to fit the object.

> **A quality bar you set after seeing your results is not a quality bar.** It is decoration. A
> threshold tuned to your model's current performance always passes.

`TIGHT_RELATIVE_WIDTH = 0.3` is the other end: ±15% around the midpoint, which is genuinely tight.

`RUPEE_ROUNDING = 1_000` is deliberately the same value as `explain/shapley.RUPEE_ROUNDING`. The
explanation's `prediction` and the band's `midpoint` are the same quantity rounded twice, and two
different rounding rules would make a response disagree with itself by a few hundred rupees for no
reason anyone could ever find.

`EXPLANATION_TOP_N = 5` — an explanation is a thing a person repeats out loud, and `findings.md`
§4.1 found only nine of the model's 37 features move held-out error at all, so a longer list would
be mostly noise dressed as reasons.

`MEASURED_RELATIVE_WIDTH_ON_SURVEY_DATA = 1.9` is documentation only — it is not used in any
calculation. **It is also stale; see the gotchas.**

#### Classes

**`ModelBundle`** — a frozen dataclass holding everything a prediction needs, loaded once and never
mutated.

| Field | Type | Plain English |
|---|---|---|
| `band` | `ConformalBand` | The calibrated band model — three quantile boosters plus the conformal q̂. |
| `version` | `str` | e.g. `"band-lgbm-cqr-seed42"`. |
| `trained_on` | `str` | Provenance, in words. For the default artifact this literally reads `"synthetic (4,000 rows, seed 42) — NOT real market data"`. |
| `n_train` | `int` | Rows the trees were fitted on. |
| `holdout_coverage` | `float \| None` | Coverage measured on the test split at training time. |
| `holdout_relative_width` | `float \| None` | Mean relative width on the same test split. |

The last two were measured on rows neither the trees nor the conformal calibration ever saw. They
are the numbers `/health` and every response report, and they are the only basis on which anyone
should decide to trust this service.

Two members:

- `known_categories` (property) → `dict[str, list[object]]`. The category vocabularies frozen at fit
  time, dug out with `getattr(self.band.model, "builder", None)` then
  `getattr(builder, "category_levels_", {})`. Reaching through the conformal wrapper to the
  underlying model is deliberate: the wrapper adjusts band *edges* and knows nothing about features.
- `to_info()` → `api_models.ModelInfo`.

**`ModelNotLoaded(RuntimeError)`** — raised when a prediction is asked for and no artifact was
found. A named exception rather than an `HTTPException`, so this module stays free of HTTP.
`app.py` decides that this maps to 503.

**`PredictionService`** — the model, the payroll rules, and the rules for talking about both.

```python
def __init__(self, bundle: ModelBundle | None = None, rules: PayrollRules | None = None,
             *, load_error: str | None = None) -> None
```

Constructed **once**, at startup, and held for the process lifetime. The usual reason given for
this is latency — loading per request would re-read a pickle, rebuild three LightGBM boosters and
re-parse a YAML config on every call. But the real cost is correctness: two requests could then be
answered by two different artifacts if the file changed underneath, and nothing in either response
would say so.

`self._explainer` and `self._explainer_error` start as `None` and are built on first use, for two
reasons: `shap` imports `numba`, which costs seconds, and a service constructed with no model has
nothing to build an explainer *from*.

Methods, in the order you should read them:

| Method | Signature | What it does |
|---|---|---|
| `from_disk` | `@classmethod (cls, model_path: Path = DEFAULT_MODEL_PATH) -> PredictionService` | Loads what is there, records what is not, and **never raises**. |
| `is_ready` | `@property -> bool` | `bundle is not None and rules is not None`. |
| `_require` | `() -> tuple[ModelBundle, PayrollRules]` | Raises `ModelNotLoaded` if either is missing. Every public method starts here. |
| `_band_for` | `(candidate) -> tuple[BandPrediction, api_models.Band]` | Frame → raw band → rounded `Band`. |
| `_confidence` | `(band, *, unknown: list[str]) -> api_models.Confidence` | |
| `explainer` | `() -> BandExplainer` | The process's one explainer, built lazily, cached **even on failure**. |
| `_explanation` | `(candidate) -> Explanation` | SHAP for one candidate. |
| `_to_api_explanation` | `@staticmethod (explanation) -> api_models.Explanation` | Truncates to `EXPLANATION_TOP_N`, recomputes nothing. |
| `_take_home` | `(midpoint: float) -> api_models.TakeHome` | |
| `_notes` | `@staticmethod (candidate) -> list[str]` | Non-fatal remarks about how the request was read. |
| `predict_band` | `(candidate, *, explain: bool = True) -> PredictBandResponse` | |
| `compa_ratio` | `(request: CompaRatioRequest) -> CompaRatioResponse` | |
| `health` | `() -> HealthResponse` | |

Two details in there are worth pulling out.

`_band_for` rounds all three edges through `_round_rupees` **before** building the `Band`, and then
computes `relative_width` from the rounded numbers. Rounding is monotone, so `lower ≤ mid ≤ upper`
survives it, and the conformal wrapper has already clipped the edges to the median so the band
cannot invert.

`explainer()` caches its own failure. A model `shap` cannot read is diagnosed once instead of on
every request.

#### Functions

##### Building and loading an artifact

```python
def train_bundle(df: pd.DataFrame | None = None, *, n_synthetic: int = 4_000,
                 seed: int = DEFAULT_SEED, label: str | None = None) -> ModelBundle
```

The whole phase 4–5 pipeline in one call: three-way split, fit `SalaryBandModel`, calibrate
`ConformalBand`, evaluate on test, package the result with its provenance.

`df=None` generates synthetic data, so the service is runnable on a clone with no data files
present — and it labels itself
`f"synthetic ({n_synthetic:,} rows, seed {seed}) — NOT real market data"`. That label travels all
the way into `ModelInfo.trained_on` and out through every response and `/health`. **The served
model trains on synthetic data by default, and says so in the payload**, so nobody can mistake a
plumbing demo for a market estimate.

The three-way split is not optional here. Calibrating q̂ on rows the trees were fitted to produces
small conformity scores, a band that barely widens, and a published coverage guarantee that is
simply false — silently, with better-looking numbers than the honest version.
`ConformalBand.calibrate` refuses to run if the sets overlap.

```python
def save_bundle(bundle: ModelBundle, path: Path = DEFAULT_MODEL_PATH) -> Path
def load_bundle(path: Path = DEFAULT_MODEL_PATH) -> ModelBundle
```

Plain `pickle`. `load_bundle` type-checks the result and raises `TypeError` if it is not a
`ModelBundle`. Pickle executes whatever it reads, so this must only ever be pointed at an artifact
you produced yourself — but it is the right tool here, because a LightGBM booster, a fitted
`FeatureBuilder` and a calibrated q̂ travel together or not at all. Splitting them into a "safe"
format invites the failure where the model and its calibration go out of sync.

```python
def load_payroll_rules(path: Path | None = None) -> PayrollRules
```

Defaults to the **latest financial year on disk** — `sorted(PAYROLL_CONFIG_DIR.glob("fy_*.yaml"))[-1]` —
because `configs/payroll/` is versioned by year and never edited in place. `PAYBANDS_PAYROLL_CONFIG`
overrides. Historical recomputation passes an explicit path.

##### Candidate → model input

```python
def candidate_to_frame(candidate: api_models.Candidate) -> pd.DataFrame
```

One request → a one-row common-schema frame. Only fields the caller *actually sent* become columns.

That is worth a beat. Omitted fields are left **out** rather than written as NaN, and the result is
identical either way, because `FeatureBuilder.transform` emits an all-NaN column for anything
absent and the feature matrix keeps its width regardless. Building it this way makes the frame a
faithful record of the request, which is exactly what you want when debugging why a band came back
wide.

Three special cases:

- `city` beats `location_tier` when both are sent (`if candidate.city is not None: ... elif ...`).
- `skills` is joined with `";"`, because `parse_skills` splits on `";"` — the survey's own format,
  kept so the serving path and the training path parse skills through exactly the same function.
- `prev_salary` is not carried across at all.

The two module-level tuples that drive it: `_PASSTHROUGH_FIELDS` (10 names that map straight onto
common-schema columns) and `_CATEGORICAL_FIELDS` (the 8 checked against learned vocabularies).

```python
def unrecognised_inputs(candidate: api_models.Candidate, bundle: ModelBundle) -> list[str]
```

Fields the caller sent whose *value* the model has never seen. Not an error and specifically not a
500: `FeatureBuilder` maps an unseen category to NaN, which is the honest encoding — the model has
no learned response to "Prompt Engineer" if no such row existed in training, so it answers the more
general question instead and the band widens to match.

Reporting it matters because that widening is the **only** trace left. A caller who sees a plausible
band for a role we have never priced would otherwise have no way to know their most important input
was discarded.

The comparison is `str(value) not in {str(v) for v in levels}` — casting both sides mirrors
`_categorical_block`, so that `3` and `"3"` cannot become two different categories.

City is handled separately and by a *different* rule: a city counts as unrecognised only when
`city_tier(candidate.city)` returns NaN, i.e. it cannot be read at all ("..." or "12345"). A real
city the alias table simply has not heard of is a different case, and it is reported in `notes`
instead.

##### The honesty layer

```python
def confidence_level(relative_width: float, *, degraded: bool = False) -> str
```

- `relative_width <= TIGHT_RELATIVE_WIDTH` (0.3) → `"high"`, or `"moderate"` if degraded
- `relative_width <= DECISION_GRADE_MAX_RELATIVE_WIDTH` (0.5) → `"moderate"`
- otherwise → `"low"`

The `degraded` cap is subtle and worth understanding. An unknown role becomes NaN; LightGBM routes
NaN down whichever branch fits training best; and all three quantile models route it the **same**
way. So the band can come back deceptively *tight* while resting on a feature the model never
actually read. Capping the label at "moderate" stops width alone from vouching for a prediction
that was made with less information than the caller supplied.

```python
def build_caveat(band: api_models.Band, *, unknown: list[str],
                 holdout_coverage: float | None) -> str
```

**The sentence that stops this API from being misleading. Never empty.** Structure is fixed so a UI
can render it verbatim or split on `". "`:

1. what the band's width means, in rupees a person recognises;
2. whether it is decision-grade, in the imperative;
3. which inputs were ignored, if any;
4. measured coverage against promised coverage, if known;
5. that the number is base salary, not take-home, not CTC.

The last one is on **every** response including the tightest ones, because "₹19L" means at least
three different things in an Indian salary conversation and the API should never be the ambiguous
party.

A real caveat, from `JOURNEY.md` Part 12:

> *"This band spans 68% of its own midpoint (₹8,39,000 to ₹15,33,000 around ₹10,15,000).
> NOT DECISION-GRADE: that is wider than the 50%-of-midpoint limit a band has to meet to be
> quotable (published HR bands typically run 80%-120% of midpoint). Use it as a rough market
> orientation only — do not put these numbers in an offer, a budget, or a pay-review decision. …"*

##### Module-level helpers

```python
def _round_rupees(value: float) -> float
```

Rounds to `RUPEE_ROUNDING`, and never below ₹1: `compute_payslip` rejects a non-positive gross, and
a band edge that rounded to zero would take the whole request down with it.

```python
def get_service() -> PredictionService
def set_service(service: PredictionService | None) -> None
```

The process-wide singleton, held in a module-level `_service`. Lazy rather than import-time so that
importing `app` — which happens in tests, in `--reload` workers, and in any tooling that inspects
the OpenAPI schema — does not touch the filesystem. `set_service` exists for tests and for the
training entry point.

```python
def main() -> None
```

`uv run python -m paybands.api.service` — trains a bundle on synthetic data, saves it, and prints
held-out coverage and mean width, with a warning line when the width exceeds the decision-grade
limit.

#### The one thing to understand here

**Decision 1 in the module docstring, generalised: JSON strips uncertainty by default.**

The calibrated band measures 2.40× its own midpoint on the real survey data. Read that in rupees: a
midpoint near ₹14L comes with edges around ₹4.9L and ₹39.8L. Both of those are real salaries for
real engineers, and the band is honest to say so — 80% coverage was *measured*, not assumed. But it
means the model is describing a market, not pricing a person.

An API that serialises that as `{"midpoint": 1395000}` and nothing else has told the truth and
communicated a lie. So the width is carried as an explicit field (`relative_width`), the verdict as
a boolean a caller can branch on (`decision_grade`), and the reasoning as a required sentence
(`caveat`) typed `str` rather than `str | None` so it cannot be quietly dropped.

Everything else in this file — the constants at the top, the `degraded` flag, `unrecognised_inputs`,
`_notes` — is the same idea applied at smaller scale: find the thing that would otherwise be
invisible in the response, and give it a field.

#### Surprises and gotchas

- **`MEASURED_RELATIVE_WIDTH_ON_SURVEY_DATA = 1.9` is stale, and the module docstring repeats the
  error.** Decision 1 says "roughly 1.9× as wide as its own midpoint" and Decision 2 says "the
  current model sits near 1.9". `findings.md` §3 measures the **raw** band at median 1.94× and the
  **calibrated** band at median **2.40×**. `JOURNEY.md` Part 15, mistake #4, records exactly this
  slip being caught elsewhere: *"1.94× is the width before calibration — the version that promises
  80% and delivers 75%. Quoting 1.94× means quoting the band that lies about itself."* The constant
  is documentation only and used in no calculation, so nothing computes wrongly — but it disagrees
  with the findings document, and it is the sort of thing that gets copied into a README.
- **`from_disk` catches bare `Exception` twice, on purpose.** A bad pickle or an unreadable payroll
  config becomes a recorded `load_error` string, not a crash. This is Decision 4: the service must
  be deployable *before* the first training run, so a missing artifact is a *state* it reports, not
  an error that stops it existing.
- **`predict_band` swallows every explanation failure.** `except Exception` around the SHAP call,
  the error is stashed in `self._explainer_error` and appended to `notes`, and the band is still
  returned. An explanation is an *addition* to a band, and a band the caller can use beats a 500
  they cannot — but a silently missing explanation is a degraded product nobody notices has
  degraded, so it goes in `notes`.
- **The `if __name__ == "__main__"` block re-imports `main` rather than calling the one defined
  directly above it.** This is not cargo cult. Run as `python -m paybands.api.service`, the module
  executes under the name `__main__`, so `ModelBundle` is `__main__.ModelBundle` — and pickle
  records that path. The artifact would then fail to load in the API process, where the class is
  `paybands.api.service.ModelBundle`. Importing the module under its real name first means the
  bundle is pickled with the name every other process will look it up by.
- **`compa_ratio` divides by the median, not by `(lower + upper) / 2`.** On a right-skewed
  distribution those differ, and the median is what HR means by "the middle of the band" — the
  salary half the market is below.
- **`_notes` warns when a city falls through to tier 3.** Tier 3 is the catch-all *and* a positive
  claim ("a real place that is not a tech metro"), so a misspelling the alias table does not cover
  lands there silently and gets priced as a non-metro. The note is the difference between a
  considered answer and a typo.

---

### `src/paybands/api/app.py`

> Three FastAPI routes and one rule about what they may return.

**Read time:** 5 minutes · **Difficulty:** easy
**Read it when:** you have read `service.py`. There is nothing here you do not already know as a
backend engineer; read it for the status-code decisions.

#### What problem it solves

Routing, and nothing else. The rule the module docstring states: **no response leaves this file
carrying a salary number without the width of the band it came from and a plain-English caveat** —
and all of that is enforced in `service.py` and typed in `models.py`, so this file is thin on
purpose.

#### Functions

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]
```

Calls `get_service()` at startup, then yields. `get_service` is idempotent, so this is a warm-up
rather than the only load path — but doing it at startup means a broken or missing artifact shows
up in the logs at boot instead of inside somebody's first prediction.

```python
def _service() -> PredictionService
```

A one-line indirection over `get_service()`. Note it is a plain function call inside each handler,
not a FastAPI `Depends`; tests swap the singleton with `service.set_service` instead.

```python
@app.get("/health", response_model=api_models.HealthResponse)
def health(response: Response) -> api_models.HealthResponse
```

```python
@app.post("/predict-band", response_model=api_models.PredictBandResponse)
def predict_band(request: api_models.PredictRequest) -> api_models.PredictBandResponse
```

```python
@app.post("/compa-ratio", response_model=api_models.CompaRatioResponse)
def compa_ratio(request: api_models.CompaRatioRequest) -> api_models.CompaRatioResponse
```

Both POST handlers catch `ModelNotLoaded` and re-raise it as
`HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, _MODEL_UNAVAILABLE)`, where `_MODEL_UNAVAILABLE`
is a module constant containing the actual command to run:

> *"No model is loaded. The service starts without one on purpose so it can be deployed before the
> first training run; train and cache one with `uv run python -m paybands.api.service`, then
> restart. See /health."*

The app itself is built with `description=__doc__`, so the module docstring becomes the OpenAPI
description.

#### The one thing to understand here

**Why `/health` returns 503 when the model is not loaded — and why it returns a body anyway.**

`health` takes `response: Response` and *mutates* `response.status_code` rather than raising an
`HTTPException`. That is the whole trick. Raising would replace the body with FastAPI's generic
`{"detail": ...}` shape. Mutating keeps the typed `HealthResponse` in the body **and** sets the
status line.

The result serves two different readers with one call:

- **A load balancer or Kubernetes probe** reads the status line. 503 means "do not send traffic
  here", and that is a machine decision that must not require parsing JSON.
- **A human** reads `detail`, which says exactly what is wrong and what to do: `"no model artifact
  at /…/models/band.pkl. Train one with: uv run python -m paybands.api.service"`.

A 200 with `"status": "no_model"` would look healthy to every piece of infrastructure in the chain.
A raised 503 would look unhealthy but say nothing useful. Setting the code and keeping the body
gives you both.

#### Surprises and gotchas

- **A wide band is a 200, not an error.** `/predict-band` returns 200 with `decision_grade: false`,
  because the band is a correct and useful answer that simply must not be quoted. Refusing to answer
  would be worse: the caller would work around it, and a workaround has no caveat attached. The
  honest design is to answer *and* to say what the answer is worth.
- **When a model *is* loaded but is wider than the threshold, `/health` still returns 200** — with
  ` — WIDER than the decision-grade limit, so predictions are advisory only` appended to `detail`.
  That is the current state of the real model. "Unhealthy" means "cannot serve"; "advisory only"
  means "serving, badly", and conflating them would make the readiness probe useless.

---

### `src/paybands/plots/style.py`

> One look for every chart in the project, plus the rupee axis formatter.

**Read time:** 10 minutes · **Difficulty:** easy
**Read it when:** before any other file in `plots/`. Every one of them imports from here.

#### What problem it solves

Three problems, and the module docstring names all three.

**1. matplotlib picks a backend when `pyplot` is first imported.** The default tries to open a
window. On a CI runner, in a Docker container, or over SSH there is no window, and the import either
crashes or hangs. `matplotlib.use("Agg")` — called at line 42, deliberately *before*
`import matplotlib.pyplot`, which is the one place in the package where an import is not at the top
of the file — draws into memory instead. It cannot show you anything on screen, which is exactly
the point: the plots here are files and figures returned to the caller, never pop-ups. There is no
`plt.show()` anywhere in `paybands.plots` and there never should be.

**2. Every plot function takes an optional `ax` and returns `(fig, ax)`.** A function that creates
its own figure can only ever produce one chart. A function that will draw into an axes *you* hand it
composes — coverage next to band width, raw distribution next to logged, six fairness panels in a
grid. It costs three lines (`new_axes`) and it is the difference between a plotting module and a
pile of scripts.

**3. Rupees are formatted in lakhs and crores.** See "the one thing to understand" below.

#### Constants

| Name | Value | Meaning |
|---|---|---|
| `SERIES` | `("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")` | Categorical colours, used **in this order, never cycled**. Blue is the main series, orange the comparison. |
| `GOOD` / `WARNING` / `CRITICAL` | `#0ca30c` / `#fab219` / `#d03b3b` | Status colours. **Reserved** — never reused as "series 5". |
| `SURFACE`, `INK`, `SECONDARY_INK`, `MUTED`, `GRID`, `AXIS` | | Chrome: text and structure, deliberately quiet so the data is the loudest thing on the page. |
| `DEFAULT_OUTPUT_DIR` | `Path("reports")` | Where `save` puts a bare filename. Gitignored. |
| `DEFAULT_FIGSIZE` | `(7.0, 4.5)` | |
| `WIDE_FIGSIZE` | `(11.0, 4.5)` | Two panels side by side. |
| `RC_PARAMS` | `dict[str, object]` | The matplotlib defaults `apply_style` installs. |

The `SERIES` palette was checked with a colour-vision-deficiency simulator rather than eyeballed:
every pair stays at least ΔE 9 apart under deuteranopia and protanopia (red-green colourblindness,
roughly 8% of men), and at least ΔE 16 apart under normal vision. Four is the cap on purpose — a
fifth hue fails the separation test, and a chart needing nine colours is a chart that should have
been small multiples.

Nothing in `RC_PARAMS` is decorative. Every entry either removes something that was not carrying
information (top and right spines are two of the four box lines and neither carries scale; heavy
gridlines) or makes something readable (sizes, tick colours). `axes.axisbelow: True` puts gridlines
behind the data, always.

#### Functions

##### Figure construction

```python
def apply_style() -> None
```

Installs `RC_PARAMS` as matplotlib's global defaults. It mutates global state, which is why it is a
function you call rather than something that happens on import.

```python
def subplots(nrows: int = 1, ncols: int = 1, *, figsize: tuple[float, float] | None = None, **kwargs)
```

A styled `plt.subplots`. It exists so that no other module has to import `pyplot` directly — and
that is not tidiness. An `import matplotlib.pyplot` sorted to the top of another file would run
*before* this module's `matplotlib.use("Agg")`, silently picking the wrong backend. One import
point, one backend decision.

```python
def new_axes(ax: Axes | None = None, *, figsize: tuple[float, float] | None = None) -> tuple[Figure, Axes]
```

**Every plotting function in this package starts with this line.** Pass `ax` and the drawing lands
in your existing multi-panel figure; pass nothing and you get a styled standalone chart. `figsize`
is ignored when `ax` is given — the figure already has a size and it is not ours to change.

Note that `new_axes` calls `apply_style()` only when it *creates* a figure, and pointedly does not
when you hand it an axes: a figure you built is a figure you style.

##### Rupees on an axis

```python
def rupee_label(value: float, _pos: int | None = None) -> str
```

```
1_500_000   →  ₹15L
12_000_000  →  ₹1.2Cr
750_000     →  ₹7.5L
48_000      →  ₹48,000
```

One lakh is 100,000; one crore is 100 lakh (10,000,000). Below a lakh it falls back to plain digit
grouping — and Western and Indian grouping are *identical* below five digits (₹48,000 either way),
so no special case is needed there. They only diverge at ₹1,00,000, which is exactly where the lakh
branch takes over.

The unused `_pos` argument is matplotlib's tick position. `FuncFormatter` passes it and we ignore
it; that is what lets the same function serve as both a formatter and something callable in an
f-string.

```python
def _trim(number: float) -> str
```

One decimal place, but drop it when it is a bare zero. ₹15L reads better than ₹15.0L, and on an axis
the extra character is the difference between labels that fit and labels that overlap.

```python
def rupee_formatter() -> FuncFormatter
def percent_formatter(decimals: int = 0) -> FuncFormatter
def rupee_axis(ax: Axes, which: str = "x") -> None
```

`percent_formatter` turns a *fraction* into a percentage: 0.8 → `80%`. Coverage, pay gaps and error
rates are all naturally fractions in the code and all naturally percentages on the page. Converting
at the axis rather than in the data means nothing downstream has to remember which scale it is on.

`rupee_axis` sets the major formatter **and silences the minor ticks** with `NullFormatter()`. That
matters on a log scale, where matplotlib will happily label every minor decade subdivision,
producing ₹2L ₹3L ₹4L ₹5L crushed together between the majors.

##### Annotation and saving

```python
def note(ax: Axes, text: str, *, loc: str = "upper right", color: str = SECONDARY_INK) -> None
```

Writes a short explanatory note **inside** the axes, at one of four corners, on a near-opaque plate
in the surface colour.

Every plot in this package carries the sentence that says what it *means* — "over-confident: the
band is too narrow", "n = 36, treat this point as noise". A chart that needs the surrounding prose
to be interpretable stops being evidence the moment someone screenshots it.

Inside the axes rather than outside, deliberately: a note drawn outside would land on the
neighbouring panel when the chart is composed into someone else's multi-panel figure. The plate is
there because a note you cannot read because a boxplot whisker runs through it is a note that is not
doing its job.

```python
def save(fig: Figure, path: str | Path, *, dpi: int = 150) -> Path
```

**The only function in this package that writes a file.** A bare filename lands in
`DEFAULT_OUTPUT_DIR`; anything with a directory in it is used exactly as given; missing parents are
created. Returns the path actually written, so callers can log or assert on it.

No plotting function writes a file as a side effect. That rule sounds fussy until you import a
plotting module inside a test suite and find PNGs scattered through your repo, or run a notebook
cell twice and silently overwrite the figure you were about to show someone. Drawing and saving are
different decisions, so they are different functions.

#### The one thing to understand here

**Rupees are formatted in lakhs and crores, not millions, and this is not a cosmetic preference.**

A chart axis reading `₹1.5M` is a correct number written in the wrong number system. An Indian
reader parses `₹15L` instantly and `₹1,500,000` slowly, and `₹1.5M` requires a conversion they have
no reason to have practised. A chart whose axis its audience has to translate is a chart nobody
reads carefully — and the whole point of the charts in this project is that they are *evidence*,
read closely enough to argue with.

It is a small detail that separates a tool built *for* its users from one built *at* them. It is
also about twelve lines of code (`rupee_label` and `_trim`), which is roughly the cost of getting it
wrong once in front of the audience you built it for.

#### Surprises and gotchas

- **`matplotlib.use("Agg")` sits between imports and will look like a lint error.** It has a
  deliberate comment, and the four imports below it carry `# noqa: E402`. Do not "fix" this.
- **`subplots` has no return type annotation** and passes `**kwargs: object` through with a
  `# type: ignore[arg-type]`. It is a thin passthrough; the typed entry point is `new_axes`.
- **`save` uses `facecolor=fig.get_facecolor()`.** Without it, `bbox_inches="tight"` can produce a
  transparent or white margin that does not match `SURFACE`.

---

### `src/paybands/plots/distributions.py`

> What the salary data actually looks like — and the visual proof that log-transforming it is
> justified.

**Read time:** 15 minutes · **Difficulty:** easy
**Read it when:** after `style.py`, and after guide 02 has explained why the model predicts
`log(salary)`.

#### What problem it solves

Two claims in the README depend on the pictures in this file, and neither should be taken on trust.

1. **"Salaries are skewed, so we model log(salary)."** Anyone can assert that.
   `salary_distribution_comparison` shows it.
2. **"Experience is the strongest signal, but it flattens."** `salary_by_experience` shows the curve
   *and shows how many people are behind each point*, which is the part almost every version of this
   chart leaves out.

#### Constants

```python
THIN_BUCKET_N = 50
```

Below this many people in a bucket, the median is more noise than signal and the chart says so out
loud. 50 is a judgement call, not a law — it is roughly where the bootstrap band on this data stops
being narrow enough to argue from.

#### Functions

##### The distribution pair

```python
def skewness(values: ArrayLike) -> float
```

**Skewness** is how lopsided a distribution is. Zero means symmetric. Positive means a long tail to
the **right** — a few very large values dragging the mean above the median, which is precisely the
shape of a salary distribution. Negative means the tail is on the left.

It is written out in numpy rather than imported from scipy for one reason: it is three lines of
arithmetic and seeing them makes the number stop being magic.

```python
deviation = x - x.mean()
spread = x.std()
return float(np.mean(deviation**3) / spread**3)
```

The average cubed deviation, divided by the cubed standard deviation. **Cubing is what makes it
lopsidedness rather than spread** — cubing keeps the sign, so a value far to the right and a value
far to the left do not cancel out the way they would if we squared them.

Rules of thumb: `|skew|` under 0.5 is roughly symmetric, over 1 is strongly skewed. Raises
`ValueError` on fewer than three finite values, or when every value is identical.

```python
def _clean_salaries(salaries: ArrayLike) -> np.ndarray
```

Drops NaNs and non-positive values, and raises `ValueError` on an empty result. Non-positive
salaries have to go before anything log-shaped happens.

```python
def salary_distribution(salaries: ArrayLike, *, log: bool = False, ax: Axes | None = None,
                        bins: int = 40, title: str | None = None) -> tuple[Figure, Axes]
```

A histogram of salaries, raw or on a log axis, with the skew annotated.

**What `log=True` actually does.** It does *not* plot `log(salary)` in log units — an axis reading
"13.2" helps nobody. It keeps the axis in rupees and spaces it logarithmically
(`np.logspace` for the bin edges, `ax.set_xscale("log")`), so ₹5L→₹10L takes the same width as
₹10L→₹20L. That is exactly what modelling `log(salary)` does to the target, drawn: *equal
multiplicative steps get equal room*, which is how pay actually moves ("a 30% raise", never "a ₹2.4L
raise").

The mean and median are both drawn as vertical rules (solid = median, dashed = mean), labelled in
the note rather than beside their own lines — on a skewed distribution the two can land almost on
top of each other or hard against the left edge, and a direct label that sometimes overlaps its
neighbour is worse than a small legend that never does.

One detail: the skew reported is the skew of the thing being *plotted* — raw rupees on the linear
chart, `np.log(values)` on the log chart. Reporting the raw skew on both panels would make the log
panel look like it changed nothing.

```python
def salary_distribution_comparison(salaries: ArrayLike, *, bins: int = 40,
                                   figsize: tuple[float, float] | None = None
                                   ) -> tuple[Figure, tuple[Axes, Axes]]
```

**The visual case for the log transform**, and the payoff of the optional-`ax` rule: the whole
function is six lines that build a two-panel figure and call `salary_distribution` twice with an
`ax` argument.

On the survey extract the left panel shows skew **+2.86** and a long tail out to about ₹2 crore; the
right panel shows **−0.23** and near-symmetry. Median ₹15L and mean ₹24.2L are both marked, so you
can *see* the mean being dragged right by the tail. Same 1,022 people in both panels, nothing
trimmed, no outliers removed — only the axis respaced.

That single image settles the "why do we log-transform?" question without a paragraph of text, and
it is worth noticing how little it took: two numbers printed on a chart instead of asserted in a
caption.

##### The experience curve, with n

```python
def bootstrap_median_ci(values: ArrayLike, *, confidence: float = 0.95, n_boot: int = 1000,
                        seed: int = 0) -> tuple[float, float]
```

**How uncertain is this median? Answered by resampling.**

The idea in one paragraph. We have 36 salaries for the `20+` bucket and we computed their median. If
we had asked 36 *different* veterans we would have got a different median — but how different? We
cannot go and ask them, so we do the next best thing: draw 36 salaries *from the 36 we have*, with
replacement, and take that median. Do it a thousand times and the spread of those thousand medians
is an honest estimate of how much our one median could have wobbled.

That is the **bootstrap**, and it needs no assumption about the shape of the distribution — which
matters here, because salary distributions are exactly the shape that breaks the textbook formulas.

The band it produces widens automatically when a bucket is thin, which is the entire reason this
plot uses it rather than showing the interquartile range. The IQR describes how *spread out people
are*, and stays wide even with 100,000 rows. This describes how *unsure we are about the middle*,
which is the question the reader is actually asking.

Returns `(low, high)`, or `(nan, nan)` when there are fewer than two finite values. `seed` is fixed
so the same data always draws the same band — a chart that moves between runs is a chart nobody can
review.

```python
def salary_by_experience(df: pd.DataFrame, *, salary_col: str = TARGET,
                         years_col: str = "years_experience", ax: Axes | None = None,
                         confidence: float = 0.95, n_boot: int = 1000, seed: int = 0,
                         thin_n: int = THIN_BUCKET_N) -> tuple[Figure, Axes]
```

Three things are on this chart, and the third is the one that makes it honest:

- the **median** salary in each experience bucket — the headline curve, steep early and flat late;
- a **bootstrap confidence band** around each median;
- the **number of people** in each bucket, printed under its tick.

Buckets are taken from an existing `experience_bucket` column if present, otherwise built with
`bucket_years(df[years_col])`. Reusing the model's own bucket definition rather than inventing edges
here matters: if the plot binned experience differently from the features, the chart would be
describing a dataset the model never saw — which is the quiet way a "supporting figure" ends up
supporting nothing.

Buckets nobody falls into are left off rather than drawn as zero. Raises `ValueError` if the salary
column is missing, if neither bucket source exists, or if no rows land in any bucket.

#### The one thing to understand here

**The sample size is not optional on `salary_by_experience`, and the code goes to some trouble to
make it inseparable from the chart.**

On the survey extract the `20+` bucket contains about **36 people**, and its median lands within a
rounding error of the `13–20` bucket. The tempting read is *"senior pay plateaus after 20 years."*
The honest read is *"we asked 36 people and cannot tell."* Those are different claims, and only the
sample sizes separate them.

Three mechanisms enforce it:

```python
ax.set_xticklabels([f"{label}\nn={n:,}" for label, n in zip(labels, counts, strict=True)])
```

The count rides along **with the tick label**. It cannot be cropped off, scrolled past, or left out
of the screenshot.

```python
thin = np.array(counts) < thin_n
ax.scatter(x[~thin], ..., color=SERIES[0], s=42, zorder=4)
ax.scatter(x[thin], ..., facecolor="white", edgecolor=WARNING, linewidth=2.0, s=52, zorder=4)
```

Solid markers where the data supports the point; **hollow** where it does not. Shape, not just
colour, so the distinction survives a greyscale printout and a colourblind reader.

And finally the note names the thin buckets in words: *"hollow marker: fewer than 50 people (20+) —
treat the level as noise, not a finding"*.

> A chart that hides `n` invites exactly the misreading it should prevent.

#### Surprises and gotchas

- **The module docstring says skew "+2.87"; `findings.md` and `JOURNEY.md` both say +2.86.** The
  chart prints whatever `skewness()` computes at run time, so no output is wrong — but the prose in
  the docstring disagrees with the findings document by 0.01.
- **`_clean_salaries`'s docstring promises to "say how many, via the caller", and no caller does.**
  Dropped rows are silently absent from the histogram. On the cleaned survey frame nothing is
  dropped, so it never bites, but the promise is not kept.
- **`bins` is the number of bins, and `edges` has `bins + 1` entries.** Easy to misread when
  comparing the `logspace` and `linspace` branches.
- **`bootstrap_median_ci` draws an `(n_boot, sample.size)` array in one go.** For 1,000 resamples of
  a large bucket that is a real allocation. Fine at this data size; worth knowing before pointing it
  at a million rows.

---

### `src/paybands/plots/calibration.py`

> Is the band honest? — **the most important plots in the project.**

**Read time:** 25 minutes · **Difficulty:** medium
**Read it when:** after guide 02's `model/conformal.py` section. You need to know that the model
outputs three quantiles and that conformal prediction widens them. If you read only one file in
this guide, read this one.

#### What problem it solves

The model does not output a salary, it outputs a **band**: *"₹18L–₹24L, and I am 80% confident."*
That second half is a promise, and **a promise nobody checks is a promise nobody should believe.**

Checking it is called **coverage**. Take a hundred people, give each their own 80% band, and count
how many of their real salaries land inside their own band.

- If the answer is **80**, the model is **calibrated**.
- If the answer is **55**, the model is **over-confident**: the bands are too narrow, and every
  recruiter using them is being told the market is tighter than it is.
- If the answer is **97**, the model is **over-cautious**: the bands are honest but so wide they give
  no advice.

Almost nobody plots this. It takes three functions, and it is the difference between a project that
reports a number and a project that audits itself.

Everything here is **model-agnostic** — plain arrays in, figure out. Nothing imports the model, so
these plots work equally on the quantile model, the conformal-adjusted bands, a competitor's output,
or a hand-built example in a test.

#### Constants

```python
CALIBRATION_TOLERANCE = 0.03
```

How far empirical coverage may sit from the promise before we call it mis-calibrated. A judgement
call, stated once here rather than re-decided per chart: 3 points either way. On 1,000 test rows the
pure sampling wobble on an 80% figure is about ±2.5 points, so a smaller tolerance would flag noise.

#### Functions

```python
def empirical_coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float
```

What fraction of true salaries actually fell inside their own band. Five lines:

```python
return float(np.mean((truth >= low) & (truth <= high)))
```

Boundaries count as inside, matching `model.metrics.coverage`. The two definitions must agree and
there is a test that they do.

It is **deliberately reimplemented here rather than imported from the model package.** These plots
must stay usable on any pair of arrays — including ones invented in a test — and a plotting module
that drags the model in behind it is a plotting module you cannot point at someone else's numbers.

Raises `ValueError` on shape mismatch, on an empty set, and if any `lower > upper` (an inverted
band).

```python
def _verdict(empirical: float, nominal: float, tolerance: float) -> tuple[str, str]
```

Turns two numbers into a sentence and a colour, **in that order, always**:

| Condition | Sentence | Colour |
|---|---|---|
| `abs(gap) <= tolerance` | `"calibrated — within 3% of the promise"` | `GOOD` |
| `gap < 0` | `"OVER-CONFIDENT by N points — the band is too narrow"` | `CRITICAL` |
| `gap > 0` | `"conservative by N points — the band is honest but wide"` | `WARNING` |

Colour alone cannot tell a colourblind reader — or a greyscale printout, or anyone glancing at a
thumbnail — that a model is lying to them. So the words are the content and the colour is
reinforcement.

```python
def coverage_plot(y_true, lower, upper, *, nominal: float = 0.8, ax: Axes | None = None,
                  tolerance: float = CALIBRATION_TOLERANCE) -> tuple[Figure, Axes]
```

**One confidence level.** Two horizontal bars: what the model **promised**, and what it
**delivered**. Same axis, same scale, stacked one above the other, because the entire content of
this chart is one comparison and any layout that makes the reader hunt for it has failed.

The measured bar sits at `y=0` (bottom) and the nominal at `y=1` (top), so: **if the lower bar is
shorter than the upper one, the model is over-confident and the band is too narrow.** A dashed
vertical rule at the promise runs through both bars, so the shortfall is a visible *distance* rather
than a subtraction the reader has to do. The verdict from `_verdict` is written on the chart in
words and the measured bar is coloured to match.

Raises `ValueError` unless `0 < nominal < 1`.

```python
def coverage_by_quantile(y_true: ArrayLike, bands: Mapping[float, tuple[ArrayLike, ArrayLike]],
                         *, ax: Axes | None = None, show_uncertainty: bool = True
                         ) -> tuple[Figure, Axes]
```

**Every level at once. The diagnostic.** See below.

`bands` is `{nominal_level: (lower_array, upper_array)}` — typically
`{0.5: (...), 0.6: (...), 0.7: (...), 0.8: (...), 0.9: (...)}`. Keeping it a plain mapping is what
keeps the plot model-agnostic: anything that can produce two arrays per level can be checked here.

`show_uncertainty=True` draws ±1.96 standard errors on each point, where the standard error of a
proportion is `sqrt(p(1-p)/n)`. Nothing exotic — but leaving it off invites the reader to treat
sampling noise as a finding.

Raises `ValueError` if `bands` is empty or any level is outside `(0, 1)`.

```python
def band_width_plot(prediction: ArrayLike, lower: ArrayLike, upper: ArrayLike, *,
                    ax: Axes | None = None, relative: bool = False, n_bins: int = 10
                    ) -> tuple[Figure, Axes]
```

**Coverage says the band is *honest*. This says whether it is *useful*.**

A band of ±₹4L is a reasonable answer for someone the model puts at ₹18L and a meaningless one for
someone it puts at ₹1.2Cr — senior pay genuinely varies more, in rupees, than junior pay does. If
this plot comes out flat, the model is applying one uncertainty to everybody, which means it is
over-confident at the top of the market and over-cautious at the bottom, while possibly scoring
perfect *average* coverage and hiding both.

The cloud is one dot per person; the orange line is the median width within each **decile of
predicted salary**. The line is the finding, the cloud is the evidence that the line is not an
artefact of a handful of points. Bins are quantile bins (`np.quantile(pred, np.linspace(0, 1, n_bins + 1))`),
not equal-width bins, so every point on the line rests on the same number of people — equal-width
bins would put three people in the top bin and draw a confident line through them. Bins with fewer
than 3 rows are skipped.

The note prints `widest decile ÷ narrowest` and calls it: `"a flat band is a bad band"` below 1.2×,
`"band widens with salary — good"` above.

`relative=True` plots width as a *fraction of the prediction* instead of rupees. Worth flipping on:
a band that widens in rupees but is flat in percentage terms is behaving exactly as a log-space
model should, and this is how you tell the two apart. Two separate charts, never two y-axes on one —
a dual-axis chart invents relationships.

On the real data both are run (`reports/band_width.png` and `reports/band_width_relative.png`) and
they say different things: rupee width grows **5.0×** from the bottom decile to the top (good, that
is learned), while *relative* width **falls** from 3.44× to 1.64× — so the band is proportionally
worst exactly where a company hires most.

#### The one thing to understand here

**What a calibration curve is, and why five levels beats one.**

`coverage_by_quantile` draws the chart the project exists to be able to draw. Here is how to read
it if you have never seen one:

- The **horizontal axis** is what the model *promised*: 50%, 60%, 70%, 80%, 90%.
- The **vertical axis** is what it *delivered* — the fraction of true salaries that actually landed
  inside the band at that level.
- The **dashed diagonal** is `delivered = promised`. A perfectly calibrated model draws its points
  exactly on that line.

And then:

| Where the points sit | What it means |
|---|---|
| **Below the diagonal** | **OVER-CONFIDENT.** It promised 80% and delivered 62%. Its bands are too narrow. **This is the dangerous failure** — a recruiter is being handed a tighter range than the evidence supports, and will negotiate as if the uncertainty were smaller than it is. |
| **Above the diagonal** | Conservative. It promised 80% and delivered 94%. Nobody is misled, but the bands are wider than they need to be, and a band from ₹8L to ₹60L is advice nobody can act on. |
| **On the diagonal at every level** | Calibrated. This is what conformal prediction is *for*, and this plot is how you check that it worked rather than assuming it did. |

The failure directions are not symmetric in consequence, so the chart is not drawn symmetrically:
the over-confident half-plane is shaded (`ax.fill_between([0, 1], [0, 0], [0, 1], color=GRID, ...)`).
The diagonal is drawn *first*, underneath everything, because the line the points are being judged
against should never be drawn on top of the evidence. And the axes are forced square
(`ax.set_aspect("equal", adjustable="box")` with identical x and y limits), because a diagonal that
is not at 45° is a diagonal you cannot eyeball against.

**Now, why five levels rather than one.**

A single coverage number is a spot check. It tells you whether *this* promise, at *this* level, held
up. That is genuinely useful and it is what `coverage_plot` draws. But it cannot distinguish between
"the model is well calibrated" and "the model happens to be right at 80% and wrong everywhere else".

Sweeping the levels turns the spot check into a **curve**, and a curve has a shape you can diagnose.
On this project's data the shape was the finding. From `findings.md` §2.3:

| promised | raw delivered | shortfall | conformal delivered | shortfall |
|---|---|---|---|---|
| 50% | 43.0% | −7.0 pts | 48.5% | −1.5 pts |
| 60% | 53.1% | −6.9 pts | 59.8% | −0.2 pts |
| 70% | 63.1% | −6.9 pts | 67.3% | −2.7 pts |
| 80% | 72.9% | −7.1 pts | 81.7% | +1.7 pts |
| 90% | 86.8% | −3.2 pts | 90.4% | +0.4 pts |

The raw band under-covers at **5 of 5** levels, and four of those five shortfalls are within 0.2
points of each other. **A constant offset like that is not noise** — it is a systematic effect, and
you cannot see it at all from a single number. (The project's best guess is pinball loss on 613 rows
with `min_child_samples=25` pulling the extreme quantiles inward. It is listed as an open question:
the symptom was fixed with conformal calibration, the cause was never diagnosed. Fine engineering,
unfinished science.)

`reports/calibration_conformal.png` is the one image to put in a portfolio README, because it is
visual proof that the uncertainty estimates mean something.

#### Surprises and gotchas

- **`coverage_plot` uses `figsize=(7.0, 3.2)` and `coverage_by_quantile` uses `(6.0, 5.6)`,
  hard-coded.** Neither takes a `figsize` argument. The square one is square for the reason above;
  the bar one is short because two bars need no height. Pass `ax` if you want a different size.
- **`ax.set_ylim(-1.5, 1.5)` on a two-bar chart is deliberate empty space**, so the verdict note has
  somewhere to sit inside the axes without landing on the data.
- **`empirical_coverage` rejects an inverted band rather than sorting it.** That is the right call
  here — an inverted band means something upstream is broken, and silently sorting would hide it —
  but note the *model* does sort, in `BandPrediction`, and counts crossings rather than hiding them.
- **`band_width_plot` filters to `pred > 0` and finite rows before plotting**, and raises if nothing
  survives. A zero prediction would blow up the `relative=True` division.
- **The bin-membership expression in `band_width_plot` is easy to misread:**
  `(pred >= start) & (pred <= end if end == edges[-1] else pred < end)`. Python's conditional
  expression binds looser than the comparison, so this reads as
  `(pred >= start) & ((pred <= end) if end == edges[-1] else (pred < end))` — half-open bins
  everywhere except the last, which is closed so the maximum value is not dropped. Correct, but
  worth a second look.

---

### `src/paybands/plots/fairness.py`

> Seeing bias, rather than asserting it.

**Read time:** 15 minutes · **Difficulty:** medium
**Read it when:** after guide 03's `fairness/audit.py` section. You need to know what a raw gap and
an adjusted gap are before the second plot means anything.

#### What problem it solves

Two plots, and between them they carry the most sensitive claim the project makes. Both are
deliberately **model-agnostic** — they take plain arrays of numbers and import nothing from
`paybands.model` or `paybands.fairness`.

That is not architectural fussiness. **A fairness chart that only works on our own audit's output is
a chart nobody can use to check our audit.**

#### Functions

```python
def residual_by_group(y_true: ArrayLike, y_pred: ArrayLike, group: ArrayLike, *,
                      ax: Axes | None = None, relative: bool = False, min_n: int = 30
                      ) -> tuple[Figure, Axes]
```

**The visual signature of bias.**

A **residual** is what the model got wrong for one person:

```
residual = actual salary − predicted salary
```

Positive means the model *under-predicted* them: they earn more than it thought. Negative means it
over-predicted.

Across everybody, residuals should scatter around zero — that is what "the model is unbiased" means,
and it is usually true by construction because the training process makes it true. **The question
this plot asks is different: is it true *within each group*?**

If one group's whole box sits below zero, the model systematically predicts too high for them; if it
sits above, too low. Either is one-sided error against a specific group, and it is what bias looks
like *before* anyone computes a statistic. It survives the "but the overall MAE is fine" defence,
because the overall MAE is exactly what is hiding it: two equal and opposite group errors average to
zero.

**What this plot does not prove.** A group difference here can come from a genuine difference in the
group's composition — different roles, different cities, different experience — that the model has
represented correctly. Residual asymmetry is a *flag*, not a verdict. The verdict needs
`gap_comparison`.

Drawing decisions worth noting:

- Groups are drawn in **sorted** order, not first-seen order, so re-running on a shuffled frame
  draws the same chart. A figure whose category order moves between runs is a figure you cannot diff
  against last week's.
- **One colour for every box.** The groups are named on the axis, so hue would carry no information
  the labels do not already carry — and a rainbow here would quietly imply the groups are ranked.
- Outliers stay visible but recede (2.5pt markers at alpha 0.4). Hiding them would be hiding data;
  letting them shout would hide the boxes.
- The group **mean** is marked separately as an orange diamond, on top of the boxplot's median. The
  median says where the typical error is; the mean is what a fairness metric will actually report,
  and when the two disagree it is because a tail is doing the work.
- `n=` rides along with every tick label, same rule as `salary_by_experience`. Groups below `min_n`
  get an explicit `⚠ fewer than 30 people in: …` line in the note.
- `relative=True` divides the residual by the true salary. Worth using: being ₹3L wrong about ₹40L
  and ₹3L wrong about ₹6L are very different mistakes, and a rupee-scale box plot makes the high
  earners' errors dominate the picture. `run_analysis.py` always passes `relative=True`.

Raises `ValueError` on shape mismatch, on an empty set, and (when `relative=True`) if any true
salary is zero.

```python
def gap_comparison(raw_gap: float, adjusted_gap: float, ci: tuple[float, float], *,
                   raw_ci: tuple[float, float] | None = None, known_gap: float | None = None,
                   labels: tuple[str, str] = ("raw gap", "adjusted gap"),
                   ax: Axes | None = None, rupees: bool = False) -> tuple[Figure, Axes]
```

**The distinction this plot exists to make.**

The *raw gap* is the plain difference in average pay between two groups. It is real, it matters, and
**it is not by itself evidence of discrimination**, because it does not hold anything else constant.
If one group is on average newer to the industry, or more concentrated in lower-paying roles or
cities, a raw gap appears without anyone ever having been paid unfairly for the same work.

The *adjusted gap* is what remains after comparing like with like — same experience, same role, same
location. That is the number that speaks to "equal pay for equal work". It is also the harder
number, because whether you have controlled for the right things is a judgement, not a calculation.

Both belong on the chart, next to each other, because **either one alone misleads**. The raw gap
alone overstates the case. The adjusted gap alone hides a real and important fact — that the groups
are not distributed equally across the well-paid roles in the first place, which is its own problem
even if every individual pay decision was defensible.

**Why the interval is not optional.** An adjusted gap of 2% with an interval running from −3% to +7%
is a measurement that has not ruled out zero. Drawn as a bare bar it would look like a finding.
Drawn with its interval crossing the zero line, it looks like what it is.

The verdict at the bottom of the chart is computed from exactly that:

```python
crosses_zero = low <= 0.0 <= high
```

→ *"the adjusted interval crosses zero — no gap demonstrated once like is compared with like"*
(`GOOD`), or *"the adjusted interval excludes zero — a gap remains after controlling for role,
experience and location"* (`CRITICAL`).

`known_gap` draws a dashed `WARNING`-coloured rule labelled "injected truth". On synthetic data you
*do* know the true gap — the generator injected it — and drawing it here is how the fairness audit
gets validated against ground truth rather than against a hunch.

Raises `ValueError` if `ci` is inverted, or if `adjusted_gap` sits outside its own interval — which
almost always means one of the two is being computed on a different scale.

#### The one thing to understand here

**These two plots are a sequence, not alternatives, and the order matters.**

`residual_by_group` is the **detector**. It is cheap, it needs no controls, no regression and no
judgement calls, and it will show you one-sided error in a group that an aggregate metric has
averaged away. Run it on everything you have — experience buckets, roles, cities, and any protected
attribute you are lucky enough to possess.

`gap_comparison` is the **adjudicator**. It costs a modelling decision (which controls?) that is
genuinely contestable, and it produces a number people will quote.

Running the second without the first means you only ever measure gaps you already suspected.
Running the first without the second means reporting a flag as a verdict — which is the single most
common mistake in this area, and the reason the raw and adjusted bars are drawn on the same axis
with the same scale.

In this project `run_analysis.py` uses `residual_by_group` twice on model error (by experience
bucket and by role, both `relative=True`) and `gap_comparison` once, on synthetic data with an 8%
injected gap, to show the audit recovering a truth it was not told.

#### Surprises and gotchas

- **`positions = [1, 0]` in `gap_comparison`.** Raw on top, adjusted below, so reading order runs
  downward — and the tick labels are set in that same non-natural order. `ax.set_ylim(-1.7, 1.9)`
  makes the room underneath where the verdict sits.
- **`raw_ci` is drawn but never validated.** The function checks that `adjusted_gap` lies inside
  `ci`, and applies no equivalent check to `raw_gap` against `raw_ci`. An inconsistent raw interval
  would be drawn without complaint.
- **`labels=("raw gap", "adjusted gap")` is only ever the default in this codebase**, but the
  parameter exists so the same chart can compare any two estimates of the same quantity.
- **The `known_gap` label is placed at `y=1.5`**, above the raw-gap row, hard-coded against the
  `ylim`. Change the limits and the label moves off-chart.
- **`residual_by_group` casts group labels through `np.asarray(group, dtype=object)` and then
  `str(label)` twice.** It is doing a linear scan per group rather than a groupby; fine for a dozen
  labels, not for thousands.

---

### `src/paybands/plots/explain.py`

> One prediction, taken apart — the chart version of `explain/shapley.py`.

**Read time:** 10 minutes · **Difficulty:** medium
**Read it when:** immediately after guide 03's `explain/shapley.py` section, while the log-space
additivity argument is still fresh.

#### What problem it solves

The claim this plot exists to prove: **"the model can show its working."**

A **waterfall** chart shows a starting value, a run of steps that push it up and down, and where it
lands. It is the natural shape for SHAP — but drawn naively in rupees it would be **wrong here**,
and quietly so.

The model works in `log(salary)`. Contributions **add** in log units and therefore **multiply** in
rupees, so a rupee waterfall's steps do not telescope to the prediction. Anyone building one has to
fudge something to make the bars meet, and `explain/shapley.py` is explicit that fudging is the one
thing not to do.

So the bars are drawn on a **log-scaled rupee axis**. Each step multiplies the running total, the
last one lands exactly on the prediction, and the arithmetic is honest with nothing adjusted. The
reading rule that falls out is the useful one anyway: **equal bar lengths are equal percentage
effects**, which is how pay actually moves — "a 20% bump" means the same thing to a fresher and to a
principal engineer, "₹4L" does not.

#### Constants

```python
DEFAULT_TOP_N = 8
```

Contributions drawn individually before the rest are pooled. Eight is about what fits on one screen
with readable labels — and `findings.md` §4.1 found only nine features in this model move held-out
error at all, so a chart with twenty rows would be nineteen rows of noise around one real one.

#### Functions

```python
def contribution_waterfall(explanation: Explanation, *, ax: Axes | None = None,
                           top_n: int = DEFAULT_TOP_N,
                           figsize: tuple[float, float] | None = None) -> tuple[Figure, Axes]
```

The only public function in the file. It takes an `explain.shapley.Explanation` — typically with the
four experience encodings already merged into one row — and draws it top to bottom: start at the
baseline, apply the largest factor, then the next, and land on the prediction. Blue pushes salary
up, orange pushes it down.

**How the maths actually happens**, which is four lines and worth reading:

```python
steps = [c.log_contribution for c in shown]
...
edges = np.exp(explanation.baseline_log + np.concatenate([[0.0], np.cumsum(steps)]))
```

Cumulative sum in **log space**, then a single `exp`. The last value of `edges` is the prediction,
exactly. Each bar is then drawn as `barh(y, width=end - start, left=start)` between consecutive
edges, on an axis set to `ax.set_xscale("log")`.

**Truncation and pooling.** `n_real` counts contributions whose `direction` is not `"no effect"`;
`cut = max(1, min(top_n, n_real))`. Anything past the cut is pooled into a single
`"everything else (N factors)"` step whose log contribution is the sum of theirs — so the chain
still lands exactly on the prediction rather than stopping short of it. The `max(1, …)` keeps the
chart drawable when *every* factor is negligible, which is itself worth seeing.

The pooled bar gets no rupee label: it is a bag of unrelated features, and `"≈ +₹40,000"` against a
row nobody can act on is clutter.

**Labels.** Each bar is annotated `×1.21  ≈ +₹4.1L`:

- `×1.21` is exact and matches the bar you are looking at.
- `≈ +₹4.1L` is `Contribution.rupees` — the same approximate figure the API and the sentence report,
  so the chart and the JSON never disagree.

**Axis detail.** matplotlib's default log ticks sit on the decades — ₹10L, then ₹1Cr. A band chart
usually spans well under one decade, so the default would leave the axis with no labels at all.
Hence `LogLocator(base=10.0, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0))`, which puts a labelled tick every
₹5L-ish wherever the data happens to sit, and then `rupee_axis(ax, "x")` for lakh/crore labels. The
reader sees rupees while the geometry stays multiplicative.

Two anchors are drawn: a quiet dashed rule at the baseline (it is context) and the darkest line on
the chart at the prediction (it is the answer). `figsize` defaults to
`(7.6, max(3.4, 1.7 + 0.46 * len(labels)))` — height scales with the number of bars, because a fixed
height either crushes ten labels together or leaves half the canvas empty for three.

Raises `ValueError` if `top_n < 1` or the explanation has no contributions.

#### The one thing to understand here

**The note is not decoration; it is the reason the chart is publishable.**

```
blue = pushes pay up, orange = pushes it down
× factors are exact and multiply to the prediction;
the ≈₹ figures overlap and do NOT add up
```

That last line is doing real work. The chart is the artefact that gets screenshotted into a hiring
thread with none of the surrounding prose. A stack of rupee bars is read as a sum unless something
**on the image** says otherwise — and here the rupee figures genuinely do not sum, because `exp` is
not linear.

This is the same principle as `caveat` in the API and `n=` on the tick labels in
`distributions.py`, applied to a third medium: whatever caveat the artefact needs, put it *inside*
the artefact, because the artefact will travel without you.

#### Surprises and gotchas

- **`contribution_waterfall` is not called anywhere in `run_analysis.py`.** The twelve charts in
  `docs/findings.md` come from `distributions`, `calibration` and `fairness` plus one hand-built bar
  chart. This one is exercised by the tests and by anyone exploring a single prediction — it is a
  tool, not part of the findings pipeline.
- **`cut` is computed from `n_real` but the slice is taken from the full list.**
  `shown = explanation.contributions[:cut]` works because contributions arrive **sorted by size**,
  so every meaningful one precedes every negligible one and a single slice separates them. If the
  sort order ever changed, this would silently draw the wrong bars.
- **`rupees` is typed `list[float | None]`** because the pooled bar has no rupee figure, and the
  label branch guards on `if rupees[y] is not None`.
- **`ax.set_ylim(len(labels) + 1.1, -0.9)`** — inverted on purpose, so the largest factor is the top
  row. The extra room underneath is where the note sits.
- **The minus sign in the label is `−` (U+2212), not `-`.** Typographically correct, occasionally
  surprising if you grep for it.

---

### `scripts/fetch_data.py`

> Downloads the public dataset this project is built on, into exactly one place, every time.

**Read time:** 5 minutes · **Difficulty:** easy
**Read it when:** first, if you are actually going to run anything.

#### What problem it solves

Why this exists as a script rather than a line in the README: **an instruction a human has to follow
by hand is an instruction that gets followed differently by different people.** "Download the survey
and put it somewhere sensible" produces a dozen slightly different filenames and one very confusing
bug report.

Why the file is not committed to git: it is 134MB, it is not ours, and it is reproducible from a
URL. Git is for things you wrote, not things you fetched.

#### Classes

**`Dataset`** — a frozen dataclass, five fields, all documentation-as-data.

| Field | Meaning |
|---|---|
| `name` | Human name, printed. |
| `filename` | The one filename it will ever have. |
| `url` | Pinned. |
| `approx_mb` | Printed before the download starts, so a 134MB fetch is not a surprise. |
| `why` | Why this dataset is here: `"49,191 responses; 1,022 usable Indian salary rows after cleaning"`. |

`DATASETS` is a one-element tuple containing the Stack Overflow Developer Survey 2025.

The URL is pinned to the 2025 archive file rather than a "latest" alias, and that is a deliberate
reproducibility decision: if the data silently changed underneath us, **every number in
`docs/findings.md` would become unreproducible without anyone noticing.**

#### Functions

```python
def _download(ds: Dataset, dest: Path, *, force: bool) -> bool
```

Fetches one dataset. Returns `True` if it downloaded, `False` if it was already present or the
download failed.

Idempotent by default: an existing file is skipped with its size printed and a hint about `--force`.
On success it prints the path, the size in MB, and the first 16 hex characters of the file's SHA-256
— so two people can check they have the same bytes without exchanging 134MB.

`urllib.error.URLError` is caught, the partial file is removed, the error goes to `stderr`, and the
function returns `False` with a message telling you to download the URL by hand. A network failure
should not need a stack trace to interpret.

```python
def main() -> int
```

Parses `--force`, creates `data/public/`, loops over `DATASETS` printing name and rationale, and
returns 0 if every destination exists afterwards or 1 otherwise. Invoked as
`raise SystemExit(main())`.

#### The one thing to understand here

**Download to a temporary name, then rename.**

```python
tmp = dest.with_suffix(dest.suffix + ".partial")
urllib.request.urlretrieve(ds.url, tmp)
...
tmp.replace(dest)
```

The file is fetched as `so_2025_raw.csv.partial` and only becomes `so_2025_raw.csv` once the
download has completed.

The reason: **a half-downloaded file wearing the correct name is worse than no file at all.** If the
connection drops at 80MB and the partial bytes sit at the real path, the next run's "already
present — skipping" check passes, the loader reads it, `pandas` fails somewhere deep in CSV parsing
with a message about an unexpected number of fields, and you go looking for the bug in the parsing
code. The bug is not in the parsing code. It is that the file is a lie.

`Path.replace` is atomic on the same filesystem, so the destination path either has the complete
file or does not exist. The failure path also calls `tmp.unlink(missing_ok=True)`, so a failed run
leaves no `.partial` litter behind either.

This is a three-line pattern and it converts a confusing, misattributed failure into a clean one.

#### Surprises and gotchas

- **`PUBLIC_DIR = Path("data/public")` is relative**, so this script must be run from the repository
  root. Every other file in the project resolves paths from `__file__`. This one does not.
- **`urlretrieve` is deprecated-ish and has no timeout**, and carries a `# noqa: S310` because the
  URL is a constant above rather than user input. On a hung connection it will sit there.
- **`_download`'s return value is discarded** in `main`; the success check is `dest.exists()`, which
  is the more robust question anyway.
- **Only `URLError` is caught.** An HTTP 404 surfaces as `HTTPError`, which *is* a subclass of
  `URLError`, so that is covered — but a disk-full `OSError` mid-write is not.

---

### `scripts/run_analysis.py`

> One script that regenerates every number and every chart in `docs/findings.md`.

**Read time:** 45 minutes · **Difficulty:** hard (long, not conceptually difficult)
**Read it when:** last. It imports from nine other modules and every one of them is covered in
guides 01–03 or above. Read it with `docs/findings.md` open beside it — each section here produces
one section there.

#### What problem it solves

```bash
uv run python scripts/run_analysis.py
```

One script, one seed list, one output directory. Run it and you get the findings document's evidence
printed to your terminal and its figures written to `reports/`. It takes about 30 seconds and it is
verified byte-identical across runs.

**Nothing in `docs/findings.md` is allowed to be a number this script does not print.** All 467
numeric values in that document were checked against the script's output.

Three design commitments are stated in the module docstring.

**Why one script rather than a notebook.** A notebook records the order you happened to click in.
Six weeks later you cannot tell whether cell 14 ran before cell 9, whether the model in memory is
the one the chart came from, or which seed produced the number in the README. This file runs top to
bottom, seeds everything explicitly, and prints what it did — so *"which settings produced this
number?"* has an answer.

**Why every result is reported over many seeds.** A single train/test split is an anecdote. On 1,022
rows the test set is about 200 people, and swapping which 200 moves the headline MAE by well over
ten percent. Every comparison is run across `--seeds` independent splits and reported as a mean with
the full range, and where a claim is "A beats B", the script counts on how many seeds that was
actually true. **A margin that only holds on 6 of 10 splits is not a margin.**

**On pooling.** The error breakdowns pool test rows across seeds so that thin slices (the 20+
experience bucket, the QA role) have enough rows to say anything about at all. Rows recur across
seeds, so a pooled `n` is not an independent sample size — it is *"how many times a row of this kind
was scored"*. The script prints both the pooled `n` and the per-split average, and the findings
document repeats the warning.

#### Settings

| Constant | Value | What it controls |
|---|---|---|
| `DEFAULT_SURVEY` | `<repo>/data/public/so_2025_raw.csv` | Overridable with `--data`. |
| `DEFAULT_OUTPUT` | `<repo>/reports` | Overridable with `--out`. |
| `DEFAULT_N_SEEDS` | `10` | Splits per headline comparison. Overridable with `--seeds`. |
| `IMPORTANCE_SEEDS` | `3` | Permutation importance is the expensive step. |
| `PERMUTATION_REPEATS` | `5` | Shuffles per feature per seed. |
| `CALIBRATION_LEVELS` | `(0.5, 0.6, 0.7, 0.8, 0.9)` | The five confidence levels swept. |
| `CALIBRATION_SEEDS` | `3` | Each level needs its own three quantile models — 15 fits per seed. |
| `INJECTED_GAPS` | `(0.0, 0.03, 0.08, 0.15, 0.25)` | Pay gaps injected for the audit's self-test. |
| `SYNTHETIC_N` | `10_000` | Rows for the fairness validation. |
| `PROXY_N` | `8_000` | Rows for the proxy demonstration. |
| `PROXY_SEEDS` | `3` | |
| `PROTECTED_ATTRIBUTE_PATTERNS` | 11 substrings | `gender`, `ethnic`, `race`, `sexuality`, `orientation`, `transgender`, `disability`, `accessibility`, `nationality`, `religion`, `caste`. |

`IMPORTANCE_SEEDS = 3` and `PERMUTATION_REPEATS = 5` are honest about their limits: three seeds ×
five shuffles is enough to separate "carries signal" from "carries noise"; it is **not** enough to
rank the middle of the table, and the report says so.

`PROTECTED_ATTRIBUTE_PATTERNS` is matched case-insensitively as **substrings** against the real
header, so a rename cannot make the check pass silently.

Note the sys.path shim at the top — `sys.path.insert(0, str(REPO_ROOT / "src"))` — so the script
runs without an install. That is why every project import carries `# noqa: E402`.

#### Printing helpers

```python
def heading(text: str) -> None
def subheading(text: str) -> None
def table(frame: pd.DataFrame, *, floatfmt: str = "%.3f") -> None
```

Cosmetic. `table` prints a frame with a fixed float format.

```python
def spread(values: np.ndarray | list[float], *, percent: bool = False, decimals: int = 0) -> str
```

**Read this one properly. It is the most interesting eight lines in the file.**

```python
return f"mean {v.mean():.1%}   range {v.min():.1%} to {v.max():.1%}"
```

It returns mean, min and max in one string, and **it physically cannot return a mean on its own.**
There is no `spread(values, range=False)`. There is no code path that produces just the average.

That is a guardrail, and it was built because the guardrail was needed. `JOURNEY.md` Part 15 records
three headline numbers that had to be corrected after being computed from too few seeds:

- Conformal coverage was reported as 70.6% → 77.2%. Across ten seeds it is actually **74.9% →
  82.3%** — the four seeds used happened to land near the bottom of the range.
- Baseline improvement was reported as ~17%; the ten-seed figure is **19.4%**.
- The band width quoted while describing conformal results was **1.94×**, which is the width
  *before* calibration. The calibrated figure is **2.40×**.

The first of those is the instructive one: the mistake that had been *taught* two sections earlier —
"one split is an anecdote" — was then committed with four splits. Four is better than one. It still
was not enough.

A mean printed alone invites the reader to treat it as the answer. On 1,022 rows the range is
usually the more informative half of the sentence. Making that structural rather than a matter of
discipline is the difference between a rule and a hope.

#### Section 1 — the data

```python
def section_data(survey_path: Path, out: Path) -> pd.DataFrame
```

Loads the raw CSV twice: once with `usecols=["Country", "Currency", "CompTotal"]` for the
non-response arithmetic, and once through `paybands.data.stackoverflow.load` for the cleaned frame
it returns.

It prints, in order:

1. **The loader's own filtering report.**
2. **Non-response.** Indian respondents in INR, how many answered the salary question, how many
   skipped it, and the percentage. Then three lines of prose naming it as **selection bias**, which
   no cleaning rule removes.
3. **The typo, and why the median is the summary we trust.** The largest salary as typed
   (`1.0000e+23`), the digit count, the mean of raw INR salaries (about 81 quintillion rupees), the
   median of the same column (unmoved), and the median and mean after cleaning. One person typed a
   run of nines; it moved the mean by eighteen orders of magnitude and the median not at all.
4. **Skew**, raw and logged, via `dist_plots.skewness`.
5. **Who is in the surviving 1,022** — the role table with counts and shares, median experience, the
   share held by the three largest roles, and the roles with fewer than 20 respondents.
6. **Pay drivers the survey does not record at all.** A dict of seven column names mapped to why
   each matters, each checked with `column in df.columns and df[column].notna().any()` and printed
   with a ✓ or ✗. All seven come back ✗.

Then it saves two charts: `dist_raw_vs_log.png` and `salary_by_experience.png`.

The closing line of the section is the one the rest of the document keeps running into:

> The model that follows is not a weak model of Indian salaries; it is an honest model of the four
> or five things a public survey happens to ask.

#### Section 2 — baselines

```python
def section_baselines(df: pd.DataFrame, seeds: range) -> None
```

For each seed: `random_split`, fit `GlobalMedianBaseline` and `GroupMedianBaseline`, and record
`mae_global`, `mae_group` and `improvement = 1 - mae_group/mae_global`.

Prints the per-seed table, then three `spread(...)` lines, then `lookup wins on N/10 seeds`.

Then two extras, both computed on the **last** split from the loop:

- `groups.coverage_report()` — where the lookup table's fallback cascade actually landed. On the
  last split, 72.5% matched a (role × experience) cell with at least 10 people, 27.5% fell back to
  experience alone. So more than a quarter of candidates are quoted a number that ignores their role
  entirely. The lookup table is coarser than it looks.
- `evaluate(...)` for both baselines, followed by the explanation of the **negative R²** on the
  global median: R² rewards predicting the *mean*, and on a right-skewed target the median is a long
  way from the mean. The metric encodes a choice, and ours is the typical candidate, not the
  average.

#### Section 3 — the band model

```python
@dataclass
class BandRun:
    seed: int
    frame: pd.DataFrame   # per-test-row predictions
    row: dict[str, float] # per-seed summary
```

```python
def _fit_one(df: pd.DataFrame, seed: int) -> BandRun
```

One seed's entire experiment. It:

1. `three_way_split(df, seed=seed)` → train / calibration / test (60/20/20).
2. Fits `SalaryBandModel(seed=seed)` and takes the **raw** quantile band, scored with `band_report`.
3. Wraps it in `ConformalBand(model).calibrate(...)` and takes the **calibrated** band, scored with
   `conformal.evaluate(...)`.
4. Fits **two** lookup baselines, deliberately:
   - `same_rows` — trained on the identical training rows, so the only difference is the model. The
     like-for-like comparison.
   - `all_rows` — trained on train **+** calibration pooled. The fair fight for a shipping decision,
     because a lookup table needs no calibration set and in production would be built on 80% of the
     data while the band model only gets 60%.
5. Builds a per-test-row frame with `actual`, `predicted`, `raw_lower`, `raw_upper`, `lower`,
   `upper`, `bucket`, `role`, and then six derived columns: `abs_error`, `ape`, `log_residual`,
   `inside`, `width`, `relative_width`.
6. Builds a per-seed summary dict with 17 keys including `coverage_raw`, `coverage_conformal`,
   `rel_width_raw`, `rel_width_conformal`, `qhat_log`, `widening`, `crossing_rate` and
   `trees_median_model`.

```python
def section_band(df: pd.DataFrame, seeds: range, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]
```

Runs `_fit_one` for every seed, builds `summary` (one row per seed) and `pooled` (every test row
from every seed concatenated), and returns both — they are the input to section 6.

Four subsections:

- **accuracy** — band midpoint against the lookup table, both the same-rows and the all-80%
  versions, each with `spread` and a `model wins on N/10 seeds` count.
- **coverage** — raw vs conformal, plus q̂ in log points, plus how far each band edge moved
  (`spread(summary['widening'] - 1.0, percent=True)`), plus the quantile crossing rate. *"The raw
  band under-covers on every seed: asking LightGBM for the 90th percentile is not the same as
  getting it."*
- **width** — median relative width raw and conformal, pooled median width and midpoint in rupees,
  and then **honesty cost** on its own line:
  ```python
  honesty_cost = summary["rel_width_conformal"].mean() / summary["rel_width_raw"].mean() - 1.0
  ```
  Reported separately because it is the trade the whole section turns on: the calibrated band is the
  truthful one **and** the less usable one, and burying that in two other numbers hides it. It comes
  out at **+23.4% more width**.

  Then a real example band, picked as the pooled row whose prediction is closest to the pooled
  median prediction — *"₹4,91,660 – ₹39,84,224 (midpoint ₹13,95,079)"* — followed by:
  *"That is not a salary band. That is the salary range of the whole industry."*
- **pooled test rows** — pooled n with the explicit warning that it is "times scored" and not an
  independent sample size, pooled coverage, pooled MAE and pooled median absolute error.

Saves four charts: `coverage_raw.png`, `coverage_conformal.png`, `band_width.png`,
`band_width_relative.png`.

#### Section 4 — the calibration sweep

```python
def section_calibration(df: pd.DataFrame, out: Path) -> None
```

*"One promise checked is a spot check. Five is a calibration curve."*

For each of the five `CALIBRATION_LEVELS`, it converts the level into three quantiles
(`tail = (1 - level) / 2`, so 0.8 → `(0.1, 0.5, 0.9)`), then for each of `CALIBRATION_SEEDS` fits a
fresh `SalaryBandModel(quantiles=quantiles, seed=seed)`, predicts the raw band, calibrates a
`ConformalBand`, predicts the calibrated band, and accumulates the lower/upper/actual arrays.

That is **15 quantile-model fits per seed**, and it is the reason `CALIBRATION_SEEDS` is 3 rather
than 10.

It prints a table with `raw_actual`, `conformal_actual`, `raw_rel_width`, `conformal_rel_width`,
`raw_shortfall` and `conformal_shortfall`, then how many of the five levels the raw band under-covers
at, the worst raw shortfall and the worst conformal shortfall. Width is reported alongside coverage
on purpose: **a band that gains coverage by getting wider has not become more accurate, only more
honest about how little it knows.**

Saves `calibration_raw.png` and `calibration_conformal.png` via `cal_plots.coverage_by_quantile`.

These two images are the ones described at length in the `calibration.py` section above. The raw
curve sits below the diagonal everywhere; the calibrated one sits on it.

#### Section 5 — feature importance

```python
def _permutation_importance(model: SalaryBandModel, X: pd.DataFrame, y: np.ndarray, *,
                            seed: int, repeats: int = PERMUTATION_REPEATS
                            ) -> tuple[pd.Series, float]
```

**Permutation importance** answers the only question anyone actually cares about: *if this column
were noise, how much money would we lose?* You shuffle one column of the held-out feature matrix,
re-predict, and measure how much worse the MAE got — in rupees, on rows the model never trained on.

A **negative** score means shuffling the column made predictions *better*, and that is a real and
useful result: the model was leaning on noise in that column and the shuffle took the noise away.

Returns `(pd.Series of per-feature scores, baseline_mae)`. The unshuffled MAE comes back too so the
caller can express importance as a *share of the error the model actually makes* rather than as a
bare rupee figure nobody has a scale for.

One implementation detail with a comment worth keeping:

```python
shuffled[name] = features[name].iloc[rng.permutation(len(features))].set_axis(features.index)
```

`.iloc[perm].set_axis(...)` rather than `.to_numpy()[perm]`, because the latter drops the pandas
`category` dtype and LightGBM then refuses to predict, since the categorical feature list no longer
matches.

The known weakness is stated in the docstring: with correlated features (there are four encodings of
experience), shuffling one leaves the others intact, so each individually looks less important than
the group really is. **Read the experience family as a block.**

```python
def section_importance(df: pd.DataFrame, out: Path) -> None
```

Fits three models (`IMPORTANCE_SEEDS`) and collects **three disagreeing importance measures**:

| Measure | Where it comes from | What it means |
|---|---|---|
| `split_%` | `booster.feature_importance("split")`, normalised | How *often* a feature was used to split. |
| `gain_%` | `booster.feature_importance("gain")`, normalised | How much loss it removed during training. |
| `permutation_rupees` | `_permutation_importance` | Held-out rupees of MAE added when shuffled. |

Plus `permutation_sd` (seed-to-seed standard deviation) and `seeds_present` — the skill vocabulary is
refitted per split (top 25 of the *training* rows), so the union across seeds is wider than any one
model, and a row with `seeds_present = 1` is a skill that made one split's top 25 and not the others.

It prints the top 15 by permutation importance, then the features that shuffling made **better or
did nothing**, then a **grouped** table using four blocks:

```python
blocks = {
    "experience (4 encodings)": [c for c in frame.index
                                 if c.startswith("experience") or c == "years_experience"],
    "individual skill flags":   [c for c in frame.index if c.startswith("skill_")],
    "n_skills":                 ["n_skills"],
    "categorical columns":      [c for c in ("role", "org_size", "remote",
                                             "education", "employment_type") if c in frame.index],
}
```

Grouping is necessary because four encodings of experience are one signal wearing four hats and 26
skill flags are one idea spread thin. Ranking them individually against `org_size` compares a
quarter of a feature with a whole one.

Then **the split-count trap, in one comparison** — three named features printed side by side:

- `n_skills` is an integer running 0–20, so a tree can split on it at many thresholds and it climbs
  to 12.3% of split count, second place, ahead of `org_size`. Shuffle it on held-out rows and the
  model loses ₹3,318 — about a quarter of one percent of its error, against +23.13% for
  `years_experience`.
- `education` is **worse than useless**: shuffling it *improves* held-out error.

> Split count measures how **many** questions a feature could answer, not how **useful** the answers
> were. Anyone reporting LightGBM's default importance is reporting cardinality as much as signal.

Finally it applies a threshold that is far more useful than "greater than zero":

```python
threshold = unshuffled_mae * 0.01
n_dead = int((frame["permutation_rupees"] <= 0).sum())
n_real = int((frame["permutation_rupees"] > threshold).sum())
```

"Worth more than 1% of the model's own error." At an MAE around ₹13.3L that is roughly ₹13,300, and
a feature below it could be deleted without anyone noticing. The result: **16 of 37 features carry no
measurable held-out signal, and only 9 move MAE by more than 1% of it** — on 613 training rows.

The section also flags `skill_php` (third by permutation importance) as a hypothesis rather than a
finding, printing its standard deviation next to its mean.

Saves `feature_importance.png` — the one chart in the script hand-built with `new_axes`, `barh` and
`note` rather than a `plots/` function, because a horizontal bar chart of twelve values did not earn
a shared abstraction.

#### Section 6 — where the model is worst

```python
def _slice_stats(group: pd.DataFrame) -> dict[str, float]
```

Seven statistics for any slice of the pooled frame: `n_pooled`, `mae`, `median_abs_error`, `mape_%`,
`coverage`, `median_rel_width`, and `bias_%`.

`bias_%` is `(exp(mean log residual) − 1) × 100`. **Positive means the model predicts *less* than
these people actually earn** (it under-pays them); negative means it predicts more. A slice whose
bias is far from zero is not noisy — it is wrong in one direction, every time.

```python
def section_error_breakdown(pooled: pd.DataFrame, n_seeds: int, out: Path) -> None
```

Three breakdowns, all on the pooled frame from section 3, all with an `n_per_split` column
(`n_pooled / n_seeds`) alongside the pooled count:

- **by experience bucket**, in `BUCKET_LABELS` order. Names the worst bucket by MAPE and reports its
  coverage, bias and people-per-split. The `20+` bucket is where everything goes wrong at once:
  highest MAE, worst MAPE, the only slice where the calibrated band breaks its 80% promise, and a
  systematic −21% bias. All four symptoms have one cause — about 7 people per split.
- **by role**, sorted by MAPE. Splits roles at 5 test rows per split and prints mean `|bias|` on each
  side: **40.8%** in the thin roles against **10.3%** in the rest. Then names the most one-sided role.
- **band width across the salary range**, using `pd.qcut(pooled["predicted"], 10)`. Reports the
  bottom and top deciles, the rupee-width growth factor, and the relative-width *fall*.

The two-sided finding in that last one is the sharpest in the document: width in rupees grows 5.0×
from bottom decile to top, which is exactly what training three separate quantile models was for and
it worked — but width **relative** to the prediction falls from 3.44× to 1.64×, so the band is
proportionally worst exactly where a company hires most: juniors.

Saves `error_by_experience.png` and `error_by_role.png`, both via
`fair_plots.residual_by_group(..., relative=True)`.

#### Section 7 — fairness

```python
def section_fairness(survey_path: Path, df: pd.DataFrame, out: Path) -> None
```

Four subsections.

**7a — what the public survey actually contains.** Reads the CSV header with `nrows=0`, matches
every `PROTECTED_ATTRIBUTE_PATTERNS` entry as a case-insensitive substring, and prints the hit count
for each. Every one returns 0 columns out of 172. The only demographic field of any kind is `Age`.
Then a boxed message: the audit cannot be run on this data, **not because the audit is broken but
because the question is not in the survey.**

**7b — the one quasi-protected attribute we do have: age band.** Runs `raw_gap` and `adjusted_gap`
on 25–34 vs 35–44 year olds with `LEGITIMATE_CONTROLS`. A **43.9% raw gap** becomes a **−5.2%
adjusted gap** whose 95% interval spans about 51 percentage points. Two lessons in one number: a raw
gap is a fact about who the two groups *are* (35–44s have more experience), not about how they are
treated; and 702 people cannot answer this question, so the honest reading is *"we cannot tell"*,
not *"no gap"*.

**7c — validating the audit against a gap we chose.** `validate_on_synthetic(INJECTED_GAPS, n=10_000,
seed=42)` injects gaps of 0/3/8/15/25% and demands them back. Largest error across all five: **0.34
percentage points**; truth inside its interval 5 of 5. The row that matters most is the first — at
an injected gap of **zero** the audit reports −0.34%, so **it does not invent bias in fair data**,
which is the test that matters most.

Then the same run with `control_for_proxy=False`: every estimate is too high by 2.2–2.9 points and
0 of 5 intervals contain the truth. Neither table is "wrong" — they answer different questions.
Controlling for career breaks says *"time out of the workforce is a legitimate reason to pay less"*;
not controlling says *"who takes that time off is itself the unfairness."* That is a policy choice,
not a statistical one, and the honest thing is to publish both numbers.

**7d — the proxy demonstration.** For three seeds it generates 8,000 rows with an 8% injected gap
and runs `compare_with_without`. Deleting the gender column removes about three fifths of the
modelled gap and leaves the rest — delivered by `career_gap_months` — at essentially no accuracy
cost. Then `proxy_correlation(scan_frame, "gender")` shows the channel is visible **before any model
is trained**.

Finally it draws `fairness_validation.png` with `fair_plots.gap_comparison`, passing
`known_gap=truth.pay_gap` so the injected truth appears as a dashed rule on the chart.

#### The entry point

```python
def main() -> int
```

Arguments: `--data` (survey CSV), `--out` (where plots go), `--seeds` (how many splits per
comparison). Exits 1 with a message if the survey file is missing. Prints the settings banner
including numpy and pandas versions and the line:

> Every random draw below comes from a locally-seeded Generator, never from numpy's global state, so
> the same command produces the same numbers.

Then calls the seven sections in order, threading `df` from section 1 into everything and
`summary, pooled` from section 3 into section 6, and prints the elapsed time.

#### The one thing to understand here

**This one script is the reason `docs/findings.md` can be trusted, and the mechanism is boring on
purpose.**

Three properties do all the work:

1. **It regenerates everything.** All 12 charts and every number. There is no second place where a
   figure could be computed, so there is no way for a chart and a paragraph to drift apart.
2. **It is seeded and deterministic.** Every random draw comes from a locally-seeded
   `np.random.default_rng`, never from numpy's global state. Verified byte-identical across runs.
   This matters more than it sounds: global-state randomness means the result depends on *what else
   ran first*, which is exactly the notebook failure the script exists to prevent.
3. **Every comparison runs across 10 splits and reports mean *and* range**, with a
   wins-on-N-of-10 count where the claim is comparative.

And then `spread()` makes the third property structural rather than a matter of remembering. That
function has no mode that prints a bare mean. It was written that way *after* three headline numbers
had been published from too few seeds and had to be corrected — which is the honest reason any such
guardrail exists.

The general lesson worth taking out of this file: **when you catch yourself making the same class of
mistake twice, change the shape of the code so the mistake is not expressible**, rather than
resolving to be more careful. `spread()` is eight lines. It is more reliable than discipline.

#### Surprises and gotchas

- **Several sections use a loop variable after the loop.** `section_baselines` prints
  `groups.coverage_report()` and two `evaluate(...)` lines using `groups`, `split`, `pred_global` and
  `pred_group` from the *last* iteration. It is intentional and the comment says "on the last split
  fitted above", but it will look like a bug to a linter and to you.
- **`section_calibration` keeps only the first level's `truths`** (`if not truths: truths = actual`).
  That is only correct because every level re-runs the same `three_way_split(df, seed=seed)` for the
  same three seeds, so the test rows and their order are identical across levels. It is a real
  assumption and nothing asserts it.
- **The pooled `n` is not a sample size.** A row appears in roughly 2 of the 10 test sets, so a
  "pooled n" of 69 is not 69 independent observations. The script prints the warning, `findings.md`
  repeats it, and it is still the easiest number in the whole document to misread.
- **`blocks["n_skills"] = ["n_skills"]` is hard-coded**, so `section_importance` would raise a
  `KeyError` if that feature ever disappeared from the matrix. The other three blocks are built by
  filtering `frame.index` and degrade gracefully.
- **`section_data` reads the 134MB CSV twice** — once with three columns, once fully through the
  loader. That is most of the script's I/O time and it is deliberate: the non-response arithmetic
  needs the rows the loader drops.
- **The script never calls `plots/explain.contribution_waterfall`.** The SHAP waterfall is not part
  of the findings pipeline.
- **`--seeds` changes the numbers.** Running with `--seeds 3` will produce a document that does not
  match `docs/findings.md`, which was generated with the default 10. Nothing warns you.

---

## Where to go next

You have now read the whole codebase. Three things worth doing with it:

1. **Run `scripts/run_analysis.py`** and check a number in `docs/findings.md` against the output.
   That is the claim the project makes about itself, and it is falsifiable in 30 seconds.
2. **Start the API and send it a deliberately misspelt field**, to watch `extra="forbid"` do its
   job:
   ```bash
   uv run python -m paybands.api.service
   uv run uvicorn paybands.api.app:app --reload
   ```
3. **Read `docs/JOURNEY.md` Part 15** — every mistake made along the way, in one place. Several of
   the design decisions explained in this guide only make sense as scar tissue, and that is where
   the wounds are recorded.
