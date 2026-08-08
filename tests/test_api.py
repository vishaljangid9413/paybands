"""Tests for the serving layer.

Three kinds of test here, and the split is deliberate.

**Contract tests** hit the endpoints through `TestClient` and assert on the JSON
a recruiter-facing UI would be built against — field names, status codes, and
the invariant that `lower <= midpoint <= upper`.

**Honesty tests** are the ones that matter most, and they are why this file
exists. The band this project currently produces is far too wide to quote, and
the single most likely way for that fact to disappear is for someone to make the
caveat conditional. So the caveat is tested as a *required* output on every
response, and `build_caveat` is tested directly with a hand-made band whose
relative width is chosen on paper — testing the wide-band branch through a
trained model would make the assertion depend on LightGBM's numbers, which tells
you nothing about whether the threshold logic is right.

**Explanation tests** (Phase 6) check that the band arrives with the reasoning
behind it, that the reasoning is in English rather than column names, and that
its rupee figures are published as approximate rather than quietly rescaled to
sum. They also pin the failure mode: a broken explainer costs the caller their
explanation, never their band.

**Arithmetic tests** check that take-home is the payroll calculator's output
applied to the midpoint we published, to the rupee. `docs/design.md` §1: the
model predicts base salary and everything below it is computed. If serving ever
starts *approximating* a deduction, that is the test that catches it.

The model is trained once for the whole module (a few seconds) and injected into
the service singleton, so no test depends on a model artifact existing on disk.
"""

from __future__ import annotations

import math
import warnings

import pytest

# The import itself emits the warning, so the filter has to wrap the import
# rather than sit at module scope. Starlette is telling us to move to `httpx2`;
# that is a dependency decision for `pyproject.toml`, not something a test file
# gets to make, and the warning is harmless in the meantime.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

from paybands.api import models as api_models
from paybands.api import service
from paybands.api.app import app
from paybands.model.band import COMPA_ABOVE_BAND, COMPA_BELOW_BAND, compa_label
from paybands.payroll.calculator import compute_payslip

# Small enough to train in a couple of seconds, large enough that the three-way
# split leaves a real calibration set — below ~40 calibration rows the conformal
# guarantee stops meaning anything (see conformal.calibrate).
TRAIN_ROWS = 1_500

#: A fully-specified candidate whose every value exists in the synthetic
#: generator's vocabulary, so nothing here should come back as unrecognised.
KNOWN_CANDIDATE: dict[str, object] = {
    "years_experience": 8,
    "role": "Backend",
    "education": "Master's degree",
    "org_size": "5000+",
    "remote": "Hybrid",
    "employment_type": "Employed, full-time",
    "institute_tier": "tier1",
    "prev_company_type": "product",
    "location_tier": 1,
}


@pytest.fixture(scope="module")
def bundle() -> service.ModelBundle:
    return service.train_bundle(n_synthetic=TRAIN_ROWS, seed=42)


@pytest.fixture(scope="module")
def rules():
    return service.load_payroll_rules()


@pytest.fixture(scope="module")
def client(bundle, rules):
    """A client backed by an in-memory model.

    The singleton is replaced rather than the artifact written to disk: a test
    that depends on `models/band.pkl` existing would pass on the developer's
    machine and fail in CI, which is the worst possible split.
    """
    service.set_service(service.PredictionService(bundle=bundle, rules=rules))
    with TestClient(app) as test_client:
        yield test_client
    service.set_service(None)


@pytest.fixture
def unhealthy_client(rules):
    """A client whose model never loaded — the fresh-deployment case.

    RESTORES the previous singleton rather than clearing it. An earlier version
    set it to None on teardown, which quietly clobbered the module-scoped
    `client` fixture: every later test in this file then fell through to
    whatever `models/band.pkl` happened to be on disk.

    Those tests passed anyway, because the cached artifact was synthetic-trained
    and behaved like the fixture. The day the artifact was retrained on survey
    data — which has no city column — three city and prev_salary tests failed,
    and the real bug turned out to be six months old.

    A test that reads state from disk is a test that passes on your machine and
    fails in CI, and this one was worse: it passed for the wrong reason.
    """
    previous = service._service
    service.set_service(service.PredictionService(rules=rules, load_error="no artifact on disk"))
    with TestClient(app) as test_client:
        yield test_client
    service.set_service(previous)


def predict(client, **overrides) -> dict:
    candidate = {**KNOWN_CANDIDATE, **overrides}
    response = client.post("/predict-band", json={"candidate": candidate})
    assert response.status_code == 200, response.text
    return response.json()


# ─────────────────────────────────────────────── health


def test_health_reports_a_loaded_model(client, bundle, rules):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"]["version"] == bundle.version
    assert body["payroll_financial_year"] == rules.financial_year
    assert body["payroll_regime"] == rules.regime


def test_health_publishes_the_measured_width(client, bundle):
    """The number that decides whether the service is worth deploying.

    A caller should be able to learn "this model's bands average 0.64x their
    midpoint" without making a prediction, because that is the fact that tells
    them not to build an offer tool on it yet.
    """
    body = client.get("/health").json()
    assert body["model"]["holdout_relative_width"] == pytest.approx(bundle.holdout_relative_width)
    assert "coverage" in body["detail"]


def test_health_is_503_and_honest_when_no_model_is_loaded(unhealthy_client):
    response = unhealthy_client.get("/health")
    assert response.status_code == 503

    body = response.json()
    assert body["status"] == "no_model"
    assert body["model_loaded"] is False
    assert body["model"] is None
    # The payroll config loads independently of the model, and saying so is the
    # difference between "something is broken" and "the model has not been
    # trained yet".
    assert body["payroll_financial_year"] is not None


# ─────────────────────────────────────────────── /schema
#
# The browser UI builds every dropdown from this endpoint rather than shipping
# its own list. That is not a convenience: a hardcoded list goes stale the
# moment the model is retrained on a different source, and a user picking a
# category the model has never seen gets the model's learned response to
# *unknown* while believing they specified something. These tests pin the
# contract the UI depends on.


def test_schema_reports_the_categories_the_model_was_fitted_on(client, bundle):
    body = client.get("/schema").json()
    assert body["trained_on"] == bundle.trained_on
    for column, levels in bundle.known_categories.items():
        assert body["categories"][column] == [str(v) for v in levels]


def test_schema_splits_fields_into_usable_and_inert(client, bundle):
    """Every candidate field lands in exactly one bucket, and neither is a lie.

    A field is inert when the training frame carried no values for it, so the
    model never learned what it means. The UI dims those. If this endpoint ever
    reported an inert field as usable, the UI would present a control that
    silently does nothing — the exact failure this project exists to refuse.
    """
    body = client.get("/schema").json()
    usable, inert = body["usable_fields"], body["inert_fields"]

    assert set(usable) | set(inert) == set(service.CANDIDATE_FIELD_SOURCE)
    assert not set(usable) & set(inert)
    assert set(usable) == set(bundle.usable_fields)


def test_schema_is_503_without_a_model(unhealthy_client):
    """There are no categories to report when nothing is loaded.

    Returning an empty schema with 200 would let the UI render a form full of
    blank dropdowns and look healthy while being useless.
    """
    assert unhealthy_client.get("/schema").status_code == 503


def test_inert_fields_are_reported_back_on_the_prediction(client):
    """Sending an inert field must produce a note, not silence.

    `/schema` warns in advance; this is the belt-and-braces check that a caller
    who ignored it still finds out. The note names the field and the training
    set, so "why did this input change nothing" is answerable without reading
    the source.
    """
    # Which fields are inert depends on the training source, so the values have
    # to be built from the field's own type rather than hardcoded — sending a
    # string where `skills` wants a list gets a 422 and tests validation, not
    # the warning.
    placeholder = {"skills": ["a-skill-nobody-has"], "location_tier": 1}

    inert = client.get("/schema").json()["inert_fields"]
    if not inert:
        pytest.skip("this bundle can use every field, so there is nothing to warn about")

    for field in inert:
        candidate = dict(KNOWN_CANDIDATE) | {field: placeholder.get(field, "anything at all")}
        response = client.post("/predict-band", json={"candidate": candidate})
        assert response.status_code == 200, f"{field}: {response.json()}"

        body = response.json()
        assert any(field in note for note in body["notes"]), f"{field} warned about nothing"
        # The warning must not cost the caller their band. An honest service
        # still answers; it just says what it could not use.
        assert body["band"]["lower"] > 0


def test_service_starts_without_an_artifact(tmp_path, rules):
    """Import and construction must not depend on a trained model existing.

    `models/` is gitignored, so a fresh clone, a fresh container and CI all
    start with nothing on disk. A service that raised here could never be
    deployed before its first training run.
    """
    del rules
    built = service.PredictionService.from_disk(tmp_path / "definitely-not-here.pkl")
    assert built.is_ready is False
    assert "definitely-not-here.pkl" in (built.load_error or "")


def test_predictions_are_503_not_500_without_a_model(unhealthy_client):
    response = unhealthy_client.post("/predict-band", json={"candidate": KNOWN_CANDIDATE})
    assert response.status_code == 503

    response = unhealthy_client.post(
        "/compa-ratio", json={"candidate": KNOWN_CANDIDATE, "actual_salary": 2_000_000}
    )
    assert response.status_code == 503


# ─────────────────────────────────────────────── the band


def test_band_is_ordered_and_positive(client):
    band = predict(client)["band"]
    assert 0 < band["lower"] <= band["midpoint"] <= band["upper"]
    assert band["currency"] == "INR"
    assert band["width"] == pytest.approx(band["upper"] - band["lower"])
    assert band["relative_width"] == pytest.approx(band["width"] / band["midpoint"])


def test_band_is_reported_in_round_rupees(client):
    """No response should imply an accuracy the band's own width disproves."""
    band = predict(client)["band"]
    for edge in ("lower", "midpoint", "upper"):
        assert band[edge] % service.RUPEE_ROUNDING == 0


def test_coverage_is_the_calibrated_promise(client):
    band = predict(client)["band"]
    confidence = predict(client)["confidence"]
    assert band["coverage"] == pytest.approx(0.8)
    assert confidence["nominal_coverage"] == pytest.approx(band["coverage"])
    # Measured on held-out rows the model never saw — the whole point of CQR.
    assert confidence["measured_coverage"] is not None


def test_more_experience_predicts_more_money(client):
    """A sanity check on the wiring, not on the model.

    If the request were being dropped somewhere between JSON and the feature
    matrix, every candidate would get the same band — and every other test here
    would still pass.
    """
    junior = predict(client, years_experience=1)["band"]["midpoint"]
    senior = predict(client, years_experience=15)["band"]["midpoint"]
    assert senior > junior


# ─────────────────────────────────────────────── the caveat


def test_every_response_carries_a_caveat(client):
    for body in (
        predict(client),
        client.post(
            "/compa-ratio", json={"candidate": KNOWN_CANDIDATE, "actual_salary": 2_000_000}
        ).json(),
    ):
        assert isinstance(body["caveat"], str)
        assert body["caveat"].strip()


def test_caveat_always_says_the_figures_are_base_salary(client):
    """ "₹19L" means at least three different things in an Indian salary
    conversation. The API must never be the ambiguous party."""
    assert "BASE salary" in predict(client)["caveat"]


def _band(relative_width: float, midpoint: float = 2_000_000.0) -> api_models.Band:
    """A band of an exactly chosen relative width, for testing the threshold."""
    half = relative_width * midpoint / 2
    return api_models.Band(
        lower=midpoint - half,
        midpoint=midpoint,
        upper=midpoint + half,
        coverage=0.8,
        width=2 * half,
        relative_width=relative_width,
    )


def test_caveat_fires_on_a_band_too_wide_to_quote():
    """The measured case: ~1.9x the midpoint, i.e. ₹19L means ₹5L to ₹42L."""
    caveat = service.build_caveat(_band(1.9), unknown=[], holdout_coverage=0.8)
    assert "NOT DECISION-GRADE" in caveat
    assert "do not put these numbers in an offer" in caveat


def test_caveat_does_not_cry_wolf_on_a_tight_band():
    caveat = service.build_caveat(_band(0.2), unknown=[], holdout_coverage=0.8)
    assert "NOT DECISION-GRADE" not in caveat
    # ...but it still refuses to bless the midpoint on its own.
    assert "midpoint on its own is not an estimate" in caveat


@pytest.mark.parametrize(
    ("relative_width", "expected"),
    [
        (service.TIGHT_RELATIVE_WIDTH, "high"),
        (service.TIGHT_RELATIVE_WIDTH + 0.01, "moderate"),
        (service.DECISION_GRADE_MAX_RELATIVE_WIDTH, "moderate"),
        (service.DECISION_GRADE_MAX_RELATIVE_WIDTH + 0.01, "low"),
    ],
)
def test_confidence_level_boundaries(relative_width, expected):
    assert service.confidence_level(relative_width) == expected


def test_unrecognised_input_caps_confidence():
    """An unknown role becomes NaN, and all three quantile models route NaN the
    same way — so the band can come back tight while resting on a feature the
    model never read. Width alone cannot detect that."""
    assert service.confidence_level(0.1) == "high"
    assert service.confidence_level(0.1, degraded=True) == "moderate"


def test_decision_grade_tracks_the_published_threshold(client):
    confidence = predict(client)["confidence"]
    assert confidence["relative_width_threshold"] == service.DECISION_GRADE_MAX_RELATIVE_WIDTH
    assert confidence["decision_grade"] == (
        confidence["relative_width"] <= service.DECISION_GRADE_MAX_RELATIVE_WIDTH
    )


def test_caveat_names_the_inputs_that_were_ignored():
    caveat = service.build_caveat(_band(0.2), unknown=["role", "city"], holdout_coverage=None)
    assert "role, city" in caveat


# ─────────────────────────────────────────────── take-home (Layer 2)


def test_take_home_is_less_than_base_salary(client):
    body = predict(client)
    take_home = body["take_home"]
    assert take_home["annual_net"] < take_home["annual_gross"]
    assert take_home["monthly_net"] < take_home["monthly_gross"]


def test_take_home_matches_the_payroll_calculator_exactly(client, rules):
    """Not "close to" — exactly.

    PF is a percentage, insurance is a flat premium and income tax is a
    published slab formula. There is no uncertainty anywhere in this
    calculation, so any discrepancy at all means serving has started
    approximating arithmetic, which is the one thing `docs/design.md` §1 forbids.
    """
    body = predict(client)
    expected = compute_payslip(body["band"]["midpoint"], rules)
    take_home = body["take_home"]

    assert take_home["annual_gross"] == expected.annual_gross
    assert take_home["annual_net"] == expected.annual_net
    assert take_home["annual_pf"] == expected.annual_pf
    assert take_home["annual_insurance"] == expected.annual_insurance
    assert take_home["annual_professional_tax"] == expected.annual_professional_tax
    assert take_home["annual_income_tax"] == expected.annual_income_tax
    assert take_home["monthly_net"] == expected.monthly_net


def test_take_home_is_computed_from_the_midpoint_we_published(client):
    """The response has to be internally consistent, or a caller who checks our
    arithmetic will find it wrong."""
    body = predict(client)
    assert body["take_home"]["annual_gross"] == body["band"]["midpoint"]


def test_take_home_names_its_config_year(client, rules):
    take_home = predict(client)["take_home"]
    assert take_home["financial_year"] == rules.financial_year
    assert take_home["regime"] == rules.regime


# ─────────────────────────────────────────────── compa-ratio


def compa(client, actual_salary: float, **overrides) -> dict:
    candidate = {**KNOWN_CANDIDATE, **overrides}
    response = client.post(
        "/compa-ratio", json={"candidate": candidate, "actual_salary": actual_salary}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_compa_ratio_is_actual_over_midpoint(client):
    midpoint = predict(client)["band"]["midpoint"]
    body = compa(client, 2_000_000)
    assert body["band"]["midpoint"] == midpoint
    assert body["compa_ratio"] == pytest.approx(2_000_000 / midpoint)


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.5, "below band"),
        (COMPA_BELOW_BAND - 1e-9, "below band"),
        (COMPA_BELOW_BAND, "in band"),  # 0.90 is inclusive — the boundary is in
        (1.0, "in band"),
        (COMPA_ABOVE_BAND, "in band"),  # 1.10 is inclusive too
        (COMPA_ABOVE_BAND + 1e-9, "above band"),
        (1.5, "above band"),
    ],
)
def test_compa_label_boundaries(ratio, expected):
    """The exact boundary behaviour, tested on the pure function.

    Driving these through the endpoint would mean multiplying a float midpoint
    and dividing it back out, and the last bit of that round trip decides the
    answer at exactly 0.90. That would be a test of IEEE 754, not of the label.
    """
    assert compa_label(ratio) == expected


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.80, "below band"), (0.95, "in band"), (1.25, "above band")],
)
def test_compa_ratio_endpoint_labels(client, fraction, expected):
    """End-to-end, at fractions far enough from the cut points that float noise
    cannot flip them."""
    midpoint = predict(client)["band"]["midpoint"]
    body = compa(client, round(midpoint * fraction))
    assert body["position"] == expected


def test_compa_ratio_returns_the_band_behind_the_denominator(client):
    """A ratio against an uncertain midpoint is itself uncertain, and the ratio
    alone hides that completely."""
    body = compa(client, 2_000_000)
    assert 0 < body["band"]["lower"] <= body["band"]["midpoint"] <= body["band"]["upper"]
    assert body["caveat"].strip()
    assert "decision_grade" in body["confidence"]


# ─────────────────────────────────────────────── input validation


@pytest.mark.parametrize(
    "candidate",
    [
        {"years_experience": -1},  # impossible
        {"years_experience": 200},  # a typo, not a career
        {"years_experience": "eight"},  # not a number
        {},  # the one required field is missing
        {"years_experience": 5, "location_tier": 9},  # only tiers 1-3 exist
        {"years_experience": 5, "role": ""},  # blank is not a category
        {"years_experience": 5, "performance_rating": 11},  # scale is 0-5
        {"years_experience": 5, "experience": 5},  # misspelt field name
    ],
)
def test_invalid_candidates_are_422_not_500(client, candidate):
    """A misspelt field is in this list on purpose.

    Silently ignoring `experience` would produce a plausible band computed from
    less information than the caller thought they sent — a wrong answer with no
    trace of being wrong. `extra="forbid"` turns that into a 422.
    """
    response = client.post("/predict-band", json={"candidate": candidate})
    assert response.status_code == 422


@pytest.mark.parametrize("actual_salary", [0, -5, 50_000, 500_000_000])
def test_implausible_actual_salaries_are_422(client, actual_salary):
    """The bounds are `data/schema.py`'s cleaning thresholds, reused.

    Rows outside them were dropped from training, so a compa-ratio against them
    would divide by a midpoint extrapolated well beyond any evidence.
    """
    response = client.post(
        "/compa-ratio", json={"candidate": KNOWN_CANDIDATE, "actual_salary": actual_salary}
    )
    assert response.status_code == 422


# ─────────────────────────────────────────────── degrading gracefully


def test_unknown_role_degrades_instead_of_crashing(client):
    """`FeatureBuilder` Decision 2 maps an unseen category to NaN, because the
    model has no learned response to a role it never saw. The API's job is to
    answer the more general question and *say* that it did."""
    body = predict(client, role="Prompt Engineer")
    assert body["band"]["midpoint"] > 0
    assert "role" in body["unrecognised_inputs"]
    assert "role" in body["caveat"]
    assert body["confidence"]["level"] != "high"


def test_unknown_city_is_priced_as_tier_3_and_says_so(client):
    """`features/location.py` treats an unknown-but-readable city as tier 3 —
    "a real place that is not a tech metro", which is a genuine claim. But it is
    also the catch-all, so a misspelling lands there silently and gets priced
    below the metro rate. The response says which happened."""
    body = predict(client, city="Atlantis", location_tier=None)
    assert body["band"]["midpoint"] > 0
    assert body["unrecognised_inputs"] == []
    assert any("tier-3" in note for note in body["notes"])


def test_unreadable_city_is_reported_as_unrecognised(client):
    """A city that normalises to nothing at all is a different case: tier NaN,
    "we were not told", rather than a claim that it is a non-metro."""
    body = predict(client, city="12345 ...", location_tier=None)
    assert body["band"]["midpoint"] > 0
    assert "city" in body["unrecognised_inputs"]


def test_known_city_aliases_are_understood(client):
    """ "Bengaluru", "BLR" and "Bangalore" are one labour market. Left unhandled,
    two of those three become tier 3 and the model learns Bangalore is cheap."""
    for spelling in ("Bangalore", "Bengaluru", "BLR", "Bangalore, KA"):
        body = predict(client, city=spelling, location_tier=None)
        assert body["unrecognised_inputs"] == []


def test_a_bare_minimum_candidate_still_gets_an_answer(client):
    """Every field but experience is optional, and omitting one is not an error.

    `FeatureBuilder` never fills a missing value with a default — NaN is the
    honest encoding of "we were not told". The band simply comes back wider.
    """
    response = client.post("/predict-band", json={"candidate": {"years_experience": 3}})
    assert response.status_code == 200
    assert response.json()["band"]["midpoint"] > 0


def test_prev_salary_is_accepted_and_deliberately_unused(client):
    """The previous-salary trap, `docs/design.md` §4.1.

    Anchoring an offer to last drawn pay makes an early underpayment follow
    someone for their whole career. The field is accepted so a caller can send
    what they have; the band must be identical with and without it, and the
    response must say we declined to use it.
    """
    without = predict(client)
    with_it = predict(client, prev_salary=400_000)

    assert with_it["band"] == without["band"]
    assert any("prev_salary" in note for note in with_it["notes"])
    assert without["notes"] == []


def test_candidate_to_frame_never_carries_prev_salary():
    """Belt and braces at the layer below the endpoint: even if the note above
    were deleted, the value must not reach a feature matrix."""
    candidate = api_models.Candidate(years_experience=5, prev_salary=400_000)
    assert "prev_salary" not in service.candidate_to_frame(candidate).columns


def test_city_takes_precedence_over_location_tier(client):
    """The city is the more specific claim; the tier is derivable from it."""
    body = predict(client, city="Bangalore", location_tier=3)
    assert any("city took precedence" in note for note in body["notes"])


# ─────────────────────────────────────────────── the explanation (Phase 6)


def test_every_prediction_explains_itself_by_default(client):
    """`explain` defaults to true, and that default is the feature.

    An explanation a caller has to opt into is an explanation nobody sees, and a
    salary number nobody can argue with is exactly the kind that causes harm
    (`ROADMAP.md` Phase 6). So the plain request — no flag at all — must come
    back with the reasoning attached.
    """
    body = predict(client)
    explanation = body["explanation"]

    assert explanation is not None
    assert explanation["contributions"], "an explanation with no factors explains nothing"
    assert explanation["summary"].startswith("Predicted ₹")
    assert explanation["n_factors"] >= len(explanation["contributions"])


def test_the_explanation_explains_the_midpoint_we_published(client):
    """Same number, both places. A response that disagrees with itself is worse
    than one that says less."""
    body = predict(client)
    assert body["explanation"]["prediction"] == body["band"]["midpoint"]


def test_explanation_contributions_are_in_plain_english(client):
    """The deliverable is a sentence a recruiter can repeat, not a feature dump."""
    top = predict(client)["explanation"]["contributions"][0]

    assert top["direction"] in {"increased", "decreased", "no effect"}
    assert "_" not in top["phrase"], f"{top['phrase']} reads like a column name"
    assert top["multiplier"] > 0
    assert top["rupees"] % service.RUPEE_ROUNDING == 0


def test_experience_leads_the_explanation_for_a_senior_candidate(client):
    """`docs/findings.md` §4.1 — experience is 53.7% of this model's gain.

    Also pins the *grouping* decision: the four experience encodings the model
    actually uses arrive as one `experience` row, because four partly
    contradictory rows about the same fact is not an explanation.
    """
    contributions = predict(client, years_experience=20)["explanation"]["contributions"]
    features = [c["feature"] for c in contributions]

    assert features[0] == "experience"
    assert contributions[0]["direction"] == "increased"
    assert "years_experience" not in features
    assert "experience_log" not in features


def test_the_explanation_admits_its_rupees_do_not_add_up(client):
    """The honesty field, and it is required rather than optional.

    ``exp`` is not linear, so per-factor rupee attributions cannot both be
    individually meaningful and sum to the total. JSON strips that caveat by
    default — a UI rendering a neat stack of rupee figures will be read as a sum
    unless the payload says otherwise. See `explain/shapley.py` Idea 2.
    """
    explanation = predict(client)["explanation"]

    assert "do NOT add up" in explanation["approximation_note"]

    attributed = sum(c["rupees"] for c in explanation["contributions"])
    move = explanation["prediction"] - explanation["baseline"]
    assert attributed != pytest.approx(move, rel=1e-6), (
        "if these ever tie out exactly, someone has rescaled the column to make "
        "it look additive — which is the one thing shapley.py refuses to do"
    )


def test_multipliers_reconstruct_the_prediction_from_the_response_alone(client):
    """The exact reading, checkable by a caller with no access to the model.

    `log_contribution` is on the wire precisely so this assertion is available to
    anyone holding the JSON. Only the top few contributions are returned, so the
    check is a lower bound rather than an equality — the response says as much
    through `n_factors`.
    """
    explanation = predict(client)["explanation"]

    for contribution in explanation["contributions"]:
        assert contribution["multiplier"] == pytest.approx(
            math.exp(contribution["log_contribution"]), rel=1e-9
        )


def test_explanation_can_be_switched_off_for_bulk_scoring(client):
    """The flag exists for the pay-equity list — ten thousand employees, nobody
    reading ten thousand sentences — not as a way for a UI to skip the caveats."""
    response = client.post("/predict-band", json={"candidate": KNOWN_CANDIDATE, "explain": False})
    assert response.status_code == 200

    body = response.json()
    assert body["explanation"] is None
    # Everything else is untouched: turning off the reasoning must not turn off
    # the honesty.
    assert body["caveat"]
    assert body["band"] == predict(client)["band"]


def test_a_broken_explainer_costs_the_explanation_not_the_prediction(bundle, rules, monkeypatch):
    """Decision 5 — an explanation is an addition to a band, not a gate on it.

    A band the caller can use beats a 500 they cannot. But the failure is
    reported in `notes` rather than swallowed, because a silently
    explanation-free response looks complete and is not.

    The explainer is sabotaged at its construction point rather than by poking a
    private attribute, so this exercises the same `except` the real failure
    would.
    """

    def refuse(_model):
        raise RuntimeError("pretend shap could not read this booster")

    monkeypatch.setattr(service, "BandExplainer", refuse)
    broken = service.PredictionService(bundle=bundle, rules=rules)

    response = broken.predict_band(api_models.Candidate(**KNOWN_CANDIDATE))

    assert response.band.midpoint > 0
    assert response.explanation is None
    assert any("could not be explained" in note for note in response.notes)


def test_the_explainer_is_built_once_and_reused(client, bundle, rules):
    """Constructing a `TreeExplainer` parses the whole booster; explaining a row
    afterwards is about a millisecond. Rebuilding per request would make the
    cheap part pay for the expensive part on every call."""
    del client
    built = service.PredictionService(bundle=bundle, rules=rules)
    assert built.explainer() is built.explainer()


def test_compa_ratio_is_unchanged_by_phase_6(client):
    """The pay-equity endpoint keeps its exact shape. Adding a field to one
    response must not quietly reshape another."""
    response = client.post(
        "/compa-ratio", json={"candidate": KNOWN_CANDIDATE, "actual_salary": 2_000_000}
    )
    assert response.status_code == 200
    assert "explanation" not in response.json()
