"""Tests for the explanation layer.

Four kinds of test, and the split says what each is for.

**The additivity test is the important one.** SHAP's whole claim is that the
contributions reconstruct the prediction exactly — and "exactly" is only true in
**log space**, which is the subtlety `explain/shapley.py` Idea 2 is about. So it
is asserted at machine precision in logs, and asserted *not* to hold in rupees.
A test that let the rupee column quietly sum correctly would be a test that
someone had rescaled the numbers to make them tie out, which is the one thing
that module refuses to do.

**Behaviour tests** check that the explanation says something true about the
world: a candidate with twenty years of experience should have experience at the
top of their list, and two candidates who differ only in city should differ in
their location contribution.

**Readability tests** check the sentence, because the sentence is the actual
deliverable. Machinery that produces "prev_company_type=services: 0.1391" has
failed even if every number in it is right.

**The plot test** checks the figure comes back and the geometry is exact — the
waterfall's last edge must land on the prediction, which is only true because the
bars are drawn on a log axis.

The model is trained once for the whole module and shared, because training and
building the explainer are the only slow things here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paybands.api import service
from paybands.api.models import Candidate
from paybands.data import synthetic
from paybands.data.schema import TARGET
from paybands.explain import shapley
from paybands.explain.shapley import (
    BandExplainer,
    Explanation,
    explain_batch,
    explain_prediction,
)
from paybands.model.band import SalaryBandModel
from paybands.plots.explain import contribution_waterfall

# Big enough that the trees find real structure and the four experience
# encodings all get used; small enough to train in a couple of seconds.
TRAIN_ROWS = 1_500
SEED = 42


@pytest.fixture(scope="module")
def training_frame() -> pd.DataFrame:
    frame, _ = synthetic.generate(n=TRAIN_ROWS, seed=SEED)
    return frame


@pytest.fixture(scope="module")
def model(training_frame) -> SalaryBandModel:
    return SalaryBandModel(seed=SEED).fit(training_frame, training_frame[TARGET])


@pytest.fixture(scope="module")
def explainer(model) -> BandExplainer:
    return BandExplainer(model)


def frame_for(**fields) -> pd.DataFrame:
    """One candidate as a common-schema frame, through the API's own converter.

    Deliberately routed through `candidate_to_frame` rather than built by hand:
    it means these tests exercise the same request → frame path the service uses,
    so an explanation that works here works there.
    """
    return service.candidate_to_frame(Candidate(**fields))


SENIOR = {
    "years_experience": 20,
    "role": "Backend",
    "org_size": "5000+",
    "city": "Bangalore",
    "prev_company_type": "product",
}
JUNIOR = {**SENIOR, "years_experience": 1}


# ─────────────────────────────────────────────── additivity — the core claim


def test_log_contributions_reconstruct_the_prediction(explainer, model):
    """baseline + sum(SHAP) == the median model's prediction. Exactly.

    This is SHAP's defining property and the reason it is worth using at all:
    the explanation is not a summary of the prediction, it *is* the prediction,
    decomposed. If this ever fails, nothing else in the module means anything.
    """
    explanation = explainer.explain(frame_for(**SENIOR))

    total = explanation.baseline_log + sum(c.log_contribution for c in explanation.contributions)
    assert total == pytest.approx(explanation.prediction_log, abs=1e-9)

    # And that log prediction is the model's own median, not a number the
    # explainer invented alongside it.
    median = model.predict_band(frame_for(**SENIOR)).median[0]
    assert float(np.exp(explanation.prediction_log)) == pytest.approx(median, rel=1e-9)


def test_multipliers_are_exact_and_multiply_to_the_prediction(explainer):
    """The multiplicative reading is exact — `shapley.py` Idea 2, option (a)."""
    explanation = explainer.explain(frame_for(**SENIOR))

    product = np.exp(explanation.baseline_log)
    for contribution in explanation.contributions:
        product *= contribution.multiplier

    assert float(product) == pytest.approx(np.exp(explanation.prediction_log), rel=1e-9)


def test_rupee_contributions_deliberately_do_not_sum(explainer):
    """The rupee column is approximate, and the module refuses to hide it.

    ``exp`` is not linear, so per-feature rupee attributions cannot both be
    individually meaningful and sum to the total. This asserts the *honest*
    behaviour: the residual is real and non-trivial, and it is published.

    If this test starts failing because the residual went to zero, the likely
    cause is that someone rescaled the column to make it tie out. That would be a
    regression, not a fix.
    """
    explanation = explainer.explain(frame_for(**SENIOR))

    attributed = sum(c.rupees for c in explanation.contributions)
    total_move = explanation.prediction - explanation.baseline

    assert explanation.rupee_residual == pytest.approx(total_move - attributed, abs=1.0)
    assert abs(explanation.rupee_residual) > 0
    assert "do NOT add up" in explanation.approximation_note


def test_grouping_experience_is_exact_addition(explainer):
    """Merging the four experience columns loses nothing — Idea 3.

    SHAP values add, so a group is the sum of its parts. Same explanation,
    grouped and ungrouped, must reach the same prediction.
    """
    grouped = explainer.explain(frame_for(**SENIOR))
    raw = explainer.explain(frame_for(**SENIOR), groups={})

    assert grouped.prediction_log == pytest.approx(raw.prediction_log, abs=1e-12)
    assert grouped.n_features < raw.n_features

    merged = next(c for c in grouped.contributions if c.feature == "experience")
    parts = [c for c in raw.contributions if c.feature in shapley.EXPERIENCE_FEATURES]
    assert len(parts) == len(shapley.EXPERIENCE_FEATURES)
    assert merged.log_contribution == pytest.approx(
        sum(c.log_contribution for c in parts), abs=1e-12
    )
    assert merged.members == shapley.EXPERIENCE_FEATURES


# ─────────────────────────────────────────────── does it say true things?


def test_experience_leads_for_a_very_senior_candidate(explainer):
    """`docs/findings.md` §4.1: experience is 53.7% of this model's gain.

    So a candidate at 20 years, holding everything else fixed, should have
    experience as their top factor and pushing upward. This is the test that
    would catch the explanation being wired to the wrong feature order — a bug
    every number in the module would survive.
    """
    explanation = explainer.explain(frame_for(**SENIOR))
    top = explanation.contributions[0]

    assert top.feature == "experience"
    assert top.direction == "increased"
    assert top.multiplier > 1.0
    assert top.rupees > 0


def test_experience_pushes_down_for_a_fresher(explainer):
    """The same feature, the other way. One year is below the training average."""
    explanation = explainer.explain(frame_for(**JUNIOR))
    experience = next(c for c in explanation.contributions if c.feature == "experience")

    assert experience.direction == "decreased"
    assert experience.multiplier < 1.0
    assert experience.rupees < 0


def test_a_metro_city_is_worth_more_than_a_small_one(explainer):
    """Two candidates differing only in city must differ in the location push.

    Not a tautology: a bug that transformed the frame with a stale builder, or
    read contributions off the wrong row, would break this while leaving every
    additivity check intact.

    **What is deliberately not asserted**, because it is not true and the reason
    is instructive: the *other* contributions do not stay fixed. SHAP splits
    credit across features that interact, and if a tree asks about location tier
    inside an experience branch, changing the city genuinely changes how much of
    the answer is attributable to experience. That is the attribution being
    correct, not drifting. Only the total is invariant to how credit is shared —
    which is what the additivity tests above pin down.
    """
    metro = explainer.explain(frame_for(**{**SENIOR, "city": "Bangalore"}))
    small_town = explainer.explain(frame_for(**{**SENIOR, "city": "Nagpur"}))

    def push(explanation: Explanation, feature: str) -> float:
        return next(c.log_contribution for c in explanation.contributions if c.feature == feature)

    assert push(metro, "location_tier") > push(small_town, "location_tier")
    assert metro.prediction > small_town.prediction


def test_a_missing_field_is_explained_rather_than_dropped(explainer):
    """`shapley.py` Idea 4 — the model has a learned response to missingness.

    Leaving `org_size` out does not remove it from the explanation; it changes
    what the explanation says about it. Hiding that would let a caller believe a
    field they never sent had no effect on their number.
    """
    explanation = explainer.explain(frame_for(years_experience=8))
    phrases = [c.phrase for c in explanation.contributions]

    assert any(p == "company size not given" for p in phrases)


# ─────────────────────────────────────────────── is it readable?


def test_the_sentence_reads_like_a_sentence(explainer):
    """The deliverable. `ROADMAP.md` Phase 6's example is the target shape."""
    sentence = explainer.explain(frame_for(**SENIOR)).sentence()

    assert sentence.startswith("Predicted ₹")
    assert " — " in sentence
    assert sentence.endswith(".")
    assert "20 years of experience added ₹" in sentence
    # No column names, no log units, no raw floats leaking into the prose.
    assert "years_experience" not in sentence
    assert "log_contribution" not in sentence


def test_the_sentence_names_at_most_the_requested_number_of_factors(explainer):
    explanation = explainer.explain(frame_for(**SENIOR))

    assert len(explanation.top(2)) <= 2
    assert explanation.sentence(2).count(" added ") + explanation.sentence(2).count(
        " subtracted "
    ) == len(explanation.top(2))


def test_phrases_are_english_not_column_names(explainer):
    """Every contribution carries something a recruiter could say out loud."""
    explanation = explainer.explain(
        frame_for(
            years_experience=8,
            role="Backend",
            org_size="5000+",
            city="Bangalore",
            prev_company_type="services",
            education="Master's degree",
        )
    )
    phrases = {c.feature: c.phrase for c in explanation.contributions}

    assert phrases["experience"] == "8 years of experience"
    assert phrases["role"] == "a Backend role"
    assert phrases["prev_company_type"] == "coming from a services company"
    assert phrases["org_size"] == "a 5000+-person company"
    assert phrases["location_tier"] == "a tier-1 (metro) location"
    assert phrases["education"] == "a Master's degree"


def test_describe_and_to_frame_are_usable(explainer):
    explanation = explainer.explain(frame_for(**SENIOR))

    text = explanation.describe()
    assert "baseline" in text
    assert "do NOT add up" in text

    frame = explanation.to_frame()
    assert list(frame.columns) == [
        "feature",
        "value",
        "phrase",
        "log_contribution",
        "multiplier",
        "rupees",
    ]
    assert len(frame) == explanation.n_features


def test_rupee_figures_are_rounded_to_the_nearest_thousand(explainer):
    """Idea 5 — the same rounding the band uses, so the two cannot disagree."""
    explanation = explainer.explain(frame_for(**SENIOR))

    assert explanation.prediction % shapley.RUPEE_ROUNDING == 0
    assert explanation.baseline % shapley.RUPEE_ROUNDING == 0
    for contribution in explanation.contributions:
        assert contribution.rupees % shapley.RUPEE_ROUNDING == 0


# ─────────────────────────────────────────────── plumbing and refusals


def test_explain_prediction_accepts_a_conformal_band(training_frame):
    """The service holds a `ConformalBand`, so that is what must work.

    Conformal calibration moves the band's *edges* and leaves the median alone,
    so unwrapping to the underlying model is not a shortcut — there is nothing
    about the midpoint for the wrapper to explain.
    """
    bundle = service.train_bundle(n_synthetic=800, seed=SEED)
    explanation = explain_prediction(bundle.band, frame_for(**SENIOR))

    median = bundle.band.predict_band(frame_for(**SENIOR)).median[0]
    assert float(np.exp(explanation.prediction_log)) == pytest.approx(median, rel=1e-9)
    assert explanation.quantile == 0.5


def test_explaining_more_than_one_row_is_refused(explainer):
    two_rows = pd.concat([frame_for(**SENIOR), frame_for(**JUNIOR)], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one row"):
        explainer.explain(two_rows)


def test_an_unfitted_model_cannot_be_explained():
    with pytest.raises(TypeError, match="fitted quantile models"):
        explain_prediction(SalaryBandModel(seed=SEED), frame_for(**SENIOR))


def test_something_that_is_not_a_model_at_all_is_refused():
    with pytest.raises(TypeError):
        explain_prediction(object(), frame_for(**SENIOR))


# ─────────────────────────────────────────────── the global view


def test_explain_batch_summarises_many_rows(model, training_frame):
    held_out, _ = synthetic.generate(n=300, seed=SEED + 1)
    batch = explain_batch(model, held_out)

    assert batch.n_rows == len(held_out)
    assert "experience" in batch.log_contributions.columns
    # Grouping applies here too, so the four encodings are one column.
    for name in shapley.EXPERIENCE_FEATURES:
        assert name not in batch.log_contributions.columns

    ranked = batch.to_frame()
    assert ranked.index[0] == "experience", "experience should dominate — findings.md §4.1"
    assert ranked["mean_abs_log"].is_monotonic_decreasing
    assert ranked["share"].sum() == pytest.approx(1.0)
    assert "experience" in batch.describe()


def test_batch_rows_agree_with_the_per_row_explanation(model, training_frame, explainer):
    """The batch path and the single-row path must be the same computation."""
    one = frame_for(**SENIOR)
    batch = explain_batch(model, one)
    single = explainer.explain(one)

    for contribution in single.contributions:
        assert batch.log_contributions.iloc[0][contribution.feature] == pytest.approx(
            contribution.log_contribution, abs=1e-12
        )


def test_explain_batch_refuses_an_empty_frame(model):
    with pytest.raises(ValueError, match="empty frame"):
        explain_batch(model, pd.DataFrame(columns=["years_experience"]))


# ─────────────────────────────────────────────── the plot


def test_waterfall_returns_a_figure(explainer):
    fig, ax = contribution_waterfall(explainer.explain(frame_for(**SENIOR)))

    assert fig is not None
    assert ax.get_xscale() == "log", "the steps are only exact on a log axis"
    assert len(ax.patches) > 0
    # `loc="left"` because `plots/style.RC_PARAMS` puts every title on the left.
    assert "Why ₹" in ax.get_title(loc="left")


def test_waterfall_lands_exactly_on_the_prediction(explainer):
    """The geometric claim behind the whole chart.

    Bars are drawn as multiplicative steps on a log axis, so the right-hand edge
    of the final bar is the prediction — not approximately, exactly. A rupee-space
    waterfall could not make this assertion without fudging a bar.
    """
    explanation = explainer.explain(frame_for(**SENIOR))
    _, ax = contribution_waterfall(explanation, top_n=3)

    edges = [patch.get_x() + patch.get_width() for patch in ax.patches]
    assert edges[-1] == pytest.approx(np.exp(explanation.prediction_log), rel=1e-9)
    # The first bar starts at the baseline, so the chain covers the whole move.
    assert ax.patches[0].get_x() == pytest.approx(np.exp(explanation.baseline_log), rel=1e-9)


def test_waterfall_pools_the_tail_so_nothing_is_silently_dropped(explainer):
    explanation = explainer.explain(frame_for(**SENIOR))
    _, ax = contribution_waterfall(explanation, top_n=2)

    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels[-1].startswith("everything else")
    assert len(ax.patches) == 3  # two named factors plus the pooled remainder


def test_waterfall_accepts_a_supplied_axes(explainer):
    """Package rule 2: every plot composes into someone else's figure."""
    from paybands.plots.style import subplots

    explanation = explainer.explain(frame_for(**SENIOR))
    fig, axes = subplots(1, 2)
    returned_fig, returned_ax = contribution_waterfall(explanation, ax=axes[0])

    assert returned_fig is fig
    assert returned_ax is axes[0]


def test_waterfall_rejects_nonsense_arguments(explainer):
    explanation = explainer.explain(frame_for(**SENIOR))
    with pytest.raises(ValueError, match="top_n"):
        contribution_waterfall(explanation, top_n=0)
    with pytest.raises(ValueError, match="no contributions"):
        contribution_waterfall(
            Explanation(
                contributions=(),
                baseline=1.0,
                prediction=1.0,
                baseline_log=0.0,
                prediction_log=0.0,
            )
        )
