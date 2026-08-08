"""The FastAPI service — three endpoints, and one rule about what they may return.

The rule: **no response leaves this file carrying a salary number without the
width of the band it came from and a plain-English caveat.** All of that is
enforced in `service.py` and typed in `models.py`; this file is routing.

    uv run python -m paybands.api.service        # train + cache a local model
    uv run uvicorn paybands.api.app:app --reload

Endpoints:

* ``POST /predict-band`` — candidate → band, take-home, confidence, caveat, and a
  SHAP explanation of the midpoint (on by default; ``"explain": false`` skips it).
* ``POST /compa-ratio``  — candidate + actual salary → ratio and position label.
* ``GET  /health``       — is a model loaded, which one, which payroll year.

**Status codes, and the one that is a modelling decision.** A wide band is not
an error: `/predict-band` returns 200 with `decision_grade: false`, because the
band is a correct and useful answer that simply must not be quoted. Refusing to
answer would be worse — the caller would work around it, and a workaround has no
caveat attached. The honest design is to answer *and* to say what the answer is
worth.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status

from . import models as api_models
from .service import ModelNotLoaded, PredictionService, get_service

_MODEL_UNAVAILABLE = (
    "No model is loaded. The service starts without one on purpose so it can be "
    "deployed before the first training run; train and cache one with "
    "`uv run python -m paybands.api.service`, then restart. See /health."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model **once**, here, before the first request.

    `get_service` is idempotent, so this is a warm-up rather than the only load
    path — but doing it at startup means a broken or missing artifact shows up
    in the logs at boot instead of inside somebody's first prediction.
    """
    get_service()
    yield


app = FastAPI(
    title="paybands",
    version="0.1.0",
    summary="Salary bands for the Indian market — a range, with its uncertainty attached.",
    description=__doc__,
    lifespan=lifespan,
)


def _service() -> PredictionService:
    return get_service()


@app.get("/health", response_model=api_models.HealthResponse)
def health(response: Response) -> api_models.HealthResponse:
    """Model status, model version, payroll config year.

    200 when a model is loaded, 503 when not — so a load balancer can read the
    status line and a human can read the reason in `detail`.
    """
    result = _service().health()
    if not result.model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@app.post("/predict-band", response_model=api_models.PredictBandResponse)
def predict_band(request: api_models.PredictRequest) -> api_models.PredictBandResponse:
    """Candidate → salary band, take-home, confidence, caveat, and *why*.

    Read `confidence.decision_grade` before `band.midpoint`. The midpoint alone
    is not an estimate of anyone's salary; it is the middle of a range whose
    width is the actual result.

    `explanation` is included by default and is the field that makes the band
    arguable — send `"explain": false` to skip it when bulk-scoring, where nobody
    is going to read the sentences.
    """
    try:
        return _service().predict_band(request.candidate, explain=request.explain)
    except ModelNotLoaded as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, _MODEL_UNAVAILABLE) from exc


@app.post("/compa-ratio", response_model=api_models.CompaRatioResponse)
def compa_ratio(request: api_models.CompaRatioRequest) -> api_models.CompaRatioResponse:
    """Actual salary ÷ predicted midpoint, plus the word HR would use for it.

    This is the pay-equity endpoint: everyone below 0.90 is the underpaid list.
    The band that produced the denominator comes back with the ratio, because a
    ratio against an uncertain midpoint is itself uncertain and the ratio alone
    hides that completely.
    """
    try:
        return _service().compa_ratio(request)
    except ModelNotLoaded as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, _MODEL_UNAVAILABLE) from exc
