"""Explanations — the layer that lets a recruiter argue with the model.

`model/` says *what* the band is. `payroll/` says what lands in the bank
account. `policy/` says what to do about the gap. This package answers the
question all three of those invite and none of them address: **why that
number?**

That is not a nice-to-have. A salary figure with no reasoning attached cannot be
challenged, and an unchallengeable salary figure is exactly the kind that causes
harm — it turns a model's guess into an authority. The same explanation is what
makes a decision defensible later, when someone asks why this candidate was
offered less than that one.

One module, `shapley`, and the two functions worth remembering::

    explain_prediction(model, one_row)   # why this candidate got this number
    explain_batch(model, many_rows)      # what the model relies on in general

The hard part is not SHAP. It is that the model is trained on ``log(salary)``, so
SHAP values arrive in **log units** — and a contribution of ``0.18`` means
nothing to anybody. `shapley`'s module docstring (Idea 2) works through the two
honest conversions, which one is exact, and why the readable one deliberately
does not add up.

`plots/explain.py` draws one of these.
"""

from . import shapley
from .shapley import (
    DEFAULT_GROUPS,
    EXPERIENCE_FEATURES,
    BandExplainer,
    BatchExplanation,
    Contribution,
    Explanation,
    explain_batch,
    explain_prediction,
    median_quantile_model,
    sentence,
)

__all__ = [
    "DEFAULT_GROUPS",
    "EXPERIENCE_FEATURES",
    "BandExplainer",
    "BatchExplanation",
    "Contribution",
    "Explanation",
    "explain_batch",
    "explain_prediction",
    "median_quantile_model",
    "sentence",
    "shapley",
]
