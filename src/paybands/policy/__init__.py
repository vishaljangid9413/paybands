"""Layer 3 — HR policy written as code, with no machine learning in it.

Layer 1 (``model``) predicts what the market pays. Layer 2 (``payroll``)
computes what lands in the bank account. This layer decides what to *do* about
the gap between the two: who is underpaid, what a rating is worth in rupees,
and how to spend a pool that is always smaller than the recommendations.

Those are decisions, not patterns — so they are rules in a config file rather
than weights in a model, and every recommendation comes with a sentence a
recruiter can argue with. See ``increment.py`` for why that matters.
"""

from . import increment
from .increment import (
    BudgetAllocation,
    EquityReview,
    IncrementRecommendation,
    IncrementRules,
    apply_budget_constraint,
    band_label,
    equity_review,
    recommend,
    recommend_batch,
)

__all__ = [
    "BudgetAllocation",
    "EquityReview",
    "IncrementRecommendation",
    "IncrementRules",
    "apply_budget_constraint",
    "band_label",
    "equity_review",
    "increment",
    "recommend",
    "recommend_batch",
]
