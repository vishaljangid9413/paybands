"""Phase 4 — the bias audit.

Two halves that answer two different questions.

``audit`` measures the gap: what one group is paid relative to another, before
and after accounting for the things a company can legitimately pay differently
for, plus whether a model's own errors lean one way. It takes plain arrays and
knows nothing about any particular model, so it works on the baseline, on the
band model, or on last year's spreadsheet of offers.

``proxy`` explains why the obvious fix does not work. Delete the gender column
and a model rebuilds the gap out of career breaks and everything else that
quietly correlates with it. That is a demonstration, not an assertion, and it is
the most useful thing in this package.

Start with :func:`paybands.fairness.audit.validate_on_synthetic`. It checks the
audit against data whose true gap we chose ourselves — because a bias detector
nobody has calibrated is an opinion with decimal places.
"""

from . import audit, proxy
from .audit import (
    CONTESTED_CONTROLS,
    LEGITIMATE_CONTROLS,
    AdjustedGap,
    FairnessReport,
    RawGap,
    ResidualCheck,
    adjusted_gap,
    raw_gap,
    residual_analysis,
    validate_on_synthetic,
)
from .proxy import (
    ProxyComparison,
    ProxyScan,
    compare_with_without,
    proxy_correlation,
)

__all__ = [
    "CONTESTED_CONTROLS",
    "LEGITIMATE_CONTROLS",
    "AdjustedGap",
    "FairnessReport",
    "ProxyComparison",
    "ProxyScan",
    "RawGap",
    "ResidualCheck",
    "adjusted_gap",
    "audit",
    "compare_with_without",
    "proxy",
    "proxy_correlation",
    "raw_gap",
    "residual_analysis",
    "validate_on_synthetic",
]
