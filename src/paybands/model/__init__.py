"""Layer 1 — the learned part of the system.

How the data is cut (``split``), how a model is scored (``metrics``), the two
dumb models a real model has to beat (``baseline``), and — the Phase 3 payload —
the band model itself (``band``) plus the calibration that makes its confidence
promise true (``conformal``).

The reading order that makes sense: ``split`` → ``metrics`` → ``baseline`` →
``band`` → ``conformal``. Each one is the reason the next exists.
"""

from . import band, baseline, conformal, metrics, split
from .band import (
    BandPrediction,
    BandReport,
    SalaryBandModel,
    band_report,
    compa_label,
)
from .baseline import GlobalMedianBaseline, GroupMedianBaseline, experience_bucket
from .conformal import ConformalBand, QuantileModel, ThreeWaySplit, three_way_split
from .metrics import MetricSet, coverage, evaluate, from_log, pinball_loss, to_log
from .split import Split, random_split, temporal_split

__all__ = [
    "BandPrediction",
    "BandReport",
    "ConformalBand",
    "GlobalMedianBaseline",
    "GroupMedianBaseline",
    "MetricSet",
    "QuantileModel",
    "SalaryBandModel",
    "Split",
    "ThreeWaySplit",
    "band",
    "band_report",
    "baseline",
    "compa_label",
    "conformal",
    "coverage",
    "evaluate",
    "experience_bucket",
    "from_log",
    "metrics",
    "pinball_loss",
    "random_split",
    "split",
    "temporal_split",
    "three_way_split",
    "to_log",
]
