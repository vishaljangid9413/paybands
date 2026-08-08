"""Plots — the evidence layer.

Every chart in this package exists to prove one specific claim the README makes.
None of them are decoration, and none of them are here because a project ought to
have graphs.

| Claim | Plot |
|---|---|
| "Salaries are skewed, so we model log(salary)" | :func:`salary_distribution_comparison` |
| "Experience flattens — but the 20+ bucket is noise" | :func:`salary_by_experience` |
| "The 80% band really does contain 80%" | :func:`coverage_plot` |
| "It holds at every confidence level, not just one" | :func:`coverage_by_quantile` |
| "The band is wider for senior roles, as it should be" | :func:`band_width_plot` |
| "The model's errors are not one-sided for any group" | :func:`residual_by_group` |
| "The raw gap and the adjusted gap are different things" | :func:`gap_comparison` |

Three rules hold across every function here, and they are worth stating once:

1. **The backend is ``Agg``, chosen before pyplot is imported.** These plots run
   in CI, over SSH, and in Docker, where there is no screen. Nothing ever calls
   ``plt.show()``.
2. **Every function takes an optional ``ax`` and returns ``(figure, axes)``.** So
   any plot composes into a multi-panel figure instead of owning the whole
   canvas. See :func:`salary_distribution_comparison` for what that buys you.
3. **Nothing writes a file as a side effect.** :func:`save` is the only function
   in the package that touches the disk, and you have to call it on purpose.

The calibration and fairness plots additionally take **plain arrays**, importing
nothing from ``paybands.model`` or ``paybands.fairness``. A calibration plot that
only works on our own model's output cannot be used to check our own model.
"""

from . import calibration, distributions, fairness, style
from .calibration import (
    band_width_plot,
    coverage_by_quantile,
    coverage_plot,
    empirical_coverage,
)
from .distributions import (
    bootstrap_median_ci,
    salary_by_experience,
    salary_distribution,
    salary_distribution_comparison,
    skewness,
)
from .fairness import gap_comparison, residual_by_group
from .style import (
    DEFAULT_OUTPUT_DIR,
    SERIES,
    apply_style,
    new_axes,
    percent_formatter,
    rupee_formatter,
    rupee_label,
    save,
    subplots,
)

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "SERIES",
    "apply_style",
    "band_width_plot",
    "bootstrap_median_ci",
    "calibration",
    "coverage_by_quantile",
    "coverage_plot",
    "distributions",
    "empirical_coverage",
    "fairness",
    "gap_comparison",
    "new_axes",
    "percent_formatter",
    "residual_by_group",
    "rupee_formatter",
    "rupee_label",
    "salary_by_experience",
    "salary_distribution",
    "salary_distribution_comparison",
    "save",
    "skewness",
    "style",
    "subplots",
]
