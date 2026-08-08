"""Feature engineering — turning a common-schema table into model inputs.

Start with `builder.FeatureBuilder`; it composes everything else.

**The idea that runs through every file here: fit on train, transform test.**
Anything that *learns* from data — which skills are common, which categories
exist, what a group's median is — must be learned from the training rows only,
then applied unchanged to the test rows. Learning it from the full dataset is
called **leakage**, it makes your test score a flattering lie, and it is the most
common serious mistake in applied machine learning.

Hence every transformer here exposes scikit-learn's `fit` / `transform` pair —
including the ones that learn nothing, where `fit` being empty is itself the
useful signal that nothing can leak.

| module | learns from data? | so it can leak? |
|---|---|---|
| `experience` | no — pure arithmetic | no |
| `location` | no — hand-written tier lists | no |
| `skills` | **yes** — the top-N vocabulary | **yes, if misused** |
| `builder` | **yes** — the category levels | **yes, if misused** |
"""

from .builder import FeatureBuilder
from .experience import ExperienceFeatures, bucket_years, clean_years, log_years
from .location import LocationFeatures, city_tier, normalise_city
from .skills import SkillFeatures, parse_skills

__all__ = [
    "ExperienceFeatures",
    "FeatureBuilder",
    "LocationFeatures",
    "SkillFeatures",
    "bucket_years",
    "city_tier",
    "clean_years",
    "log_years",
    "normalise_city",
    "parse_skills",
]
