# CBH documentation

Documentation for the cloud base height (CBH) retrieval work. The top-level
[README](../../README.md) holds the validated results and the reproduction
steps.

Contents:

- [MODEL_CARD.md](MODEL_CARD.md) - model details, training data, validated
  metrics, limitations (v2.0.0 restudy).
- [requirements_pinned.txt](requirements_pinned.txt) - pinned environment.
- [requirements_production.txt](requirements_production.txt) - dependencies
  for `train_production_model.py`.

History note: earlier Sprint-6 guides reported pooled metrics (R2 = 0.744)
that cross-flight leakage inflated. The v2.0.0 restudy supersedes them. The
removed guides remain in git history.
