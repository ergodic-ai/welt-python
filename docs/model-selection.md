# Selecting and swapping foundation models

Check that each model is ready for your task, then compare several models on the
same held-out rows. Each model you try needs its own fit.

!!! need "You'll need"
    - The Iris split from [Your first prediction](start.md): `train`, `target`, `query`
      and `y_test`.
    - A connected Welt workspace. See [Connect your workspace](connect.md).

## Check readiness, then compare

You select an exact model family and task through the service catalogue. Readiness
is per task: a listed family can have a classifier ready while its regressor is
unavailable. Before fitting, check the requested task's `task_profiles` entry for
`status`, `execution_available`, `version`, configuration and limits.

The example below has two parts. The first stops if any requested classifier is not
available. The second fits each model on the same training rows and scores it on the
same test rows.

!!! caution "This comparison is optional and uses hosted resources"
    Four fits and predictions use hosted resources. Each fit retains a new predictor.

```python
from welt import Client, Classifier

with Client() as client:  # WELT_BASE_URL and WELT_API_KEY configured locally
    entries = {entry["id"]: entry for entry in client.models()}
    for model_id in ("tabicl-v2", "kumo-medium", "tabdpt-1.3", "mitra-v2"):
        profile = next((p for p in entries[model_id]["task_profiles"]
                        if p["task"] == "classification"), None)
        if not profile or profile["status"] != "available" or not profile["execution_available"]:
            raise RuntimeError(f"Requested classifier unavailable: {model_id}")

# Continue with Start's Iris train/test split. Each fit retains a new predictor.
from sklearn.metrics import accuracy_score
for model_id in ("tabicl-v2", "kumo-medium", "tabdpt-1.3", "mitra-v2"):
    model = Classifier(model=model_id, random_state=9)
    model.fit(train, target=target)
    predictions = model.predict(query)
    print(model_id, "test accuracy:", accuracy_score(y_test, predictions))
```

!!! note "Readiness can change"
    Worker availability can change after you inspect it. Typed service errors at
    submission time remain authoritative.

For a numeric outcome, use `Regressor` with the same pattern, and inspect the
regression profiles first.

## What stays fixed when you swap models

- **A new model needs a new fit.** Existing predictors keep their exact model,
  configuration and seed identities. Changing the `model` parameter requires a new fit.
- **Reopening never switches models.** Reopening by predictor ID reuses that pinned
  identity.
- **DataFrame columns are matched by name.** Named DataFrames preserve the fitted columns and
  allow safe reordering.
- **Idle caches don't lose state.** Predictors keep their support and preprocessing
  state when idle native caches are evicted.

## Compare models fairly

These are small teaching holdout comparisons, not publisher benchmark results. When
you compare models for real:

- Evaluate every model on the same appropriate held-out rows and the same metric.
- Keep preprocessing inside the training folds.
- Separate model selection from unbiased assessment.

!!! caution "Scores are not calibrated confidence"
    Native classifier probabilities and regression mean points do not promise
    calibrated confidence.

Automatic search, enrichment, conformal coverage and larger workloads remain
separate capabilities.

<details markdown="1"><summary>How does Mitra-v2 predict?</summary>

Mitra-v2 uses an explicit zero-shot recipe without prediction-time fine-tuning. Its
native regression histogram mean is a point in the original target units. It does
not provide a calibrated interval or publisher benchmark guarantee.

</details>

## Next steps

<div class="cards" markdown="1">

- [Models and limits](capabilities.md): every qualified task and its limits.
- [Predict a number](regression.md): the same pattern with `Regressor`.
- [Reuse a predictor](reuse.md): reopen a fitted predictor without another fit.

</div>
