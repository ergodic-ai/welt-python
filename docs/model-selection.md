# Selecting and swapping foundation models

Select an exact model family and task through the service catalogue. A listed
family can have a classifier ready while its regressor is unavailable. Check the
requested `task_profiles` entry for `status`, `execution_available`, `version`,
configuration and limits before fitting. Worker availability can change after
inspection; typed service errors remain authoritative at submission time.

```python
from welt import Client, Classifier

with Client() as client:  # WELT_BASE_URL and WELT_API_KEY configured locally
    entries = {entry["id"]: entry for entry in client.models()}
    for model_id in ("tabicl-v2", "kumo-medium", "tabdpt-1.3"):
        profile = next((p for p in entries[model_id]["task_profiles"]
                        if p["task"] == "classification"), None)
        if not profile or profile["status"] != "available" or not profile["execution_available"]:
            raise RuntimeError(f"Requested classifier unavailable: {model_id}")

# X_train, y_train and X_query are your consistently prepared table/split.
# Fit a new immutable predictor for each model, even on the same training table.
models = {
    model_id: Classifier(model=model_id, random_state=42).fit(X_train, y_train)
    for model_id in ("tabicl-v2", "kumo-medium", "tabdpt-1.3")
}
points = {model_id: estimator.predict(X_query) for model_id, estimator in models.items()}
```

Use `Regressor` with the same pattern and inspect regression profiles first.
Named DataFrames preserve fitted columns and allow safe reordering. Existing
predictors retain their exact model/configuration/seed identities; changing the
model parameter requires a new fit. Reopening by predictor ID reuses that pinned
identity and never switches models. Predictors retain support/preprocessing state
when idle native caches are evicted.

Comparing these predictions is not a benchmark. Evaluate on the same appropriate
held-out rows and metric, keep preprocessing inside training folds, and separate
model selection from unbiased assessment. Automatic search, enrichment, conformal
coverage and larger workloads remain separate capabilities. Native classifier
probabilities and regression mean points do not promise calibrated confidence.
