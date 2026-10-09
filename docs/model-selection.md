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

Use `Regressor` with the same pattern and inspect regression profiles first.
Named DataFrames preserve fitted columns and allow safe reordering. Existing
predictors retain their exact model/configuration/seed identities; changing the
model parameter requires a new fit. Reopening by predictor ID reuses that pinned
identity and never switches models. Predictors retain support/preprocessing state
when idle native caches are evicted.

These are small teaching holdout comparisons, not publisher benchmark results.
Four fits and predictions use hosted resources; this comparison is optional. Evaluate on the same appropriate
held-out rows and metric, keep preprocessing inside training folds, and separate
model selection from unbiased assessment. Automatic search, enrichment, conformal
coverage and larger workloads remain separate capabilities. Native classifier
probabilities and regression mean points do not promise calibrated confidence.

Mitra-v2 uses an explicit zero-shot recipe without prediction-time fine-tuning.
Its native regression histogram mean is a point in the original target units;
it does not provide a calibrated interval or publisher benchmark guarantee.
