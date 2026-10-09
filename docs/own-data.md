# Predict with your own table

Start with a labelled training DataFrame and a separate query DataFrame. Choose
classification for discrete labels or regression for numeric points; Welt does not
guess the task or target.

## Declare the target

`fit(train, target="churn")` removes the named target from features before upload.
Pass exactly one of `y` or `target`. The original sklearn form `fit(X, y)` remains
supported, including cloning, Pipeline and cross-validation within checked bounds.

```python
# First run Start's real Iris walkthrough to learn the full six-step workflow.
# For your own approved DataFrame, choose its target explicitly:
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from welt import Classifier

print(data.head())
data.info()
target = "churn"  # Replace with your own recorded label column.
print(data[target].value_counts())
train, test = train_test_split(data, test_size=0.2, random_state=9,
                             stratify=data[target])
query = test.drop(columns=target)
y_test = test[target]
model = Classifier(model="tabicl-v2", random_state=9)
model.fit(train, target=target)
predictions = model.predict(query)
print("Test accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
```

This continuation expects your approved `data` DataFrame with a `churn` label.
[Start's real research examples](start.md) are complete, requiring no local dataset.
For numeric targets use `Regressor` and regression metrics instead. Hold back test
answers before fitting; do not fit preprocessing on the test rows.

## Match the schema

DataFrame columns must be unique strings. Named query columns can arrive in a
different order: they are safely aligned to the fitted names. Missing or extra
features reject. Arrays use positional features, so preserve their order.
Feature cells can be scalar numeric, categorical or missing values. Infinite
values and nested objects reject. Targets cannot be missing; regression targets
must be finite numeric values. Target-column convenience requires a DataFrame;
read a CSV/Parquet into one before using `target=`. CSV/Parquet estimator paths
with explicit `y` are convenience in-memory reads; Parquet needs the optional
`parquet` extra. File reading is not resumable upload or arbitrary-size execution.

## Keep evaluation explicit

Separate training and query rows yourself. Respect the task's
[qualified limits](capabilities.md); Welt does not silently sample your table.
Preprocessing belongs inside each training fold during cross-validation. Use the
same held-out rows and metric when comparing models and keep selection separate
from unbiased assessment. Process-based parallel CV workers reauthenticate through
`WELT_API_KEY` or an explicitly saved origin credential; process-only connections
are not serialized. an explicit constructor key is intentionally not serialized.

## Inspect the result

`predict` returns an array in query row order. Classification probabilities use
`classes_` order; regression returns target-unit mean points. For operation and
version provenance, call `model.predict_details(query)`; inspect pinned fit metadata
with `model.metadata()`. Native probabilities are not conformal coverage and
regression mean points are not calibrated intervals.

A new fit creates a new immutable predictor. Reuse the existing one for subsequent
queries or [reopen it later](reuse.md). Switching model, configuration or task
requires a new fit; [model selection](model-selection.md) explains the task profiles.
