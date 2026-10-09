# Predict with your own table

Start with a labelled training DataFrame and a separate query DataFrame. Choose
classification for discrete labels or regression for numeric points; Welt does not
guess the task or target.

## Declare the target

`fit(train, target="churn")` removes the named target from features before upload.
Pass exactly one of `y` or `target`. The original sklearn form `fit(X, y)` remains
supported, including cloning, Pipeline and cross-validation within checked bounds.

```python
import pandas as pd
from welt import Classifier

# Replace these paths with your own approved local tables.
train = pd.read_csv("labelled_customers.csv")
query = pd.read_csv("new_customers.csv")
model = Classifier(model="tabicl-v2", random_state=9)
model.fit(train, target="churn")
predictions = model.predict(query)
assert predictions.shape == (len(query),)
```

This template requires your files: `train` includes a unique string column
`churn`; `query` includes the same feature columns and excludes `churn`.
For a numeric target use `Regressor` explicitly. Start's [complete synthetic
examples](start.md) run without local data files.

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
`WELT_API_KEY`; an explicit constructor key is intentionally not serialized.

## Inspect the result

`predict` returns an array in query row order. Classification probabilities use
`classes_` order; regression returns target-unit mean points. For operation and
version provenance, call `model.predict_details(query)`; inspect pinned fit metadata
with `model.metadata()`. Native probabilities are not conformal coverage and
regression mean points are not calibrated intervals.

A new fit creates a new immutable predictor. Reuse the existing one for subsequent
queries or [reopen it later](reuse.md). Switching model, configuration or task
requires a new fit; [model selection](model-selection.md) explains the task profiles.
