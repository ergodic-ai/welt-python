# Predict with your own table

Train a model on your own labelled table and check it on rows it hasn't seen. You
choose the task and name the target column; Welt doesn't guess either.

!!! need "You'll need"
    - A labelled training DataFrame and a separate query DataFrame, from data you're
      approved to use.
    - The six-step workflow from [Your first prediction](start.md).

Choose classification (`Classifier`) for discrete labels, or regression (`Regressor`)
for numeric points.

## Declare the target

Pass the name of your label column to `fit`. `fit(train, target="churn")` removes the
named target from the features before upload. Pass exactly one of `y` or `target`.

The original sklearn form `fit(X, y)` is still supported, including cloning, Pipeline
and cross-validation within checked bounds.

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
[Start's real research examples](start.md) are complete and need no local dataset.
For numeric targets, use `Regressor` and regression metrics instead.

!!! caution "Keep test answers out of training"
    Hold back test answers before fitting. Don't fit preprocessing on the test rows.

## Match the schema

- DataFrame columns must be unique strings.
- Named query columns can arrive in a different order: they are safely aligned to the
  fitted names. Missing or extra features are rejected.
- Arrays use positional features, so keep their order.
- Feature cells can be scalar numeric, categorical or missing values. Infinite values
  and nested objects are rejected.
- Targets can't be missing. Regression targets must be finite numeric values.

`target=` needs a DataFrame. Read a CSV or Parquet file into one before you use it.

<details markdown="1"><summary>Can I pass a CSV or Parquet file instead?</summary>

Estimators accept CSV and Parquet paths with an explicit `y`, as a convenience
in-memory read. Parquet needs the optional `parquet` extra. File reading is not
resumable upload or arbitrary-size execution.

</details>

## Keep evaluation explicit

Separate training and query rows yourself. Respect the task's
[qualified limits](capabilities.md): Welt does not silently sample your table.

- During cross-validation, preprocessing belongs inside each training fold.
- When you compare models, use the same held-out rows and metric.
- Keep model selection separate from unbiased assessment.

<details markdown="1"><summary>Running cross-validation in parallel processes?</summary>

Process-based parallel CV workers reauthenticate through `WELT_API_KEY` or an
explicitly saved origin credential. Process-only connections are not serialized, and
an explicit constructor key is intentionally not serialized either.

</details>

## Inspect the result

`predict` returns an array in query row order. Classification probabilities follow
`classes_` order; regression returns mean points in the target's units.

For operation and version provenance, call `model.predict_details(query)`. To
inspect pinned fit metadata, call `model.metadata()`.

!!! caution "Not calibrated uncertainty"
    Native probabilities are not conformal coverage, and regression mean points are
    not calibrated intervals.

A new fit creates a new immutable predictor. Reuse the existing one for later queries,
or [reopen it later](reuse.md). Switching model, configuration or task requires a new
fit; [model selection](model-selection.md) explains the task profiles.

## Next steps

<div class="cards" markdown="1">

- [Reuse a predictor](reuse.md): reopen a fitted predictor without another fit.
- [Select a model](model-selection.md): compare task profiles fairly.
- [Models and limits](capabilities.md): the qualified limits for each task.

</div>
