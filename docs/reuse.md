# Reuse a fitted predictor

Fit once, then predict on new rows using the same private workspace identity.
Reopening fetches the pinned metadata; it does not upload training data or fit again.

## Save the identity

After a successful fit, retain `model.predictor_id_` with your experiment metadata.
A predictor ID is an identifier, not a credential or portable executable model.

```python
predictor_id = model.predictor_id_
metadata = model.export_metadata()
print(predictor_id, metadata["model_version"])
```

Metadata export returns authorized JSON-compatible server provenance. It contains
no API key, model weights or fitted training context. Do not save prediction
payloads, customer rows or credentials into your repository.

## Reopen in another session

Configure your key again for the same workspace. Use the identity you saved and the feature-only test table from [Start](start.md).

```python
from welt import Classifier
from sklearn.metrics import accuracy_score

# Continue with the feature-only Iris test rows from Start.
reopened = Classifier.from_predictor(predictor_id)
repeat = reopened.predict(query)
print("Test accuracy after reopening:", accuracy_score(y_test, repeat))
```

Use `Regressor.from_predictor(...)` for a regression predictor. A mismatched task
rejects instead of changing the prediction semantics. Reopening keeps model,
configuration, version and seed pinned; changing a constructor parameter does not
upgrade a fit. A new task/model/configuration needs a new fit.

## Inspect your workspace

```python
from welt import Client

with Client() as client:
    predictors = client.predictors()
    metadata = client.predictor(predictor_id)
    events = client.usage()
```

Predictors and datasets persist until explicit deletion. Idle native cache eviction
is execution state, distinct from durable ownership. A reconnect check alone does
not prove a worker restart or cold restoration. Usage records show operation,
attempt and lifecycle workload metadata; several events may represent one fit.
Count distinct `operation_id` values when counting logical operations. Usage is
not a monetary price, credit balance or accuracy report.

See the [artifacts and usage notebook](notebooks.md) for executable checks and
[waiting and recovery](async-and-errors.md) if a fit has not returned an identity.
