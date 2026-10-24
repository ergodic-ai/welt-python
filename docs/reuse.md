# Reuse a fitted predictor

Fit once, then predict on new rows with the same predictor in your private workspace.
Reopening fetches the pinned metadata; it doesn't upload training data or fit again.

A predictor is the fitted model Welt keeps in your workspace after a successful `fit`.
You reach it again through its predictor ID.

## Save the predictor ID {#save-the-identity .step}

After a successful fit, keep `model.predictor_id_` with your experiment metadata.

```python
predictor_id = model.predictor_id_
metadata = model.export_metadata()
print(predictor_id, metadata["model_version"])
```

A predictor ID is an identifier, not a credential or a portable executable model.
Metadata export returns authorized, JSON-compatible server provenance. It contains no
API key, model weights or fitted training context.

!!! caution "Keep data out of your repository"
    Don't save prediction payloads, customer rows or credentials into your repository.

## Reopen in another session {.step}

Configure your key again for the same workspace. Then use the ID you saved and the
feature-only test table from [Start](start.md):

```python
from welt import Classifier
from sklearn.metrics import accuracy_score

# Continue with the feature-only Iris test rows from Start.
reopened = Classifier.from_predictor(predictor_id)
repeat = reopened.predict(query)
print("Test accuracy after reopening:", accuracy_score(y_test, repeat))
```

For a regression predictor, use `Regressor.from_predictor(...)`. A mismatched task is
rejected instead of changing what the predictions mean.

Reopening keeps the model, configuration, version and seed pinned. Changing a
constructor parameter does not upgrade a fit. A new task, model or configuration needs
a new fit.

## Inspect your workspace

List your predictors, fetch one predictor's metadata and read usage events:

```python
from welt import Client

client = Client()
predictors = client.predictors()
metadata = client.predictor(predictor_id)
events = client.usage()
```

!!! note "What persists"
    Predictors and datasets persist until they are explicitly deleted. Idle native cache
    eviction is execution state, separate from durable ownership.

Usage records show operation, attempt and lifecycle workload metadata. Several events
may represent one fit, so count distinct `operation_id` values when you count logical
operations. Usage is not a monetary price, credit balance or accuracy report.

<details markdown="1"><summary>Does a successful reconnect prove the worker restarted?</summary>

No. A reconnect check alone does not prove a worker restart or cold restoration.

</details>

## Next steps

<div class="cards" markdown="1">

- [Notebook examples](notebooks.md): the artifacts and usage notebook runs these checks.
- [Wait and recover](async-and-errors.md): what to do if a fit hasn't returned an ID.
- [Use your own table](own-data.md): fit a predictor on your data.

</div>
