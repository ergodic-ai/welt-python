# SDK 0.1.0 guide

`Client()` reads `WELT_BASE_URL` and `WELT_API_KEY`. Always close it or use a context
manager. `Client.models()` reports actual capabilities, not blanket task support.

```python
from welt import Client

with Client() as client:
    data = client.upload(columns=columns, rows=rows, target=target_name)
    job = client.submit_fit(data['id'], model='tabicl-v2', task='classification', seed=9)
    predictor = job.result(timeout=600)
    metadata = client.predictor(predictor['id'])
```

`Classifier` and `Regressor` support fit/predict and `predict_details`; Classifier
supports predict_proba. Labels/probabilities follow persisted class order. DataFrame
prediction columns may be safely reordered; missing/extra names or width/type
changes reject. Arrays follow positional order. CSV/Parquet paths are convenient
in-memory reads today, not resumable streaming upload. `submit_fit(X,y)` returns
a durable Job without blocking; it owns an HTTP client, so close `job.client` in
a finally block when finished. Blocking estimator fit closes that client internally. Explicit Client uploads permit dataset-ID reuse.

Inspect `job.id`/`job.inspect()`, reconnect with `client.job(id)` and retrieve with
`result()`. A polling deadline is local; it does not cancel durable server execution.
`job.cancel()` explicitly requests cancellation. Typed errors distinguish invalid
input, unavailable model, authentication, capacity and execution failure; IDs allow
operator diagnosis without exposing data values. PredictionPendingError carries
a job_id for reconnect after server-side small-prediction timeout.

Predictors retain exact model/config/seed identity. `Classifier.from_predictor(id)`
reopens a persisted fit. A new fit creates a new predictor; cache eviction does not
mean deletion. Current SDK resources include datasets(), predictors(), jobs() and
usage(); metadata export/delete/recipes are future SDK work.

Only controlled tests establish current clone/Pipeline/CV and construction behavior.
The full official estimator/task/version matrix, async resources, batch/resumable
files, causal results/ergodic and executable Jupyter notebooks remain qualification
and implementation work. No conformal uncertainty or full catalogue availability
is claimed by this SDK release.

No automatic HTTP retry policy is implemented. Reusing a Client submission requires
explicit `idempotency_key` preservation by the caller; omitted keys are freshly
generated per call. Estimator fit/submit_fit makes a new dataset upload and logical
fit, so call Client.submit_fit on the existing dataset for an intentional request replay.
