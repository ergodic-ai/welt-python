# SDK guide

`Client()` defaults to `https://welt.ergodic.dev`; explicit `base_url` overrides
`WELT_BASE_URL`. Authenticated methods use your connected/saved workspace credential,
`WELT_API_KEY` or an explicit key. See [Connect](connect.md) for precedence. Always close it or use a context
manager. `Client.models()` reports actual capabilities, not blanket task support.

```python
# Continue with the real Iris train/test tables from Start.
from welt import Client
from sklearn.metrics import accuracy_score

with Client() as client:
    dataset = client.upload(columns=list(train.columns), rows=train.values.tolist(), target=target)
    job = client.submit_fit(dataset["id"], model="tabicl-v2", task="classification", seed=9)
    print("Keep the job ID:", job.id)
    predictor = job.result(timeout=600)
    result = client.predict(predictor["id"], columns=list(query.columns), rows=query.values.tolist())
print("Test accuracy:", accuracy_score(y_test, result["predictions"]))
```

`Classifier` and `Regressor` support fit/predict and `predict_details`; Classifier
supports predict_proba. Labels/probabilities follow persisted class order. DataFrame
prediction columns may be safely reordered; missing/extra names or width/type
changes reject. Arrays follow positional order. CSV/Parquet paths are convenient
in-memory reads today, not resumable streaming upload. `submit_fit(X, y)` or `submit_fit(train, target=target)` returns
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
usage(); metadata inspection/export_metadata() returns authorized JSON-compatible server
provenance only, never weights/context/credentials. Explicit owned dataset/causal-result
deletion is available; recipe/deletion-plan transport remains separate.

`Client` and `AsyncClient` also provide resumable upload sessions, `upload_file`,
`submit_batch_predict` and `batch_payload`. Keep the upload ID before sending bytes
and inspect batch manifests for failed row ranges. Enabled file formats, batch
limits and model readiness depend on the connected service. See [CLI recipes](cli.md)
and [capabilities](capabilities.md) for those boundaries. Native ArrowFM/CDFM/AVICI
results retain their graph semantics; public Ergodic continuation and remaining
engines have separate availability requirements. See the [research catalogue](research-datasets.md) for tables to try.

## Usage and workspace credits

Modelling and research downloads may consume workspace credits. Before running a
walkthrough, open **Workspace → Billing** in [Welt](https://welt.ergodic.dev) and
check available credits, pending reservations, plan status and spend caps.
`client.pricing()` reads the current public rate card; it does not buy credits,
change a plan or authorize a payment. Billing changes require browser authority.
The current public rate card has a five-credit minimum for a priced fit,
prediction, batch prediction or discovery, and a 25-credit research download.
Verify the current offers in Billing or `client.pricing()` before budgeting;
request size and applicable accounting rules determine the charge.

```python
with Client() as client:
    events = client.usage()
for event in events:
    print(event["operation_id"], event["status"],
          event.get("credits"), event.get("rate_card_version"))
```

Usage contains lifecycle/attempt workload metadata and nullable accounting fields:

- `credits` is the operation's net credit charge, or its reservation until settled.
- `rate_card_version` identifies the accounting rules used for that operation.
- `credits: null` means unpriced; it is distinct from zero and does not promise
  that later operations will be free.

Several lifecycle events can share one `operation_id` and repeat its accounting
fields. Count distinct operations and do not sum credits across every event.
Use Workspace Billing for the workspace ledger and available balance. Usage
contains no prediction payloads and does not measure model accuracy.

HTTP 402 is exposed as `WeltError` with its `code`, including
`insufficient_credits`, `spend_cap_reached` or `plan_past_due`. Resolve the billing
condition before a deliberate retry; see [Wait and recover](async-and-errors.md#recover-from-the-cause).

Safe metadata GET requests have bounded automatic retries (default two,
maximum five), respecting Retry-After up to the configured wait bound; a larger
hint returns an actionable typed error rather than retrying early. POST mutations
never auto-retry. Reusing a Client submission requires
explicit `idempotency_key` preservation by the caller; omitted keys are freshly
generated per call. Estimator fit/submit_fit makes a new dataset upload and logical
fit, so call Client.submit_fit on the existing dataset for an intentional request replay.

The [capability boundary](capabilities.md), [onboarding guide](onboarding.md),
[executed notebook examples](notebooks.md) and generated installed-package reference
explain supported behavior. The source examples use real research data and hosted calls; Local release checks patch transport
only in disposable test copies with authorized pinned research files; public CI
validates source/contracts without hosted data or model-quality claims.
