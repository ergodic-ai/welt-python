# SDK guide

Use the lower-level `Client` to upload a table, submit a fit and predict, then inspect
your workspace's jobs, predictors and usage. This page also explains how usage relates
to workspace credits.

## Fit and predict with Client

`Client()` connects to `https://welt.ergodic.dev` by default. Always close it, or use
it as a context manager. `Client.models()` reports actual capabilities, not blanket
task support.

This example continues with the real Iris `train` and test tables from
[Start](start.md):

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

<details markdown="1"><summary>Which endpoint and credential does Client use?</summary>

An explicit `base_url` overrides `WELT_BASE_URL`. Authenticated methods use your
connected or saved workspace credential, `WELT_API_KEY` or an explicit key. See
[Connect](connect.md) for precedence.

</details>

## Estimators and jobs

- `Classifier` and `Regressor` support `fit`, `predict` and `predict_details`.
  `Classifier` also supports `predict_proba`.
- Labels and probabilities follow the persisted class order.
- You may safely reorder DataFrame prediction columns. Missing or extra names, or
  width or type changes, are rejected. Arrays follow positional order.
- CSV and Parquet paths are convenient in-memory reads today, not resumable streaming
  upload.
- `submit_fit(X, y)` or `submit_fit(train, target=target)` returns a durable Job
  without blocking. The job owns an HTTP client, so close `job.client` in a `finally`
  block when finished. Blocking estimator fit closes that client internally.
- Explicit Client uploads permit dataset-ID reuse.

## Reconnect and handle errors

Inspect `job.id` or `job.inspect()`, reconnect with `client.job(id)` and retrieve the
result with `result()`. A polling deadline is local; it does not cancel durable server
execution. `job.cancel()` explicitly requests cancellation.

Typed errors distinguish invalid input, unavailable model, authentication, capacity
and execution failure. Their IDs allow operator diagnosis without exposing data
values. `PredictionPendingError` carries a `job_id`, so you can reconnect after a
server-side small-prediction timeout. [Wait and recover](async-and-errors.md) lists
the next action for each error.

## Predictors and workspace resources

- Predictors retain exact model, configuration and seed identity.
  `Classifier.from_predictor(id)` reopens a persisted fit.
- A new fit creates a new predictor. Cache eviction does not mean deletion.
- Current SDK resources include `datasets()`, `predictors()`, `jobs()` and `usage()`.
- Metadata inspection and `export_metadata()` return authorized, JSON-compatible
  server provenance only, never weights, context or credentials.
- Explicit deletion of datasets and causal results you own is available. Recipe and
  deletion-plan transport remain separate.

## Files and batch prediction

`Client` and `AsyncClient` also provide resumable upload sessions, `upload_file`,
`submit_batch_predict` and `batch_payload`. Keep the upload ID before sending bytes,
and inspect batch manifests for failed row ranges.

Enabled file formats, batch limits and model readiness depend on the connected
service. See [CLI recipes](cli.md) and [capabilities](capabilities.md) for those
boundaries. Native ArrowFM, CDFM and AVICI results retain their graph semantics;
public Ergodic continuation and the remaining engines have separate availability
requirements. See the [research catalogue](research-datasets.md) for tables to try.

## Usage and workspace credits

Modelling and research downloads may consume workspace credits. Before running a
walkthrough, open **Workspace → Billing** in [Welt](https://welt.ergodic.dev) and
check available credits, pending reservations, plan status and spend caps.

`client.pricing()` reads the current public rate card. It does not buy credits, change
a plan or authorize a payment; billing changes require browser authority. The current
public rate card has a five-credit minimum for a priced fit, prediction, batch
prediction or discovery, and a 25-credit research download.

!!! note "Check current prices before budgeting"
    Verify the current offers in Billing or `client.pricing()`. Request size and the
    applicable accounting rules determine the charge.

List your usage events:

```python
with Client() as client:
    events = client.usage()
for event in events:
    print(event["operation_id"], event["status"],
          event.get("credits"), event.get("rate_card_version"))
```

Usage contains lifecycle and attempt workload metadata, plus nullable accounting
fields:

- `credits` is the operation's net credit charge, or its reservation until settled.
- `rate_card_version` identifies the accounting rules used for that operation.
- `credits: null` means unpriced. It is distinct from zero and does not promise
  that later operations will be free.

!!! caution "Count operations, not events"
    Several lifecycle events can share one `operation_id` and repeat its accounting
    fields. Count distinct operations and do not sum credits across every event.

Use Workspace Billing for the workspace ledger and available balance. Usage contains
no prediction payloads and does not measure model accuracy.

HTTP 402 is exposed as `WeltError` with its `code`, including `insufficient_credits`,
`spend_cap_reached` or `plan_past_due`. Resolve the billing condition before a
deliberate retry; see [Wait and recover](async-and-errors.md#recover-from-the-cause).

## Retries and replays

- Safe metadata GET requests have bounded automatic retries: two by default, five at
  most. They respect Retry-After up to the configured wait bound. A larger hint
  returns an actionable typed error rather than retrying early.
- POST mutations never auto-retry.
- Reusing a Client submission requires you to preserve an explicit `idempotency_key`.
  Omitted keys are freshly generated per call.
- Estimator fit and `submit_fit` make a new dataset upload and logical fit. For an
  intentional request replay, call `Client.submit_fit` on the existing dataset.
- `submit_evaluation` with a table uploads it each time. To replay an evaluation, pass
  its dataset ID and the same `idempotency_key`; a dropped submission's error names both.

## Next steps

The [capability boundary](capabilities.md), [onboarding guide](onboarding.md),
[executed notebook examples](notebooks.md) and generated installed-package reference
explain supported behavior.

<details markdown="1"><summary>How are these examples checked?</summary>

The source examples use real research data and hosted calls. Local release checks
patch transport only in disposable test copies with authorized pinned research files.
Public CI validates source and contracts without hosted data or model-quality claims.

</details>

<div class="cards" markdown="1">

- [Wait and recover](async-and-errors.md): keep the job ID and act on each error.
- [Models and limits](capabilities.md): check readiness and limits before you submit.
- [Agent CLI](cli.md): the same resources from a terminal.

</div>
