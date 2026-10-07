# Welt Python SDK

Python client for Welt's structured-data foundation-model API. Installable package
`welt-client`, import `welt`. This public repository owns SDK source and developer
documentation; the hosted service is maintained separately.

The current 0.1.0 client offers synchronous HTTP resources, durable job polling,
sklearn-shaped `Classifier`/`Regressor`, named DataFrames/arrays/CSV/Parquet input,
probabilities, predictor reopening and sanitized typed errors. Actual execution is
subject to service catalogue capability/rights/limits. The hosted initial preview
qualifies narrow TabICLv2 classification only. A Regressor class does not imply a
qualified hosted regression model.

Full sklearn matrix qualification, async client, resumable uploads, large batch
jobs, causal discovery/ergodic integration and executed notebook documentation
are tracked for subsequent releases. File input currently loads into memory.
Private API state and model artifacts are never bundled with this SDK.

## Install and use

```sh
pip install .
# Optional Parquet reader:
pip install '.[parquet]'
```

Set `WELT_BASE_URL` to your service URL and `WELT_API_KEY` through your local secret
configuration. Do not commit credentials or serialize credential-bearing clients/
estimators. Signup/key onboarding depends on your deployed service's account release.

```python
from welt import Classifier

model = Classifier(model='tabicl-v2', random_state=9)
model.fit(X_train, y_train)   # uploads, submits durable fit, waits
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
reopened = Classifier.from_predictor(model.predictor_id_)
```

A local wait timeout leaves server work running. Reconnect using `Client.job(id)`;
`job.cancel()` requests explicit server cancellation. New intentional fits use new
identities. There are no automatic HTTP request retries today. Repeating an explicit
Client.submit_fit request requires the caller to reuse its idempotency_key; omitting
it generates a fresh key for each call. Estimator refit creates a new upload/job. Model availability and maximum
rows/features/classes are visible through `Client.models()`.

See [SDK guide](docs/usage.md) and [contributor notes](CONTRIBUTING.md).

## Checks

```sh
uv sync --frozen
uv run --frozen pytest -q
uv build
```

Tests use an independent in-memory HTTP contract fixture. They neither contact the
hosted service nor execute a foundation model, and make no scientific/latency claim.
CI does not require access to private backend source, credentials or checkpoints.

For explicit estimator `submit_fit(X, y)`, close `job.client` in a finally block
after waiting/submitting; blocking `fit` closes that internal client automatically.

## License

Licensed under Apache 2.0. See [LICENSE](LICENSE). Model weights and the hosted
service are separate and are not licensed by this SDK repository.
