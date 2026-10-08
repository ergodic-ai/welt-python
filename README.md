# Welt Python SDK

Python client for Welt's structured-data foundation-model API. Installable package
`welt-client`, import `welt`. This public repository owns SDK source and developer
documentation; the hosted service is maintained separately.

The incremental 0.6.0 client retains the released 0.3 interfaces and adds bounded
sync/async research catalogue search, inspection and verified file downloads.
It does not include the separate unreleased resumable CSV0.4 or batch0.5 candidates.
The client offers sync/async HTTP resources, durable job polling,
sklearn-shaped `Classifier`/`Regressor`, named DataFrames/arrays/CSV/Parquet input,
probabilities, predictor reopening/metadata, credential-safe serialization and
sanitized typed errors. Five examples use controlled synthetic transport, with separate bounded
live acceptance. Actual execution is subject to service catalogue capabilities,
rights and limits. TabICLv2, Kumo Medium, TabDPT1.3 and Mitra-v2 classification/regression tasks are separately qualified
for the narrow hosted preview; each additional task requires its own evidence.

ArrowFM observational discovery returns durable native DAGs and separate scores,
with explicit owned result/dataset lifecycle and typed graph conversion. Its
bounded hosted preview is qualified. CDFM adds native directed graphs with its
original adaptive threshold and preserved scores; unsupported conversion is typed.
AVICI adds original native directed scores with strict probability>0.5 decoding,
including cycles/reciprocals, within the same bounded observational envelope.
Its native inference key is fixed at0; request seeds remain provenance.
Local Ergodic continuation for declared DAGs uses the approved
private co-release artifact until a public distribution is available.

Full sklearn matrix qualification, resumable uploads, large batch jobs,
the remaining causal engines/public Ergodic co-release and the complete all-model notebook pack
remain later gates. File input currently loads into memory.
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
identities. Safe GET requests have bounded retries; POST mutations never auto-retry. Repeating an explicit
Client.submit_fit request requires the caller to reuse its idempotency_key; omitting
it generates a fresh key for each call. Estimator refit creates a new upload/job. Model availability and maximum
rows/features/classes are visible through `Client.models()`.

See [research datasets](docs/research-datasets.md), [SDK guide](docs/usage.md), [model selection](docs/model-selection.md) and
[causal discovery](docs/causal-discovery.md) and [contributor notes](CONTRIBUTING.md).

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

## Developer guides and Jupyter examples

See [onboarding](docs/onboarding.md), [capabilities](docs/capabilities.md) and
[notebook execution](docs/notebooks.md). Versioned static SDK docs are generated
from these guides and installed package signatures; broader feature/model gates
remain explicit. Notebook sources have cleared outputs and default to no live API
access. Run them from a built wheel via `scripts/check_notebooks.py`.
