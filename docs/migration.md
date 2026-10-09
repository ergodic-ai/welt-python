# Move from SDK 0.6 to 0.7

SDK 0.7 keeps the released supervised, research and causal interfaces and adds a
simpler hosted setup plus explicit target-column convenience.

## Select the endpoint explicitly when local

Precedence is constructor `base_url` → `WELT_BASE_URL` →
`https://welt.ergodic.dev`. The default changed from localhost to the hosted origin.
If you develop against a local server, select it deliberately:

```python
from welt import Client

with Client(base_url="http://localhost:8000") as client:
    models = client.models()
```

Public model-catalogue reads remain keyless. Authenticated methods validate missing
credentials before issuing requests and explain how to create a workspace key.
Invalid/expired credentials and insufficient permissions remain typed server
failures. Keys are never saved implicitly.

## Use a DataFrame target

Existing `fit(X, y)` remains valid. The additive form accepts exactly one target
source, requires a directly supplied DataFrame, and removes its named target from
features before upload:

```python
# Continue with the real train table and target name from Start.
model.fit(train, target=target)
```

Select `Classifier` or `Regressor` explicitly. Conflicting `y`/`target`, missing or
duplicate target names and invalid schema reject locally before upload. Query rows
still contain features only. Existing fitted predictor identities remain pinned;
upgrading the client does not refit or change a remote model.

## Reconnect after waiting

`Job.result(timeout=...)` budgets initial inspection, polling, GET retry waits and
final result retrieval. It never cancels or resubmits work. Async requests are
bounded using the remaining wait budget. Synchronous HTTP phases receive the
remaining budget and the deadline is checked after each request; separate network
phases and custom blocking transports do not provide a hard wall-clock interrupt.
Use the preserved `job_id` with the same endpoint/current workspace credentials.
See [Wait and recover](async-and-errors.md).

## Keep the release boundary

0.7 excludes the held resumable CSV and large batch candidates. CSV/Parquet
convenience reads still load into memory. Full all-model qualification, public
Ergodic distribution and calibrated/conformal confidence retain their own gates.
The [0.6 documentation](../v0.6.0/index.html) remains available with its original
contract and pinned notebook sources.
