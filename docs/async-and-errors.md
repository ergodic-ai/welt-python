# Wait and recover

A durable job is server work with an identity. A local wait is how long this Python
call chooses to observe it. Stopping the wait leaves the job running.

## Submit and keep the job ID

For an estimator, submit instead of blocking when you want the identity immediately.
This template uses the real Iris `train` DataFrame and `target` name from [Start](start.md).

```python
from welt import Classifier, JobTimeoutError

model = Classifier(model="tabicl-v2", random_state=9)
job = model.submit_fit(train, target=target)
print("Keep this job ID:", job.id)
try:
    predictor = job.result(timeout=120)
except JobTimeoutError as error:
    print("Wait ended; reconnect with:", error.job_id)
finally:
    job.client.close()
```

`submit_fit` uploads and creates a new logical fit. The job owns an HTTP client,
which callers close; blocking estimator `fit` closes its temporary client itself.
Closing the client does not cancel or delete anything. Server stages come from
`job.inspect()`; the SDK does not invent progress percentages.

## Reconnect without another fit

Replace the ID with your recorded job and use the same endpoint/current workspace
credentials. Retrieval below makes no training-table upload or fit submission.

```python
from welt import Client, Classifier

with Client() as client:
    job = client.job("your-recorded-job-id")
    predictor = job.result(timeout=600)
model = Classifier.from_predictor(predictor["id"])
# model.predict(query) now reuses this completed classification fit.
```

A polling budget includes initial inspection, retry waits and final result access.
Async awaited requests are bounded with the remaining wait budget. Synchronous
HTTP phases receive the remaining budget and deadlines are checked after requests;
custom blocking sync transports and multiple network phases are not a hard
wall-clock interrupt. This is a local waiting limit, not a GPU latency guarantee.
`PredictionPendingError` likewise preserves a prediction `job_id`; estimator
prediction conveniences wait for it without silently creating another prediction.

## Use async resources

Async requests and durable background jobs are distinct. `client.job(id)` is a
regular factory; resource methods and `AsyncJob.result()` are awaited.

```python
from welt import AsyncClient

async def resume_fit(job_id):
    async with AsyncClient() as client:
        job = client.job(job_id)
        return await job.result(timeout=600)
```

Local task cancellation/timeout leaves the server job intact. Explicit
`await job.cancel()` or synchronous `job.cancel()` requests remote cancellation.
A completion/cancellation race follows the current server job state.

## Recover from the cause

| Error / condition | Next action |
| --- | --- |
| Missing `WELT_API_KEY` | Open Welt → Account; create a workspace key and configure it locally. No authenticated request was sent. |
| `AuthenticationError`: invalid, expired or revoked key | Replace the key through Account; avoid printing the secret. These conditions require current server authorization. |
| `PermissionDeniedError` | Use a key with **Allow writes, fits and predictions** enabled in the intended workspace. |
| `ValueError` / `InvalidInputError` | Check unique feature names, exactly one explicit target, matching query schema and task limits. |
| `ModelUnavailableError` | Inspect the requested `task_profiles` entry and worker readiness. There is no hidden model or task fallback. |
| `CapacityError` / `RateLimitError` | Respect `retry_after`; inspect queued work before submitting again. |
| `JobTimeoutError` / pending prediction | Keep `job_id` and reconnect. Do not blindly refit. |
| `ExecutionError` / `JobCancelledError` | Inspect the durable state and safe request/job IDs; decide explicitly whether a new attempt is appropriate. |
| `TransportError` | Inspect existing work before replaying a mutation; use safe IDs for support. |

Typed errors expose available `code`, `request_id`, `job_id`, `status_code`,
`retryable` and `retry_after`. Bodies are sanitized; never add keys or raw rows to
logs. A temporary `ResultExpiredError` concerns access, not proof of physical blob
deletion. Explicitly deleted native causal reads use `ResultDeletedError`.

## Retries and repeated submissions

Safe GET requests use bounded retries (default two, maximum five), respecting
`Retry-After` within the configured wait bound. A larger hint returns a typed
retryable error rather than retrying early. POST mutations never auto-retry.
For an intentional replay of the same Client fit request, reuse the existing
dataset ID and preserve an explicit `idempotency_key`; omitted keys are freshly
generated. Calling estimator `fit` again uploads again and creates a new fit.

## Credential-safe estimators

Explicit keys are redacted in estimator repr, nested Pipeline HTML and parameters.
In-memory cloning retains the redacted runtime credential; pickle drops it and
later execution reauthenticates from `WELT_API_KEY`. Process-based parallel CV
workers need that environment configuration. Failed refit clears stale fitted
identity. These safeguards do not make saving secrets in notebook cells safe.
