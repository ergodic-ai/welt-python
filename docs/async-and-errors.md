# Wait and recover

A durable job is server work with its own ID. Your local wait only decides how long
this Python call watches it; stopping the wait leaves the job running.

## Submit and keep the job ID

With an estimator, call `submit_fit` instead of blocking when you want the job ID right
away. This template uses the real Iris `train` DataFrame and `target` name from
[Start](start.md).

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

- `submit_fit` uploads your table and creates a new logical fit.
- The job owns an HTTP client, which you close, as the `finally` block does. Blocking
  estimator `fit` closes its temporary client itself.
- Closing the client does not cancel or delete anything.
- Server stages come from `job.inspect()`. The SDK does not invent progress percentages.

## Reconnect without another fit

Replace the ID with the job ID you recorded. Use the same endpoint and your current
workspace credentials. This retrieval uploads no training table and submits no fit.

```python
from welt import Client, Classifier

client = Client()
job = client.job("your-recorded-job-id")
predictor = job.result(timeout=600)
model = Classifier.from_predictor(predictor["id"])
# model.predict(query) now reuses this completed classification fit.
```

The `timeout` is a local waiting limit, not a GPU latency guarantee.
`PredictionPendingError` likewise preserves a prediction `job_id`. Estimator
prediction conveniences wait for it without silently creating another prediction.

<details markdown="1"><summary>What exactly does the timeout cover?</summary>

A polling budget includes initial inspection, retry waits and final result access.
Async awaited requests are bounded by the remaining wait budget. Synchronous HTTP
phases receive the remaining budget, and deadlines are checked after requests. Custom
blocking sync transports and multiple network phases are not a hard
wall-clock interrupt.

</details>

## Use async resources

Async requests and durable background jobs are different things. `client.job(id)` is
a regular factory; you await resource methods and `AsyncJob.result()`.

```python
from welt import AsyncClient

async def resume_fit(job_id):
    async with AsyncClient() as client:
        job = client.job(job_id)
        return await job.result(timeout=600)
```

Cancelling or timing out a local task leaves the server job intact. To request remote
cancellation, call `await job.cancel()` or, in synchronous code, `job.cancel()`. If
completion and cancellation race, the result follows the current server job state.

## Recover from the cause

Find the error you received, then take the next action.

| Error or condition | Next action |
| --- | --- |
| Missing `WELT_API_KEY` | Open Welt → Account; create a workspace key and configure it locally. No authenticated request was sent. |
| `AuthenticationError`: invalid, expired or revoked key | Replace the key through Account; avoid printing the secret. These conditions require current server authorization. |
| `PermissionDeniedError` | Use a key with **Allow writes, fits and predictions** enabled in the intended workspace. |
| `WeltError`, HTTP 402, `code="insufficient_credits"` | Open **Workspace → Billing**. Check available credits and pending reservations; wait for accepted work to settle or have an authorized person add credits. |
| `WeltError`, HTTP 402, `code="spend_cap_reached"` | Ask a workspace billing administrator to review the monthly spend cap. Retry after the cap is raised or the calendar month changes. |
| `WeltError`, HTTP 402, `code="plan_past_due"` | An authorized person must resolve the unpaid plan through Workspace Billing and its customer portal. Retry after the workspace's billing status recovers. |
| `ValueError` or `InvalidInputError` | Check unique feature names, exactly one explicit target, matching query schema and task limits. |
| `ModelUnavailableError` | Inspect the requested `task_profiles` entry and worker readiness. There is no hidden model or task fallback. |
| `CapacityError` or `RateLimitError` | Respect `retry_after`; inspect queued work before submitting again. |
| `JobTimeoutError` or pending prediction | Keep `job_id` and reconnect. Do not blindly refit. |
| `ExecutionError` or `JobCancelledError` | Inspect the durable state and safe request and job IDs; decide explicitly whether a new attempt is appropriate. |
| `TransportError` | Inspect existing work before replaying a mutation; use safe IDs for support. |

Typed errors expose `code`, `request_id`, `job_id`, `status_code`, `retryable` and
`retry_after` when available. Bodies are sanitized; never add keys or raw rows to logs.

!!! note "Billing errors use the base WeltError"
    The three billing conditions have no dedicated payment exception: check
    `status_code == 402` and `code`. Repeating the same request does not repair a
    balance, spend cap or unpaid plan, and the SDK does not purchase credits or change
    billing settings. See [usage accounting](usage.md#usage-and-workspace-credits).

<details markdown="1"><summary>Does an expired or deleted result mean the data is gone?</summary>

A temporary `ResultExpiredError` concerns access, not proof of physical blob deletion.
Reads of explicitly deleted native causal results raise `ResultDeletedError`.

</details>

## Retries and repeated submissions

- Safe GET requests use bounded retries: two by default, five at most. They respect
  `Retry-After` within the configured wait bound. A larger hint returns a typed
  retryable error rather than retrying early.
- POST mutations never auto-retry.
- To replay the same Client fit request on purpose, reuse the existing dataset ID and
  preserve an explicit `idempotency_key`. Omitted keys are freshly generated.
- Calling estimator `fit` again uploads again and creates a new fit.

## Credential-safe estimators

- Explicit keys are redacted in estimator repr, nested Pipeline HTML and parameters.
- In-memory cloning keeps the redacted runtime credential. Pickle drops it, and later
  execution reauthenticates from `WELT_API_KEY`. Process-based parallel CV workers
  need that environment configuration.
- A failed refit clears stale fitted identity.

!!! caution "Keep secrets out of notebook cells"
    These safeguards do not make saving secrets in notebook cells safe.

## Next steps

<div class="cards" markdown="1">

- [Reuse a predictor](reuse.md): reopen a fitted predictor without another fit.
- [Workspace resources](usage.md): inspect jobs, usage and workspace credits.
- [Agent CLI](cli.md): wait on and recover jobs from a terminal.

</div>
