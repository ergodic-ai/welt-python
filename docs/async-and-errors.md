# Async, retry and error behavior in SDK 0.2

Use `AsyncClient` separately from synchronous sklearn estimators. Resource methods
are awaited; `client.job(id)` is a regular factory yielding an `AsyncJob` whose
inspect/result/cancel methods are awaited. Use an async context manager or aclose().
The [async notebook](notebooks.md) executes upload, durable submission, reconnect
and predict against a controlled fixture; live mode is explicitly separate.

Local cancellation/timeout leaves the server job intact. JobTimeoutError includes
job_id for reconnect; explicit await job.cancel() requests remote cancellation.
Server-side prediction timeout likewise preserves job_id rather than silently
cancelling. A fit creates durable immutable state; cloned estimators never clone
remote fitted identity. Failed refit clears stale fitted identity.

Safe GET requests have bounded retries: default2, maximum5, respecting Retry-After
within the configured wait bound. A larger server hint returns a typed retryable
error instead of retrying before the advised time. POST mutations are never retried
automatically. Preserve an explicit idempotency key to replay the same intended fit
request on an existing dataset; omitting it creates a fresh key. Estimator refit
uploads again and creates a new logical fit.

Authentication/permission/not-found/conflict/rate/capacity/input/model/result-expiry/
execution/cancellation/transport errors have distinct types with available request,
job, status and retry metadata. Error bodies remain sanitized; they do not expose
raw rows or credentials. A result-expiry error concerns result access, not proof of
physical blob deletion. Unsupported routes/tasks are actionable errors, not fallback.

Explicit API keys are wrapped as redacted credentials, including nested Pipeline
repr/get_params/HTML. Clone keeps the in-memory credential so execution can continue;
pickle drops it and later execution reauthenticates from local environment. Metadata
export returns authorized pinned provenance, not model weights/context or credentials.
Keep keys out of saved notebook outputs regardless of these SDK safeguards.

Process-based parallel CV/joblib serializes estimators, so each worker must
reauthenticate from WELT_API_KEY; an explicit constructor key alone is intentionally
not serialized. Serial/in-memory cloning retains its redacted runtime credential.
