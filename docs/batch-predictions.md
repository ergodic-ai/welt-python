# Durable batch predictions

SDK 0.5 source candidate under API-049/050. Batch is disabled in the current
service, and no scientific task has qualified its chunk protocol. The examples
below describe the reviewed interface; publication and live execution remain held
for task, SDK, database and resource acceptance.

A batch predicts an ordered, target-free query dataset using an existing fitted
predictor. The predictor keeps its training dataset, model version, configuration,
seed and class order. Query columns must match its fitted schema; batch does not
sample rows, drop columns or upgrade the predictor.

```python
from welt import Client

with Client() as client:
    job = client.submit_batch(
        query_dataset_id,
        predictor_id=predictor_id,
        probabilities=True,  # Classification only.
        idempotency_key=stable_key,
    )
    job_id = job.id
    result = job.result(timeout=600)
    print(result.status, result.successful_ranges, result.failed_ranges)
```

`Client.batch()` combines submission and waiting. `submit_batch()` returns the
durable job immediately. Keep its ID to reconnect with `client.job(job_id)` or
`client.batch_result(job_id)`. Reuse an idempotency key only with the same predictor,
query dataset and probability request; replay returns the original job.

Waiting returns an immutable `BatchResult` manifest for every terminal batch:
`succeeded`, `partially_completed`, `failed` or `cancelled`. It does not download
prediction payloads. Half-open ranges `[start, stop)` refer to positions in the
original query table. Successful and failed ranges cover it exactly. A cancelled
job can retain all successful ranges if cancellation won after the final range
commit; inspect both status and ranges.

Retrieve numerical output explicitly using the owned job and manifest reference:

```python
with Client() as client:
    result = client.batch_result(job_id)
    payload = client.batch_payload(
        job_id, result_reference=result.result_reference
    )
    predictions = payload.predictions
    probabilities = payload.probabilities
    errors = payload.errors
```

`BatchPayload` preserves query row order. Failed positions are `None`, including
their probability vectors. Classification vectors use `payload.classes` order;
regression has no classes or probabilities. Sanitized errors identify exact failed
ranges and outcomes. Both result objects expose immutable nested values;
`to_dict()` returns a detached JSON-shaped copy. No result object stores a hidden
client or fetches an arbitrary download URL.

Payload access expires seven days after terminal publication with typed
`ResultExpiredError`; the manifest, job and usage remain readable. This is logical
HTTP expiry, not physical object cleanup. A completed query dataset can be deleted
explicitly without removing the retained batch payload; active jobs block dataset
deletion. No example performs hosted deletion.

A local wait timeout carries `job_id` and leaves the server job running. Async
wait cancellation also carries the acknowledged job ID and stops only the client
wait. Remote cancellation is explicit:

```python
with Client() as client:
    client.job(job_id).cancel()
    result = client.job(job_id).result()
```

Cancellation prevents subsequent ranges and late publication; it does not promise
preemption of a native call already running. Committed successes remain available
through the terminal manifest and payload until expiry.

```python
from welt import AsyncClient

async with AsyncClient() as client:
    job = await client.submit_batch(query_dataset_id, predictor_id=predictor_id)
    result = await job.result(timeout=600)
    payload = await client.batch_payload(job.id)
```

The proposed preview admits up to 10,000 rows within existing dataset and output
budgets. Each serial call uses at most 100 rows and the pinned task/direct limits.
Chunking must qualify separately for each task; it does not imply equivalence to
one large native call. At most two whole-job claims and two consumed reservations
per range apply, with a 600-second claim budget. Only explicitly retryable
allowlisted infrastructure failures retry. Native schema/version/scientific
failures are terminal. Whole-table admission and payload serialization share the
API process capacity slot; busy responses require retry guidance. PostgreSQL
concurrency, canonical usage, native chunk and hosted capacity evidence remain
separate release gates.
