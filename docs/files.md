# Upload and resume a CSV file

SDK 0.4 source candidate under API-049/050. Publication is held until the corresponding
API/SDK slice and bounded live checks pass independent review.

`Client.upload_file()` sends a file in bounded binary chunks and completes it as
one immutable reusable dataset. It does not fit a model. Choose the returned
dataset ID for an explicit fit or discovery job, subject to that model's limits.

```python
from welt import Client

schema = {
    "columns": ["customer_code", "amount", "flag", "target"],
    "types": ["string", "number", "boolean", "number"],
}
with Client() as client:
    dataset = client.upload_file("training.csv", target="target", schema=schema)
```

The file header must match the schema names and order exactly. Explicit strings
preserve lexical identifiers such as `0012`. Numbers use finite decimal notation;
booleans use lowercase `true` and `false`. An empty field is null. Without a
schema, the server infers each complete column as numeric when every nonempty
field is a valid finite decimal; otherwise it preserves the column as strings.
Numeric inference does not preserve the original numeric spelling. Duplicate or
empty headers, malformed UTF-8/CSV and unequal row widths reject explicitly.

When an accepted upload is interrupted, the error carries an opaque `upload_id` after creation has been acknowledged.
A failure before that acknowledgement has no safe resumable identity.
Keep that ID and the unchanged source file and resume using current credentials:

```python
with Client() as client:
    dataset = client.resume_upload(upload_id, "training.csv",
                                   target="target", schema=schema)
```

Resume verifies the file's size, SHA-256, format, target and schema against the
original declaration before skipping accepted chunks. A changed file is refused.
Identical chunk replay is safe; changed bytes at an accepted index conflict. Only
successful complete-file digest and schema validation publishes the dataset.
Replaying completion returns the same dataset, and cannot recreate a deleted one.

Preview defaults are 1 MiB/chunk, 64 MiB/file, four active uploads/workspace and
24 hours to complete an upload, with at most 4096 chunks. Each CSV field is bounded
to 131072 characters. Each header/data record also has a conservative 10 MiB
budget counted as JSON-escaped lexical field text before parsing. This includes
Unicode/control-character expansion; long numeric spellings can reject even when
the normalized value would be small. The existing normalized dataset size, row/column
allowances and individual model envelopes still apply. Raw CSV size does not
expand a model's admitted context. Only one completion is admitted per API process by default across workspaces. A
busy completion returns retryable `upload_busy` (429) with `Retry-After: 30`;
respect the retry guidance and resume the same acknowledged upload ID rather than
creating another upload.

Logical upload expiry is distinct from physical
storage cleanup. This slice performs no hosted deletion.

`AsyncClient` exposes the same upload methods with `await`; local cancellation
stops the client operation without removing the accepted upload. Reconnect with
its ID. Explicit file transport is separate from estimator file convenience
inputs, which currently read through pandas and submit JSON. Resumable Parquet
and batch prediction remain separate acceptance gates.
