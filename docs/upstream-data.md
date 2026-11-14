# Upstream data

SDK 0.16.4 provides `client.upstream` for deployments that explicitly expose
qualified upstream datasets. This client release does not itself enable the server
feature or qualify any source. Dataset policies and capabilities are authoritative;
being listed does not mean a source is available for retrieval.

```python
from welt import Client

client = Client()  # Uses your existing Welt connection; no provider key in code.
catalogue = client.upstream.search(source="world_bank")
description = client.upstream.describe("world_bank:SP.POP.TOTL")

# Run only where this dataset's retrieval/export/retention policies are qualified.
result = client.upstream.fetch(
    description["id"],
    selections=[
        {"column": "economy", "operator": "eq", "values": ["USA"]},
        {"column": "year", "operator": "eq", "values": ["2020"]},
    ],
    idempotency_key="population-usa-2020-v1",
)
frame = result.to_pandas()
print(result.receipt, result.provenance)
result.save("population-result.json")  # Refuses an existing destination.
```

`fetch` describes the dataset, submits a durable retrieval, waits, and downloads a
complete JSON result. It validates the result's content and snapshot hashes against
metadata; it never treats a preview as a full dataset. Decimal columns become exact
Python `Decimal` objects in `to_pandas()`, and native period/date labels remain
strings. The frame's `attrs["welt_upstream"]` retains schema and provenance. Numeric
conversion or period-to-date conversion is an explicit caller choice.

Use `submit(...)` and `wait(retrieval_id)` separately to reconnect after interruption.
The wait timeout stops polling, not the server job; use `cancel(retrieval_id)` to
cancel publication. The `timeout` argument bounds polling; describe/submit/download
use the client's configured HTTP transport timeouts. Reuse an explicit idempotency
key to retry an uncertain submission. `preview(result_id, limit=20)` is explicitly
bounded; `result(result_id)` and `download(result_id, path)` retrieve complete data,
up to a configurable ceiling of 10 MiB, and refuse corruption or partial transfer.
Downloads do not follow redirects. Export requires original and current source rights.

`AsyncClient().upstream` provides the same operations with `await` (including
`fetch`, `wait`, `result`, and `download`). Result conversion with `to_pandas()` stays
local and synchronous. Source credentials are not accepted as query arguments;
account-owned source connections and production activation remain service gates.

The namespace also provides `client.upstream.materialize(result.id, name="Population",
decimal_encoding="float64", target="value", idempotency_key="population-copy-v1")`
to save an ordinary workspace dataset. Use `decimal_encoding="string"` to preserve
exact decimals as categorical text; `float64` explicitly permits numeric rounding.
The dataset description must record `materialization_evidence` qualifying ongoing
retention/modelling, and export/retention rights must permit the copy. No target is
inferred. The saved dataset keeps source provenance and both content hashes; source
result deletion/revocation does not delete the independent copy. It follows ordinary
dataset deletion, and a retry cannot recreate a deleted copy. Async clients expose
the same method.

Submission reads the current dataset description to apply its lower retrieval
ceilings. Normal client defaults are never increased automatically. Partial explicit
`limits` override those defaults; values above the dataset ceiling are refused by
the service. A supplied `schema_revision` remains authoritative and is never
silently replaced by the current description.

The namespace also supports account-owned source connections through
`client.upstream.connections()`, `client.upstream.connect("fred", api_key=source_key)`
and `client.upstream.revoke_connection("fred")`; use `"census"` for Census or `"open_meteo"` for a qualified Open-Meteo commercial account; use `"eia"` for an EIA API key or `"comtrade"` for a UN Comtrade subscription key.
A Comtrade key does not establish commercial redistribution permission.
Historical weather requires an appropriate archive-enabled subscription. A stored
key does not qualify source rights or enable a pending dataset.
The async client exposes the same methods with `await`. Load `source_key` from your
secret store or a local prompt; the SDK does not infer provider keys from environment
variables, query arguments or the Welt API key. SDK `Credential` wrappers are accepted.
Only connection metadata is returned. The service must have its encrypted vault
configured, and your Welt key must represent an account with write access.

Repeating the same active key is a no-op. Changing it rotates the version; revoking
or rotating prevents queued retrievals from using the old version. Reconnecting a
revoked connection creates a new version. Each connection belongs to your account
within the current workspace, not to every workspace member. Existing completed
snapshots remain subject to their original and current source-access policies. A connection does
not activate an unqualified dataset or establish upstream usage rights. Server access and source qualification remain required.


Dataset descriptions also expose `filter_guidance` when supported by the server.
Use `required` and `requires_bounds` to build selections, and display native value/label
choices without substituting labels for codes. `choices_complete` describes the
supported profile domain, not guaranteed observations. Examples are incomplete;
World Bank reference-data links identify ordinary datasets to retrieve explicitly.

## Result retention and explicit deletion

On servers implementing persistent upstream retention, results remain until explicitly
deleted. New receipts have null `expires_at`; old receipt timestamps remain unchanged
for reproducibility and do not impose an access cutoff. Preview lifecycle metadata
reports `retention="until_deleted"` and null expiry. Retention does not make a snapshot
current or override source rights and revocation. SDK0.16.4 reads both receipt forms.

SDK 0.16.5 adds `client.upstream.delete(result.id)` and the corresponding
async method. Use it only when you intend to delete that specific server result.
Independent materialized datasets and previously downloaded local files remain.
A `result_delete_incomplete` error means result access was removed but storage cleanup
failed: retry deletion of the same result ID. Repeated deletion does not restore data.
SDK0.16.4 can invoke the endpoint using
`client.request("DELETE", f"/v1/data/results/{result.id}")`.
This endpoint requires an enabled server supporting explicit upstream-result deletion;
the client method does not enable it or qualify source rights.
