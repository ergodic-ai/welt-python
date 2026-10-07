# Causal discovery

SDK0.3.0 supports the independently accepted ArrowFM observational preview:
2–500 rows and2–20 finite continuous variables, original native DAG decoder and
separate scores. The service catalogue reports exact task/version/readiness.
Other causal engines and a usable public Ergodic co-release remain gated.
No automatic Ergodic/model download is performed by these interfaces.

`Client` and `AsyncClient` provide `submit_discover(dataset_id, model='arrow',
configuration='default', seed=0, constraints=None, model_version=None,
idempotency_key=None)` and blocking/awaitable `discover(..., timeout=600)`.
Submission returns the existing durable Job/AsyncJob handle. A successful discover
job returns `CausalResult`; successful supervised jobs retain their current return
types. Timeout does not cancel server work; reconnect with client.job(job_id).

Use target-free observational datasets: discovery preserves every uploaded column
in original order. It rejects target-labelled datasets instead of dropping their
target. Nonempty constraints and unsupported input reject before model work.
Capability/readiness checks come from the current /v1/models task profile, not a
client-side list or a model being present in the catalogue.

```python
from welt import Client

# X is the user's finite observational DataFrame. Use this only after the selected
# causal model/task is enabled and its published input bounds are satisfied.
with Client() as client:
    profile = next(m for m in client.models() if m["id"] == "arrow")
    ready = any(p["task"] == "causal_discovery" and p["execution_available"]
                for p in profile["task_profiles"])
    if ready:
        dataset = client.upload(columns=list(X.columns), rows=X.values.tolist(), target=None)
        job = client.submit_discover(dataset["id"], model="arrow", seed=42)
        result = job.result()
        scores = client.causal_scores(job.id)
        reopened = client.causal_result(job.id)
```

`CausalDiscovery` has sklearn-style constructor parameters and cloning/get_params;
its modelling operation is discover, not supervised fit/predict. `discover(X)` and
`submit_discover(X)` accept DataFrame/2D arrays/CSV/Parquet via the existing table
converter, or keyword dataset_id without upload. Supply exactly one table or
reference. Arrays receive x0,x1,... names; named columns retain their original
order. Blocking discover closes its temporary HTTP client. A submitted job owns
its client until the caller finishes/reconnects and closes job.client explicitly.

`CausalResult` preserves immutable ordered variables, marks, native graph type,
model/configuration and opaque score reference; to_dict returns a fresh ordinary
wire dictionary. Immutable job_id/diagnostics/configuration_version/assumptions/
decoder accessors expose original provenance. Blocking convenience calls return
result.job_id, so a new client can fetch scores or reopen after the facade closes
its HTTP client. Client reopening verifies this owned job identity. Use explicit
client.causal_result(job_id)/causal_scores(job_id)
for authenticated reconnecting/score access, so the result carries no hidden
client or credentials. Score access checks result identity/reference/original
axes and finite native probability semantics; scores never threshold/rewrite the
graph. Typed malformed-result errors apply without repair.

`result.to_ergodic()` converts a declared native DAG locally, retains every isolate
and native tail/arrow direction. It refuses every graph not explicitly declared a
native DAG, including acyclic directed graphs and cycles, without projection.
The actual compatible Ergodic0.1.1 co-release wheel has been
checked privately. Its repository is private and the public package-index name
currently refers to another project; there is no public causal extra yet. Install
only the approved co-release artifact when available. Missing/unrelated dependency
has typed OptionalDependencyError before importing an unrelated package. Welt
never downloads a causal dependency during conversion.

MixedGraph stores node sets: inspect adjacency with
`graph.to_adjacency(order=list(result.variables))` to retain original axis order.
Conversion and local identification/effect estimation are separate scientific
steps; a model's discovered edges do not establish causal truth. Native edge
scores are neither calibrated confidence nor graph posterior probabilities.
Primary graph/score retention/deletion and result quotas are server contracts,
independent of seven-day downloadable prediction payloads.

## Explicit owned lifecycle

`client.delete_causal_result(result.id)` explicitly deletes the retained primary
graph and scores. `client.delete_dataset(result.dataset_id)` separately deletes
an upload only if it has no retained predictor/result or queued/running job
dependencies. AsyncClient exposes the same methods with `await`. There is no
implicit deletion, cascade or expiry when a client closes or discovery times out.
These explicit server routes are deployed; dependency/race/tenant contracts were
checked against temporary local state. Live acceptance retains its synthetic
results and does not delete hosted state.

A successful deletion returns `None`; repeat deletion of the same owned resource
is safe. A foreign or missing identity raises `NotFoundError`. Dataset dependencies
raise `ConflictError` with code `dependency_conflict`. Explicitly deleted primary
result/score reads raise `ResultDeletedError`, while job/provenance/usage metadata
remain. If payload removal returns retryable `deletion_pending`, retry the same
DELETE; quota and dataset dependency remain reserved until it completes.
