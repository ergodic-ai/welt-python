# Causal discovery

Discover a causal graph from an observational table with a qualified model, then
inspect its native edges and scores.

!!! need "You'll need"
    - A target-free observational table. The example uses the Iris measurements you
      downloaded in [Your first prediction](start.md) (`data` and `target`).
    - A connected Welt workspace. See [Connect your workspace](connect.md).

## Choose a model

SDK 0.7 retains the independently accepted ArrowFM observational preview. CDFM and
AVICI also qualify the same input envelope: 2–500 rows and 2–20 finite continuous
variables. The service catalogue reports the exact task, version and readiness.

| Model | Select with | Decoding | Native output |
| --- | --- | --- | --- |
| ArrowFM | `model="arrow"` | Original native DAG decoder, with separate scores. | Native DAG. |
| CDFM | `model="cdfm"` | Original native adaptive threshold, with separate native scores. | Directed output, including possible cycles and reciprocal edges. |
| AVICI | `model="avici"` | Preserves full observations and the native population z-score; strict probability > 0.5 decoding. | Directed output, including possible cycles and reciprocal edges. |

You select every model through the same client interfaces.

- CDFM does not relabel an acyclic directed result as a declared DAG.
- AVICI's native PRNG key stays 0; request seeds are retained as provenance. It also
  refuses every non-declared-DAG Ergodic conversion.

!!! caution "Scores are not causal confidence"
    Native scores are not calibrated causal confidence or effects.

Other causal engines and a usable public Ergodic co-release remain gated. These
interfaces perform no automatic Ergodic or model download.

## Prepare observational data

Use target-free observational datasets. Discovery preserves every uploaded column in
its original order. It rejects target-labelled datasets instead of dropping their
target. Nonempty constraints and unsupported input are rejected before model work.

The [causal notebook](notebooks.md) downloads and inspects the real data first.

!!! caution "Iris has no ground-truth causal graph"
    Inspect native edges and assumptions, without pretending class labels evaluate a
    graph or establish causal truth.

## Run discovery

`Client` and `AsyncClient` provide `submit_discover(dataset_id, model='arrow',
configuration='default', seed=0, constraints=None, model_version=None,
idempotency_key=None)`, plus a blocking (or awaitable) `discover(..., timeout=600)`.

Check readiness first. Capability and readiness checks come from the current
`/v1/models` task profile, not a client-side list or a model being present in the
catalogue. The example drops the encoded class column, checks ArrowFM's
`causal_discovery` profile, uploads the four measurements and runs discovery.

```python
from welt import Client

# Continue after downloading/inspecting the real Iris data in Start.
# Exclude the encoded classification target; keep all four measurements/150 rows.
X = data.drop(columns=target)
print(X.head())
print(X.shape)
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

Submission returns the existing durable `Job` or `AsyncJob` handle. A successful
discover job returns a `CausalResult`; successful supervised jobs keep their current
return types.

!!! tip "A timeout doesn't stop the job"
    Timeout does not cancel server work. Reconnect with `client.job(job_id)`.

## Use the estimator interface

`CausalDiscovery` has sklearn-style constructor parameters, cloning and `get_params`.
Its modelling operation is discover, not supervised fit and predict.

- `discover(X)` and `submit_discover(X)` accept a DataFrame, a 2D array, CSV or
  Parquet through the existing table converter. Or pass the keyword `dataset_id`
  without uploading. Supply exactly one table or reference.
- Arrays receive `x0`, `x1`, … names. Named columns keep their original order.
- Blocking `discover` closes its temporary HTTP client. A submitted job owns its
  client until you finish or reconnect and close `job.client` explicitly.

## Read the result

`CausalResult` preserves immutable ordered variables, marks, the native graph type,
model and configuration, and an opaque score reference. `to_dict` returns a fresh
ordinary wire dictionary. The immutable `job_id`, `diagnostics`,
`configuration_version`, `assumptions` and `decoder` accessors expose the original
provenance.

Blocking convenience calls return a result with `result.job_id`, so a new client can
fetch scores or reopen it after the facade closes its HTTP client. Client reopening
verifies this owned job identity. Use `client.causal_result(job_id)` and
`client.causal_scores(job_id)` explicitly for authenticated reconnecting and score
access. That way the result carries no hidden client or credentials.

Score access checks the result identity, reference, original axes and finite native
probability semantics. Scores never threshold or rewrite the graph. Typed
malformed-result errors apply without repair.

## Convert to an Ergodic graph

`result.to_ergodic()` converts a declared native DAG locally. It keeps every isolate
and the native tail and arrow directions. It refuses every graph not explicitly
declared a native DAG, including acyclic directed graphs and cycles, without
projection.

!!! caution "Install only the approved Ergodic artifact"
    Its repository is private and the public package-index name currently refers to
    another project; there is no public causal extra yet. Install only the approved
    co-release artifact when available.

<details markdown="1"><summary>How is the Ergodic dependency checked?</summary>

The actual compatible Ergodic 0.1.1 co-release wheel has been checked privately. A
missing or unrelated dependency raises a typed `OptionalDependencyError` before an
unrelated package is imported. Welt never downloads a causal dependency during
conversion.

</details>

`MixedGraph` stores node sets. To keep the original axis order, inspect adjacency with
`graph.to_adjacency(order=list(result.variables))`.

!!! caution "Discovered edges are not causal truth"
    Conversion and local identification or effect estimation are separate scientific
    steps; a model's discovered edges do not establish causal truth. Native edge scores
    are neither calibrated confidence nor graph posterior probabilities.

## Delete results and datasets {#explicit-owned-lifecycle}

Retention and deletion of the primary graph and scores, and result quotas, are server
contracts. They are independent of seven-day downloadable prediction payloads.
Two explicit calls delete owned state:

- `client.delete_causal_result(result.id)` explicitly deletes the retained primary
  graph and scores.
- `client.delete_dataset(result.dataset_id)` separately deletes an upload, only if it
  has no retained predictor or result, and no queued or running job dependencies.

`AsyncClient` exposes the same methods with `await`. There is no implicit deletion,
cascade or expiry when a client closes or discovery times out.

| Situation | Result |
| --- | --- |
| Successful deletion | Returns `None`. Repeat deletion of the same owned resource is safe. |
| Foreign or missing identity | Raises `NotFoundError`. |
| Dataset still has dependencies | Raises `ConflictError` with code `dependency_conflict`. |
| Reading an explicitly deleted primary result or scores | Raises `ResultDeletedError`. Job, provenance and usage metadata remain. |
| Payload removal returns retryable `deletion_pending` | Retry the same DELETE. Quota and dataset dependency remain reserved until it completes. |

<details markdown="1"><summary>How were the deletion routes checked?</summary>

These explicit server routes are deployed; dependency, race and tenant contracts were
checked against temporary local state. Earlier bounded live acceptance retains its
qualification results and does not delete hosted state.

</details>

## Next steps

<div class="cards" markdown="1">

- [Notebook examples](notebooks.md): the causal notebook inspects the real data first.
- [Models and limits](capabilities.md): every qualified causal model and its limits.
- [Wait and recover](async-and-errors.md): reconnect to jobs and handle typed errors.

</div>
