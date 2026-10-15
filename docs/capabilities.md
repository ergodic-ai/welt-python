# Implemented and planned capabilities

See which models and tasks Welt runs today, the limits each one is qualified for, and
what is still planned.

!!! note "The catalogue has the final word"
    The service catalogue is authoritative for actual task, model and configuration
    limits. A Python class or a named catalogue entry alone does not qualify hosted
    execution.

The bounded upstream TabICLv2, Kumo Tabular Medium, TabDPT 1.3 and Mitra-v2 previews
each have separately qualified classification and regression tasks. Arrow / ArrowFM,
CDFM and AVICI each qualify observational discovery.

| Task | Qualified models | Notes |
| --- | --- | --- |
| Classification | TabICLv2, Kumo Tabular Medium, TabDPT 1.3, Mitra-v2 | Predictions and class probabilities. |
| Regression | TabICLv2, Kumo Tabular Medium, TabDPT 1.3, Mitra-v2 | Returns mean points with its explicit constant-target policy. |
| Observational causal discovery | Arrow / ArrowFM, CDFM, AVICI | Finite continuous variables, no supervised target. |

Each model's context shapes, prediction rows per call and classes are in
[Available models](#available-models).

The catalogue also reports matching-worker readiness for each task. Other selected
families require their own gates; no classical or older-generation fallback is
permitted.

Mitra-v2 uses the explicit zero-shot recipe: prediction does not fine-tune weights.
This is not a claim to reproduce publisher benchmark recipes.

## Available models

{{models-table}}

What "context" means depends on the task:

- **Classification and regression:** the training table you pass to `fit`, counted
  as rows × feature columns without the target. The model conditions on it when it
  predicts. Prediction rows are separate: up to 100 per direct call, more with batch
  prediction.
- **Causal discovery:** the whole table, with every column a variable.
- **Forecasting:** the history of one series. Forecasting uses the hidden `chronos-2`
  model, which your workspace must be granted access to; see
  [forecast a time series](forecasting.md) for its history and horizon limits.

A shape caps both axes at once. TabICLv2 refuses 6,000 rows × 30 features (plus the
target): 8,000 × 20 is too narrow and 4,000 × 40 too short. Use 4,000 rows or 20
features instead.

## Behavior boundaries

| Behavior | Current boundary |
| --- | --- |
| Classification: preparation, prediction, probabilities and reopening | Available with the qualified classifier. Metadata pins the model, configuration and seed. |
| Regression: preparation, prediction and reopening | Qualified TabICLv2, Kumo Tabular Medium, TabDPT 1.3 and Mitra-v2 tasks return finite points in target units; classes and probabilities are absent. See the task-specific [regression guide](regression.md). |
| DataFrame, array, CSV and Parquet input | Scalar cells. A named schema aligns columns safely. Files are currently loaded into memory; Parquet needs the optional reader. |
| Durable jobs | Explicit submission, polling, reconnection and cancellation. A local wait timeout leaves durable work running. |
| Usage | Logical operation IDs, lifecycle and attempt workload metadata, and nullable `credits` and `rate_card_version`. Event count is not fit count. See [usage accounting](usage.md#usage-and-workspace-credits). |
| Model swapping | Available between the qualified TabICLv2, Kumo Tabular Medium, TabDPT 1.3 and Mitra-v2 tasks. A new model needs a new fit. See [model selection](model-selection.md). |
| Cross-validated evaluations | Up to 8 classification or regression models (or every eligible one) on the same seeded folds, 2–10 folds, with a leaderboard, fold metrics, seven-day out-of-fold predictions, an HTML report and signed report links. Each training fold must fit the model limits above; larger tables are refused unless you opt into `sample_rows`. No time-ordered splits yet. See [evaluations](evaluations.md). |
| Resumable files and true large batch jobs | Released sync and async upload-session, chunk and file resume, and batch methods. The server sets the enabled formats, parser envelopes and batch limits. Estimator file convenience reads still load into memory. |
| Metadata export, recipes and deletion plans | `export_metadata()` returns pinned server provenance. Recipes and deletion plans require released routes. There is no executable, context or credential export. |
| Causal discovery and Ergodic | ArrowFM is a bounded observational preview (limits in the table above), native DAG, owned scores and durable reopening. `to_ergodic()` converts a declared native DAG into an [ergodic]({{ergodic-docs}}) graph with `welt-client[ergodic]` (ergodic>=0.2,<0.3). CDFM preserves native directed output and its adaptive threshold. AVICI preserves native directed output and strict >0.5 decoding with fixed native PRNG 0. CDFM and AVICI both refuse to convert any graph that is not a declared DAG, including acyclic directed output. The remaining engines are gated. |
| Confidence and conformal | Native probabilities are not calibrated coverage or conformal intervals; conformal comes later. |

<details markdown="1"><summary>What do the scikit-learn, version and cache checks cover?</summary>

The sklearn cloning, Pipeline and cross-validation checks establish specific supported
behavior. They do not cover all formats and tasks, or the complete scikit-learn
1.6–<2 matrix.

Version retirement requires an explicit migration path. Existing predictors remain
pinned, and a changed model needs a new fit. Cache presence is derived execution
state, distinct from durable predictor ownership and cold restoration.

</details>

<a id="resource-completion-source-candidate"></a>

## Upload and batch resources

Both `Client` and `AsyncClient` have these methods:

- **Workspace:** `workspace()` and `pricing()`.
- **Uploads:** `create_upload(format=..., size_bytes=..., sha256=..., target=None, schema=None)`,
  `upload_session(id)`, `put_upload_chunk(id, index, bytes)`, `complete_upload(id)` and
  `upload_file(path, target=None, schema=None, format=None, upload_id=None)`.
- **Batch prediction:** `submit_batch_predict(predictor_id, dataset_id, probabilities=False, idempotency_key=None)`,
  `job_result(job_id)` and `batch_payload(job_id, result_reference=None)`.

On `AsyncClient` you await the equivalents, including upload, file and batch
submission. `Job.result()` and `AsyncJob.result()` return batch manifests with
explicit partial row ranges. `batch_payload()` retrieves authenticated JSON until it
expires.

!!! tip "Keep the upload ID"
    Keep upload IDs before sending bytes, so you can recover an interrupted upload.

File reads and hashing in the async helper are local blocking I/O. The low-level
chunk methods let you schedule disk I/O independently.

!!! note "These methods need server support"
    These released methods require server support. Inspect workspace limits and model
    readiness before submitting work; a client method does not enable a server format
    or model. Account and payment controls remain browser-authorized.

See the [agent CLI reference](cli.md).

## Compatibility with ergodic

[ergodic](https://docs.ergodic.dev) identifies and estimates effects on a Welt graph,
locally. This is the same table as the
[ergodic compatibility page](https://docs.ergodic.dev/compatibility.html). The link between
the packages is a versioned graph format, not a package version.

| ergodic | welt-client | Graph format | Python for the Welt link |
| --- | --- | --- | --- |
| 0.2.x | 0.13 or later | `ergodic.graph/1` | 3.11, 3.12, 3.13 |
| 0.1.x | none | none | none |

- `pip install "welt-client[ergodic]"` adds `ergodic>=0.2,<0.3`.
- `pip install "ergodic[welt]"` adds `welt-client>=0.13` on Python 3.11 to 3.13.

Only a graph that Welt declares a native DAG converts:

| Welt model | Native graph | In ergodic 0.2.0 |
| --- | --- | --- |
| `arrow` | DAG | Converts. `ergodic.welt.discover` returns it as a `DiscoveryResult`. |
| `cdfm` | Directed, may hold cycles | Refused before paying by default. With an explicit `decoder=`, its scores are decoded locally into a DAG labelled decoded. |
| `avici` | Directed, may hold cycles | Refused before paying by default. With an explicit `decoder=`, its scores are decoded locally into a DAG labelled decoded. |

`to_ergodic()` in this SDK never decodes: it refuses `cdfm` and `avici` results. A
decoded graph is the decoder's choice, not the model's claim. See
[Discover in Welt, estimate in ergodic]({{ergodic-docs}}).

## Next steps

<div class="cards" markdown="1">

- [Select a model](model-selection.md): check readiness and compare models fairly.
- [Hidden model runs](model-runs.md): models you can run by exact ID.
- [Workspace resources](usage.md): usage records and workspace credits.

</div>
