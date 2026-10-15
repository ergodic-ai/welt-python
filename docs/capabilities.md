# Implemented and planned capabilities

See which models and tasks Welt runs today, the limits each one is qualified for, and
what is still planned.

!!! note "The catalogue has the final word"
    The service catalogue is authoritative for actual task, model and configuration
    limits. A Python class or a named catalogue entry alone does not qualify hosted
    execution.

The bounded upstream TabICLv2, Kumo Medium, TabDPT1.3 and Mitra-v2 previews each have
separately qualified classification and regression tasks. ArrowFM, CDFM and AVICI each
qualify observational discovery.

| Task | Qualified models | Limits |
| --- | --- | --- |
| Classification | TabICLv2, Kumo Medium, TabDPT1.3, Mitra-v2 | At most 500 training rows, 20 features and 100 query rows. Up to 10 classes. |
| Regression | TabICLv2, Kumo Medium, TabDPT1.3, Mitra-v2 | At most 500 training rows, 20 features and 100 query rows. Returns mean points with its explicit constant-target policy. |
| Observational causal discovery | ArrowFM, CDFM, AVICI | Arrow and CDFM: up to 8,000 rows × 20 variables (measured 4,000 × 40 and 2,000 × 80). AVICI: up to 1,000 × 20 (or 500 × 40). At least 2 rows and 2 finite continuous variables. |

The catalogue also reports matching-worker readiness for each task. Other selected
families require their own gates; no classical or older-generation fallback is
permitted.

Mitra-v2 uses the explicit zero-shot recipe: prediction does not fine-tune weights.
This is not a claim to reproduce publisher benchmark recipes.

## Available models

{{models-table}}

## Behavior boundaries

| Behavior | Current boundary |
| --- | --- |
| Classification: preparation, prediction, probabilities and reopening | Available with the qualified classifier. Metadata pins the model, configuration and seed. |
| Regression: preparation, prediction and reopening | Qualified TabICLv2, Kumo Medium, TabDPT1.3 and Mitra-v2 tasks return finite points in target units; classes and probabilities are absent. See the task-specific [regression guide](regression.md). |
| DataFrame, array, CSV and Parquet input | Scalar cells. A named schema aligns columns safely. Files are currently loaded into memory; Parquet needs the optional reader. |
| Durable jobs | Explicit submission, polling, reconnection and cancellation. A local wait timeout leaves durable work running. |
| Usage | Logical operation IDs, lifecycle and attempt workload metadata, and nullable `credits` and `rate_card_version`. Event count is not fit count. See [usage accounting](usage.md#usage-and-workspace-credits). |
| Model swapping | Available between the qualified TabICLv2, Kumo Medium, TabDPT1.3 and Mitra-v2 tasks. A new model needs a new fit. See [model selection](model-selection.md). |
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

## Next steps

<div class="cards" markdown="1">

- [Select a model](model-selection.md): check readiness and compare models fairly.
- [Hidden model runs](model-runs.md): models you can run by exact ID.
- [Workspace resources](usage.md): usage records and workspace credits.

</div>
