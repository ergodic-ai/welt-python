# Implemented and planned capabilities

The service catalogue is authoritative for actual task/model/configuration limits.
A Python class or named catalogue entry alone does not qualify hosted execution.
The bounded upstream TabICLv2 and Kumo Medium previews each have separately
qualified classification and regression tasks: at most 500 training rows,
20 features and 100 query rows.
Classification supports up to 10 classes; regression returns mean points with its
explicit constant-target policy. The catalogue also reports matching-worker
readiness for each task. Other selected families require their own gates; no
classical or older-generation fallback is permitted.

| Behavior | Current boundary |
| --- | --- |
| Classification preparation/prediction/probabilities/reopen | Available with the qualified classifier; metadata pins model/configuration/seed. |
| Regression preparation/prediction/reopen | Qualified TabICLv2 and Kumo Medium tasks return finite target-unit points, with classes/probabilities absent. See the task-specific regression guide. |
| DataFrame/array/CSV/Parquet | Scalar cells; named schema safe alignment. Files are currently loaded into memory; Parquet needs optional reader. |
| Durable jobs | Explicit submission, polling, reconnect, cancellation. A local wait timeout leaves durable work running. |
| Usage | Logical operation IDs plus lifecycle/attempt workload metadata; event count is not fit count or monetary price. |
| Model swapping | Available between qualified TabICLv2 and Kumo Medium tasks; new models need a new fit. See model selection. |
| Resumable files / true large batch jobs | Future transport implementation, not implied by file convenience reads. |
| Metadata export / recipes / deletion plans | export_metadata() returns pinned server provenance; recipes and deletion plans require released routes. No executable/context/credential export. |
| Causal discovery / ergodic | Planned learned engines, native graph semantics and valid local continuation; no placeholder result or invented orientation. |
| Confidence / conformal | Native probabilities are not calibrated coverage or conformal intervals; conformal is later. |

Sklearn cloning/Pipeline/CV checks establish specific supported behavior, not all
formats/tasks or the complete 1.6–<2 matrix. Version retirement requires an explicit
migration path; existing predictors remain pinned and a changed model needs a new
fit. Cache presence is derived execution state, distinct from durable predictor
ownership and cold restoration.
