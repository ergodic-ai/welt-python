# Implemented and planned capabilities

The service catalogue is authoritative for actual task/model/configuration limits.
A Python class or named catalogue entry alone does not qualify hosted execution.
The initial available classifier is upstream TabICLv2: at most 500 training rows,
20 features, 10 classes and 100 query rows per prediction in the bounded preview.
Other selected families and its regressor require separate gates; no classical or
older-generation fallback is permitted.

| Behavior | Current boundary |
| --- | --- |
| Classification preparation/prediction/probabilities/reopen | Available with the qualified classifier; metadata pins model/configuration/seed. |
| Regressor class | HTTP/estimator interface exists; hosted regression still needs an eligible qualified task. |
| DataFrame/array/CSV/Parquet | Scalar cells; named schema safe alignment. Files are currently loaded into memory; Parquet needs optional reader. |
| Durable jobs | Explicit submission, polling, reconnect, cancellation. A local wait timeout leaves durable work running. |
| Usage | Logical operation IDs plus lifecycle/attempt workload metadata; event count is not fit count or monetary price. |
| Model swapping | Intended compatible task API; each exact artifact/task must qualify first. |
| Resumable files / true large batch jobs | Future transport implementation, not implied by file convenience reads. |
| Metadata export / recipes / deletion plans | export_metadata() returns pinned server provenance; recipes and deletion plans require released routes. No executable/context/credential export. |
| Causal discovery / ergodic | Planned learned engines, native graph semantics and valid local continuation; no placeholder result or invented orientation. |
| Confidence / conformal | Native probabilities are not calibrated coverage or conformal intervals; conformal is later. |

Sklearn cloning/Pipeline/CV checks establish specific supported behavior, not all
formats/tasks or the complete 1.6–<2 matrix. Version retirement requires an explicit
migration path; existing predictors remain pinned and a changed model needs a new
fit. Cache presence is derived execution state, distinct from durable predictor
ownership and cold restoration.
