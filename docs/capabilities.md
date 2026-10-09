# Implemented and planned capabilities

The service catalogue is authoritative for actual task/model/configuration limits.
A Python class or named catalogue entry alone does not qualify hosted execution.
The bounded upstream TabICLv2, Kumo Medium, TabDPT1.3 and Mitra-v2 previews each have separately
qualified classification and regression tasks: at most 500 training rows,
20 features and 100 query rows.
Classification supports up to 10 classes; regression returns mean points with its
explicit constant-target policy. The catalogue also reports matching-worker
readiness for each task. ArrowFM, CDFM and AVICI each qualify observational discovery at 2–500 rows/2–20 finite
continuous variables. Other selected families require their own gates; no
classical or older-generation fallback is permitted. Mitra-v2 uses the explicit
zero-shot recipe: prediction does not fine-tune weights. This is not a claim to
reproduce publisher benchmark recipes.

| Behavior | Current boundary |
| --- | --- |
| Classification preparation/prediction/probabilities/reopen | Available with the qualified classifier; metadata pins model/configuration/seed. |
| Regression preparation/prediction/reopen | Qualified TabICLv2, Kumo Medium, TabDPT1.3 and Mitra-v2 tasks return finite target-unit points, with classes/probabilities absent. See the task-specific regression guide. |
| DataFrame/array/CSV/Parquet | Scalar cells; named schema safe alignment. Files are currently loaded into memory; Parquet needs optional reader. |
| Durable jobs | Explicit submission, polling, reconnect, cancellation. A local wait timeout leaves durable work running. |
| Usage | Logical operation IDs plus lifecycle/attempt workload metadata; event count is not fit count or monetary price. |
| Model swapping | Available between qualified TabICLv2, Kumo Medium, TabDPT1.3 and Mitra-v2 tasks; new models need a new fit. See model selection. |
| Resumable files / true large batch jobs | Future transport implementation, not implied by file convenience reads. |
| Metadata export / recipes / deletion plans | export_metadata() returns pinned server provenance; recipes and deletion plans require released routes. No executable/context/credential export. |
| Causal discovery / ergodic | ArrowFM bounded observational preview:2–500 rows/2–20 finite continuous variables, native DAG/owned scores/durable reopen. Literal local continuation passes with the private approved Ergodic 0.1.1 artifact. CDFM preserves native directed output/adaptive threshold; AVICI preserves native directed output/strict>0.5 decoding with fixed native PRNG0. Both refuse every non-declared-DAG conversion, including acyclic directed output. Usable public co-release and remaining engines are gated. |
| Confidence / conformal | Native probabilities are not calibrated coverage or conformal intervals; conformal is later. |

Sklearn cloning/Pipeline/CV checks establish specific supported behavior, not all
formats/tasks or the complete 1.6–<2 matrix. Version retirement requires an explicit
migration path; existing predictors remain pinned and a changed model needs a new
fit. Cache presence is derived execution state, distinct from durable predictor
ownership and cold restoration.
