# Hidden model runs and univariate forecasting

Hidden models are available by exact ID where your workspace has access. They do not appear in `Client.models()` or the console catalogue.
Inspect `client.model(ID)` for tasks, limits, pinned configuration and readiness.
Seven entries permit authenticated workspaces; LimiX-1_2M, FoundCause and LCM need
operator-approved internal testing access. Availability requires an active native
worker as well as permission. A listed checkpoint is not an accuracy guarantee.

For the high-level `Forecast` estimator, native arrays and typed durable results,
see [forecast a time series](forecasting.md).

## Exact model IDs

| Model ID | Tasks | Access |
| --- | --- | --- |
| `kumo-small` | classification, regression | authenticated |
| `kumo-large` | classification, regression | authenticated |
| `nori-30m` | regression | authenticated |
| `orion-msp-1.0` | classification | authenticated |
| `tabdpt-turbo-1.2` | classification, regression | authenticated |
| `tabstar-1.1.16` | classification, regression | authenticated |
| `limix-1-2m` | classification, regression | allowlisted internal testing |
| `foundcause-1.0` | causal_discovery | allowlisted internal testing |
| `lcm-2.5m` | temporal_causal_discovery | allowlisted internal testing |
| `chronos-2` | univariate_forecasting | authenticated |

## Submit, wait and reconnect

This small protocol fixture illustrates the wire fields and makes no quality
claim. For assessment, supply real chronological observations and reserve future
values for rolling-origin evaluation.

```python
from welt import Client

with Client() as client:
    print(client.model("chronos-2"))
    # A small illustrative hourly series. Use your own chronological observations.
    payload = {
        "model_id": "chronos-2", "task": "univariate_forecasting",
        "series": {"values": [12, 14, 15, 13, 12, 15, 16, 14,
                              13, 15, 17, 15, 14, 16, 18, 16],
                   "start": "2026-01-01T00:00:00Z", "frequency_seconds": 3600},
        "horizon": 4, "quantile_levels": [0.1, 0.5, 0.9],
    }
    job = client.submit_model_run(payload, idempotency_key="forecast-00000001")
    print(job.id)  # retain before waiting
    result = job.result(timeout=600)
    print(result["timestamps"], result["predictions"], result["quantiles"])
```

`AsyncClient` offers the same methods with `await`; `client.job(ID)` reconnects
without resubmission. A local timeout leaves accepted work running. Reusing the
same replay key and identical payload returns the same job; changed input conflicts.
Use `job.cancel()` for explicit remote cancellation. Admission/runtime failures
retain typed API errors; raw native logs are not returned.

## Task inputs and outputs

| Task | Payload | Result |
| --- | --- | --- |
| classification/regression | `model_id`, `task`, numeric `support` and `query` tables (`columns`, `rows`), `target`, optional classification `probabilities` | Predictions in query order, class order/probabilities when requested, exact version/configuration. |
| causal_discovery (FoundCause) | numeric `table` | Directed and latent-confounder scores; source/target axes. No guaranteed DAG or joint posterior. |
| temporal_causal_discovery (LCM) | chronological numeric `table` | Target/source/reverse-lag axes, lag order `[3,2,1]`, self-lags preserved. |
| univariate_forecasting (Chronos-2) | regular finite `series.values`, UTC `start`, positive integer `frequency_seconds`, `horizon`, optional quantile levels | Future UTC timestamps, median points and horizon-by-quantile values; no calibrated coverage promise. |

Forecasting accepts 16–512 observations and horizon 1–64. Tabular runs accept
2–500 support rows, 1–100 query rows and 1–20 numeric features; classifiers accept
2–10 classes with labels exactly representable in FP32. Quantile levels must be
unique and increasing members of `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Nori is regression-only and uses request seed42/native predictor
seed0. Orion is classification-only. Consult metadata for each causal input bound.
TabSTAR fine-tunes for up to ten epochs with patience3 in the pinned recipe within each run. All tabular
support and query are submitted together; the API retains no reusable native fit.
Query labels must be absent. Preserve chronological/held-out assessment yourself;
these calls do not implement model selection or benchmark ranking.

## Privacy and retention

Inputs/results remain private to the credential's workspace. The bounded preview
allows up to100 retained runs per workspace, with the shared queued-job limit.
Runs are currently unpriced (`usage.credits` is null), with no monetary guarantee
for future versions. Purge deliberately using `client.delete_model_run(job.id)`;
only terminal runs may be purged. This removes input/result but retains job,
idempotency and usage metadata. Retrieving a purged result returns404; replay returns the same job with
`purged:true`. Replaying a purged job does not recreate its data;
use a new replay key for a deliberate new run.

Model/version/configuration identities are pinned at submission. Hidden status
controls discovery; workspace access controls execution. FoundCause/LCM
scores retain their scientific semantics and cannot be converted to an invented
certain graph through `CausalResult.to_ergodic()`.
