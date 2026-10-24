# Hidden model runs and univariate forecasting

Run a hidden model by its exact ID with a low-level payload, then wait for the result
or reconnect to it later.

Hidden models are available by exact ID where your workspace has access. They do not
appear in `Client.models()` or the console catalogue. Inspect `client.model(ID)` for
tasks, limits, pinned configuration and readiness.

Seven entries permit authenticated workspaces; LimiX-1_2M, FoundCause and LCM need
operator-approved internal testing access. Availability requires an active native
worker as well as permission. Hidden status controls discovery; workspace access
controls execution. Model, version and configuration identities are pinned at
submission.

!!! note "No accuracy guarantee"
    A listed checkpoint is not an accuracy guarantee.

!!! tip "Forecasting? Start with the forecasting guide"
    Panels, several targets, past and known-future covariates and quantiles go through
    the `Forecast` estimator and `POST /v1/forecasts`, not a model run. See
    [forecast time series with Chronos-2](forecasting.md). The `chronos-2`
    `univariate_forecasting` run below stays supported for code written against SDK
    0.11–0.13. Timestral (`timestral-209m`, `timestral-28m`) is forecasting only and has
    no model run.

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
| `chronos-2` | univariate_forecasting (legacy run; panels and covariates use [`Forecast`](forecasting.md)) | authenticated |

With the forecasting release (SDK 0.14), Chronos-2 is also a public catalogue model for
the `forecasting` task, so it appears in `Client.models()`. Its univariate run is still
submitted here by exact ID.

## Submit, wait and reconnect

The example inspects `chronos-2`, submits a short hourly series and keeps the job ID
before waiting.

!!! note "A protocol fixture, not a quality test"
    This small protocol fixture illustrates the wire fields and makes no quality claim.
    For assessment, supply real chronological observations and reserve future values
    for rolling-origin evaluation.

```python
from welt import Client

client = Client()
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

- `AsyncClient` offers the same methods with `await`.
- `client.job(ID)` reconnects without resubmission. A local timeout leaves accepted
  work running.
- Reusing the same replay key and identical payload returns the same job; changed
  input conflicts.
- Use `job.cancel()` for explicit remote cancellation.
- Admission and runtime failures keep their typed API errors; raw native logs are not
  returned.

## Task inputs and outputs

| Task | Payload | Result |
| --- | --- | --- |
| classification, regression | `model_id`, `task`, numeric `support` and `query` tables (`columns`, `rows`), `target`, optional classification `probabilities` | Predictions in query order, class order and probabilities when requested, exact version and configuration. |
| causal_discovery (FoundCause) | numeric `table` | Directed and latent-confounder scores; source and target axes. No guaranteed DAG or joint posterior. |
| temporal_causal_discovery (LCM) | chronological numeric `table` | Target, source and reverse-lag axes; lag order `[3,2,1]`; self-lags preserved. |
| univariate_forecasting (Chronos-2) | regular finite `series.values`, UTC `start`, positive integer `frequency_seconds`, `horizon`, optional quantile levels | Future UTC timestamps, median points and horizon-by-quantile values; no calibrated coverage promise. |

!!! caution "Causal scores are not a certain graph"
    FoundCause and LCM scores retain their scientific semantics and cannot be
    converted to an invented certain graph through `CausalResult.to_ergodic()`.

### Limits

- **Forecasting (univariate run):** 16–512 observations and horizon 1–64. Quantile
  levels must be unique and increasing members of
  `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. These limits, the request and the
  result schema are unchanged by the `forecasting` task, whose larger published
  envelopes apply only to [`Forecast`](forecasting.md#limits) requests.
- **Tabular runs:** support rows and numeric features follow each model's measured
  shapes in its metadata (`limits` and `shapes`): most hidden tabular models take up to
  8,000 × 20, 4,000 × 40 or 2,000 × 80; TabSTAR and LimiX differ. At least 2 support
  rows and 1–100 query rows.
  Classifiers accept 2–10 classes with labels exactly representable in FP32.
- **Causal runs:** consult metadata for each causal input bound.

Some models behave differently:

- Nori is regression-only and uses request seed 42 and native predictor seed 0.
- Orion is classification-only.
- TabSTAR fine-tunes for up to ten epochs with patience 3 in the pinned recipe,
  within each run.

All tabular support and query rows are submitted together; the API retains no
reusable native fit. Query labels must be absent.

!!! caution "You own the evaluation"
    Preserve chronological or held-out assessment yourself; these calls do not
    implement model selection or benchmark ranking.

## Privacy and retention

Inputs and results remain private to the credential's workspace. The bounded preview
allows up to 100 retained runs per workspace, with the shared queued-job limit.

!!! note "Most runs are unpriced"
    Runs are currently unpriced (`usage.credits` is null), with no monetary guarantee
    for future versions. The exception, from the forecasting release, is the `chronos-2`
    univariate run on the rate card that prices forecasts: it costs the same as a one-series, one-target forecast,
    (observations + horizon) tokens at 50 credits per million with a 5-credit minimum,
    so it is not a free route around [forecast pricing](forecasting.md#pricing).
    `client.pricing()` shows the card in force.

Purge deliberately using `client.delete_model_run(job.id)`; only terminal runs may be
purged. This removes the input and result but retains job, idempotency and usage
metadata.

- Retrieving a purged result returns 404.
- Replay returns the same job with `purged:true`. Replaying a purged job does not
  recreate its data; use a new replay key for a deliberate new run.

## Next steps

<div class="cards" markdown="1">

- [Forecast time series](forecasting.md): panels, covariates and quantiles with `Forecast`.
- [Models and limits](capabilities.md): catalogue models and their qualified limits.
- [Wait and recover](async-and-errors.md): reconnect to jobs and handle typed errors.

</div>
