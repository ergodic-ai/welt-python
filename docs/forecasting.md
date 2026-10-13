# Forecast a univariate time series

Forecast the next values of one regularly spaced series with the hidden `chronos-2`
model. You get median points and native quantiles for each future step.

!!! need "You'll need"
    - `welt-client>=0.11.0`.
    - A workspace with access to the hidden `chronos-2` model, and an available worker.
    - A server advertising `supports_version_expectations` for the model.
    - Your own regular observations with timezone-aware timestamps.

`Forecast` provides the same constructor → `fit` → prediction flow as `Classifier`
and `Regressor`. The initial estimator accepts `configuration="default"` only. Model
metadata may set narrower context and horizon limits within the bounds below. `fit`
validates its context against those limits, and prediction enforces the resolved
horizon bound.

!!! tip "Run it from a terminal"
    Download the [local CSV example](downloads/forecasting.py) and run
    `python forecasting.py history.csv --time timestamp --target value --horizon 24`
    with your own observations and configured workspace credentials.

## Prepare your history {#prepare-an-explicit-chronological-history}

This example reads a local CSV containing `timestamp` and `value`. Supply 16–512
observations, and keep future observations separate for evaluation.

```python
import pandas as pd
from welt import Forecast

history = pd.read_csv("history.csv")
history["timestamp"] = pd.to_datetime(history["timestamp"])
forecast = Forecast(model="chronos-2", timeout=600)
forecast.fit(history, time="timestamp", target="value")
result = forecast.predict_details(horizon=24)
print(result.to_frame())  # future UTC rows: median, q_0.1, q_0.5, q_0.9
print(result.job_id, result.model_version, result.configuration)
```

`fit` copies only the selected values and time index, validates them locally, then
reads workspace and model metadata. It uploads no observations and creates no remote
predictor or training job: Chronos uses zero-shot history context.

- Changing the original table afterwards does not change that context.
- Constructor changes invalidate the context. Fit again explicitly to use another
  workspace, connection, model, configuration or timeout.
- Cloning an estimator returns an unfitted copy.

You can also fit an indexed series, or an array with a start time and spacing:

```python
forecast.fit(history.set_index("timestamp")["value"])
forecast.fit(history["value"].to_numpy(),
             start="2026-01-01T00:00:00Z", frequency_seconds=3600)
```

The history must follow these rules:

- For arrays, both `start` and integer `frequency_seconds` are required.
- Indexed history must be strictly increasing, unique and evenly spaced in whole
  elapsed seconds (1–31,536,000).
- Explicit aware offsets convert to UTC while preserving instants.
- Extra DataFrame columns are ignored and never uploaded.

`fit` rejects naive times; missing, nonfinite, nonnumeric or boolean values; FP32
overflow; subsecond spacing; and irregularity.

!!! caution "Welt doesn't repair your series"
    No sorting, filling, resampling or truncation happens. Calendar months and
    daylight-saving irregularities require explicit preparation.

## Choose one output for each run

Each prediction method submits one durable run with the copied history.

- The horizon is an integer from 1–64. The future begins one frequency step after the
  final observation.
- Native quantiles must be unique, increasing selections from `0.1, 0.2, …, 0.9`. The
  default is `(0.1, 0.5, 0.9)`.

```python
points = forecast.predict(24)                         # ndarray (24,): native medians
quantiles = forecast.predict_quantiles(24)            # ndarray (24, 3)
result = forecast.predict_details(24)                 # immutable ForecastResult
frame = result.to_frame()                             # owned UTC-indexed DataFrame
wire = result.to_dict()                               # independent native JSON payload
```

!!! tip "Get every view from one run"
    These three prediction calls are three separate runs. Use `predict_details` once
    and read its `predictions`, `quantiles`, `quantile_levels` and `timestamps` to
    obtain all views from the same run.

Arrays and dictionaries returned by the result are independent copies. `job_id` is
separate from the native `to_dict()` payload. Results retain model and configuration
versions and native median semantics.

!!! caution "Quantiles are not intervals"
    Quantiles are **not calibrated prediction intervals**. Crossing, nonfinite or
    mismatched results raise `InvalidForecastResultError`; they are never repaired.
    Some native level selections (including all nine levels) can fail on crossings.

There is no automatic scoring, cross-validation, history update or backtest.

## Keep the job and reconnect {#retain-durable-work-and-reconnect}

Keep the job ID before you wait. If the wait times out, the accepted work keeps
running; reconnect later instead of resubmitting.

```python
from welt import Client, JobTimeoutError

saved_job_id = None
with forecast.submit_predict(24, idempotency_key="my-forecast-20260101") as job:
    print(job.id)  # retain before waiting
    print(job.inspect()["status"])
    try:
        result = job.result(timeout=60)
    except JobTimeoutError:
        saved_job_id = job.id
        # Accepted work remains running; reconnect later. Do not resubmit here.

if saved_job_id is not None:
    with Client() as client:
        with client.forecast_job(saved_job_id) as job:
            result = job.result(timeout=600)
            print(result.to_frame())
```

The reconnect block applies when the first wait timed out.

- Estimator submissions own their new HTTP client: use a context manager or
  `job.close()`.
- Reconnected handles borrow the enclosing `Client`; closing the handle leaves that
  client open.
- `job.cancel()` explicitly checks the forecast task and identity before asking the
  server to cancel.

!!! caution "Timeouts don't cancel"
    Local timeout and close never cancel or delete remote work.

Typed reconnects validate the durable task, model, configuration and result
structure. Without the original submitted context they cannot reconstruct the
original history, future start, horizon or requested levels. Estimator-created
handles also validate those exact request expectations.

<details markdown="1"><summary>Can I use forecasts from async code?</summary>

`AsyncClient.forecast_job(id)` is a regular factory returning `AsyncForecastJob`.
Await `inspect()`, `result()` and `cancel()`, and use `async with` for the handle. It
likewise borrows its `AsyncClient`. There is no asynchronous estimator in this first
version. Existing low-level sync and async `submit_model_run` and generic jobs retain
their dictionary contracts.

</details>

## Version guards and privacy

`fit` freezes the model and configuration versions, and the effective workspace,
origin and credential. Prediction rechecks the connection and workspace before
sending observations, and submits paired model and configuration expectation headers.

- Version drift returns a typed conflict before new work is created. Refit explicitly
  to accept the new version.
- An exact replay key, payload and guard reconnects to the original job, even after a
  later model change.
- Changed input or guards conflict; a new key deliberately starts another run.
- Older servers fail closed at fit.

<details markdown="1"><summary>Can I use version guards with low-level runs?</summary>

Advanced `Client.submit_model_run` and `AsyncClient.submit_model_run` accept paired
`expected_model_version` and `expected_configuration_version`. Omitting both
preserves the older unguarded submission behavior. These are HTTP headers, not
scientific input fields; they do not change the frozen worker payload or digest.

</details>

History and results are private to the authenticated workspace. The server retains
them until explicit terminal-run deletion. Deletion remains an explicit
`client.delete_model_run(id)` action. Protect local history and avoid logging it or
credentials.

## Next steps

<div class="cards" markdown="1">

- [Hidden model runs](model-runs.md): rights, limits, retention and accounting.
- [Wait and recover](async-and-errors.md): typed errors and waiting.

</div>
