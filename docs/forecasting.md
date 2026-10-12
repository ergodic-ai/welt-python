# Forecast a univariate time series

`Forecast` provides the same constructor → `fit` → prediction flow as `Classifier`
and `Regressor`. It uses the hidden `chronos-2` model where your authenticated
workspace has access and a worker is available. Requires `welt-client>=0.11.0`
and a server advertising `supports_version_expectations` for the model.
The initial estimator accepts `configuration="default"` only. Model metadata may
set narrower context/horizon limits within the bounds below; fit validates its
context against those limits and prediction enforces the resolved horizon bound.

Download the [local CSV example](downloads/forecasting.py) and run
`python forecasting.py history.csv --time timestamp --target value --horizon 24`
with your own observations and configured workspace credentials.

## Prepare an explicit chronological history

Use your own regular observations, with timezone-aware timestamps. This example
reads a local CSV containing `timestamp` and `value`; supply 16–512 observations.
Keep future observations separate for evaluation.

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

`fit` copies only the selected values and time index, validates locally, then reads
workspace and model metadata. It uploads no observations and creates no remote
predictor or training job: Chronos uses zero-shot history context. Changing the
original table afterwards does not change that context. Constructor changes
invalidate the context; explicitly fit again to use another workspace, connection,
model, configuration or timeout. Cloning an estimator returns an unfitted copy.

Other supported inputs:

```python
forecast.fit(history.set_index("timestamp")["value"])
forecast.fit(history["value"].to_numpy(),
             start="2026-01-01T00:00:00Z", frequency_seconds=3600)
```

For arrays, both `start` and integer `frequency_seconds` are required. Indexed
history must be strictly increasing, unique and evenly spaced in whole elapsed
seconds (1–31,536,000). Explicit aware offsets convert to UTC while preserving
instants. Naive times, missing/nonfinite/nonnumeric/boolean values, FP32 overflow,
subsecond spacing and irregularity are rejected. No sorting, filling, resampling
or truncation happens. Extra DataFrame columns are ignored and never uploaded.
Calendar months and daylight-saving irregularities require explicit preparation.

## Choose one output for each run

Each prediction method submits one durable run with the copied history. Horizon
is an integer from 1–64. The future begins one frequency step after the final
observation. Native quantiles must be unique, increasing selections from
`0.1, 0.2, …, 0.9`; default is `(0.1, 0.5, 0.9)`.

```python
points = forecast.predict(24)                         # ndarray (24,): native medians
quantiles = forecast.predict_quantiles(24)            # ndarray (24, 3)
result = forecast.predict_details(24)                 # immutable ForecastResult
frame = result.to_frame()                             # owned UTC-indexed DataFrame
wire = result.to_dict()                               # independent native JSON payload
```

These three prediction calls are three separate runs. Use `predict_details` once
and read its `predictions`, `quantiles`, `quantile_levels` and `timestamps` to obtain
all views from the same run. Arrays and dictionaries returned by the result are
independent copies. `job_id` is separate from the native `to_dict()` payload.
Results retain model/configuration versions and native median semantics;
quantiles are **not calibrated prediction intervals**. Crossing, nonfinite or
mismatched results raise `InvalidForecastResultError`; they are never repaired.
Some native level selections (including all nine levels) can fail on crossings.
There is no automatic scoring, cross-validation, history update or backtest.

## Retain durable work and reconnect

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

The reconnect block applies when the first wait timed out. Estimator submissions
own their new HTTP client: use a context manager or `job.close()`. Reconnected
handles borrow the enclosing `Client`; closing the handle leaves that client open.
`job.cancel()` explicitly checks the forecast task/identity before asking the
server to cancel. Local timeout and close never cancel or delete remote work.
Typed reconnects validate durable task/model/configuration and result structure;
without the original submitted context they cannot reconstruct the original
history, future start, horizon or requested levels. Estimator-created handles
also validate those exact request expectations.

`AsyncClient.forecast_job(id)` is a regular factory returning `AsyncForecastJob`;
await `inspect()`, `result()` and `cancel()`, and use `async with` for the handle.
It likewise borrows its `AsyncClient`. There is no asynchronous estimator in this
first version. Existing low-level sync/async `submit_model_run` and generic jobs
retain their dictionary contracts.

## Version guards and privacy

Fit freezes model/configuration versions and the effective workspace, origin and
credential. Prediction rechecks the connection and workspace before sending
observations, and submits paired model/configuration expectation headers. Version
drift returns a typed conflict before new work is created; explicitly refit to
accept the new version. An exact replay key/payload/guard reconnects to the
original job even after a later model change. Changed input or guards conflict;
a new key deliberately starts another run. Older servers fail closed at fit.

Advanced `Client.submit_model_run` and `AsyncClient.submit_model_run` accept paired
`expected_model_version` and `expected_configuration_version`; omitting both
preserves the older unguarded submission behavior. These are HTTP headers, not
scientific input fields; they do not change the frozen worker payload or digest.

History/results are private to the authenticated workspace and retained by the
server until explicit terminal-run deletion. Deletion remains an explicit
`client.delete_model_run(id)` action. Read [hidden runs](model-runs.md) for rights,
limits, retention and accounting, and [waiting and recovery](async-and-errors.md)
for typed errors. Protect local history and avoid logging it or credentials.
