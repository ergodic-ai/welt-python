# Forecast time series with Chronos-2

Forecast one series or a whole panel with Chronos-2. Add several targets, drivers you
only observe in the past, and drivers you know ahead of time. You get a median and
native quantiles for every series, target and future step.

!!! need "You'll need"
    - `welt-client>=0.14.0`: `pip install "welt-client>=0.14"`.
    - A Welt workspace with credits. Every `forecast` call is one priced request; see
      [what a forecast costs](#pricing).
    - A Welt server that offers the `forecasting` task for `chronos-2`. Older servers
      keep only the [univariate API](#forecast-a-univariate-time-series).

The steps build one script, `forecast_sales.py`. Run them in order in one Python
session: each step reuses the data and model from the step before. The sales are
simulated so the page runs anywhere; use your own history for real forecasts.

!!! tip "Prefer a notebook?"
    The [forecasting notebook](notebook-forecasting-with-covariates.html) builds a
    realistic retail panel, holds out the last four weeks, and scores Chronos-2
    against a seasonal-naive baseline with MASE and interval coverage. The same
    tutorial runs as a script: [forecasting_covariates.py](downloads/forecasting_covariates.py).

## Forecast one series {.step}

Make a table with a time column and a value column, fit, then ask for a horizon.

```python file="forecast_sales.py"
import numpy as np
import pandas as pd
from welt import Forecast

# A year of simulated daily sales for three stores. Use your own history.
rng = np.random.default_rng(0)
days = pd.date_range("2025-01-01", periods=365, freq="D")
stores = {"north": 120, "south": 90, "east": 60}
panel = pd.concat([pd.DataFrame({
    "date": days,
    "store": store,
    "sales": level * (1 + 0.3 * (days.dayofweek >= 5)) + rng.normal(0, 5, len(days)),
}) for store, level in stores.items()], ignore_index=True)

north = panel[panel["store"] == "north"]
model = Forecast(model="chronos-2").fit(north, time="date", target=["sales"])
result = model.forecast(14)
print(result.to_frame().head())
```

- `fit` copies your columns, checks them locally and reads the model's published
  limits. It uploads nothing and starts no job: Chronos-2 is zero-shot, so the history
  is the context for each forecast.
- `forecast(14)` sends one request for the 14 days after the last row and waits for
  the result.
- Pass `target` as a list. A single string target selects the older
  [univariate API](#forecast-a-univariate-time-series), which needs timezone-aware
  times and complete values.
- Columns you don't name, like `store` here, are not sent.

## Forecast a panel {.step}

Name the column that identifies each series with `id`. Every store becomes its own
series, and all of them go in one request.

```python file="forecast_sales.py"
model = Forecast(model="chronos-2").fit(panel, time="date", id="store", target=["sales"])
result = model.forecast(14)
frame = result.to_frame()
print(frame.groupby("id")["median"].mean())
```

- Rows can come in any order. Each (id, time) pair must appear once.
- Series can differ in length and start date, but they share one frequency. Each
  series' forecast starts one step after its own last row.
- IDs are strings or integers. Results return them as strings.
- Already uploaded the history as a CSV? Pass its dataset ID instead of a DataFrame,
  for example `fit("ds_…", time="date", id="store", target=["sales"])`, and give
  `forecast(future=...)` a dataset ID too. Those rows travel by reference, so only the
  server checks them, and fitted sizes such as `n_series_` are `None`. Use `forecast`
  for a dataset history; `predict` needs a local table.

## Forecast several targets {.step}

Pass more than one target. Chronos-2 forecasts a series' targets jointly, so related
targets can inform each other.

```python file="forecast_sales.py"
# Visits move with sales; forecast both together.
panel["visits"] = (3 * panel["sales"] + rng.normal(0, 15, len(panel))).round()

model = Forecast(model="chronos-2").fit(
    panel, time="date", id="store", target=["sales", "visits"])
result = model.forecast(14)
print(result.to_frame().groupby(["id", "target"]).size())  # 14 rows per store and target
```

The result has one entry per series and target: series in the order their IDs first
appear, then targets in the order you listed them.

## Add covariates {.step}

Covariates are drivers the model reads but doesn't forecast. Tell Welt when you know
each one:

- **Past covariates** are observed only up to now, like temperature or footfall. They
  appear in the history only.
- **Future covariates** are known ahead, like planned promotions, holidays or prices
  you set. They appear in the history and in a separate future table.
- **Categorical** names the covariates whose cells are labels (strings or integers)
  rather than amounts.

```python file="forecast_sales.py"
def holidays(dates):
    """Public holidays are on the calendar long before they happen."""
    return np.where(dates.dt.strftime("%m-%d").isin(["01-01", "05-01", "12-25", "12-26"]),
                    "holiday", "regular")

panel["promo"] = (panel["date"].dt.dayofweek == 4).astype(int)  # planned Friday promotions
panel["holiday"] = holidays(panel["date"])
panel["temperature"] = (12 + 9 * np.sin(2 * np.pi * (panel["date"].dt.dayofyear - 110) / 365)
                        + rng.normal(0, 2, len(panel)))  # known only once measured

model = Forecast(model="chronos-2").fit(
    panel, time="date", id="store", target=["sales", "visits"],
    past_covariates=["temperature"],
    future_covariates=["promo", "holiday"],
    categorical=["holiday"])
```

Then give the known values for the forecast days in a future table, and pass it to
`forecast`:

```python file="forecast_sales.py"
# The next 14 days for every store, with the values you already know.
ahead = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=14, freq="D")})
future = pd.concat([ahead.assign(store=store) for store in stores], ignore_index=True)
future["promo"] = (future["date"].dt.dayofweek == 4).astype(int)
future["holiday"] = holidays(future["date"])

result = model.forecast(14, future=future)
```

- The future table has exactly the time column, the id column and every future
  covariate: no targets, no past covariates.
- It has exactly `horizon` rows for each series, at the steps right after that
  series' last row. The SDK checks this before sending anything.
- Future covariates must be complete, in the history and the future table.
- Chronos-2 encodes categories itself. A category that first appears in the future
  table is allowed; the native encoder handles it.

## Read quantiles and the table {.step}

Ask for the quantile levels you need, then read the long table or each series.

```python file="forecast_sales.py"
result = model.forecast(14, future=future, quantile_levels=[0.05, 0.1, 0.5, 0.9, 0.95])
frame = result.to_frame()
print(frame.columns.tolist())
# ['id', 'target', 'timestamp', 'median', 'q_0.05', 'q_0.1', 'q_0.5', 'q_0.9', 'q_0.95']
print("Sorted steps:", result.rearranged_steps)

for series in result.series:
    print(series.id, series.target, series.context_length, series.predictions[:3])
```

- Levels are any increasing subset of the 21 native levels: 0.01, 0.05, 0.10, 0.15, …,
  0.95, 0.99. Model metadata publishes them as `quantile_levels`. The default is
  `[0.1, 0.5, 0.9]`.
- `median` is the native 0.5 quantile. It is always returned, even when you don't
  request 0.5.
- `to_frame()` has one row per series, target and step. `timestamp` is UTC.
- `result.series` holds one `ForecastSeries` per series and target, with `id`,
  `target`, `context_length`, `timestamps`, `predictions` and `quantiles`. For one
  series and one target, `result.predictions`, `result.quantiles` and
  `result.timestamps` read it directly.

!!! caution "Intervals are not calibrated"
    Every result reports `calibrated_intervals: false`. The band from `q_0.1` to
    `q_0.9` is the model's estimate, not a promise that 80% of future values fall
    inside it. Measure coverage on a holdout of your own data, as the
    [notebook](notebook-forecasting-with-covariates.html) does.

!!! note "When native quantiles cross, Welt sorts them and tells you"
    Chronos-2 can return quantiles out of order at a step, for example a 0.1 value
    above the 0.5 value. Welt then sorts that step's 21 native values (monotone
    rearrangement) and takes your levels and the median from the sorted values.
    `result.rearranged_steps` counts the (series, step) pairs it sorted. When it is 0,
    every value is exactly what the model returned. A non-finite native value fails
    the job instead.

## Share information across series {.step}

By default each series is forecast on its own. Set `cross_learning=True` to let the
series in one request inform each other.

```python file="forecast_sales.py"
shared = model.forecast(14, future=future, cross_learning=True)
print(shared.cross_learning)  # True
```

- With `cross_learning=False`, a series' forecast is the one you would get by sending
  that series alone. Welt checks this when it qualifies the model.
- With `cross_learning=True`, forecasts depend on which series you send together. It
  can help with many short, related series. Keep the panel fixed when you compare
  results.
- It's priced the same. This call is a second request.

## Check the limits {#limits}

Limits come from the model's published metadata. Read them before you send a large
panel:

```python file="forecast_sales.py"
from welt import Client

with Client() as client:
    chronos = client.model("chronos-2")
print("Minimum rows per series:", chronos["min_context"])
for corner in chronos["forecast_envelopes"]:
    print(corner)  # max_context, max_variates, max_series, max_horizon
```

Each envelope is a measured corner. A request fits when one corner covers it on all
four axes at once:

- **Context:** the history rows of the longest series.
- **Variates:** targets plus past covariates plus future covariates. Step 4 uses
  2 + 1 + 2 = 5.
- **Series:** the number of IDs.
- **Horizon:** the steps you ask for.

Welt doesn't interpolate between corners. Every series also needs at least
`min_context` rows, and at most 8,192, Chronos-2's native context. Histories are
never truncated. `fit` checks the history against the corners, and `forecast` checks
the horizon. Both raise `InvalidInputError` with code `workload_limit`, naming the
published corners, before anything is sent. A very large request can still be
refused by the server if its encoded input or output exceeds the transport bounds.

<!-- release: replace TBD-API-160 with the corners API-160 measured and API-164 published, then delete this comment. -->

At release, the measured limits are: **TBD-API-160**. The published metadata always
has the final word.

## What a forecast costs {#pricing}

A forecast is priced in tokens, at the same rate as tabular prediction: 50 credits per
million tokens, with a 5-credit minimum per request.

`tokens = Σ over series (history rows + horizon) × (targets + covariates)`

```python file="forecast_sales.py"
horizon, variates = 14, 2 + 1 + 2  # targets + past covariates + future covariates
tokens = sum((len(rows) + horizon) * variates for _, rows in panel.groupby("store"))
print(tokens, "tokens")  # 3 × (365 + 14) × 5 = 5,685
print(max(5, tokens * 50 / 1_000_000), "credits")  # under the minimum, so 5
```

- Columns you don't name are not sent and not counted.
- Each `forecast` call is its own request. Step 6 costs a second one.
- Credits are reserved when the request is queued and settled when it finishes. Usage
  events have kind `forecast`; see [usage and credits](usage.md#usage-and-workspace-credits).
- Check current prices with `client.pricing()` or in **Workspace → Billing**.

## Prepare your data {#prepare-your-data}

Welt forecasts the table you send. It doesn't fill, resample or truncate anything.

- **Shape:** one row per series and time, in any order. A repeated (id, time) pair is
  refused.
- **Time:** a datetime column or ISO-8601 strings. Naive times, like the output of
  `pd.date_range` or `read_csv`, are read as UTC. Times with an offset are converted
  to UTC. Results are always UTC.
- **Local clocks:** daily dates work as naive dates. Hourly data recorded on a local
  clock with daylight saving is irregular in UTC; convert it to UTC and decide how to
  handle the shifted hours before you fit.
- **No gaps:** every step between a series' first and last row must be present. Add
  missing rows yourself, with a missing target if you have no value.
- **Missing values:** an empty (NaN or None) target or past-covariate cell is passed
  to Chronos-2 as missing, which it handles natively. A target that is missing for a
  whole series is refused. Time, id and future covariates must be complete.
- **Values:** targets and numeric covariates are finite numbers that fit in 32-bit
  floats. Booleans and text are refused in numeric columns; convert booleans with
  `.astype(int)`.
- **IDs:** strings or integers. Mixing `1` and `"1"` in one column is refused because
  both read as `"1"`.

```python file="forecast_sales.py"
gappy = panel.copy()
gappy.loc[gappy.index[100], "sales"] = np.nan  # a missed reading stays missing
gappy_model = Forecast(model="chronos-2").fit(
    gappy, time="date", id="store", target=["sales"], past_covariates=["temperature"])
```

Welt infers the frequency when every series has one constant spacing. Pass
`frequency=` to state it:

```python file="forecast_sales.py"
daily = Forecast(model="chronos-2").fit(
    panel, time="date", id="store", target=["sales"], frequency="D")
print(daily.frequency_)  # {'unit': 'day', 'every': 1}
```

- Aliases: `"s"`, `"min"`, `"h"`, `"D"`, `"W"`, `"M"` (also `"MS"` and `"ME"`),
  `"Q"` and `"Y"`, with an optional multiple such as `"15min"` or `"2W"`. Or pass a
  dictionary such as `{"unit": "hour", "every": 6}`.
- Months, quarters and years use calendar steps. Every observation must fall on the
  same day of the month (1 to 28), or every observation on a month-end.

## Handle errors {#errors}

Most problems are caught before anything is sent:

```python file="forecast_sales.py"
from welt import InvalidInputError

try:
    model.forecast(100_000, future=future)
except InvalidInputError as error:
    print(error.code)  # workload_limit: no published corner admits this horizon
```

| Error | When | What to do |
| --- | --- | --- |
| `ValueError` | A local check failed: a gap, a repeated time, a missing future value, wrong future rows. | Fix the table. Nothing was sent. |
| `InvalidInputError`, code `workload_limit` | No published corner admits the request, a series is shorter than `min_context`, or the request body is above 10 MiB. | Shorten the history, use fewer variates or series, or ask for a shorter horizon. For a large table, upload it and pass its dataset ID. |
| `InvalidInputError`, code `invalid_input` | The server refused the request. | Read the message. It never repeats your cell values. |
| `InvalidInputError`, code `unsupported_capability` | The server has no `forecasting` task, but you asked for a panel, covariates, a dataset history, a calendar frequency or `forecast()`. | Use the [univariate API](#forecast-a-univariate-time-series), or a current server. |
| `ConflictError`, code `model_version_mismatch` or `configuration_version_mismatch` | The model changed after your `fit`. | Fit again. |
| `ConflictError`, code `idempotency_conflict` | You reused a replay key with a different request. | Use a new key. |
| `JobTimeoutError` | Your local wait ended. | [Reconnect](#submit-now-collect-later). The forecast keeps running. |
| `ExecutionError` | The forecast job failed. | Retry later, or contact support with the job ID. |
| `InvalidForecastResultError` | The result failed the SDK's checks. | Report it. Nothing was repaired. |

`InvalidInputError` is also a `ValueError`, so catch it first. Billing refusals such
as `insufficient_credits` are described in [wait and recover](async-and-errors.md).

## Submit now, collect later {#submit-now-collect-later}

`submit_forecast` returns a job without waiting. Keep its ID, then wait, or reconnect
from another session.

```python file="forecast_sales.py"
from welt import JobTimeoutError

job = model.submit_forecast(14, future=future, idempotency_key="sales-forecast-2026-01-01")
print("Keep this job ID:", job.id)
try:
    result = job.result(timeout=60)
except JobTimeoutError:
    print("Still running. Reconnect later with the job ID.")
finally:
    job.close()

with Client() as client:
    with client.forecast_job(job.id) as reopened:
        result = reopened.result(timeout=600)
print(result.to_frame().shape)
```

- A local timeout or `job.close()` never cancels the forecast. Call `job.cancel()` to
  cancel it.
- Replaying the same key with the same request reconnects to the same job. The same
  key with a different request raises `ConflictError`.
- A reconnected job checks the task and the model and configuration versions. It
  can't rebuild your original request, so it doesn't check the series against it.
- `AsyncClient.forecast_job(id)` works the same way with `await`.
- The SDK sends DataFrame tables inline in the request and never uploads them for
  you, so replaying a key sends the same body. A body above the API's 10 MiB limit
  (`welt.forecast.MAX_REQUEST_BYTES`) is refused with `InvalidInputError`
  (`workload_limit`) before anything is sent: upload the table, for example with
  `client.upload_file`, and pass its dataset ID.
- `Client.submit_forecast(body)` and `welt forecast --input request.json` send a raw
  `POST /v1/forecasts` body without the SDK's checks.
- Forecast inputs and results are private to your workspace until you delete them.
  `client.delete_forecast(job.id)` or `welt forecasts delete ID --yes` removes a
  finished forecast's input and result. The job, its replay key and its usage record
  stay, and workspace datasets it read are not deleted.

!!! tip "Backtest and reconcile in ergodic"
    Use Chronos-2 inside ergodic's backtest and reconcile:
    [Forecast with Welt's Chronos-2](https://docs.ergodic.dev/forecast-with-welt.html).

## Forecast one series with the univariate API {#forecast-a-univariate-time-series}

The univariate API from SDK 0.11 stays supported: `predict`, `predict_quantiles`,
`predict_details` and `submit_predict` for one series and one target. It fits an
array, a Series with a `DatetimeIndex`, or a DataFrame with one string `target` and no
`id` or covariates.

- On a server with the `forecasting` task, these calls use `POST /v1/forecasts`.
  Limits and quantile levels then come from the published envelopes.
- On older servers they use the univariate model run described below, with its
  16–512 observation and 1–64 horizon limits.
- Either way, `predict_details().to_frame()` keeps the UTC-indexed layout with
  `median` and `q_<level>` columns.
- For a panel or several targets they raise `ValueError` and point you to
  `forecast()`.

!!! tip "Run it from a terminal"
    Download the [local CSV example](downloads/forecasting.py) and run
    `python forecasting.py history.csv --time timestamp --target value --horizon 24`
    with your own observations and configured workspace credentials.

### Prepare your history {#prepare-an-explicit-chronological-history}

This example reads a local CSV containing `timestamp` and `value`. On servers with only
the univariate run, supply 16–512 observations. Keep future observations separate for
evaluation.

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
overflow; subsecond spacing; and irregularity. To read naive times as UTC and allow
missing values, pass the target as a list, as in [step 1](#forecast-one-series).

!!! caution "Welt doesn't repair your series"
    No sorting, filling, resampling or truncation happens. Calendar months and
    daylight-saving irregularities require explicit preparation.

### Choose one output for each run

Each prediction method submits one durable run with the copied history.

- The future begins one frequency step after the final observation.
- On servers with only the univariate run, the horizon is an integer from 1–64 and
  native quantiles are unique, increasing selections from `0.1, 0.2, …, 0.9`. On
  servers with the `forecasting` task, the published envelopes and the 21 native
  levels apply instead. The default is `(0.1, 0.5, 0.9)`.

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
    Quantiles are **not calibrated prediction intervals**. On the univariate model
    run, crossing, nonfinite or mismatched results raise `InvalidForecastResultError`;
    they are never repaired, and some native level selections (including all nine
    levels) can fail on crossings. On the `forecasting` task, crossed quantiles are
    sorted and counted in `rearranged_steps` instead.

There is no automatic scoring, cross-validation, history update or backtest.

### Keep the job and reconnect {#retain-durable-work-and-reconnect}

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

### Version guards and privacy

`fit` freezes the model and configuration versions, and the effective workspace,
origin and credential. Prediction rechecks the connection and workspace before
sending observations, and submits paired model and configuration expectation headers.
`forecast` and `submit_forecast` send the same guards.

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
`Client.submit_forecast` accepts the same pair.

</details>

History and results are private to the authenticated workspace. The server retains
them until explicit terminal-run deletion: `client.delete_model_run(id)` for a
univariate model run, `client.delete_forecast(id)` for a forecast. Protect local history and avoid
logging it or credentials.

## Next steps

<div class="cards" markdown="1">

- [Forecasting notebook](notebook-forecasting-with-covariates.html): a retail panel with a holdout, a baseline, MASE and coverage.
- [Chronos-2 in ergodic](https://docs.ergodic.dev/forecast-with-welt.html): backtest against baselines and reconcile a hierarchy.
- [Models and limits](capabilities.md): every model Welt runs and its published limits.
- [Wait and recover](async-and-errors.md): typed errors and waiting.

</div>
