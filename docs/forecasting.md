# Forecast time series with Chronos-2 and Timestral

For historical walk-forward accuracy and optional final forecasts, see
[Backtest a time series](backtesting.md).

Forecast one series or a whole panel with Chronos-2. Add several targets, drivers you
only observe in the past, and drivers you know ahead of time. You get a median and
native quantiles for every series, target and future step. Timestral, listed after
Chronos-2, forecasts one series per request through the same calls; see
[forecast with Timestral](#timestral).

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
    realistic retail panel, holds out the last four weeks, and scores Chronos-2 with
    and without covariates against a seasonal-naive baseline, with MASE and interval
    coverage. The same
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
- Pass `target` as a list. With no `id` and no covariates, a single string target
  selects the older [univariate API](#forecast-a-univariate-time-series), which needs
  timezone-aware times and complete values.
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
- Already uploaded the history? Pass its dataset ID instead of a DataFrame:
  `fit("ds_…", time="date", id="store", target=["sales"])`. See
  [dataset histories](#dataset-histories) for how to upload it.

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

The same arguments work with `model="chronos-2"`, `"timestral-209m"` and
`"timestral-28m"`. Start with one series and `target=["sales"]` to use the same
example across all three models; Chronos-2 also accepts panels within its published
envelopes. Timestral accepts one series per request and needs one target when any
covariate is categorical. From SDK 0.14.2, `fit` checks the model's categorical/
multiple-target restriction locally, and `forecast(cross_learning=True)` checks
its cross-learning restriction before submitting a job. SDK 0.14.0 and 0.14.1
receive the same `unsupported_capability` error from the server.

The downloadable [covariate script](downloads/forecasting_covariates.py) accepts
the same model choices. To run one simulated store with Timestral:

```sh
python forecasting_covariates.py --model timestral-28m --store berlin
# Or use --model timestral-209m, or --model chronos-2.
```

This sends one priced forecast request. The script requires an explicit store for
Timestral before making any request; it does not fan out into multiple paid calls.

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
    `result.rearranged_steps` counts the (series, target, step) triples it sorted. When it is 0,
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

At release, Chronos-2 publishes these seven measured corners, and every series needs
at least 16 rows (`min_context`):

| Context (rows) | Variates | Series | Horizon |
| --- | --- | --- | --- |
| 8,000 | 50 | 1 | 1,024 |
| 8,000 | 5 | 8 | 1,024 |
| 4,000 | 10 | 8 | 1,024 |
| 2,000 | 20 | 8 | 1,024 |
| 2,000 | 5 | 32 | 256 |
| 1,000 | 50 | 8 | 256 |
| 512 | 20 | 32 | 256 |

Step 4's panel (365 rows × 5 variates × 3 series × 14 steps) fits the 2,000 × 5 × 32
× 256 corner, for example.

- Welt doesn't interpolate between corners: one corner must cover all four numbers.
- Larger panels are bounded by request size, not by the GPU: the 10 MiB request body
  and the server's table intake. That is why wide or long requests have fewer series.
- `fit` checks the history against the corners and `forecast` checks the horizon.
  Both raise `InvalidInputError` with code `workload_limit`, naming the corners,
  before anything is sent. Histories are never truncated.
- The corners can change. Your code should read them from
  `client.model("chronos-2")`, as above, rather than copy this table.

## What a forecast costs {#pricing}

A forecast is priced in tokens, at the same rate as tabular prediction: 50 credits per
million tokens, rounded up, with a 5-credit minimum per request.

`tokens = Σ over series (history rows + horizon) × (targets + covariates)`

```python file="forecast_sales.py"
import math

horizon, variates = 14, 2 + 1 + 2  # targets + past covariates + future covariates
tokens = sum((len(rows) + horizon) * variates for _, rows in panel.groupby("store"))
print(tokens, "tokens")  # 3 × (365 + 14) × 5 = 5,685
print(max(5, math.ceil(tokens * 50 / 1_000_000)), "credits")  # under the minimum, so 5
```

- Columns you don't name are not sent and not counted.
- Each `forecast` call is its own request. Step 6 costs a second one.
- Credits are reserved when the request is queued and settled when it finishes. Usage
  events have kind `forecast`; see [usage and credits](usage.md#usage-and-workspace-credits).
- Check current prices with `client.pricing()` or in **Workspace → Billing**. A usage
  event with `credits: null` was not priced.

## Prepare your data {#prepare-your-data}

Welt forecasts the table you send. It doesn't fill, resample or truncate anything.

- **Shape:** one row per series and time, in any order. A repeated (id, time) pair is
  refused.
- **Time:** a datetime column or ISO-8601 strings. Naive times, like the output of
  `pd.date_range` or `read_csv`, are read as UTC. Times with an offset are converted
  to UTC. Results are always UTC.
- **Daylight saving:** keep daily dates naive; with a local offset, days become 23
  or 25 hours apart in UTC and are refused. Hourly local-clock times without an
  offset have a missing and a repeated hour at each change; add the offset or convert
  to UTC first.
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

### Large tables and dataset histories {#dataset-histories}

The SDK sends DataFrames inline and never uploads them for you. A request body above
10 MiB (`welt.forecast.MAX_REQUEST_BYTES`) is refused with `InvalidInputError`
(`workload_limit`) before anything is sent. Upload such a table once, then pass its
dataset ID to `fit`, and to `forecast(future=...)` if the future table is large too:
`fit("ds_…", time="date", id="store", target=["sales"])`.

- Write times as ISO-8601 text: a CSV, or string columns in Parquet. Parquet
  timestamp columns are refused.
- Declare the id column `string` in the upload schema, for example
  `client.upload_file("sales.csv", schema={"columns": [...], "types": [...]})` or
  `welt uploads file sales.csv --schema schema.json`. Without it, a CSV ID such as
  `001` becomes the number 1 and no longer matches `"001"` in your future table.
- Only the server checks dataset rows, so fitted sizes such as `n_series_` are `None`.
  Use `forecast`; `predict` needs a local table.

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
| `InvalidInputError`, code `unsupported_capability` | The server has no `forecasting` task, but you asked for a panel, covariates, a dataset history, a calendar frequency or `forecast()`. Or the model doesn't offer what you asked for: a panel, `cross_learning=True`, or categorical covariates with several targets on [Timestral](#timestral). | Use the [univariate API](#forecast-a-univariate-time-series), or a current server. For Timestral, send one request per series, with `cross_learning=False`, and one target when you use categorical covariates. |
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

- A timeout or `job.close()` never cancels the forecast; `job.cancel()` does.
- The same key with the same request reconnects to the same job. The same key with a
  different request raises `ConflictError`.
- A reconnected job checks that the job and its result agree on task, model and
  configuration versions. It doesn't know your original request.
- `AsyncClient.forecast_job(id)` works the same way with `await`.

## Delete a forecast {#delete-a-forecast}

Forecast inputs and results stay private in your workspace until you delete them:

- `client.delete_forecast(job.id)` or `welt forecasts delete ID --yes` removes a
  finished forecast's input and result.
- The job, its replay key and its usage record stay. Deleting a dataset doesn't
  delete forecasts that read it, and deleting a forecast doesn't delete the dataset.

!!! tip "Backtest and reconcile in ergodic"
    Use Chronos-2 inside ergodic's backtest and reconcile:
    [Forecast with Welt's Chronos-2](https://docs.ergodic.dev/forecast-with-welt.html).

## Forecast with Timestral {#timestral}

Timestral is Mistral AI's forecasting model, in two sizes: `timestral-209m` (primary)
and `timestral-28m` (small). The catalogue lists both after Chronos-2. They use the same
`Forecast` calls, request, result and quantile levels as Chronos-2, with one
difference that matters most: **one series per request**.

```python file="forecast_timestral.py"
import numpy as np
import pandas as pd
from welt import Client, Forecast

# A year of simulated daily sales for one store, with planned Friday promotions.
rng = np.random.default_rng(0)
days = pd.date_range("2025-01-01", periods=365, freq="D")
north = pd.DataFrame({"date": days,
                      "sales": 120 * (1 + 0.3 * (days.dayofweek >= 5)) + rng.normal(0, 5, len(days))})
north["promo"] = (north["date"].dt.dayofweek == 4).astype(int)

model = Forecast(model="timestral-209m").fit(
    north, time="date", target=["sales"], future_covariates=["promo"])
ahead = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=14, freq="D")})
ahead["promo"] = (ahead["date"].dt.dayofweek == 4).astype(int)
result = model.forecast(14, future=ahead)
print(result.to_frame().head())

with Client() as client:
    timestral = client.model("timestral-209m")
print(timestral["forecast_envelopes"])  # the published single-series corners
print(timestral["attribution"]["notice"])  # 2026 - Mistral AI
```

A panel is refused before anything is sent. Fit and forecast each series on its own
instead:

```python file="forecast_timestral.py"
from welt import InvalidInputError

stores = pd.concat([north.assign(store="north"),
                    north.assign(store="south", sales=0.75 * north["sales"])], ignore_index=True)
try:
    Forecast(model="timestral-209m").fit(stores, time="date", id="store", target=["sales"])
except InvalidInputError as error:
    print(error.code)  # unsupported_capability: one series per request

forecasts = {}
for store, rows in stores.groupby("store"):
    single = Forecast(model="timestral-209m").fit(
        rows, time="date", id="store", target=["sales"], future_covariates=["promo"])
    forecasts[store] = single.forecast(14, future=ahead.assign(store=store))
print({store: r.series[0].predictions[:3] for store, r in forecasts.items()})
```

- **One series per request.** The model's forecasting configuration publishes
  `supports_panels: false`. `fit` raises `InvalidInputError` with code
  `unsupported_capability` for a history with more than one series. A dataset history
  is checked by the server, which refuses a panel with the same code. An `id` column
  that names a single series is fine. The reason: when Welt qualified Timestral, a
  series forecast inside a multi-series call didn't always match the same series
  forecast alone, so no panel corner is published.
- **No cross-learning.** `cross_learning=True` is refused (`unsupported_capability`;
  checked locally before submission from SDK 0.14.2).
- **Categorical covariates need one target.** With several targets, categorical
  covariates are refused (`unsupported_capability`; checked in `fit` from SDK
  0.14.2, including when history is a dataset ID). Forecast each target separately.
- **Longer histories.** A series can have up to 32,000 rows (the native context is
  32,768 steps); a longer one is refused, never truncated.
- **Non-negative targets stay non-negative.** Welt keeps the publisher's defaults:
  the forecast is averaged with the mirrored forecast of the negated history, and a
  target whose history is entirely non-negative is clipped at 0, so it never gets a
  negative quantile. The resolved configuration records both
  (`symmetric_averaging`, `clip_nonnegative`).
- **Same price.** A Timestral forecast is priced exactly like a Chronos-2 forecast
  (rate card `2026-10-v3`, the [same formula](#pricing)). Each series is its own
  request, so each pays the 5-credit minimum.
- SDK 0.14.0 also forecasts single series with Timestral, but reports a DataFrame
  panel as `workload_limit` (its local corner check); a panel given as a dataset ID is
  sent and gets the server's `unsupported_capability`. 0.14.1 refuses a DataFrame panel
  locally with `unsupported_capability` too.

At release, both sizes publish these three measured corners, single series only, and
every series needs at least 16 rows (`min_context`):

| Context (rows) | Variates | Series | Horizon |
| --- | --- | --- | --- |
| 32,000 | 10 | 1 | 1,024 |
| 16,000 | 20 | 1 | 1,024 |
| 8,000 | 50 | 1 | 1,024 |

The example above (365 rows × 2 variates × 1 series × 14 steps) fits the first. As for
Chronos-2, read the corners from `client.model("timestral-209m")` rather than copying
this table.

!!! note "Timestral attribution"
    Timestral weights: 2026 - Mistral AI. Licensed under the Modified MIT licence:
    [Timestral-209M licence](https://huggingface.co/mistralai/Timestral-209M/blob/9874d252ab0aae16f7f7c999f98f98d6ad7d8365/LICENSE),
    [Timestral-28M licence](https://huggingface.co/mistralai/Timestral-28M/blob/180698bbf5a50dda3516aa6419798437fb03a430/LICENSE).
    Model metadata carries the same notice and link as `attribution`.

## Forecast one series with the univariate API {#forecast-a-univariate-time-series}

The univariate API from SDK 0.11 stays supported: `predict`, `predict_quantiles`,
`predict_details` and `submit_predict`, for one series and one target. Use it with an
array, a Series with a `DatetimeIndex`, or a DataFrame with one string `target` and no
`id` or covariates.

- On a server with the `forecasting` task these calls are forecasts
  (`POST /v1/forecasts`): the published envelopes and the 21 native levels apply.
- On older servers they are univariate model runs: 16–512 observations, a horizon of
  1–64 and levels from `0.1, 0.2, …, 0.9`.
- Either way, `to_frame()` keeps the 0.13 layout: UTC-indexed rows with `median` and
  `q_<level>` columns.
- Panels and several targets raise `ValueError`; use `forecast()`.

!!! tip "Run it from a terminal"
    Download the [local CSV example](downloads/forecasting.py) and run
    `python forecasting.py history.csv --time timestamp --target value --horizon 24`
    with your own observations and configured workspace credentials.

### Prepare your history {#prepare-an-explicit-chronological-history}

This example reads a local CSV with `timestamp` and `value` columns. Keep later
observations aside for evaluation.

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

You can also fit an indexed series, or an array with a start time and spacing:

```python
forecast.fit(history.set_index("timestamp")["value"])
forecast.fit(history["value"].to_numpy(),
             start="2026-01-01T00:00:00Z", frequency_seconds=3600)
```

- Times must be timezone-aware; offsets convert to UTC. To read naive times as UTC
  and allow missing values, pass the target as a list, as in
  [step 1](#forecast-one-series).
- Values must be complete, finite numbers, strictly increasing in time and evenly
  spaced in whole seconds (1 second to 1 year). On a server with the `forecasting`
  task, calendar frequencies (months, quarters, years) and longer spacing are also
  accepted. Arrays need `start` and `frequency_seconds`.
- `fit` copies the values and uploads nothing. Changing the table afterwards doesn't
  change the fitted context; changing a constructor parameter needs a new `fit`.

### Choose one output for each run

Each call below is one separate request, starting one step after the last observation.
The default levels are `(0.1, 0.5, 0.9)`.

```python
points = forecast.predict(24)                         # ndarray (24,): native medians
quantiles = forecast.predict_quantiles(24)            # ndarray (24, 3)
result = forecast.predict_details(24)                 # immutable ForecastResult
frame = result.to_frame()                             # owned UTC-indexed DataFrame
wire = result.to_dict()                               # independent native JSON payload
```

Call `predict_details` once and read its `predictions`, `quantiles`,
`quantile_levels` and `timestamps` to get every view from one request. Quantiles are
not calibrated intervals. On the univariate model run, crossed quantiles raise
`InvalidForecastResultError` instead of being sorted.

### Keep the job and reconnect {#retain-durable-work-and-reconnect}

Keep the job ID before you wait. A timeout leaves the job running; reconnect instead
of submitting again.

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

- A job from the estimator owns its HTTP client: use `with` or `job.close()`. A
  reconnected job borrows the `Client` you pass.
- `AsyncClient.forecast_job(id)` returns an `AsyncForecastJob` to `await`.
- A reconnected job cannot reconstruct your original request, so it doesn't check the
  result against it.

### Version guards and privacy

`fit` pins the model and configuration versions, your workspace and credential. Every
request sends them as guard headers, so a changed model raises a `ConflictError`
before any work starts; fit again to accept it. `Client.submit_model_run` and
`Client.submit_forecast` accept the same pair as `expected_model_version` and
`expected_configuration_version`.

Delete with `client.delete_forecast(id)`. Only runs created on servers without the
`forecasting` task are model runs, deleted with `client.delete_model_run(id)`.

## Next steps

<div class="cards" markdown="1">

- [Forecasting notebook](notebook-forecasting-with-covariates.html): a retail panel with a holdout, a baseline, MASE and coverage.
- [Chronos-2 in ergodic](https://docs.ergodic.dev/forecast-with-welt.html): backtest against baselines and reconcile a hierarchy.
- [Forecast with Timestral](#timestral): one series per request, longer histories.
- [Models and limits](capabilities.md): every model Welt runs and its published limits.
- [Wait and recover](async-and-errors.md): typed errors and waiting.

</div>
