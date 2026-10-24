# Backtest a time series

Evaluate forecasting models at historical walk-forward origins, then optionally
forecast beyond the history. This needs a server supporting `/v1/backtests`.
Tabular `Client.evaluate()` and forward `Forecast.predict()` / `forecast()` keep
their existing behavior.

```python
import pandas as pd
from welt import Client

history = pd.read_csv("sales.csv")  # date, sales
with Client() as client:
    result = client.backtest(
        history, time="date", target="sales",
        models=["chronos-2", "timestral-209m"],
        folds=5, horizon=24, frequency="h", seasonal_period=24,
        final_horizon=24,   # omit for historical tests only
        max_credits=60,     # includes every fold and final call
    )
    print(result.to_frame())
    predictions = result.predictions()
    metrics = result.metrics()
    future = result.final_predictions()
    report = result.report_html()
```

Use explicit `chronos-2`, `timestral-209m` or `timestral-28m` model IDs. Every call
must fit that model's existing limits. Timestral accepts one series per request;
Chronos panels must share a complete time grid. Unsupported inputs are refused
before work is queued; no model is silently skipped or selected as a winner.

The latest N complete test windows each contain `horizon` observations. `step`
defaults to horizon; smaller steps overlap. Context expands by default; explicit
`context_length=512` uses a fixed rolling history and must fit the first origin.
No history is silently truncated. Long histories can use a rolling context that
fits the model's published limits. For T=1000, folds=3 and horizon/step=24, train
stops are 928, 952 and 976, with tests `[928:952]`, `[952:976]`, `[976:1000]`.
Optional final forecasts use history through 999, run for each requested model,
and never enter historical scores. Fold execution does not update pretrained weights.

## Dates or positions

Bare arrays and ordinary Series use integer positions, with no invented dates:

```python
with Client() as client:
    result = client.backtest(
        [float(i) for i in range(100)],
        models=["chronos-2"], folds=3, horizon=12,
    )
    print(result.predictions()[["cutoff", "index", "lead"]])
```

A Series/DataFrame with a DatetimeIndex automatically supplies its past index.
Alternatively pass arrays with `start="2026-01-01T00:00:00Z", frequency="h"`, or
name a DataFrame's `time` column. Future timestamps are generated using the existing
[forecasting frequency/calendar rules](forecasting.md). Array start times need
a timezone; naive table/DatetimeIndex times are UTC. Nonzero sub-microsecond
precision, duplicates, gaps and irregular spacing are refused. Nothing is filled
or resampled. Positional inputs have no frequency. Panels need equal grids/lengths.

Time metadata is separate from model features. Calendar feature generation is a
future opt-in extension needing an explicit local timezone. Current fixed daily
steps are 24 elapsed UTC hours, including across daylight-saving transitions.

## Covariates

Use `past_covariates`, `future_covariates` and `categorical` as in Forecast.fit.
Observed covariates stop at each origin. Only declared known-future columns from
the historical test window are passed ahead. Their declaration asserts they were
available at that origin: realized weather or revised measurements are not valid
known-ahead features merely because they appear in your table. Your preprocessing
must also respect cutoffs. Historical vintage/as-of snapshots are outside v1.

With `final_horizon` and known-future covariates, supply `future` containing exactly
the expected future time/ID and those covariate columns. For positional input its
ID/covariate row order defines future positions within each series; no time column
is supplied. Without a final forecast, no separate future table is needed.

## Metrics and results

MAE/RMSE score native medians. MASE uses your `seasonal_period` (default 1) and a
scale computed only from each fold's actual training context. Zero/unavailable
scales yield null with a reason. Frequency never implicitly selects seasonality.
Missing outcomes are excluded with eligible counts.

Pinball loss averages across requested quantile levels; it is not exact CRPS.
Coverage/width use the outermost requested levels with nominal coverage
`q_max-q_min`; coverage is descriptive, not a calibration claim. Naive and
seasonal-naive references use training values only and have no quantile scores.

`predictions()` returns model/fold/series/target, cutoff, index, original sorted
position, lead, training bounds, actual, median and quantiles. Overlapping folds
retain each origin/lead event. `metrics()` exposes overall, fold, series/target
and lead views; pooled metrics weight valid events equally. No independence or
significance test is assumed. Only models with every historical fold completed
enter the leaderboard; incomplete models retain diagnostics without being ranked.
Final forecasts are in `final_predictions()`, never historical scores.

`to_dict()` includes job/operation IDs, pinned versions, split hash,
protocol/metric versions and estimated/settled credits. Frames and dictionaries
are owned copies. `summary` is a locally held property; `to_frame()` builds the
pooled leaderboard without a remote call.

## Submit and reconnect

```python
with Client() as client:
    pending = client.submit_backtest(
        history, time="date", target="sales", models=["chronos-2"],
        folds=5, horizon=24, idempotency_key="sales-backtest-v1",
    )
    print(pending.id)  # retain before waiting
    result = client.reopen_backtest(pending.id).wait(timeout=600)
```

Local timeout never cancels server work. Reopen by ID. After an uncertain POST,
replay the identical request and idempotency key; the SDK reports that key after a
submission transport error. Do not create a new key for an uncertain submission.
`cancel()` stops unfinished children and preserves completed results/charges.
`delete()` removes a terminal group's retained data, never datasets or usage.

One bounded backtest counts as one workspace workload/rate-limited submission.
Each model/fold/final call contributes to lane load and ordinary forecast pricing.
`max_credits` refuses excessive estimates before submission. Orchestration bounds
are ten folds/eight models, 10 MiB plan, 50 MiB combined native inputs and 10,000
prediction events including references/final outputs, in addition to model limits.

Predictions, metric details and reports expire seven days after the last child
finishes; summary metadata remains. Partial success emits BacktestWarning;
complete failure/cancellation raises a typed error on wait. Metadata still shows
individual outcomes. There is no new public report-sharing link.

AsyncClient has the same submission/reconnect methods: await them and `wait()`,
`predictions()`, `final_predictions()`, `metrics()`, `report_html()` and mutations.
`to_frame()`, `to_dict()` and `summary` use locally held metadata.

A fitted Forecast exposes `model.backtest(horizon=24, folds=3)` with its copied
context and pinned versions, without changing that context. Its returned handle
owns only an estimator-created client: use `with model.backtest(...) as result:`
or call `result.close()` after reading. A supplied Forecast(client=...) stays open.
