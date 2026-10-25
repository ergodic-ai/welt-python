# Backtest a time series

Evaluate forecasting models at historical walk-forward origins, then optionally
forecast beyond the history. This needs a server supporting `/v1/backtests`.
Tabular `Client.evaluate()` and forward `Forecast.predict()` / `forecast()` keep
their existing behavior.

```python
import pandas as pd
from welt import Client

history = pd.read_csv("sales.csv")  # date, sales
client = Client()
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
client = Client()
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

Native foundation-model time metadata is separate from model features. The
optional CPU linear benchmarks below generate calendar features in UTC only.
Local business timezones remain a future extension. Fixed daily steps are 24
elapsed UTC hours, including across daylight-saving transitions.

## Fast CPU benchmarks

SDK 0.16 adds an explicit optional suite on compatible servers:

```python
result = client.backtest(
    history, time="date", target="sales", frequency="h",
    models=["chronos-2"], folds=3, horizon=24,
    seasonal_period=24, context_length=168, final_horizon=24,
    benchmark_suite="fast-v1", max_credits=20,
)
print(result.to_frame(include_benchmarks=True))
cpu_predictions = result.benchmark_predictions()
cpu_future = result.benchmark_predictions(final=True)
selected_features = result.feature_selection()
diagnostics = result.benchmarks()
```

`fast-v1` contains Laplace (skaters 0.16.0), AutoTheta, AutoETS and MFLES
(StatsForecast 2.1.1), direct auto-feature Elastic Net and Ridge (scikit-learn
1.9.1), and drift. These are independent per-series/target CPU references; they
share **20 seconds of compute wall time for the entire suite**, including imports,
all folds and optional final forecasts. Queueing, HTTP and native GPU time are
separate. No additional benchmark credits are charged. A timeout preserves whole
completed units and reports unfinished/skipped units explicitly. Only models
with every historical unit completed enter the benchmark leaderboard.

Laplace accepts up to 512 training rows and 64 leads; the other benchmarks up to
4,096 rows and 64 leads. There are at most 32 fold/series/target units, and the
existing 10,000 combined prediction-event limit includes CPU output. These
bounds do not promise every admitted workload finishes before the shared budget.
The shared input/fold history is retained; model-specific lookbacks are explicit
configuration, not adaptive truncation at the time limit. Statistical models require finite observed
targets and at least two declared seasonal periods; unsupported units are skipped.

The fast Laplace configuration is **Laplace (season-aware, lookback 64)**, revision
`declared-seasonality-lookback64-v1` in the `fast-v1:laplace` family. It runs upstream
single-scale Laplace (`scales=[1]`, default CRPS/sticky/GPD-tail settings) on the
most recent `min(64, available_context_length)` target observations inside each
fold. When the declared seasonal period is greater than one, the publisher’s
seasonal-difference conjugation is applied and forecasts are inverted to target
levels. Two complete seasons must fit in the effective 64-point window; otherwise
that unit is unsupported. Period1 retains the identity transform. This explicit
seasonal preset can improve or worsen accuracy depending on the series; its
quantile intervals have not been qualified as calibrated confidence intervals.
The configured GPD tail splice requires 500 warm-up observations and never
activates inside this 64-point window. Beyond a seasonal cycle, the publisher
uses recovered means and independent Gaussian anchor variance as an approximation
to the joint forecast path.
Its parameters include `seasonality="declared-period"`; unit details record
`seasonal_period` and `transform`. Its parameters record `lookback=64`; per-unit details expose
`available_context_length`, `effective_context_length`, `effective_train_start`
and exclusive `effective_train_stop`. Event `train_start`/`train_stop` still
identify the shared available fold, and MASE uses that complete fold's training
scale for every model. Other models keep their supplied context. Final Laplace
forecasts use the same explicit preset and remain unscored. This is a fast
configured variant, not the publisher's full-context/multi-scale default.
Retained older configurations keep their original revision and parameters; inspect
`benchmarks()['metadata']['models']` and `benchmarks()['details']` for provenance.

Laplace publishes predictive medians and requested quantiles. AutoTheta/AutoETS/
MFLES publish their native point means; linear models publish regression point
estimates and drift publishes its deterministic path. Their interval metrics are
unavailable. All use the same outer folds, actuals, training-only MASE scale and
metric definitions. Final forecasts never enter backtest scores.

Laplace, AutoTheta, AutoETS and drift ignore covariates. MFLES uses known-future
numeric covariates. Elastic Net/Ridge use target lags, recent summaries/changes,
declared observed covariates at the origin, known-future covariates at the
destination, and UTC cyclical calendar features when actual timestamps exist.
Positional arrays get no calendar features. Each lead is fitted directly to its
future value; outputs are levels, not cumulative totals.

Elastic Net selects sparse coefficients; Ridge provides dense shrinkage. Three
fixed alphas use a chronological inner holdout with training labels ending before
validation origins. Imputation, category encoding and scaling fit on training
only, then refit on the fold's complete training window. Category vocabularies
retain at most 16 sorted training levels; other/unknown levels use all-zero
encoding. `benchmarks()['details']` records vocabulary, settings and coefficients;
`feature_selection()` provides one row per selected feature/fold/lead, including
final fits with a null fold. Coefficients use standardized inputs and describe
predictive associations, not causal effects.

`wait()` waits for both native and requested CPU work; local timeout never cancels
or resubmits. `finished` and `status` retain native-job semantics for compatibility.
Inspect `to_dict()['benchmarks']` or `benchmarks()['metadata']` for CPU status.
Partial CPU outcomes emit `BacktestWarning` while native results remain available.
`predictions()`, `metrics()` and default `to_frame()` keep their native contracts;
CPU results use the separate methods above. Async handles expose the same methods
with `await`; `to_frame()` remains local. Reports include already-retained CPU
scores without starting computation. Results expire with the parent after seven
days; cancellation also stops unfinished CPU work, and explicit parent deletion
removes its CPU artifacts. Reopen the existing ID to recover an uncertain read.

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
client = Client()
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

## Evaluation entry and shareable reports

The dashboard now lists tabular and time-series runs together under Evaluations.
Each forecasting entry opens a self-contained interactive report with historical
observations and date/position zoom. Zoom changes the view, not the held-out
scores or prediction exports. The Forward walk still scores only historical
out-of-sample forecasts; final forecasts remain separate and unscored.

The existing SDK `report_html()` continues to return HTML. To create a shareable
link with the current client’s generic request method:

```python
link = client.request(
    "POST", f"/v1/backtests/{result.id}/report-links",
    json={"expires_in_days": 7},
)
print(link["url"])  # only returned on creation; treat it as a secret
```

Anyone holding the link can read this report, including target history and
forecasts. It does not authorize model execution, other workspace data or raw API
endpoints. Private input hashes and categorical vocabulary values are excluded.
Links expire after 1–7 days and cannot outlive retained artifacts or the creating
key. List them with `GET /v1/backtests/{id}/report-links`; revoke one with
`DELETE /v1/backtests/{id}/report-links/{link_id}`. Revocation/expiry stops future
access, and cannot erase a report that somebody already downloaded.

Earlier retained Laplace runs preserve their recorded configuration. The raw
`single-scale-lookback64-v1` variant did not apply explicit seasonal differencing.
Inspect the returned manifest instead of assuming old results were recomputed.
No client method, runtime dependency, native score or charging contract changes
in this documentation patch.
