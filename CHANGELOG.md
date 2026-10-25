# 0.16.1 (2026-10-25) — Explicit fast Laplace context documentation

- Clarify Laplace's named 64-observation model lookback, single-scale configuration
  revision and effective-context provenance. Shared folds/actuals and full-fold
  MASE remain unchanged; final forecasts use the same preset and remain unscored.
- Retain the complete published 0.16 documentation and downloads when publishing
  the corrected guide. SDK methods, runtime dependencies and numerical behavior
  are unchanged; 0.16 can read the additive server metadata.

# 0.16.0 (2026-10-25) — Fast CPU backtest benchmarks

- Opt in with `benchmark_suite="fast-v1"`; sync/async waits include the bounded CPU
  suite on compatible servers. Native jobs, results, accounting and model limits
  retain their existing contracts.
- Added validated `benchmarks()`, `benchmark_predictions(final=...)`,
  `benchmark_scores()` and `feature_selection()` results. Optional
  `to_frame(include_benchmarks=True)` combines already-fetched complete scores.
- Documented the shared 20-second compute deadline, partial/unsupported outcomes,
  point/quantile semantics, training-only direct linear feature selection and
  covariate/calendar availability. SDK runtime dependencies are unchanged.
- API-201/202 and DEC-108 track qualification/publication/deployment separately.

# 0.15.0 (2026-10-24) — Walk-forward forecasting backtests

- Sync/async `Client.backtest`, `submit_backtest`, `reopen_backtest` and `backtests`
  add durable multi-model walk-forward evaluation, metrics, reference forecasts,
  optional final forecasts and authenticated HTML reports on compatible servers.
- `Forecast.backtest` / `submit_backtest` evaluate a copied fitted context with
  pinned model/configuration versions; existing forward methods stay compatible.
- Backtests accept bare positional arrays/Series, datetime-indexed data, explicit
  time-column tables and dataset references. They never invent public calendar
  dates for positional inputs, fill gaps or silently truncate model context.
- Typed Backtest/AsyncBacktest handles validate request identities, fold/job counts,
  result indices and quantile invariants. Invalid wait options are refused before
  submission; timeouts preserve backtest-specific reconnect instructions.
- Added the backtesting guide. Runtime dependencies are unchanged. API-194 /
  DEC-103 record qualification/release status separately from this source change.

- Guides, README, examples and notebooks create a `Client` with `client = Client()`
  (or `Client().connect()`) instead of wrapping one-off calls in `with Client() as
  client:`. The SDK guide explains when closing matters: long-running services and
  code that creates many clients. No SDK behavior changes (API-199).

# 0.14.2 (2026-10-24) — Forecast covariate capability checks

- `Forecast.fit` now honors the forecasting task's
  `supports_categorical_with_multiple_targets: false` flag locally, including
  dataset references, with `unsupported_capability` and no submitted job.
- `Forecast.forecast` and `submit_forecast` honor
  `supports_cross_learning: false` before reconnecting or submitting. Missing
  capability flags and older saved fitted routes preserve server validation.
- The covariate guide names all three supported model IDs, separates model limits,
  and recommends one target per request for Timestral categorical covariates.
- No model recipe, forecast result shape, runtime dependency or server contract
  changes. Low-level sync/async requests remain server-validated.

# 0.14.1 (2026-10-17) — Timestral forecasting

SDK side of the public Timestral release (API-174, DEC-088). `timestral-209m` (primary) and
`timestral-28m` (small) use the same `POST /v1/forecasts` contract, `Forecast` calls, result
and pricing as Chronos-2, and the catalogue lists them after Chronos-2. They forecast one series
per request.

## Changed

- **Panel refusal names the capability.** When a model's forecasting configuration publishes
  `supports_panels: false` (Timestral), `Forecast.fit` refuses a history with more than one
  series before any request with `InvalidInputError` code `unsupported_capability` and the hint
  to send one request per series. 0.14.0 refused the same panel as `workload_limit` ("reduce
  series"), because the single-series corners admit no panel. The field is read from the
  forecasting task profile like the envelopes; only an explicit `false` refuses, so Chronos-2,
  models without the field and older servers keep panels.
- Dataset histories (whose series only the server sees) and `welt forecast` send the request
  as before; the server refuses a panel with 422 `unsupported_capability`, raised as
  `InvalidInputError` with that code.

## Documentation

- Forecasting guide: [forecast with Timestral](https://ergodic-ai.github.io/welt-python/forecasting.html#timestral), with a
  one-series example, a panel sent as one request per series, the differences from Chronos-2
  (no cross-learning, categorical covariates need one target, histories up to 32,000 rows, the
  publisher's symmetric averaging and non-negative clipping), the three published
  single-series corners (32,000 × 10, 16,000 × 20 and 8,000 × 50 context × variates, horizon
  1,024, `min_context` 16), the price (the Chronos-2 rate, rate card `2026-10-v3`) and the
  attribution notice: "2026 - Mistral AI", Modified MIT licence, linked for both sizes.
- Capabilities, README, CLI and model-runs pages name Timestral and its one-series rule; the
  docs fixture serves both Timestral entries, and the docs tests pin their corners and
  attribution.
- Unchanged: the request and result schemas, the univariate API, runtime dependencies.

# 0.14.0 (2026-10-17) — Forecasting panels and covariates

SDK side of public Chronos-2 forecasting (DEC-085 §7, API-161). It needs a server that
advertises the `forecasting` task. Against older servers, univariate forecasting behaves as in
0.13; the remaining differences are listed under "Changed".

## Added

- `Forecast.fit` accepts a long-format DataFrame:
  - `time`;
  - an optional `id` (one series per ID; string or integer cells, rendered as strings in
    results);
  - `target` as a column name or a list (targets are forecast jointly);
  - `past_covariates`, `future_covariates` and `categorical` (a subset of the covariates;
    string or integer cells).
- Rows may come in any order. Timezone-naive timestamps (for example from `pd.date_range`
  without `tz` or `read_csv`) are read as UTC, and offsets are converted to UTC (DEC-085 §1),
  in the history and in the future table.
- Missing cells are sent as null (native missing-value handling) in numeric targets, numeric
  past covariates and categorical past covariates. Future covariates must be complete.
- These are refused before any request:
  - duplicate (id, time) pairs and gaps;
  - booleans, or text in numeric columns;
  - nonfinite or non-FP32 values;
  - incomplete future covariates;
  - a target that is missing for a whole series.
- `fit` uploads nothing.
- **Dataset histories:** `fit` also takes an existing workspace dataset ID (for example one
  created with `Client.upload_file`), or a dataset dictionary with `"id"`, together with
  `time=` and `target=`. `forecast(future=...)` takes a dataset ID as well.
  - Such rows travel by reference and only the server validates them. The fitted sizes and
    times (`n_series_`, `context_length_`, `series_ids_`, `cutoff_`) are `None`.
  - `predict()` needs a local history; use `forecast()` instead.
  - A file path is refused with a hint to read it or upload it.
- **Frequencies:** `frequency=` takes `{"unit", "every"}` (second, minute, hour, day, week,
  month, quarter, year) or an alias, optionally with a multiple such as `"15min"`:

  | Unit | Aliases |
  | --- | --- |
  | second | `s`, `S`, `sec` |
  | minute | `min`, `T` |
  | hour | `h`, `H` |
  | day | `D`, `d` |
  | week | `W` |
  | month | `M`, `MS`, `ME` |
  | quarter | `Q`, `QS`, `QE` |
  | year | `Y`, `YS`, `YE`, `A`, `AS` |

  The unit names themselves are also accepted.
  - Month, quarter and year use calendar arithmetic. Each series is anchored on a day of the
    month (1–28) or on month-ends, taken from its observations.
  - A frequency is sent in its largest whole unit: `{"unit": "hour", "every": 24}` is sent as
    one day.
  - Without `frequency=`, the spacing is checked locally (one frequency shared by every
    series) and the server infers it. Only the univariate form sends its fixed spacing
    explicitly.
- **New methods:** `Forecast.forecast(horizon, *, future=None, quantile_levels=(0.1, 0.5, 0.9),
  cross_learning=False, idempotency_key=None)` and `submit_forecast(...)` submit one guarded
  `POST /v1/forecasts` request.
  - `future` is required exactly when the fit named future covariates. It is either the
    known-future table (the time, id and future covariate columns, with exactly `horizon` rows
    per series at its next timestamps) or a dataset ID.
  - `quantile_levels` is any increasing subset of the levels the model publishes. When the
    model publishes none, the allowed levels are the 21 native levels 0.01, 0.05, 0.10, …,
    0.95, 0.99. The request default stays (0.1, 0.5, 0.9).
- **Limits** come from model metadata, never from SDK constants.
  - A request must fit inside one published `forecast_envelopes` corner: context, variates
    (targets + covariates), series and horizon.
  - Every series needs `min_context` observations.
  - Refusals raise `InvalidInputError` (`workload_limit`, a `ValueError`) and name the
    published corners.
  - Wherever the server publishes larger envelopes, this lifts the 512-observation and
    64-step univariate limits.
- **Transport:** DataFrame tables always travel inline in the request body.
  - A body above `welt.forecast.MAX_REQUEST_BYTES` (10 MiB of compact JSON, the Welt API's
    default HTTP body limit) is refused locally with `InvalidInputError` (`workload_limit`)
    before any request. Upload such a table as a dataset and pass its ID.
  - The constant follows the server's default limit. The server does not publish its
    limit, so a deployment that overrides `WELT_HTTP_BODY_BYTES` is not reflected: a lower
    override answers HTTP 413 (`payload_limit`), and a higher one is still refused locally
    above 10 MiB.
  - The SDK never uploads datasets on its own, so replaying a call with the same
    `idempotency_key` sends the same body and reconnects to the original job.
  - Requests carry `Idempotency-Key` and the paired `X-Welt-Model-Version` /
    `X-Welt-Configuration-Version` guards exactly like model runs (DEC-066).
- **`ForecastResult` also holds `forecasting` results.**
  - Every documented field is required and validated: finite values, sorted quantiles, a
    median equal to q_0.5, and timestamps stepping at the reported frequency.
  - The result is bound to the request: column roles, series order (first appearance of the
    ID, then target order), context lengths, horizon, levels, `cross_learning`, frequency and
    each series' future timestamps.
  - Additional fields the server may add later are ignored.
  - New properties: `task`, `series`, `rearranged_steps`, `horizon`, `frequency`,
    `cross_learning` and `columns`.
    - `series` is a tuple of `ForecastSeries`, with `id`, `target`, `context_length`,
      `timestamps`, `predictions` and `quantiles`.
    - `rearranged_steps` counts the steps whose crossing native quantiles the server sorted.
  - For long-format and dataset fits, `to_frame()` is long: `id`, `target`, `timestamp` (UTC),
    `median`, `q_<level>`.
- **Univariate fits keep the 0.13 result shape.** This covers an array, a Series, or a
  DataFrame with one string target and no id or covariates.
  - `predict_details()`, `forecast()` and `submit_predict()` results keep the
    `univariate_forecasting` `to_dict()` schema and the UTC-indexed `to_frame()`, on either
    route.
  - On the forecasting route, `frequency_seconds` is `None` for calendar frequencies.
    `rearranged_steps` stays available on the property.
  - The fitted `model_version_`/`configuration_version_` are the forecasting identity when
    that route serves the fit.
- `predict(horizon, future=None)` still returns a `(horizon,)` median array.
  `predict_quantiles`, `predict_details` and `submit_predict` accept `future`. For a panel or
  several targets they raise `ValueError` pointing to `forecast()`.
- `Forecast(client=...)` borrows an existing synchronous `welt.Client`, for example one with
  your own transport.
  - It cannot be combined with `base_url`, `api_key` or `workspace`.
  - It is never closed by the estimator or its jobs, and `sklearn.clone` copies share it.
  - A replacement client is checked at the next call: it must reach the same origin,
    credential and workspace.
  - Pickling drops it (credentials are never pickled). After unpickling a fitted estimator,
    set `estimator.client` again. A fitted estimator now pickles, with or without `client=`.
- **Client methods:**
  - `Client.submit_forecast` and `AsyncClient.submit_forecast` submit a raw `/v1/forecasts`
    body with optional paired version guards.
  - `Client.delete_forecast(job_id)` and `AsyncClient.delete_forecast` explicitly purge a
    terminal forecast's stored input and result (`DELETE /v1/forecasts/{id}`). Job metadata
    and usage remain.
  - `Job.result()` returns the forecasting result dictionary.
  - `forecast_job(id)` reconnects to either task with typed validation. A reconnect has no
    request context, so it returns the native `forecasting` result.
- **CLI:** `welt forecast --input request.json [--idempotency-key KEY]` mirrors `welt run`;
  `welt forecasts delete ID --yes` mirrors `welt model-runs delete`.
- New fitted attributes: `task_`, `n_series_`, `series_ids_`, `targets_` and `frequency_`.
  They join `context_length_` (the longest series), `frequency_seconds_` (None for calendar
  frequencies) and `cutoff_` (the latest observation).

## Changed

- **Route:** on servers with the `forecasting` task, single-series prediction uses
  `POST /v1/forecasts`. The admitted context, horizon and levels follow the published
  envelopes and levels, not 512/64 and 0.1–0.9.
- **Limits after metadata:** the univariate form still refuses fewer than 16 observations
  before any request. The upper context and horizon limits are checked after the metadata
  read and need a credential. They raise `InvalidInputError` (`workload_limit`), which is
  still a `ValueError`.
- **Univariate-only servers:** on a server without the `forecasting` task, these raise
  `InvalidInputError` (`unsupported_capability`, a `ValueError`) at fit instead of a plain
  `ValueError` before any request:
  - calendar-spaced (monthly, annual) history;
  - fixed spacing above one year;
  - panels;
  - covariates;
  - dataset histories.

  `forecast()` raises the same error there.
- **Unchanged (DEC-085 §8):** the legacy `univariate_forecasting` model-run request, result
  schema, limits and levels. Runtime dependencies are also unchanged.
- **ergodic extra:** `pip install "welt-client[ergodic]"` now admits ergodic 0.3
  (`ergodic>=0.2,<0.4`; ergodic 0.3 still reads and writes `ergodic.graph/1`, DEC-087). CI
  tests conversion against the oldest and latest ergodic in that range.

Documentation (API-162):

- The forecasting guide is rewritten as numbered steps: one series, a panel, several
  targets, past and known-future (numeric and categorical) covariates, quantiles and
  `to_frame()`, `cross_learning`, then limits read from model metadata, pricing,
  data preparation, errors and the job form. The univariate API section stays, with
  its examples unchanged. Its anchors still resolve.
- New notebook `forecasting-with-covariates` and script `forecasting_covariates.py`: a
  simulated five-store retail panel, a chronological 28-day holdout, a seasonal-naive
  baseline, a covariate-free Chronos-2 ablation (notebook), MASE, 80% interval coverage
  and a plot. It is the only notebook on
  simulated data, because it needs drivers known ahead of time.
- The Models page lists forecasting; the hidden model-runs page points to the guide
  and notes the priced `chronos-2` univariate run. The CLI page documents
  `welt forecasts delete ID --yes`. Links to ergodic's
  [Forecast with Welt's Chronos-2](https://docs.ergodic.dev/forecast-with-welt.html).
- The notebook fixture gains a `POST /v1/forecasts` wire fixture (seasonal-naive
  numbers, not model output) so the guide, script and notebook run in tests.

# 0.13.1 (2026-10-15) — ergodic documentation address

- The ergodic documentation moved to https://docs.ergodic.dev before it went live.
  The README, the causal discovery and Models pages, the product switcher and the
  `to_ergodic()` error message now link there; 0.13.0 linked the unassigned
  `ergodic.dev`. No API change. The 0.13.0 documentation was never published; this
  release's documentation replaces it.
- The Models table lists every accepted (rows × columns) shape as each model's
  context, not only the largest-row one, and the Models page defines context per task.
  The evaluation guide's fold limits are corrected from the old 500 × 20 to each
  model's shapes. Documentation only.

# 0.13.0 (2026-10-15) — ergodic 0.2 bridge

- `CausalResult.to_ergodic()` now checks the graph format the installed ergodic reads
  (`ergodic.GRAPH_FORMAT`, supported: `ergodic.graph/1`) instead of requiring
  ergodic 0.1.1, so it works with the public ergodic 0.2 release (DEC-078). The
  payload passed to `MixedGraph.from_dict` carries `"format"`. Conversion stays
  DAG-only, preserves every node and native direction, never projects, and keeps the
  typed errors: a missing ergodic, or one without a supported format, raises
  `OptionalDependencyError` naming the extra.
- Add the `ergodic` extra: `pip install "welt-client[ergodic]"` installs
  `ergodic>=0.2,<0.3`. CI tests conversion against the oldest and latest supported
  ergodic from PyPI once ergodic 0.2.0 is published.
- Documentation: the causal discovery and Models pages drop the private-repository and
  gated co-release wording and link the ergodic documentation at `ergodic.dev` (moved to docs.ergodic.dev in 0.13.1).
  Model limits in the guides, README and Models table now follow the measured
  envelopes live since API-137 (DEC-077): for example TabICLv2 and Kumo Medium up to
  8,000 × 20, Mitra-v2 4,000 × 20, TabDPT1.3 500 × 20, Arrow and CDFM 8,000 × 20,
  AVICI 1,000 × 20; 100 rows per direct prediction. The Models table snapshot is
  refreshed from the public catalogue.
  The version picker keeps the 0.12.0 documentation, rebuilt from its exact release
  commit with its published wheel. Model semantics, metrics and runtime dependencies
  are unchanged.
- Documentation cross-links with ergodic (API-145): a Products switcher in the top bar
  (Welt, Welt console, ergodic; styles in `docs/assets/site.css`, `docs.css`
  unchanged), the causal discovery guide's first Next-steps card goes to "Discover in
  Welt, estimate in ergodic", the Models page gains a "Compatibility with ergodic"
  section with the same version and conversion tables as the ergodic docs, and the
  SDK `llms.txt` links ergodic's.

# 0.12.0 (2026-10-14) — Cross-validated evaluations

- Add `Client.submit_evaluation`, `evaluate`, `evaluation(id)` and `evaluations()`
  (and the `AsyncClient` equivalents) for the `/v1/evaluations` resource (DEC-076).
  A dataset ID, DataFrame, array or CSV/Parquet table is accepted; tables are
  uploaded with their target. Options are checked before upload; empty `models` lets
  Welt choose. Submissions carry an `Idempotency-Key`; a dropped submission's error
  names the dataset ID and key to replay with. Mutations still never auto-retry. A
  server without evaluations raises `NotFoundError` (`evaluations_unavailable`).
- Add `Evaluation` and `AsyncEvaluation`: `wait`/`refresh` with the same polling
  backoff as jobs, `cancel`, `delete`, `summary()` (leaderboard with
  `<metric>_mean`/`_sd` and rank), `folds()` and `predictions()` DataFrames,
  `report_html()`, notebook display in a sandboxed iframe (no scripts, no network
  loads, no notebook restyling), settled `credits` and metric `runtime`, and an
  explicit `fit(model_id)` that submits a new full-data fit and warns if the fitted
  model version differs from the evaluated one. `partially_completed` returns with an
  `EvaluationWarning`; failed and cancelled evaluations raise `ExecutionError` /
  `JobCancelledError`. Probability columns use `repr()` labels when labels of
  different types print alike.
- Add `ReportLink` with `report_link(days)`, `report_links()` and
  `revoke_report_link()`; its repr hides the signed URL.
- `WeltError.details` exposes the server's structured, data-free error context (for
  example `budget_exceeded` estimates or `no_eligible_models` reasons), redacting the
  configured key. Existing errors keep their types and fields.
- Document evaluations (leakage boundary, model choice, splits, results, links,
  pricing and limits); add a local CSV example and an Iris evaluation notebook.
  The version picker keeps the 0.11.1 and 0.11.0 documentation, each rebuilt from
  its exact release commit with its published wheel. Runtime dependencies, existing
  model semantics and estimator behavior are unchanged.

# 0.11.1 (2026-10-14) — Package description and agent index

- Correct the package README's Forecast section, which was missing its text in 0.11.0
  and appears on the PyPI page.
- The documentation's `llms.txt` now points agents to the product-level index at
  https://welt.ergodic.dev/llms.txt for the hosted MCP server, the CLI and agent docs.
- Keep the 0.11.0 documentation as a retained version.
- Runtime behavior, dependencies, model semantics and evaluation protocols are
  unchanged from 0.11.0; only the declared version changes in the package.

# 0.11.0 (2026-10-13) — Forecast estimator and redesigned documentation

- Add zero-shot `Forecast.fit`, `predict`, `predict_quantiles`, `predict_details`
  and `submit_predict` with owned regular UTC history and frozen workspace/model/
  configuration identities. Fit creates no remote predictor or training job.
- Add immutable `ForecastResult`, typed durable `ForecastJob` and
  `AsyncForecastJob`, explicit cancellation, timeout/reconnect support and
  client-ownership rules. Native medians/quantiles are validated without repair.
- Add paired model/configuration expectation headers to sync/async low-level
  model-run submission; scientific JSON/digests and generic jobs remain compatible.
  The estimator requires the server guard capability; older servers fail closed.
- Document chronological inputs, shapes, provenance, retention and recovery.
  Dependencies, model weights/configurations and evaluation protocols are unchanged.
- Restyle the documentation on the Welt console's Graphite design system with
  self-hosted fonts, a system-following theme, ⌘K search, a reference class index
  with constructor parameter tables, previous/next navigation and a Models table
  rendered from a reviewed public-catalogue snapshot. Pages also publish their
  Markdown source and an `llms.txt` index. Every existing URL and anchor still
  resolves; older documentation versions keep their original presentation.

# 0.10.1 (2026-10-12) — Current documentation and runnable examples

- Correct the package README, causal CLI recipe, workspace credit guidance and
  upgrade instructions to match the current released client and service contract.
- Load current Iris data on the homepage; expose inherited scores and batch
  results in the reference, link search directly to methods and preserve old
  deep links when redirecting to current guides.
- Runtime behavior, dependencies, model semantics and evaluation protocols are
  unchanged. Historical packages and documentation remain available.

# 0.10.0 (2026-10-12) — Workspace-aware agent CLI and hidden model runs

- Workspace connection profiles and workspace-targeted browser/CLI login; complete
  authenticated resource commands from API-089/090.
- Explicit hidden model lookup, `submit_model_run`, terminal `delete_model_run`,
  synchronous/async Job.result support for one-shot tabular, causal and forecast
  results. CLI `welt run --input` and `welt model-runs delete --yes`.
- Main model listing stays unchanged; hidden availability and internal research
  access are decided by the API's qualified deployment and workspace rights.

# 0.9.1 (2026-10-11) — Research downloads up to the service bound

- `download_research_dataset` and `load_research_dataset` (`Client` and `AsyncClient`)
  default to `max_bytes=128 MiB`, the service's per-file research download bound, so
  every catalogue file downloads without an explicit limit (previously 64 MiB; the
  four largest catalogue files needed `max_bytes`). Explicit `max_bytes` behaves as
  before (up to 2 GiB); content larger than the limit is still refused before download.

# 0.9.0 (2026-10-10) — Research dataset metadata (API-078)

- `download_research_dataset(..., metadata=True)` also writes the catalogue detail
  to `<path>.metadata.json`, create-only after the verified data file, and returns a
  frozen `ResearchDownload` (path, metadata_path, version, sha256, metadata). Both
  destinations are refused before any request when either exists. The default
  `metadata=False` behavior and `pathlib.Path` return are unchanged.
- Add `load_research_dataset(dataset_id, *, version=None, metadata=False,
  max_bytes=64 MiB)` on `Client` and `AsyncClient`: verified download into a
  private temporary directory, read with pandas/pyarrow, returning a DataFrame or
  `(DataFrame, ResearchMetadata)`. Notice-bearing ZIPs must contain exactly
  `dataset.parquet`, `LICENSE.txt` and `ATTRIBUTION.txt`; notices are exposed as
  `license_text`/`attribution_text` with a `ResearchNoticeWarning`.
- Add the `research` extra (`welt-client[research]`: pandas, pyarrow). Missing
  dependencies raise `OptionalDependencyError` (an `ImportError`) naming it.
- `version=` on `download_research_dataset`/`load_research_dataset` accepts the
  current version or any version listed in the detail's `previous_content`; the SDK
  requests exactly that version and verifies it against that descriptor's size and
  SHA-256. Other versions still raise `research_version_mismatch` before any download.
- Notice ZIP tables are checked against `provenance.delivery.table_sha256` for a
  current restored-header archive, otherwise `provenance.canonical_sha256`.
- Detail fields such as `display_name`, `summary`, `tags`, column descriptions and
  `enrichment` are passed through unchanged; unknown future fields are tolerated.

# 0.8.0 — Explicit browser connection

- Add Client.connect(): browser approval, remote/headless verification link,
  process-only origin credentials by default, explicit private save=True.
- Add welt login CLI with optional no-browser/no-save, and exact-origin saved-key
  loading after explicit/environment credentials. Constructors never sign in.
- Simplify the homepage to title and numbered steps; show the training DataFrame
  before modelling. Keep the 0.7.1 route compatible with its immutable package.
- Preserve real research teaching recipes and existing model/task semantics.

# 0.7.1 — Real-data learning examples

Documentation-only point release; model, API and runtime behavior are unchanged.
Standalone examples/notebooks now download pinned Iris/Yacht research data, inspect
DataFrames, split held-out rows, fit, predict and measure actual quality. Test
transport stays exclusively in developer harnesses. Prior releases stay immutable.

# Changelog

## 0.7.0

- `Client`, `AsyncClient`, `Classifier` and `Regressor` now default to
  `https://welt.ergodic.dev`. An explicit `base_url` takes precedence over
  `WELT_BASE_URL`, which takes precedence over the hosted default. Local service
  users must pass `base_url="http://localhost:8080"` or set `WELT_BASE_URL` explicitly.
- Authenticated operations reject missing workspace keys before sending a request.
  Public model and research catalogue reads remain keyless. Authentication and
  permission errors explain workspace-key setup; current keys are redacted from
  server error messages and diagnostic attributes.
- `Classifier` and `Regressor` accept `fit(df, target="column")` and
  `submit_fit(df, target="column")`. Exactly one of `y` or `target` is required;
  targets require a direct pandas DataFrame with unique, nonempty string columns.
  The declared target is removed from features and transmitted as labels using the
  existing dataset contract. Separate `fit(X, y)`, array inputs, sklearn cloning,
  Pipeline/CV, immutable predictor versions and explicit task selection remain.
- Named prediction-schema mismatches report bounded missing/extra column names.
  Errors display only bounded opaque request/job identifiers.
- `Job.result(timeout=...)` and `AsyncJob.result(timeout=...)` account for polling,
  read retries and the final result read. Zero timeout sends no read. HTTP timeout
  phases are capped at the remaining local wait budget. Async requests have a
  cancellable total-await bound; synchronous transports cannot be forcibly
  interrupted, and separate HTTP phases can exceed a strict wall-clock bound.
  Elapsed time is checked after each read. This wait starts after submission and
  does not bound uploads or submission. Timeout preserves the durable job and
  supplies reconnect guidance using the same origin and current credentials.

No new large-file, resumable-upload or Parquet batch API is included in this release.
Existing in-memory CSV/Parquet estimator input behavior is retained.


## Unreleased — workspace and agent CLI source candidate

- Exact workspace login/selection, multiple origin-bound saved credentials and local logout.
- JSON CLI for existing modelling/resource/job/research operations; browser entry points for account controls.
- Sync/async resumable upload and batch manifest/payload methods; partial batch wait support.
- Workspace verification before private input and cross-origin request rejection.
- Login diagnostics move to stderr; successful CLI commands emit JSON on stdout.
