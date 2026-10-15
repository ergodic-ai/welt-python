# 0.14.0 (unreleased) — Forecasting panels and covariates

SDK side of public Chronos-2 forecasting (DEC-085 §7, API-161). Needs a server that
advertises the `forecasting` task; against older servers the 0.13 univariate behaviour
is unchanged.

- `Forecast.fit` accepts a long-format DataFrame: `time`, optional `id` (one series per
  ID; string or integer cells, rendered as strings in results), `target` as a column
  name or a list (targets are forecast jointly), `past_covariates`,
  `future_covariates` and `categorical` (a subset of the covariates; string or integer
  cells). Rows may come in any order. Timezone-naive timestamps (for example from
  `pd.date_range` without `tz` or `read_csv`) are read as UTC and offsets are converted
  to UTC (DEC-085 §1), in the history and the future table. Duplicate (id, time) pairs,
  gaps, booleans, text in numeric columns, nonfinite or non-FP32 values,
  incomplete future covariates or categories, and a target missing for a whole series
  are refused before any request. Missing target and numeric past-covariate cells are
  sent as null (native missing-value handling). The fit uploads nothing.
- `frequency=` takes `{"unit", "every"}` (second, minute, hour, day, week, month,
  quarter, year) or an alias, optionally with a multiple such as `"15min"`:
  `s`/`S`/`sec` second, `min`/`T` minute, `h`/`H` hour, `D`/`d` day, `W` week,
  `M`/`MS`/`ME` month, `Q`/`QS`/`QE` quarter, `Y`/`YS`/`YE`/`A`/`AS` year (also the
  unit names). Month, quarter and year use calendar arithmetic anchored on a day of
  the month (1–28) or on month-ends, taken from the observations. It is sent in its
  largest whole unit (`{"unit": "hour", "every": 24}` is sent as one day). Without it
  the spacing is checked locally (one frequency shared by every series) and the
  server infers it; a single regular series sends its spacing explicitly.
- New `Forecast.forecast(horizon, *, future=None, quantile_levels=(0.1, 0.5, 0.9),
  cross_learning=False, idempotency_key=None)` and `submit_forecast(...)` submit one
  guarded `POST /v1/forecasts` request. `future` is the known-future table (time, id
  and future covariates; exactly `horizon` rows per series at its next timestamps),
  required exactly when the fit named future covariates. Quantile levels are any
  increasing subset of the levels the model publishes (default the 21 native levels
  0.01, 0.05, 0.10, …, 0.95, 0.99).
- Limits come from model metadata, never from SDK constants: a request must be
  dominated by one published `forecast_envelopes` corner (context, variates = targets +
  covariates, series, horizon) and every series needs `min_context` observations.
  Refusals raise `InvalidInputError` (`workload_limit`, a `ValueError`) naming the
  published corners. This also lifts the 512-observation/64-step univariate limits
  wherever the server publishes larger envelopes.
- Transport: the history and future tables are sent inline when each is at most
  `welt.forecast.INLINE_TABLE_BYTES` (1 MiB of compact JSON); a larger table is first
  uploaded as a workspace dataset (`Client.upload`) and sent by ID. Uploaded datasets
  remain in the workspace and are listed on `job.datasets`. A transport failure after an
  upload names the datasets so the same body and `idempotency_key` can be replayed.
  Requests carry `Idempotency-Key` and paired `X-Welt-Model-Version` /
  `X-Welt-Configuration-Version` guards exactly like model runs (DEC-066).
- `ForecastResult` also holds `forecasting` results, validated strictly (exact fields,
  finite values, sorted quantiles, median equal to q_0.5, timestamps stepping at the
  reported frequency) and against the request (column roles, series order by first ID
  appearance then target order, context lengths, horizon, levels, cross_learning,
  frequency and each series' future timestamps). New properties: `task`, `series`
  (`ForecastSeries` with `id`, `target`, `context_length`, `timestamps`, `predictions`,
  `quantiles`), `rearranged_steps` (steps whose crossing native quantiles the server
  sorted; 0 for legacy results), `horizon`, `frequency`, `cross_learning`, `columns`.
  For `forecasting` results `to_frame()` is long: `id`, `target`, `timestamp` (UTC),
  `median`, `q_<level>`. `timestamps`, `predictions` and `quantiles` remain for a single
  series and target.
- `predict(horizon, future=None)` still returns a `(horizon,)` median array, and
  `predict_quantiles`/`predict_details`/`submit_predict` accept `future`; for a panel or
  several targets they raise `ValueError` pointing to `forecast()`. On servers with the
  `forecasting` task, single-series prediction uses `POST /v1/forecasts` and returns a
  `forecasting` result, so `predict_details().to_frame()` is in the long layout there;
  on older servers it keeps the legacy model run and layout. `forecast()` needs the
  `forecasting` task and raises `WeltError` (`unsupported_capability`) otherwise; panels,
  covariates and calendar frequencies are refused the same way at fit.
- The univariate form (an array, a Series, or a DataFrame with one string target and no
  id or covariates) keeps the 0.13 validation: ordered, complete, finite FP32 history.
  Its timestamps must still be timezone-aware. Pass the target as a list for
  long-format semantics, including missing values and naive times read as UTC. The 16-observation minimum
  is now the server's published `min_context`, read at fit.
- `Client.submit_forecast` and `AsyncClient.submit_forecast` submit a raw
  `/v1/forecasts` body with optional paired version guards; `Job.result()` returns its
  result dictionary and `forecast_job(id)` reconnects to either task with typed
  validation. `welt forecast --input request.json [--idempotency-key KEY]` mirrors
  `welt run`.
- `Forecast(client=...)` borrows an existing synchronous `welt.Client` (for example one
  with your own transport) instead of opening a client per call. It is mutually
  exclusive with `base_url`, `api_key` and `workspace`, is never closed by the
  estimator or its jobs, and is shared by `sklearn.clone` copies; replacing it requires
  a new fit. Without it the estimator opens and closes its own clients as before.
- Fitted attributes: `task_`, `n_series_`, `series_ids_`, `targets_`, `frequency_` join
  `context_length_` (the longest series), `frequency_seconds_` (None for calendar
  frequencies) and `cutoff_` (latest observation).
- The legacy `univariate_forecasting` model-run request, result schema, limits and
  levels are unchanged (DEC-085 §8). Runtime dependencies are unchanged.

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
