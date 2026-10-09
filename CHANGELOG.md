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
