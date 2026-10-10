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
