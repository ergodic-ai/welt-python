# Learn with real research data

Each notebook is standalone: install the SDK's `parquet` extra, approve your workspace using
the first `Client.connect()` cell, and run the cells in order. No support.py or mock context is
needed. Do not save keys or outputs into Git.

- Classification: real Iris, inspect, seed9 stratified120/30 split, fit, predict,
  held-out accuracy and per-class report.
- Regression: real Yacht, inspect, seed9 random246/62 split, fit, predict,
  held-out MAE/RMSE/R². Related experiments can make random-split scores optimistic.
- Async jobs: the Iris workflow with durable submission, a retained job ID and async waiting.
- Artifacts and usage: the Iris workflow, then reopen that fit and inspect operation metadata.
- Research datasets: search, select a fixed reviewed Iris asset, then the complete modelling workflow.
- Causal discovery: real Iris measurements, all150 rows/four continuous variables;
  native Arrow output inspection, no ground-truth causal accuracy or effect claim.

The Iris distribution is CC BY4.0 (R.A.Fisher1936, UCI DOI10.24432/C56C76), and
selected Yacht distribution is CC0. Programs print retained licence notices before
use. Recorded targets, column names and canonical bytes remain unchanged. Teaching
splits are explicit local choices, not source-provided benchmark splits. There is
no promised score, latency or calibrated interval. Each fit uses hosted resources;
uploads, predictors and causal results persist. Reopening does not refit.

Local release checks execute disposable notebook copies with test-only transport
patching and the same authorized SHA-pinned research bytes to check code/schema/
splits. Public CI validates standalone sources and SDK contracts without fetching
data or holding account credentials. Fixture outputs do not measure FM quality;
real hosted acceptance is separate.
