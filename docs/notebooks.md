# Learn in a notebook

Start with classification. Each notebook downloads real research data, prints the
table, and explains what happens next. No helper file or testing context is needed.
The first cell installs the reader; the next cell connects your workspace.

Open in Colab to run an example, download the `.ipynb` file for Jupyter,
or choose View to read its formatted cells here. Each action uses the same notebook.

## [Classification](notebook-classification.html)

Iris → inspect → train/test → fit → predict → accuracy and per-class report.

{{actions:classification}}

## [Regression](notebook-regression.html)

Yacht → inspect → train/test → fit → predict → MAE, RMSE and R².

{{actions:regression}}

## [Async jobs](notebook-async-jobs.html)

The Iris workflow with durable submission and async waiting.

{{actions:async-jobs}}

## [Artifacts and usage](notebook-artifacts-and-usage.html)

Evaluate Iris, reopen the same predictor and inspect operation events.

{{actions:artifacts-and-usage}}

## [Research datasets](notebook-research-datasets.html)

Search, inspect a fixed eligible asset and complete your first classification.

{{actions:research-datasets}}

## [Causal discovery](notebook-causal-discovery.html)

Inspect real Iris measurements and native proposed graph structure.

{{actions:causal-discovery}}

## Run in Colab

Choose **Open in Colab** and run the cells in order. Each notebook starts with:

```python
!pip install "welt-client[parquet]"
```

If Colab asks you to restart the session, restart it after installation, then
continue with the connection cell. Open its printed verification link and approve
your workspace. View pages let you read the same cells without running them.

Use a fresh Python3.11–3.13 environment with `welt-client[parquet]`. Outputs in
notebook sources are cleared; your own run shows actual data and measurements.
Keep credentials in `WELT_API_KEY`, never in saved notebook cells. Every modelling
call goes to Welt; created private training data/predictors/results persist.
Check **Workspace → Billing** for available credits and spend caps before running
the notebooks. Modelling and research downloads may consume credits; see
[usage and recovery](usage.md#usage-and-workspace-credits).

The causal example has no ground-truth graph and makes no causal-accuracy claim.
Native scores are not calibrated confidence; approved public Ergodic continuation
remains gated. Local release checks patch transport outside the public notebook and execute
using authorized SHA-pinned research files. Public CI validates sources/contracts
without downloading research bytes or using account credentials. Fixture scores
never measure model quality; actual hosted acceptance is separately recorded.
