# Learn in a notebook

Work through Welt's workflows as notebooks on real research data. Start with
classification.

Each notebook downloads real research data, prints the table and explains what happens
next. The forecasting notebook is the exception: it simulates its store panel, because
it needs drivers known ahead of time. You don't need a helper file or a testing context. The first cell installs the
reader; the next cell connects your workspace.

For each notebook, choose **Open in Colab** to run it, download the `.ipynb` file for
Jupyter, or choose **View** to read its formatted cells here. All three use the same
notebook.

!!! caution "Notebooks use workspace credits"
    Modelling and research downloads may consume credits. Check **Workspace → Billing**
    for available credits and spend caps before you run the notebooks. See
    [usage and recovery](usage.md#usage-and-workspace-credits).

## [Classification](notebook-classification.html)

Download Iris, inspect it, split train and test rows, fit, predict, then read accuracy
and a per-class report.

{{actions:classification}}

## [Regression](notebook-regression.html)

Download Yacht, inspect it, split train and test rows, fit, predict, then read MAE,
RMSE and R².

{{actions:regression}}

## [Async jobs](notebook-async-jobs.html)

The Iris workflow with durable submission and async waiting.

{{actions:async-jobs}}

## [Artifacts and usage](notebook-artifacts-and-usage.html)

Evaluate Iris, reopen the same predictor and inspect its operation events.

{{actions:artifacts-and-usage}}

## [Compare models](notebook-evaluation.html)

Hold out Iris test rows, cross-validate models on the rest, read the leaderboard and the
report, then fit the chosen model and check it once on the test rows.

{{actions:evaluation}}

## [Research datasets](notebook-research-datasets.html)

Search the catalogue, inspect a fixed eligible asset and complete your first
classification.

{{actions:research-datasets}}

## [Forecasting with covariates](notebook-forecasting-with-covariates.html)

Simulate a five-store retail panel with promotions, holidays and temperature, hold out
the last four weeks, and score Chronos-2 against a seasonal-naive baseline with MASE,
80% interval coverage and a plot. Prefer a script? Download
[forecasting_covariates.py](downloads/forecasting_covariates.py).

{{actions:forecasting-with-covariates}}

## [Causal discovery](notebook-causal-discovery.html)

Inspect real Iris measurements and the graph structure Welt's native model proposes.

{{actions:causal-discovery}}

## Before you run

- Use a fresh Python 3.11–3.13 environment with `welt-client[parquet]`.
- Outputs in the notebook sources are cleared. Your own run shows actual data and
  measurements.
- Keep credentials in `WELT_API_KEY`, never in saved notebook cells.
- Every modelling call goes to Welt. The private training data, predictors and results
  you create persist.

!!! note "About the causal example"
    It has no ground-truth graph and makes no causal-accuracy claim. Native scores are
    not calibrated confidence. To continue in ergodic, install `welt-client[ergodic]` and
    call `result.to_ergodic()` (declared DAGs only); see
    [Convert to an Ergodic graph](causal-discovery.md#convert-to-an-ergodic-graph).

<details markdown="1"><summary>How are these notebooks checked?</summary>

Local release checks patch transport outside the public notebook and execute using
authorized SHA-pinned research files. Public CI validates sources and contracts
without downloading research bytes or using account credentials. Fixture scores never
measure model quality; actual hosted acceptance is recorded separately.

</details>

## Run in Colab

Choose **Open in Colab** and run the cells in order. Each notebook starts with:

```python
!pip install "welt-client[parquet]"
```

If Colab asks you to restart the session, restart it after installation. Then
continue with the connection cell: open its printed verification link and approve
your workspace. View pages let you read the same cells without running them.

## Next steps

<div class="cards" markdown="1">

- [Your first prediction](start.md): the same classification and regression steps as a page.
- [Use your own table](own-data.md): run the workflow on your data.
- [Connect your workspace](connect.md): connection, saved access and remote use.

</div>
