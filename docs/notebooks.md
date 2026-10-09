# Learn in a notebook

Start with classification. Each notebook downloads real research data, prints the
table, and explains what happens next. No helper file or testing context is needed.
Install the existing Parquet extra and configure your workspace key locally first.

| Notebook | What you learn |
| --- | --- |
| [Classification](downloads/classification.ipynb) | Iris → inspect → train/test → fit → predict → accuracy and per-class report |
| [Regression](downloads/regression.ipynb) | Yacht → inspect → train/test → fit → predict → MAE, RMSE and R² |
| [Async jobs](downloads/async-jobs.ipynb) | The Iris workflow with durable submission and async waiting |
| [Artifacts and usage](downloads/artifacts-and-usage.ipynb) | Evaluate Iris, reopen the same predictor and inspect operation events |
| [Research datasets](downloads/research-datasets.ipynb) | Search, inspect a fixed eligible asset and complete your first classification |
| [Causal discovery](downloads/causal-discovery.ipynb) | Inspect real Iris measurements and native proposed graph structure |

Use a fresh Python3.11–3.13 environment with `welt-client[parquet]`. Outputs in
notebook sources are cleared; your own run shows actual data and measurements.
Keep credentials in `WELT_API_KEY`, never in saved notebook cells. Every modelling
call goes to Welt; created private training data/predictors/results persist.

The causal example has no ground-truth graph and makes no causal-accuracy claim.
Native scores are not calibrated confidence; approved public Ergodic continuation
remains gated. Local release checks patch transport outside the public notebook and execute
using authorized SHA-pinned research files. Public CI validates sources/contracts
without downloading research bytes or using account credentials. Fixture scores
never measure model quality; actual hosted acceptance is separately recorded.
