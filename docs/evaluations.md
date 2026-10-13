# Compare models with cross-validation

An evaluation cross-validates one or more models on your table and ranks them. You
get a leaderboard, per-fold metrics, out-of-fold predictions and an HTML report you
can display in a notebook or share with a link.

!!! need "You'll need"
    - `welt-client>=0.12.0`.
    - A workspace key with write permission, and credits for the evaluation.
    - A table with a target column, within the limits of the models you compare.

## Run an evaluation

Pass a DataFrame and its target column. Welt uploads the table, assigns folds,
evaluates every eligible model and returns when the results are ready.

```python
import pandas as pd
from welt import Client

data = pd.read_csv("churn.csv")
with Client() as client:
    evaluation = client.evaluate(data, "churned", task="classification", seed=7)
    print(evaluation.summary())   # leaderboard: one row per model, then the baseline
    folds = evaluation.folds()    # long table: model_id, fold, rows, metric, value
    oof = evaluation.predictions()
    evaluation.show()             # in Jupyter, displays the report
```

The first argument can also be an existing dataset ID (`"ds_…"`, which already names
its target), a CSV or Parquet path, or a feature array with separate target values:
`client.evaluate(X, y, task="regression")`. A string ending in `.csv`, `.parquet` or
`.pq` is read as a file; any other string is a dataset ID.

`evaluate` waits up to `timeout` seconds (600 by default). The evaluation keeps
running if the wait stops; see [Wait, reconnect and cancel](#wait-reconnect-and-cancel).

## What each model sees

Welt assigns every row to one fold, once, from your `seed`. Every model is scored on
the same folds; `to_dict()["fold_sha256"]` identifies them. For each fold, the model
gets only the other folds as its context and predicts the held-out rows. **A row is
never predicted by a model that has seen it.** Fold contexts are discarded; they are
never saved as predictors.

!!! caution "The boundary covers Welt's folds, not your preparation"
    Anything you compute across the whole table before uploading, such as target
    encodings, scaling or features built from future rows, already contains the
    held-out answers. Prepare rows independently, or the scores will be optimistic.
    An ID column that tracks the target leaks the same way; drop it first.

## Choose models, or let Welt choose

Name up to 8 models with `models=["tabicl-v2", "tabdpt-1.3"]`. Each one must be able
to evaluate your table, otherwise the request is refused with that model's reason and
nothing runs.

Leave `models` empty and Welt evaluates every available catalogue model for the task
whose limits admit your data. Hidden research models are never evaluated in this
version. The others are listed with a reason in `evaluation.skipped`:

```python
for model in evaluation.skipped:
    print(model["model_id"], model["reason"], model["message"])
```

If no model can evaluate the table, submission raises `InvalidInputError` with
`code="no_eligible_models"`; `error.details["skipped"]` explains each model.

## Choose a split

`folds` sets the number of folds, from 2 to 10 (default 5). The `split` options are:

| `split` | Folds |
| --- | --- |
| `"auto"` (default) | Stratified for classification, shuffled k-fold for regression. |
| `"kfold"` | Shuffled k-fold. |
| `"stratified"` | Keeps class shares in every fold. Classification only. |
| `"group"` | With `group_column`: all rows of a group stay in one fold. |
| `"custom"` | With `fold_column`: each distinct value is one fold. `folds`, if given, must equal their count. |

Group and fold columns are removed from the features. The same seed and data give the
same folds. For classification, every class must appear in every training fold;
otherwise the request is refused before anything runs. Time-ordered splits are not
available in this version.

## Read the results

`summary()` makes no request. Its index is `model_id`, ranked models first, then the
baseline:

- `name` and `rank` (the baseline has no rank);
- `<metric>_mean` and `<metric>_sd` across folds, for every metric of the task.

Classification reports `roc_auc`, `log_loss`, `brier`, `accuracy`, `balanced_accuracy`,
`f1_macro` and `ece`. Regression reports `rmse`, `mae`, `r2`, `mape` (only when no
target is zero) and `median_ae`. Undefined values are `NaN`.

Models are ranked by `primary_metric`: ROC AUC for binary classification, log loss
for multiclass and RMSE for regression, unless you choose another. For binary tasks,
`positive_class` sets the class used by ROC, precision–recall and calibration; the
default is the second class in sorted order.

- The **baseline** predicts the training-fold class shares or mean. It is a reference
  point, not a catalogue model.
- `evaluation.comparison` compares the top two models fold by fold.
  `separated: False` means they are tied within fold-to-fold variation.
- `evaluation.warnings` notes small datasets, class imbalance, skipped models and
  models that don't beat the baseline.

`predictions()` returns one row per evaluated row:

- `row_index` is the row's position in the uploaded table, then `fold` and `actual`;
- one column per model (and `baseline`) with its out-of-fold prediction;
- for classification, `<model>:p(<class>)` probability columns. If labels of different
  types print alike (for example `1` and `"1"`), these columns use `repr()`:
  `p(1)` and `p('1')`.

!!! note "Predictions expire after seven days"
    After seven days `predictions()` raises `ResultExpiredError`. The summary, fold
    metrics and report stay until you delete the evaluation.

## Display the report in a notebook

`evaluation.show()` displays the report in Jupyter. An evaluation that is the last
expression of a cell also displays it while its client is open. The report appears in
a sandboxed frame: it can't run code, load anything or restyle your notebook. Scroll
inside the frame to read all of it. An unfinished evaluation shows its status instead.

`report_html()` returns the report as one self-contained HTML document. It contains
no script and loads nothing. Save it as UTF-8:

```python
from pathlib import Path

Path("report.html").write_text(evaluation.report_html(), encoding="utf-8")
```

## Share the report with a link

```python
link = evaluation.report_link(days=7)
print(link.url, link.expires_at)  # the URL is shown once; store it safely
```

!!! caution "Anyone with the link can read the report"
    The link opens the report without signing in, so share it like a password. It
    never opens the predictions, folds or other API data. The report itself includes
    the leaderboard, fold metrics and charts; for regression, the charts plot the
    actual and predicted values of up to 1,500 rows, which a reader can make out
    approximately. Unlike `predictions()`, the report is kept until you delete the
    evaluation.

A link lasts the chosen 1–7 days (default 7). A link created from the SDK with an API
key, or through an OAuth connection, never outlives that key and stops if the key is
revoked. A link created in the console lasts the chosen days, even after an ordinary sign-out; "sign out everywhere", a password reset or account disablement revoke every link you created in the console.
Every link stops working when it expires, when it is revoked, or when its creator is
no longer a verified member of the workspace. Deleting the evaluation removes the
report and its links.

`report_links()` lists the evaluation's links without their URLs, and
`revoke_report_link(link)` revokes one. Revoking needs a key with write permission; a
read-only key can create links but not revoke them. Printing a `ReportLink` hides its
URL, so it does not end up in saved notebook output by accident.

## Fit the model you chose

```python
model = evaluation.fit()          # the leaderboard's first model
model = evaluation.fit("tabdpt-1.3")
predictions = model.predict(new_rows)
```

`fit()` submits a **new fit** of that model on the full stored dataset, with the
evaluation's seed, and returns a fitted `Classifier` or `Regressor`. It doesn't
promote a fold context, and it is charged as a fit.

The fit uses the model's current version. If the catalogue has moved on since the
evaluation, `fit()` warns with an `EvaluationWarning` naming both versions; the
evaluation's scores then describe the earlier version, so evaluate again to compare
the current one. `model.model_version_` shows the fitted version.

- If the evaluation used a `group_column` or `fold_column`, `fit()` refuses, because
  the stored dataset still contains that column. Fit an estimator on a table without
  it instead.
- After a `sample_rows` evaluation, the fit uses every row of the dataset, which must
  be within the model's limits.

## Wait, reconnect and cancel

Submit without waiting to keep the evaluation ID first:

```python
from welt import Client, JobTimeoutError

with Client() as client:
    evaluation = client.submit_evaluation(data, "churned", task="classification",
                                          idempotency_key="churn-eval-0001")
    saved_id = evaluation.id  # retain before waiting
    print(saved_id, evaluation.estimated_credits)
    try:
        evaluation.wait(timeout=60)
    except JobTimeoutError:
        pass  # The evaluation keeps running. Reconnect later; don't resubmit.

with Client() as client:
    evaluation = client.evaluation(saved_id).wait()
```

- `wait()` raises `ExecutionError` when no model finished and `JobCancelledError`
  after cancellation. Some failed models give `partially_completed`: `wait()` returns
  normally with an `EvaluationWarning`, and `evaluation.models` explains each error.
- `cancel()` stops every unfinished model; `delete()` removes a finished evaluation's
  results, report and links. `client.evaluations()` lists the workspace's evaluations.
- An evaluation borrows its client. Read its results while the client is open, or
  reopen it with `client.evaluation(id)`.

!!! tip "Replay a submission with its key"
    The same `idempotency_key` with the same dataset and options returns the original
    evaluation instead of starting another. If the connection drops while submitting,
    the error names the dataset ID and key to replay with. A DataFrame is uploaded again
    on every call, so replay with the dataset ID from that message.

## Pricing and limits

An evaluation is one priced operation (usage `kind: evaluate`) on the fit and predict
rates of the active rate card:

- each model that produced out-of-fold predictions costs the fit rate on
  (folds − 1) × rows × columns, target included, plus the predict rate on
  rows × features;
- the total is rounded up once, with one five-credit minimum for the whole evaluation;
- models that failed cost nothing, and a failed or cancelled evaluation releases its
  whole reservation.

For example, on the current `2026-10-v2` card, 600 rows with 18 features plus the
target, 5 folds and 4 models come to 20.4, so 21 credits. Read the current rates with
`client.pricing()`, and see [usage and workspace credits](usage.md#usage-and-workspace-credits)
for balances and usage events.

`evaluation.estimated_credits` is reserved at submission, and `evaluation.credits` is
the settled charge once the evaluation finishes (`None` until then). `max_credits`
refuses an evaluation whose estimate is higher (`InvalidInputError`,
`code="budget_exceeded"`).

Each training fold must fit the model's [limits](capabilities.md): at most 500
training rows, 20 features and 10 classes for the current tabular models. With 5
folds, each training fold holds about four fifths of the rows. A larger table is
refused, never silently sampled. To evaluate on a subset, opt in with
`sample_rows=500`: Welt draws a seeded sample (stratified for classification) and the
report records it.

<details markdown="1"><summary>Can I evaluate from async code?</summary>

`AsyncClient` has the same `submit_evaluation`, `evaluate`, `evaluation` and
`evaluations` methods, returning `AsyncEvaluation`. Await its `wait()`, `refresh()`,
`folds()`, `predictions()`, `report_html()`, `report_link()`, `show()` and `fit()`;
`summary()` needs no await. In a notebook, an `AsyncEvaluation` displays its report
after you have awaited `show()` or `report_html()` once.

</details>

<details markdown="1"><summary>What if the server doesn't offer evaluations?</summary>

Evaluations need a Welt server release with the `/v1/evaluations` routes. On an older
server, `submit_evaluation`, `evaluation` and `evaluations` raise `NotFoundError` with
`code="evaluations_unavailable"`. If a table was already uploaded, the message names
its dataset ID so you can reuse or delete it.

</details>

## Next steps

<div class="cards" markdown="1">

- [Select a model](model-selection.md): readiness, swapping and fair comparisons.
- [Models and limits](capabilities.md): every qualified task and its limits.
- [Reuse a predictor](reuse.md): reopen the model you fitted.

</div>
