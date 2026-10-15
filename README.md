# Welt Python SDK

From a DataFrame to a prediction. Use familiar `Classifier` and `Regressor`
estimators backed by durable predictors in your private Welt workspace.
Install `welt-client`, import `welt`. Python 3.11–3.13.

[Start here](https://ergodic-ai.github.io/welt-python/) ·
[Guides](https://ergodic-ai.github.io/welt-python/own-data.html) ·
[Reference](https://ergodic-ai.github.io/welt-python/reference.html) ·
[Models and limits](https://ergodic-ai.github.io/welt-python/capabilities.html)

## Install

```sh
python -m pip install "welt-client[parquet]"
```

Run `welt login` and approve your workspace in the browser. This saves a private,
origin-bound credential for later Python sessions. In a notebook, connect once:

```python
from welt import Client

with Client() as client:
    client.connect()
```

Open its printed verification link on your own computer if Python is remote.
Python connection stays in this process; use `save=True` explicitly to persist it.
Constructors never open a browser. Explicit keys and `WELT_API_KEY` still work and
have precedence over connected/saved credentials. [Connection guide](https://ergodic-ai.github.io/welt-python/connect.html).

Modelling and research downloads may consume workspace credits. Check **Workspace
→ Billing** for available credits, pending reservations and spend caps before
running the examples. [Usage and recovery](https://ergodic-ai.github.io/welt-python/usage.html#usage-and-workspace-credits)
explains accounting; payments and account changes require browser authorization.

## Learn with real data

Download Iris from our research inventory, look at the table, split train/test,
fit a classifier, predict unseen rows, and measure the result:

```python
from pathlib import Path
from tempfile import mkdtemp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from welt import Client, Classifier

# 1. Get a real table from Welt's research inventory.
dataset_id = "asset-d839644b6de36fc5cafdda18c5daeca3"
path = Path(mkdtemp(prefix="welt-iris-")) / "iris.parquet"
with Client() as client:
    info = client.research_dataset(dataset_id)
    print("Dataset:", info["name"])
    client.download_research_dataset(dataset_id, path)
data = pd.read_parquet(path)
target = "target"  # The outcome column in this table.

# 2. Look at the rows before modelling.
print(data.head())
print("Rows and columns:", data.shape)
data.info()
print(data[target].value_counts())

# 3. Set aside test rows before fitting; never train on their answers.
train, test = train_test_split(
    data, test_size=0.2, random_state=9, stratify=data[target])
query = test.drop(columns=target)
y_test = test[target]
print("Training rows:", len(train), "— Test rows:", len(test))

# 4. Create a model and train it on the training table.
model = Classifier(model="tabicl-v2", random_state=9)
model.fit(train, target=target)

# 5. Predict the held-out rows, without passing their target column.
predictions = model.predict(query)
print(pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predictions}).head())

# 6. Compare predictions with the answers we kept aside.
print("Test accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
print("Keep this predictor ID:", model.predictor_id_)
```

Accuracy and per-class precision/recall/F1 describe your held-out run, not a
promised benchmark score. [Start](https://ergodic-ai.github.io/welt-python/start.html) explains each step and teaches
numeric predictions with real Yacht data. `fit` uploads only your training rows;
your workspace retains the resulting dataset and predictor for reuse.

### Research metadata and one-step loading

Keep the catalogue detail (schema, column descriptions and provenance)
beside a download with `metadata=True`, or load a table directly with the
`research` extra (`pip install "welt-client[research]"`):

```python
with Client() as client:
    saved = client.download_research_dataset(
        dataset_id, path.with_name("iris-2.parquet"), metadata=True)
    # saved.path, saved.metadata_path ("iris-2.parquet.metadata.json"), saved.metadata
    data, meta = client.load_research_dataset(dataset_id, metadata=True)
print(meta.name, meta.column_descriptions)
```

Without `metadata=True` the download returns its local path. Omitting `version`
selects the currently published file; the SDK verifies its size and checksum.
`load_research_dataset` removes its temporary file after loading. See
[Research datasets](https://ergodic-ai.github.io/welt-python/research-datasets.html).

The released client includes sync/async resources, resumable file uploads, durable
batch jobs, workspace selection and native causal objects. Estimator CSV/Parquet
convenience reads load files into memory; `Client.upload_file` uses upload sessions
and supports recovery with a retained upload ID. Available formats, models, tasks
and limits depend on the connected server. Inspect its catalogue before executing.
Each TabICLv2, Kumo Tabular Medium, TabDPT 1.3 and Mitra-v2 classification and regression
profile has measured shapes: TabICLv2 and Kumo Tabular Medium up to 8,000 training rows × 20
features (or 4,000 × 40, 2,000 × 80), Mitra-v2 up to 4,000 × 20, TabDPT 1.3 up to 500 × 20.
Direct predictions take up to 100 rows per call, and classification up to 10 classes.
`Client().models()` reports the current limits. [Capabilities](https://ergodic-ai.github.io/welt-python/capabilities.html) explains scope.


A local wait timeout leaves server work running. Reconnect by its job ID rather
than fitting again; [recovery](https://ergodic-ai.github.io/welt-python/async-and-errors.html) describes typed failures,
authentication, cancellation and retry behavior. Probabilities are not calibrated
coverage. ArrowFM, CDFM and AVICI discovery preserve native graph and score semantics;
[causal conversion](https://ergodic-ai.github.io/welt-python/causal-discovery.html) is restricted to declared native DAGs
and needs the `welt-client[ergodic]` extra; [ergodic](https://docs.ergodic.dev) then identifies
and estimates effects locally. The complete all-model catalogue remains gated.

## Examples and development

[Runnable script](https://ergodic-ai.github.io/welt-python/downloads/first_prediction.py) · [Jupyter notebooks](https://ergodic-ai.github.io/welt-python/notebooks.html) ·
[Research datasets](https://ergodic-ai.github.io/welt-python/research-datasets.html) · [Contributing](https://github.com/ergodic-ai/welt-python/blob/main/CONTRIBUTING.md)

```sh
uv sync --frozen
uv run --frozen pytest -q
uv build
```

Offline tests patch HTTP transport outside user examples; they neither contact the
hosted service nor measure foundation-model quality. Notebook sources have cleared outputs; clean-wheel execution and
bounded authorized live acceptance are separate evidence. CI requires no private
backend, credentials or checkpoints.

Agent workflows, workspace selection and the released command reference: [CLI](https://ergodic-ai.github.io/welt-python/cli.html).

## Explicit hidden-model runs

`client.model("chronos-2")` inspects an explicit hidden ID while
`client.models()` keeps the main catalogue. `client.submit_model_run(payload,
idempotency_key="your-key")` returns a durable Job; `job.result()` returns the
native task result. AsyncClient has matching methods. The payload follows the
API's tabular support/query, causal table or univariate forecasting contract;
check the model metadata for enabled tasks, limits, readiness and workspace access.

```python
from welt import Client
with Client() as client:
    job = client.submit_model_run({
        "model_id": "chronos-2", "task": "univariate_forecasting",
        "series": {"values": [float(i) for i in range(48)],
                   "start": "2026-01-01T00:00:00Z", "frequency_seconds": 3600},
        "horizon": 12,
    }, idempotency_key="forecast-example-1")
    result = job.result()
    print(result["predictions"])
```

CLI equivalents: `welt models get chronos-2`, `welt run --input request.json`,
`welt jobs wait JOB_ID`. Use `welt model-runs delete JOB_ID --yes` or
`client.delete_model_run(id)` to explicitly remove a terminal run's input/result.
Deleting keeps the idempotency/usage record; a new run needs a fresh key. Native
model libraries remain server-side. Restricted models need an internal testing
workspace; hidden status alone does not grant execution access. TabSTAR's trained
state stays alive for the single run, and these runs do not publish reusable
support-only fitted predictors. LimiX query splitting can change its transductive
results. API errors, cancellation and timeouts retain existing typed semantics.

## Compare models with cross-validation

`client.evaluate(data, "target", task="classification")` cross-validates every
eligible model (or up to 8 you name with `models=[...]`) on the same seeded folds and
returns an `Evaluation`: `summary()` is the leaderboard against a baseline, `folds()`
and `predictions()` are DataFrames, `show()` displays the HTML report in Jupyter and
`report_link()` creates a signed link anyone holding it can open for up to seven days.
`evaluation.fit()` then fits the chosen model on the full dataset. It requires SDK 0.12
or later and a server release with evaluations; an older server raises `NotFoundError` with
`code="evaluations_unavailable"`. [Evaluations guide](https://ergodic-ai.github.io/welt-python/evaluations.html).

## Forecast a time series

`Forecast` snapshots regular timezone-aware history and submits guarded zero-shot
Chronos-2 runs. It requires SDK 0.11 or later and the matching server capability.

Supply your own 16–512 finite observations and a horizon of 1–64. Fit uploads no history;
each prediction call creates one run. Native quantiles are not calibrated
intervals. [Forecast guide](https://ergodic-ai.github.io/welt-python/forecasting.html).
