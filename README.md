# Welt Python SDK

From a DataFrame to a prediction. Use familiar `Classifier` and `Regressor`
estimators backed by durable predictors in your private Welt workspace.
Install `welt-client`, import `welt`. Python 3.11–3.13.

[Start here](https://ergodic-ai.github.io/welt-python/) ·
[Guides](https://ergodic-ai.github.io/welt-python/v0.8.0/own-data.html) ·
[Reference](https://ergodic-ai.github.io/welt-python/v0.8.0/reference.html) ·
[Models and limits](docs/capabilities.md)

## Install

```sh
python -m pip install "welt-client[parquet]==0.8.0"
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
have precedence over connected/saved credentials. [Connection guide](docs/connect.md).

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
version = "d839644b6de36fc5cafdda18c5daeca3284a5ce8bb6169aff2fcb3fbb4a12021"
path = Path(mkdtemp(prefix="welt-iris-")) / "iris.parquet"
with Client() as client:
    info = client.research_dataset(dataset_id)
    print("Dataset:", info["name"], "— License:", info["license"])
    for notice in info["license_notices"]:
        print(notice)
    client.download_research_dataset(dataset_id, path, version=version)
data = pd.read_parquet(path)
target = "__target__#0"  # The target recorded in this research snapshot.

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
promised benchmark score. [Start](docs/start.md) explains each step and teaches
numeric predictions with real Yacht data. `fit` uploads only your training rows;
your workspace retains the resulting dataset and predictor for reuse.

SDK 0.7 adds explicit DataFrame target convenience and hosted credential guidance
while preserving `fit(X, y)`, sync/async resources, durable jobs, research downloads
and native causal objects. Available hosted tasks and workers follow the service
catalogue. Bounded TabICLv2, Kumo Medium, TabDPT1.3 and Mitra-v2 classification and
regression previews are separately qualified: 500 training rows, 20 features,
100 query rows, and 10 classes for classification. Other tasks need separate
qualification. File reads load into memory. Held resumable CSV and large batch
APIs are excluded.

A local wait timeout leaves server work running. Reconnect by its job ID rather
than fitting again; [recovery](docs/async-and-errors.md) describes typed failures,
authentication, cancellation and retry behavior. Probabilities are not calibrated
coverage. ArrowFM, CDFM and AVICI discovery preserve native graph and score semantics;
[causal conversion](docs/causal-discovery.md) is restricted to declared native DAGs
and the separately approved Ergodic co-release artifact. Public Ergodic distribution
and the complete all-model catalogue remain gated.

## Examples and development

[Runnable script](examples/first_prediction.py) · [Jupyter notebooks](docs/notebooks.md) ·
[Research datasets](docs/research-datasets.md) · [Contributing](CONTRIBUTING.md)

```sh
uv sync --frozen
uv run --frozen pytest -q
uv build
```

Offline tests patch HTTP transport outside user examples; they neither contact the
hosted service nor measure foundation-model quality. Notebook sources have cleared outputs; clean-wheel execution and
bounded authorized live acceptance are separate evidence. CI requires no private
backend, credentials or checkpoints.

Apache 2.0: see [LICENSE](LICENSE). Model weights and hosted service are separate
from this SDK licence.
