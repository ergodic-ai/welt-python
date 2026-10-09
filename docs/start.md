# Your first prediction

Let's predict a flower's class from its measurements. We will download real Iris
observations from Welt's research inventory, inspect them, hold some back, and see
how well a classifier predicts those unseen rows.

## Install

Python 3.11–3.13. Use a virtual environment, then install the SDK and its Parquet reader:

```sh
python -m pip install "welt-client[parquet]==0.8.0"
```

## Connect your workspace

In a terminal, run `welt login` and approve your workspace in the browser. This
saves an origin-bound credential privately on this computer. For a notebook,
connect explicitly in Python instead:

```python
from welt import Client

with Client() as client:
    client.connect()
```

Open the printed verification link on your own computer if Python is running on
a remote server. No callback to localhost is required. Python connection keeps
access only in this process by default, so the next `Classifier()` works without
copying a key. Use `client.connect(save=True)` only if you want it saved for future
processes. [Connection details](connect.md) explains expiry and key precedence.
Manual `WELT_API_KEY` configuration remains supported; never save a secret in code.

## Classification: predict a flower's class

Iris has 150 rows, four measurements and three encoded classes. The canonical
snapshot calls its target `__target__#0`; keep that name rather than guessing from
a column's position. This is the eligible p10k/PMLB Iris asset, attributed to
R. A. Fisher (1936), UCI Iris DOI10.24432/C56C76, CC BY4.0.

Run the complete example. The numbered comments follow the six steps:

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

You now have 120 training rows and 30 test rows. The same seed makes the split
repeatable; stratification keeps all three classes represented. Accuracy is the
fraction of test labels predicted correctly. The report shows precision, recall
and F1 for each encoded class. Read it to see where mistakes occur.

```text
Your run prints the first five data rows, the schema, and the first five
actual/predicted pairs. It evaluates all 30 held-out rows for test accuracy
and a classification report.
```

These are your run's measurements, not a promised score or the source's benchmark.
Do not repeatedly tune on this small test set; reserve another validation split if
choosing configurations. `fit` uploads only the training table into your private
workspace and waits for preparation. `predict` executes remotely; each operation
appears in usage. Your downloaded research table is local, while the uploaded
training dataset and fitted predictor remain retained.

## Regression: predict a number

For a numeric outcome, use `Regressor`. This small Yacht Hydrodynamics research
snapshot has 308 real observations and six numeric features. Its recorded target
is residuary resistance (`__target__#0`); the selected distribution is CC0. We use
all observations and an explicitly declared random 80/20 teaching split (246/62),
not a source-provided benchmark split. Closely related experimental rows can make
random-split scores optimistic; a study should choose a split matching its goal.

```python
from pathlib import Path
from tempfile import mkdtemp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from welt import Client, Regressor

# 1. Get a real table from Welt's research inventory.
dataset_id = "asset-bc7dddd884c5e798c1b2abfc559a36ab"
version = "bc7dddd884c5e798c1b2abfc559a36ab79a6d09c4e2ea3e9bbd95c06539c7342"
path = Path(mkdtemp(prefix="welt-yacht-")) / "yacht.parquet"
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
print(data[target].describe())

# 3. Set aside test rows before fitting; never train on their answers.
train, test = train_test_split(
    data, test_size=0.2, random_state=9)
query = test.drop(columns=target)
y_test = test[target]
print("Training rows:", len(train), "— Test rows:", len(test))

# 4. Create a model and train it on the training table.
model = Regressor(model="tabicl-v2", random_state=9)
model.fit(train, target=target)

# 5. Predict the held-out rows, without passing their target column.
predictions = model.predict(query)
print(pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predictions}).head())

# 6. Compare predictions with the answers we kept aside.
print("Test MAE:", mean_absolute_error(y_test, predictions))
print("Test RMSE:", root_mean_squared_error(y_test, predictions))
print("Test R²:", r2_score(y_test, predictions))
print("Keep this predictor ID:", model.predictor_id_)
```

MAE is the average absolute error; RMSE penalizes larger misses more. Both use the
recorded target's units. R² compares with a constant test-set-mean baseline and can
be negative. No calibrated interval or minimum quality is promised. The task's
qualified limits are 500 training rows, 20 features and 100 prediction rows.

## Keep your work

You can [reopen a predictor](reuse.md) without another fit. The same workflow also
works with [your own table](own-data.md). For [async jobs](async-and-errors.md), keep
the job ID before waiting. [Notebooks](notebooks.md) teach the same real-data steps.
