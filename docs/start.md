# Your first prediction

Train a classifier on a real table, then check how well it predicts rows it has never seen.
You'll use Iris flower measurements from Welt's research catalogue.

!!! need "You'll need"
    - Python 3.11, 3.12 or 3.13 in a virtual environment.
    - A Welt workspace. Check its credits in **Workspace → Billing** before you start.

## Install {.step}

Install the SDK with its Parquet reader:

```sh
python -m pip install "welt-client[parquet]"
```

## Connect your workspace {.step}

In a terminal, run `welt login` and approve your workspace in the browser. In a
notebook, connect from Python instead:

```python
from welt import Client

with Client() as client:
    client.connect()
```

!!! note "Running Python on a remote server?"
    Open the printed verification link on your own computer. Nothing needs to call
    back to localhost.

<details markdown="1"><summary>Where is my access stored?</summary>

`welt login` saves an origin-bound credential privately on this computer. A Python
connection keeps access only in this process by default, so the next `Classifier()`
works without copying a key. Use `client.connect(save=True)` only if you want access
saved for future processes. Manual `WELT_API_KEY` configuration still works; never
save a secret in code. [Connection details](connect.md) explains expiry and which key
wins.

</details>

!!! caution "This uses workspace credits"
    Modelling and research downloads may consume workspace credits. Check available
    credits, pending reservations and spend caps in **Workspace → Billing** first. See
    [usage and billing recovery](usage.md#usage-and-workspace-credits).

## Classification: predict a flower's class {.step}

Iris has 150 rows, four measurements and three encoded classes. The outcome column
is named `target`. Run the complete example; its numbered comments follow six steps:
get the data, look at it, hold rows back, train, predict and compare.

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

You now have 120 training rows and 30 test rows. Your run prints the first five data
rows, the schema and the first five actual and predicted pairs. It then evaluates all
30 held-out rows.

- **Accuracy** is the fraction of test labels predicted correctly.
- **The classification report** shows precision, recall and F1 for each encoded class.
  Read it to see where mistakes occur.
- **The same seed** makes the split repeatable, and stratification keeps all three
  classes in both parts.

!!! caution "Thirty rows is a small test set"
    These are your run's measurements, not a promised score or the source's benchmark.
    Don't tune repeatedly against this test set; keep another validation split for
    choosing configurations.

<details markdown="1"><summary>What did this create in my workspace?</summary>

`fit` uploads only the training table into your private workspace and waits for
preparation. `predict` runs remotely, and each operation appears in usage. The
research table you downloaded stays local. The uploaded training dataset and the
fitted predictor are retained.

</details>

## Regression: predict a number

For a numeric outcome, use `Regressor`. This small Yacht Hydrodynamics research
snapshot has 308 real observations and six numeric features. Its recorded target is
`Residuary.resistance`.

```python
from pathlib import Path
from tempfile import mkdtemp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from welt import Client, Regressor

# 1. Get a real table from Welt's research inventory.
dataset_id = "asset-bc7dddd884c5e798c1b2abfc559a36ab"
path = Path(mkdtemp(prefix="welt-yacht-")) / "yacht.parquet"
with Client() as client:
    info = client.research_dataset(dataset_id)
    print("Dataset:", info["name"])
    client.download_research_dataset(dataset_id, path)
data = pd.read_parquet(path)
target = "Residuary.resistance"  # The outcome column in this table.

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

- **MAE** is the average absolute error. **RMSE** penalizes larger misses more. Both
  use the recorded target's units.
- **R²** compares with a constant test-set-mean baseline and can be negative.

No calibrated interval or minimum quality is promised. The task's qualified limits
are 500 training rows, 20 features and 100 prediction rows.

!!! caution "About this split"
    The example uses all observations and an explicitly declared random 80/20 teaching
    split (246/62), not a source-provided benchmark split. Closely related experimental
    rows can make random-split scores optimistic; a study should choose a split that
    matches its goal.

## Keep your work

<div class="cards" markdown="1">

- [Reuse a predictor](reuse.md): reopen a fitted predictor without another fit.
- [Use your own table](own-data.md): run the same workflow on your data.
- [Wait and recover](async-and-errors.md): keep the job ID before waiting on async jobs.
- [Notebook examples](notebooks.md): the same real-data steps in Jupyter or Colab.

</div>
