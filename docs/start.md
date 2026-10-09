# From a DataFrame to a prediction

Use a familiar Python estimator to prepare a durable predictor in your private
Welt workspace. Start with a complete classification or regression example.

## Install

Use Python 3.11–3.13. Create a virtual environment, activate it, and install the
reviewed release wheel. No Git installation is required.

```sh
python -m venv .venv
# macOS / Linux
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install 'welt-client @ https://github.com/ergodic-ai/welt-python/releases/download/v0.7.0/welt_client-0.7.0-py3-none-any.whl'
```

## Set up your key

Open [Welt](https://welt.ergodic.dev), sign in, then create a key on your workspace's
**Account** page with **Allow writes, fits and predictions** selected. The secret
is shown once. A read-only key can browse research data but cannot fit or predict.

Use a hidden prompt in your Python session so the key stays out of notebook source
and shell history. Existing local secret configuration may provide `WELT_API_KEY`.

```python
import getpass
import os

if not os.environ.get("WELT_API_KEY"):
    os.environ["WELT_API_KEY"] = getpass.getpass("Welt API key (hidden): ")
```

`fit` uploads the training table to your private workspace and waits for remote
preparation. `predict` executes remotely. Datasets and predictors persist until
explicit deletion; closing Python does not delete them. Keep keys and outputs out
of source control. Browser sessions and SDK keys are separate credentials.

## Classify a table

This complete synthetic example uses seed9, 128 labelled training rows, four
features and 32 separate query rows. The query has feature columns only.

```python
import numpy as np
import pandas as pd
from welt import Classifier

rng = np.random.default_rng(9)
values = rng.normal(size=(160, 4))
train = pd.DataFrame(values[:128], columns=["a", "b", "c", "d"])
train["label"] = np.where(values[:128, 0] + 0.5 * values[:128, 1] > 0,
                          "positive", "negative")
query = pd.DataFrame(values[128:], columns=["a", "b", "c", "d"])
model = Classifier(model="tabicl-v2", random_state=9)
model.fit(train, target="label")
predictions = model.predict(query)
print(type(predictions).__name__, predictions.shape)
```

Expected output contract:

```text
ndarray (32,)
```

The array contains one class label per query row in the same row order. Actual
labels depend on the qualified model; the output above describes type and shape,
not captured model predictions. `model.predict_proba(query)` returns a
`(32, n_classes)` array with columns ordered like `model.classes_`. Native
probabilities are not a calibration or coverage guarantee.

## Predict numeric values

Choose `Regressor` explicitly for a numeric target. This example preserves the
seed42 planted recipe: 128 training rows, four features and 32 query rows.

```python
import numpy as np
import pandas as pd
from welt import Regressor

rng = np.random.default_rng(42)
values = rng.normal(size=(160, 4))
weights = rng.normal(size=4)
train = pd.DataFrame(values[:128], columns=["a", "b", "c", "d"])
train["target"] = values[:128] @ weights + rng.normal(scale=0.05, size=128)
query = pd.DataFrame(values[128:], columns=["a", "b", "c", "d"])
model = Regressor(model="tabicl-v2", random_state=42)
model.fit(train, target="target")
predictions = model.predict(query)
print(type(predictions).__name__, predictions.shape)
```

Expected output is `ndarray (32,)`: one finite mean point in target units per query
row. This interface has no class probabilities or calibrated prediction intervals.
Neither example is a scientific benchmark or accuracy claim.

## Reopen your predictor

Keep `model.predictor_id_` with your experiment's non-secret metadata. In another
Python session, authenticate to the same workspace and use the matching task class:

```python
from welt import Classifier

# Use the identity from your completed classification fit.
reopened = Classifier.from_predictor(model.predictor_id_)
repeat = reopened.predict(query)
print(repeat.shape)  # (32,); no training-table upload or new fit
```

For the regression example, use `Regressor.from_predictor(...)`. Reopening retains
the fitted model version, configuration, task and seed. It does not export weights
or training context. See [Reuse a predictor](reuse.md).

## Continue

[Use your own DataFrame](own-data.md), [wait or recover](async-and-errors.md), or
[download the runnable first-prediction script](https://raw.githubusercontent.com/ergodic-ai/welt-python/v0.7.0/examples/first_prediction.py).
The script prompts safely, prints shapes and checks repeated predictions.

TabICLv2's separately qualified classification and regression previews accept up
to 500 training rows, 20 features and 100 query rows; classification supports up to
10 classes. Current task profile and worker availability remain authoritative.
See [Models and limits](capabilities.md) before changing the model or workload.
SDK 0.7 does not include the held resumable CSV or large batch APIs.
