# Welt Python SDK

From a DataFrame to a prediction. Use familiar `Classifier` and `Regressor`
estimators backed by durable predictors in your private Welt workspace.
Install `welt-client`, import `welt`. Python 3.11–3.13.

[Start here](https://ergodic-ai.github.io/welt-python/) ·
[Guides](https://ergodic-ai.github.io/welt-python/v0.7.0/own-data.html) ·
[Reference](https://ergodic-ai.github.io/welt-python/v0.7.0/reference.html) ·
[Models and limits](docs/capabilities.md)

## Install

```sh
python -m pip install 'welt-client @ https://github.com/ergodic-ai/welt-python/releases/download/v0.7.0/welt_client-0.7.0-py3-none-any.whl'
```

Create a workspace API key at [Welt](https://welt.ergodic.dev) with **Allow writes,
fits and predictions** enabled. Configure `WELT_API_KEY` through local secret
configuration or a hidden `getpass` prompt. Keep the secret out of source, notebooks,
logs and shell history. SDK 0.7 defaults to `https://welt.ergodic.dev`; an explicit
`base_url` overrides `WELT_BASE_URL`, which overrides that default. Local development
must explicitly select localhost.

## Predict

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
print(predictions.shape)  # (32,)
reopened = Classifier.from_predictor(model.predictor_id_)
repeat = reopened.predict(query)
```

This synthetic recipe returns one label per query row, in order; actual labels
come from the hosted model. It is not an accuracy benchmark. Use `Regressor`
explicitly for numeric targets; the [complete regression example](docs/start.md)
returns one finite target-unit point per row. `fit` uploads the table and waits for
remote preparation; `predict` executes remotely. Created datasets/predictors remain
retained. Reopening uses the pinned identity without another fit or training upload.

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

Controlled synthetic transport tests neither contact the hosted service nor execute
foundation models. Notebook sources have cleared outputs; clean-wheel execution and
bounded authorized live acceptance are separate evidence. CI requires no private
backend, credentials or checkpoints.

Apache 2.0: see [LICENSE](LICENSE). Model weights and hosted service are separate
from this SDK licence.
