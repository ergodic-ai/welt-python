# Learn in a notebook

Download a notebook **and its matching support.py into the same directory**. The
helper is portable and pinned to SDK 0.7; imports do not depend on your checkout.
Sources have cleared outputs and default to no-network synthetic transport.

## Install and open

Install the [0.7 release wheel](start.md#install), then add Jupyter to your local
environment. This is an optional learning tool, not an SDK runtime dependency.
Download [support.py](https://raw.githubusercontent.com/ergodic-ai/welt-python/v0.7.0/examples/notebooks/support.py)
and a notebook below. Open it from their shared directory and restart/run all.

```sh
python -m pip install jupyterlab
python -m jupyter lab
```

| Notebook | What you learn |
| --- | --- |
| [Classification](https://raw.githubusercontent.com/ergodic-ai/welt-python/v0.7.0/examples/notebooks/classification.ipynb) | A DataFrame with explicit target, probabilities, shape and reopen. |
| [Regression](https://raw.githubusercontent.com/ergodic-ai/welt-python/v0.7.0/examples/notebooks/regression.ipynb) | Explicit numeric task, finite points, column alignment and pinned reuse. |
| [Async jobs](https://raw.githubusercontent.com/ergodic-ai/welt-python/v0.7.0/examples/notebooks/async-jobs.ipynb) | Await resources, keep durable job ID and reconnect. |
| [Artifacts and usage](https://raw.githubusercontent.com/ergodic-ai/welt-python/v0.7.0/examples/notebooks/artifacts-and-usage.ipynb) | Predictor provenance and logical operation accounting, without pricing. |
| [Research datasets](https://raw.githubusercontent.com/ergodic-ai/welt-python/v0.7.0/examples/notebooks/research-datasets.ipynb) | Inspect metadata and download checksum-pinned bytes without private upload. |
| [Native causal discovery](https://raw.githubusercontent.com/ergodic-ai/welt-python/v0.7.0/examples/notebooks/causal-discovery.ipynb) | Target-free observations, immutable native graph, separate scores and reconnect. |

[Execution instructions](https://github.com/ergodic-ai/welt-python/blob/v0.7.0/examples/notebooks/README.md)
explain fixture and live modes. Static docs also retain release-local copies in
`downloads/`; the linked release tag remains the canonical notebook source.

## Know what ran

Default fixture mode validates notebook/client plumbing with controlled HTTP
transport. It does not execute a foundation model, prove backend security or make
scientific/latency claims. Classification uses seed9 and regression seed42, each
with 128 training rows, four features and 32 query rows. Native discovery uses the
full seed42 128×4 chain/isolate observations; scores and graph semantics remain
separate. Research fixture bytes are clearly synthetic and are never called a real
Parquet download.

Live mode is an explicit separate opt-in with current task qualification/readiness,
locally configured key and HTTPS endpoint. Fits/discoveries retain the created
synthetic dataset/predictor/result. No notebook deletes hosted state or stores keys.
Clear outputs before saving. Local Ergodic conversion requires the separately
approved co-release artifact; absence is shown and never counted as successful
causal continuation. Full catalogue, true large batch and five-engine causal packs
remain separate gates.

## Reproduce checks from source

CI builds the wheel, installs it, then executes actual disposable Jupyter kernels.
It rejects saved outputs and never persists executed predictions.

```sh
uv sync --frozen
uv build
uv pip install --python .venv/bin/python --reinstall --no-deps dist/*.whl
.venv/bin/python scripts/check_notebooks.py
.venv/bin/python scripts/build_docs.py
```

Use the direct environment Python after wheel installation; `uv run` can restore
editable source. The runner forces fixture mode unless `--live` is supplied;
`--notebook regression` selects one example without counting other notebooks as
checked. [0.6 examples](https://github.com/ergodic-ai/welt-python/tree/v0.6.0/examples/notebooks)
remain pinned for users of that release.
