# Executable notebooks

The notebooks demonstrate available classification, async job reconnect, predictor
reopening and usage. Regression, true batch and the five-engine causal/ergodic pack follow their
actual implementation/qualification; no placeholder notebook is claimed working.

Run from this directory using the repository's locked development environment,
or run `python scripts/check_notebooks.py` from its root. Default mode uses only
`httpx.MockTransport`, synthetic inputs (seed9) and contract assertions. The fixture
is not an FM, a backend security test or scientific evidence. Source notebooks
always have cleared outputs/execution counts; execution never writes outputs here.

For bounded live execution, explicitly set `WELT_NOTEBOOK_MODE=live`, the HTTPS
`WELT_BASE_URL` of your Welt service and `WELT_API_KEY` via local secret configuration.
Do not paste credentials into cells. Use a qualified TabICLv2 classification entry;
unavailable tasks reject without fallback. Each notebook prepares 128 rows/four features
and predicts 32 rows, with a 120 second preparation wait. Created synthetic state is
retained; no deletion is performed. Check service readiness/capacity before opting
in. Operator live evidence is separate from default CI execution.

A local wait timeout leaves durable work running. Keep/reconnect using job identity;
explicit cancellation differs from local timeout. Native probability outputs are
not conformal confidence or calibrated coverage guarantees. After interactive use,
clear outputs before saving and never commit credentials or generated predictions.

From the repository root, install the built wheel into the locked development
environment before running the checker (uv sync alone installs editable source):

```sh
uv sync --frozen
uv build
uv pip install --python .venv/bin/python --reinstall --no-deps dist/*.whl
.venv/bin/python scripts/check_notebooks.py
.venv/bin/python scripts/build_docs.py
```

Use the direct environment Python after wheel installation; uv run may restore
editable source. The runner forces fixture mode by default even if the shell has
WELT_NOTEBOOK_MODE=live. Live runner execution requires the explicit `--live` flag
as well as configured credentials/origin. The environment mode variable applies
to interactive notebook cells. No live CI execution or credentials are configured.
