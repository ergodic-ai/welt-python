# Executable notebooks

The notebooks demonstrate task-gated classification/regression, async job reconnect,
predictor reopening and usage, plus causal and SDK 0.4 resumable CSV source candidates. The causal
notebook validates native-result/score plumbing using a synthetic transport; live
ArrowFM and optional approved Ergodic conversion remain separate gates. The SDK
0.5 batch candidate demonstrates partial manifests, explicit payloads and expiry
through controlled transport. Native batch and the five-engine causal/ergodic pack
follow their actual qualification; no placeholder model execution is claimed.

Run from this directory using the repository's locked development environment,
or run `python scripts/check_notebooks.py` from its root. Default mode uses only
`httpx.MockTransport`, synthetic inputs (classification seed9, regression seed42) and contract assertions. The fixture
is not an FM, a backend security test or scientific evidence. Source notebooks
always have cleared outputs/execution counts; execution never writes outputs here.

For bounded live execution, explicitly set `WELT_NOTEBOOK_MODE=live`, the HTTPS
`WELT_BASE_URL` of your Welt service and `WELT_API_KEY` via local secret configuration.
Do not paste credentials into cells. Use separately qualified TabICLv2 classification and regression task profiles;
unavailable tasks reject without fallback. The supervised notebooks prepare 128 rows/four features
and predict 32 rows, with a 120 second preparation wait. The causal candidate
requires a separately available ArrowFM task and submits the full128x4 seed42
observational chain/isolate input, with no target or predictions. Optional local
graph conversion requires the approved Ergodic co-release artifact; its absence
is explicit and is not counted as executed causal continuation. Created synthetic state is
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

To validate only the separately qualified regression task without preparing the
other examples again, use `.venv/bin/python scripts/check_notebooks.py --live --notebook regression` with the same locally configured credentials. The optional
`--notebook` selector also works in default fixture mode. Other examples are not
counted as checked by a selected run.

The files-and-resume candidate writes only a tiny local synthetic CSV. Default
fixture mode accepts a chunk then loses its acknowledgement, demonstrating resume
without creating a second upload. Live mode may complete without interruption; it
retains synthetic hosted uploads/datasets and performs no deletion or model jobs.
Use the matching built SDK 0.4 wheel for this new notebook. Async await cancellation
is local and differs from explicit job cancellation. Backend tenant/capacity and
actual model reuse require separate hosted evidence.

The batch-predictions candidate uses the matching SDK 0.5 wheel. Its synthetic
fixture illustrates one failed range, immutable row placement, client reopening,
operation usage and typed payload expiry. It does not test native predictions,
database races, cancellation preemption or seven days of hosted retention. Live
mode refuses before any request because no task has qualified batch. Select a
separately qualified existing notebook when running the checker with `--live`;
there is no silent batch skip to claim a complete live notebook pack.
