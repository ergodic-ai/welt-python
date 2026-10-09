# Install to your first real request

Open [Welt](https://welt.ergodic.dev), sign in with Google or your verified
email/password, and use your workspace's Account page. Google login is accepted;
GitHub is currently paused. Create an API key with **Allow writes, fits and
predictions** checked. Copy its secret once; it is not shown again. Read-only keys
can browse/download research data but cannot prepare or run your first model.

Use Python 3.11–3.13. Create a fresh environment and install the released wheel,
including the existing Parquet reader extra. No Git executable is needed:

```sh
python -m venv .venv
# macOS/Linux:
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install 'welt-client[parquet] @ https://github.com/ergodic-ai/welt-python/releases/download/v0.7.0/welt_client-0.7.0-py3-none-any.whl'
```

Download [first_request.py](https://raw.githubusercontent.com/ergodic-ai/welt-python/v0.7.0/examples/first_request.py)
from the same reviewed release, then run:

```sh
python first_request.py
```

The script prompts for the key without echoing it or adding it to shell history.
It also accepts an existing `WELT_API_KEY` environment value. Never paste keys into
source, notebook cells, command URLs, logs or screenshots. Do not persist an SDK
key in browser storage. Browser sessions and SDK keys are separate credentials.

This is a **real native request**, not a mock or metadata ping. The script downloads
exactly one reviewed p10k Iris Parquet asset (7,332 bytes, 150 rows), checks its
immutable version/SHA, prints its attribution and uses all four recorded features
and the recorded target. Its explicit stratified seed9 recipe uses 120 training
rows and 30 query rows. It prepares the qualified `tabicl-v2` classification
configuration and returns 30 predictions within that task's declared envelope.
The recipe is onboarding, not a source-provided split, scientific evaluation or
publisher benchmark. Your private fit dataset and predictor persist; the script
does not delete them or silently change the full research corpus.

The source asset is `asset-d839644b6de36fc5cafdda18c5daeca3`, version
`d839644b6de36fc5cafdda18c5daeca3284a5ce8bb6169aff2fcb3fbb4a12021`.
Iris is attributed to R. A. Fisher (1936), [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris),
DOI10.24432/C56C76, CC BY4.0; the p10k snapshot retains its PMLB delivery and
canonical Parquet representation. Dataset license and model rights are separate.

Return to your signed-in workspace. Onboarding checks a successful native prediction
attributed by the server to your account/workspace/SDK key. A shared workspace's
unrelated success, a metadata GET, queued/failed job or client flag does not qualify.
Celebration follows this server acknowledgement, with a reduced-motion alternative.
If the operation succeeds but the console is temporarily unavailable, reconnect
later; the server evidence remains durable.

If an output file already exists, the script reuses it only after checking the exact
size/SHA. It never overwrites a mismatching file: choose a new `--output` path.
If a local wait times out, server work continues; use the supplied job ID and
`Client.job(id)` to reconnect rather than blindly creating another fit. Errors
from unavailable workers, capacity or revoked credentials remain actionable and
do not produce a success celebration. No latency guarantee is implied.

SDK 0.7 contains research methods and the released0.3 interfaces. It excludes the
separate unreleased CSV0.4 and batch0.5 candidates. No PyPI release is claimed.
See [Research datasets](research-datasets.md), [SDK guide](usage.md) and
[model capabilities](capabilities.md). Broader managed signup/recovery and all-model
completion gates remain distinct from this bounded walkthrough.
