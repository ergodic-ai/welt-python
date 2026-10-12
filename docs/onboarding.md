# Install to your first real request

Sign in to [Welt](https://welt.ergodic.dev) with Google or verified email/password.
GitHub sign-in is paused. The SDK's browser connection asks you to approve your
workspace and permissions; you no longer need to copy a key for the default path.
Manual workspace keys remain available in Account when needed.

Use Python 3.11–3.13. Create a fresh environment and install the released wheel,
including the existing Parquet reader extra. No Git executable is needed:

```sh
python -m venv .venv
# macOS/Linux:
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "welt-client[parquet]"
```

Download [first_request.py](downloads/first_request.py)
from this reviewed documentation revision, then run:

```sh
python first_request.py
```

The script connects through the browser or its printed verification link. It also
reuses a valid configured/saved key. No API secret is printed or written into the
notebook/source. See [Connect](connect.md) for remote/headless use and optional
private persistence. Browser sessions and SDK credentials remain distinct.

This is a **real native request**, not a mock or metadata ping. The script downloads
the current Iris table (150 rows), verifies it and uses all four features
and the `target` column. Its explicit stratified seed9 recipe uses 120 training
rows and 30 query rows. It prepares the qualified `tabicl-v2` classification
configuration and returns 30 predictions within that task's declared envelope.
It prints the DataFrame/schema, compares predictions with held-out labels, and
reports accuracy plus per-class precision/recall/F1. These measurements describe
your run, not a source-provided split or publisher benchmark. Your private fit dataset and predictor persist; the script
does not delete them or silently change the full research corpus.

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

See [Research datasets](research-datasets.md), [SDK guide](usage.md) and
[model capabilities](capabilities.md). Broader managed signup/recovery and all-model
completion gates remain distinct from this bounded walkthrough.
