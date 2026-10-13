# Install to your first real request

Install the SDK, run one real prediction from a script, then see your workspace
confirm it. The script uses the Iris table from Welt's research catalogue.

!!! need "You'll need"
    - Python 3.11–3.13.
    - A [Welt](https://welt.ergodic.dev) account. Sign in with Google or a verified
      email and password; GitHub sign-in is paused.
    - Credits in your workspace. You'll check them in **Workspace → Billing**.

## Install the SDK {.step}

Create a fresh environment and install the released wheel with the existing Parquet
reader extra. You don't need Git:

```sh
python -m venv .venv
# macOS/Linux:
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "welt-client[parquet]"
```

## Check your credits {.step}

Open **Workspace → Billing** and check available credits, pending reservations and
spend caps before you run the script.

!!! caution "This uses workspace credits"
    Modelling and research downloads may consume credits.
    [Usage and recovery](usage.md#usage-and-workspace-credits) explains HTTP 402 and
    the browser billing authority it requires.

## Run the first request {.step}

Download [first_request.py](downloads/first_request.py) from this reviewed
documentation revision, then run it:

```sh
python first_request.py
```

The script connects through your browser or its printed verification link. You
approve your workspace and permissions there, so you don't need to copy a key on
this default path. If you already have a valid configured or saved key, the script
reuses it. It never prints an API secret or writes one into a notebook or source file.

This is a **real native request**, not a mock or a metadata ping. The script:

- downloads the current Iris table (150 rows) and verifies it;
- uses all four features and the `target` column;
- splits the rows with an explicit stratified recipe (seed 9) into 120 training rows
  and 30 query rows;
- prepares the qualified `tabicl-v2` classification configuration and returns 30
  predictions within that task's declared envelope;
- prints the DataFrame and its schema, compares predictions with the held-out labels,
  and reports accuracy plus per-class precision, recall and F1.

These measurements describe your run. They are not a source-provided split or a
publisher benchmark.

!!! note "What stays in your workspace"
    Your private fit dataset and predictor persist. The script doesn't delete them or
    silently change the full research corpus.

<details markdown="1"><summary>Running remotely, or want to save access? Need a manual key?</summary>

See [Connect](connect.md) for remote or headless use and optional private
persistence. Browser sessions and SDK credentials stay separate. Manual workspace
keys are still available in Account when you need them.

</details>

## See it in your workspace {.step}

Return to your signed-in workspace. Onboarding checks for a successful native
prediction that the server attributes to your account, workspace and SDK key.

These don't count: an unrelated success in a shared workspace, a metadata GET, a
queued or failed job, or a client flag. The celebration follows the server's
acknowledgement, with a reduced-motion alternative.

If the operation succeeds but the console is temporarily unavailable, reconnect
later. The server evidence remains durable.

## If something goes wrong

- **The output file already exists.** The script reuses it only after checking its
  exact size and SHA. It never overwrites a mismatching file: choose a new `--output`
  path.
- **A local wait times out.** Server work continues. Use the supplied job ID and
  `Client.job(id)` to reconnect, rather than blindly creating another fit.
- **Unavailable workers, capacity or revoked credentials.** These errors stay
  actionable and don't produce a success celebration.

No latency guarantee is implied.

## Next steps

<div class="cards" markdown="1">

- [Research datasets](research-datasets.md): find more tables to model.
- [SDK guide](usage.md): workspace resources, usage and credits.
- [Model capabilities](capabilities.md): available tasks and current limits.

</div>
