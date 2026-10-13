# Upgrade your client

Move your scripts and notebooks to the current `welt-client` release. Upgrading the
SDK does not refit a model or change an existing remote predictor.

Install the current release in the environment used by your script or notebook:

```sh
python -m pip install --upgrade "welt-client[parquet]"
```

!!! tip "Using a notebook?"
    Restart the kernel after upgrading if it has already imported `welt`.

<a id="select-the-endpoint-explicitly-when-local"></a>
<a id="move-from-07-to-08"></a>

## Connect to the intended workspace

- `Client()` defaults to `https://welt.ergodic.dev`. An explicit `base_url` overrides
  `WELT_BASE_URL`; select `http://localhost:8000` deliberately for a local server.
- Public catalogue reads need no key. Authenticated work needs a workspace credential:
  run `welt login` in a terminal or call `Client().connect()` in a notebook.
- Constructors never start authentication. Explicit keys and `WELT_API_KEY` keep
  precedence. A Python connection lasts only for the process unless `save=True` is
  explicit.

See [Connect](connect.md) for private persistence and remote or headless use.

The current release supports multiple saved workspace profiles and explicit `workspace`
selection. A key stays bound to its original tenant; a workspace option cannot retarget
it. See [CLI workspace selection](cli.md#connect-and-choose-a-workspace).

<a id="use-a-dataframe-target"></a>

## Keep your existing supervised workflow

`fit(X, y)` remains valid. You can also name the target in a DataFrame you pass
directly. Give exactly one target source:

```python
# Continue with the real train table and target name from Start.
model.fit(train, target=target)
```

- Select `Classifier` or `Regressor` explicitly.
- Missing or duplicate target names, conflicting `y` and `target`, and invalid schemas
  are rejected locally before upload.
- Query rows still contain features only.
- Predictors retain their exact model and configuration identity. Changing models
  requires an intentional new fit.

<a id="keep-the-release-boundary"></a>

## Use the released resource methods

The current client includes:

- research metadata sidecars and one-step table loading;
- resumable file upload sessions;
- durable batch prediction;
- workspace inspection;
- hidden-model runs;
- cross-validated [evaluations](evaluations.md) with DataFrame results and report links (0.12).

Both sync and async clients expose the corresponding resources. Estimator CSV and
Parquet convenience reads still load into memory. For resumable file transfer, use
`Client.upload_file` with a retained upload ID.

!!! note "A client method does not enable a server feature"
    Client availability does not enable a server parser, model or task. Inspect
    workspace limits and the model's task profile and readiness before submission.

`Job.result()` returns a batch manifest for successful or partial batches. Retrieve
its payload separately and inspect failed ranges. See [capabilities](capabilities.md),
[CLI recipes](cli.md) and [hidden-model runs](model-runs.md).

<a id="reconnect-after-waiting"></a>

## Preserve durable identities when recovering

`Job.result(timeout=...)` budgets initial inspection, polling, GET retry waits and
final result retrieval. It never cancels or resubmits work. Keep `job_id` and reconnect
using the same endpoint and workspace credentials. See
[Wait and recover](async-and-errors.md).

<details markdown="1"><summary>Is the timeout a hard deadline?</summary>

Async requests use the remaining wait budget. Synchronous HTTP phases receive the
remaining budget, and the deadline is checked afterward; a custom blocking transport
is not a hard wall-clock interrupt.

</details>

Usage now includes nullable `credits` and `rate_card_version`. HTTP 402 remains a
`WeltError` whose `code` explains the balance, spend-cap or unpaid-plan condition.
Check Workspace Billing before rerunning work; see
[usage accounting](usage.md#usage-and-workspace-credits).

!!! note "Older versions"
    Historical documentation remains available from the version menu with its own SDK
    contract. Redirected guides identify that they now lead to the current docs.

## Next steps

<div class="cards" markdown="1">

- [Connect](connect.md): saved access, expiry and which key wins.
- [Wait and recover](async-and-errors.md): reconnect to a job instead of refitting.
- [Workspace resources](usage.md): usage fields, credits and billing errors.

</div>
