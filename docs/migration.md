# Upgrade your client

Install the current release in the environment used by your script or notebook:

```sh
python -m pip install --upgrade "welt-client[parquet]"
```

Restart a notebook kernel after upgrading if it has already imported `welt`.
Upgrading the SDK does not refit a model or change an existing remote predictor.

<a id="select-the-endpoint-explicitly-when-local"></a>
<a id="move-from-07-to-08"></a>

## Connect to the intended workspace

`Client()` defaults to `https://welt.ergodic.dev`. An explicit `base_url` overrides
`WELT_BASE_URL`; select `http://localhost:8000` deliberately for a local server.
Public catalogue reads are keyless. Authenticated work needs a workspace credential:
run `welt login` in a terminal or call `Client().connect()` in a notebook.
Constructors never initiate authentication. Explicit keys and `WELT_API_KEY` retain
precedence; Python connection is process-only unless `save=True` is explicit.
See [Connect](connect.md) for private persistence and remote/headless use.

The current release supports multiple saved workspace profiles and explicit
`workspace` selection. A key remains bound to its original tenant; a workspace
option cannot retarget it. See [CLI workspace selection](cli.md#connect-and-choose-a-workspace).

<a id="use-a-dataframe-target"></a>

## Keep your existing supervised workflow

`fit(X, y)` remains valid. You can also name the target in a directly supplied
DataFrame; exactly one target source is required:

```python
# Continue with the real train table and target name from Start.
model.fit(train, target=target)
```

Select `Classifier` or `Regressor` explicitly. Missing/duplicate target names,
conflicting `y`/`target` and invalid schema reject locally before upload. Query
rows still contain features only. Predictors retain their exact model/configuration
identity; changing models requires an intentional new fit.

<a id="keep-the-release-boundary"></a>

## Use the released resource methods

The current client includes research metadata sidecars and one-step table loading,
resumable file upload sessions, durable batch prediction, workspace inspection and
hidden-model runs. Both sync and async clients expose the corresponding resources.
Estimator CSV/Parquet convenience reads still load into memory; use
`Client.upload_file` with a retained upload ID for resumable file transfer.

Client availability does not enable a server parser, model or task. Inspect
workspace limits and the model's task profile/readiness before submission.
`Job.result()` returns a batch manifest for successful or partial batches;
retrieve its payload separately and inspect failed ranges. See [capabilities](capabilities.md),
[CLI recipes](cli.md) and [hidden-model runs](model-runs.md).

<a id="reconnect-after-waiting"></a>

## Preserve durable identities when recovering

`Job.result(timeout=...)` budgets initial inspection, polling, GET retry waits and
final result retrieval. It never cancels or resubmits work. Async requests use the
remaining wait budget. Synchronous HTTP phases receive the remaining budget and
the deadline is checked afterward; a custom blocking transport is not a hard
wall-clock interrupt. Keep `job_id` and reconnect using the same endpoint and
workspace credentials. See [Wait and recover](async-and-errors.md).

Usage now includes nullable `credits` and `rate_card_version`. HTTP 402 remains a
`WeltError` whose `code` explains the balance, spend-cap or unpaid-plan condition.
Check Workspace Billing before rerunning work; see [usage accounting](usage.md#usage-and-workspace-credits).

Historical documentation remains available from the version menu with its own
SDK contract. Redirected guides identify that they now lead to the current docs.
