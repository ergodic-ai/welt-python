# Welt CLI for agents

Run Welt from a terminal, a script or an agent. Successful commands print one JSON
value on stdout. Check the exit status before you parse it.

Install the current `welt-client` release to use these commands. Model readiness, file
formats and limits depend on the connected service.

`welt --help`, `welt COMMAND --help` and `welt capabilities` describe the complete
command tree. `python -m welt` uses the same entry point.

## Connect and choose a workspace

```sh
welt workspaces create                 # opens /workspaces/new; finish the browser form
welt login --workspace acme-research   # exact slug or workspace ID
welt login --workspace ws_other --no-browser
welt workspaces list                   # saved local profiles, expiry and active status
welt workspaces use acme-research      # verifies saved key then changes local default
welt whoami                           # actual key workspace, role and limits
welt datasets list --workspace ws_other
welt logout --workspace ws_other       # forgets local profile; does not revoke server key
```

To create a Team or Academic workspace and switch browser context, use the console's
**New workspace** navigation entry or its topbar switcher. Your Personal workspace
already exists.

- A new shared workspace is billed separately and starts with zero credits. Creating
  one does not require a payment method.
- Invite colleagues in Members; they must accept explicitly.
- Owners, admins and members can authorize their own client. Approval shows the
  requested workspace and rejects a different one.

Every command accepts `--base-url`, `--workspace` and `--timeout`. `WELT_BASE_URL` and
`WELT_WORKSPACE` provide defaults. Credential precedence is: an explicit SDK key, then
`WELT_API_KEY`, then the process or saved profile.

- The workspace option verifies that credential's tenant before sending private input.
  It cannot retarget a bearer key.
- Unset `WELT_API_KEY` and `WELT_WORKSPACE` if they override the default you selected
  with `workspaces use`.
- Use explicit per-command selection for concurrent agents. The shared active profile
  is a convenience for interactive use.
- CLI origins must be HTTPS or explicit localhost HTTP. Redirects and cross-origin
  requests never receive bearer credentials.

<details markdown="1"><summary>Where are my credentials stored?</summary>

Credentials are stored privately in `WELT_CONFIG_DIR` (default `~/.config/welt`): the
POSIX directory is 0700 and files are 0600, with atomic writes, private per-origin lock
files and no symlink following. Profile updates are serialized across CLI processes.

Each origin retains multiple workspace keys; `login` and `use` select the local
default. Old single-credential files still load. When you save a login, the CLI links
any old single credential to its tenant with a read-only API probe and keeps that
previous key. If the probe is unavailable, the CLI keeps the unknown credential
privately until it can be linked.

`--no-save` retains access only during the CLI process. `Client().connect()` keeps
credentials in the Python process unless `save=True` is explicit.

</details>

!!! caution "Paid work uses workspace credits"
    Before paid work, check **Workspace → Billing** for available credits, pending
    reservations and spend caps. [Usage and recovery](usage.md#usage-and-workspace-credits)
    explains current accounting and HTTP 402 conditions.

## Durable modelling workflow

```sh
welt models list
welt models get tabicl-v2
welt datasets upload --input train.json  # {columns, rows, target, name}
welt fit --dataset ds_train --model tabicl-v2 --task classification --seed 19 --idempotency-key fit-run-00000001
welt jobs get job_fit
welt jobs wait job_fit --timeout 600
welt predictors get pred_fitted
welt predict --predictor pred_fitted --input query.json
welt batch --predictor pred_fitted --dataset ds_query --idempotency-key batch-run-000001
welt jobs result job_batch               # row coverage, failures, version, expiry
welt jobs payload job_batch              # authenticated payload, original row positions
welt usage --model tabicl-v2 --since 2026-10-11T00:00:00Z
```

- Use the IDs returned by earlier commands.
- `models list` reports actual availability, qualified tasks and configurations, and
  limits. Commands do not activate held models.
- You can supply table JSON on stdin with `--input -`. Prediction input contains
  `columns` and `rows`; request probabilities explicitly with `--probabilities`.
- Fit returns a durable job ID and replay key immediately. `jobs wait` times out
  locally while accepted work continues; reconnect with the job ID.

!!! caution "Unsure whether a submission went through?"
    Inspect jobs before submitting again. To retry, reuse the original idempotency key
    and an identical payload. Replay keys, including generated ones, are printed to
    stderr before submission.

Writes are not automatically retried; bounded transient retries apply to GETs.

<a id="file-uploads-and-causal-discovery"></a>

## Resume a supervised file upload

```sh
welt uploads create train.csv --target outcome --schema schema.json
welt uploads get upload_id
welt uploads file train.csv --target outcome --schema schema.json --upload-id upload_id
```

Use `uploads create` first when recovery matters, and keep the upload ID before sending
bytes. `uploads file` verifies the file's checksum, size, format, schema and target
against a resumed session. It skips acknowledged chunks and completes validation.
`uploads chunk ID INDEX FILE` and `uploads complete ID` expose the lower-level contract.

A schema looks like `{"columns":["x","outcome"],"types":["number","string"]}`. Types
follow the API file schema (`number`, `boolean`, `string`); use `string` for
categorical columns.

The completed upload returns a dataset ID for a supervised fit. That target-labelled
dataset cannot be used for causal discovery.

!!! note "File formats vary by deployment"
    CSV and Parquet support depends on the deployed server configuration and its
    qualified parser and envelope. Do not assume every deployment enables Parquet.

## Causal discovery with target-free observations

Prepare a separate table containing only the numeric variables to study. For example,
save the Iris measurements loaded in [Start](start.md), dropping the class label
before you create the upload JSON:

```python
# Continue with the Iris data and target from Start.
from pathlib import Path
import json

observations = data.drop(columns=target)
Path("observations.json").write_text(json.dumps({
    "columns": list(observations.columns),
    "rows": observations.values.tolist(),
    "target": None,
    "name": "Iris measurements",
}), encoding="utf-8")
```

Then upload the table and run discovery:

```sh
welt datasets upload --input observations.json
# Replace ds_observations with the dataset ID returned above.
welt discover --dataset ds_observations --model arrow --seed 19 --idempotency-key graph-run-000001
welt jobs result job_graph
welt jobs scores job_graph
```

Arrow accepts at least 2 rows and 2 finite continuous variables, up to 8,000 × 20
(or 4,000 × 40, 2,000 × 80; see `welt models get arrow`), without a supervised target. Its current path does not support nonempty required or forbidden edges, or
temporal constraints, so this recipe omits `--constraints`.

!!! caution "There is no ground truth here"
    A graph inferred from these measurements has no ground-truth causal graph for
    accuracy evaluation. Native scores are method-defined, not normalized graph
    posterior probabilities.

Python provides validated graph objects and optional local Ergodic conversion; see
[Causal discovery](causal-discovery.md).

## Other commands and authority

Inspect resources with `datasets list/get`, `predictors list/get`, `jobs list/get`,
`research list/get/download` and `pricing`.

- Research search is paginated; pass its `next_cursor` as `--cursor`.
- Preserve the exact version and checksum provenance of downloads.
  `research download ID PATH --max-bytes LIMIT` selects the current published file and
  makes a verified download to a new local file. The configured server may meter
  downloads.
- `jobs cancel ID --yes`, `datasets delete ID --yes` and `causal-results delete ID --yes`
  are explicit mutations.
- Dataset deletion refuses retained dependencies; the CLI never cascades. Primary
  result deletion retains job and usage metadata.

!!! caution "Deletion needs an exact request"
    Never delete real customer state without an exact user request.

Account changes, workspace creation, invitations and member administration, key
administration and billing need browser account authority. Use
`welt console account|workspace|members|keys|billing|create [--no-browser]` to open
the corresponding control.

- Signed-in browser state determines the console's active workspace, independently of
  CLI profiles.
- `workspaces list` shows local saved connections. It is not an account-wide membership
  query.
- Payments require explicit completion of browser Checkout or Portal. `pricing` only
  reads public offers.

## Output, errors and exit codes

Successful commands emit one JSON value on stdout. Login links and codes, submission
replay keys and errors go to stderr; secrets never appear. A script should parse stdout
only when the exit status is zero. `capabilities` is a machine-readable command
inventory.

Error JSON has `error.code`, `message`, `status_code`, `request_id`, `job_id`,
`retryable`, `retry_after` and `idempotency_key` where available.

| Exit status | Meaning |
| --- | --- |
| 0 | Success. If the command submitted a job, that job is still pending; inspect it separately. |
| 1 | API, local file, credential, validation or wait failure. |
| 2 | Command syntax error (argparse usage text on stderr). |
| 130 | Interrupted locally; server job state is unchanged. |

Batch wait returns a manifest for `succeeded` or `partially_completed`. Inspect failed
ranges and payload errors rather than counting missing rows as predictions. An entirely
failed or cancelled job raises its terminal error.

Manifests remain inspectable after payload retention expires; payload retrieval
returns explicit expiry errors. Use the server's `expires_at` and version fields.

## Hidden model runs and forecasting

```sh
welt models get chronos-2
welt run --input forecast.json --idempotency-key forecast-00000001
welt jobs wait job_returned --timeout 600
welt model-runs delete job_returned --yes
```

`forecast.json` uses the task payload in [Hidden model runs](model-runs.md).
`models list` stays curated; inspect a hidden model by its exact ID. The server
enforces workspace access.

`run` returns a durable job, and `jobs wait` returns its native task-specific JSON
result. These bounded runs create no reusable fitted predictor. Deleting a terminal run
explicitly purges both its private input and result, while job, replay and usage
metadata remain.

## Next steps

<div class="cards" markdown="1">

- [Wait and recover](async-and-errors.md): what each error means and what to do next.
- [Workspace resources](usage.md): usage events, credits and HTTP 402 conditions.
- [Causal discovery](causal-discovery.md): validated graph objects in Python.
- [Hidden model runs](model-runs.md): task payloads for bounded runs.

</div>
