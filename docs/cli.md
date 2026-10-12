# Welt CLI for agents

Status: source candidate for API-089/090; package publication and backend rollout
are separate. Install the reviewed release before using these new commands.
`welt --help`, `welt COMMAND --help`, and `welt capabilities` describe the complete
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

Use the console's **New workspace** navigation entry or its topbar switcher to
create a Team/Academic workspace and switch browser context. Personal workspace
already exists. A new shared workspace is separately billed, starts with zero
credits and does not require a payment method for creation. Invite colleagues in
Members; acceptance is explicit. Owners, admins and members can authorize their
own client. Approval shows the requested workspace and rejects a different one.

Credentials are stored privately in `WELT_CONFIG_DIR` (default `~/.config/welt`):
POSIX directory 0700, files 0600, atomic writes, private per-origin lock files and no symlink following. Profile updates are serialized across CLI processes. Each origin
retains multiple workspace keys; login/use selects the local default. Old single
credential files still load. A saved login links their tenant with a read-only API
probe and preserves the previous key; an unavailable probe retains its unknown
credential privately until it can be linked.
`--no-save` retains access only during the CLI process. `Client().connect()` keeps
credentials in the Python process unless `save=True` is explicit.

Every command accepts `--base-url`, `--workspace`, `--timeout`. `WELT_BASE_URL` and
`WELT_WORKSPACE` provide defaults. Credential precedence: explicit SDK key, then
`WELT_API_KEY`, then process/saved profile. The workspace option verifies that
credential's tenant before sending private input; it cannot retarget a bearer key.
Unset `WELT_API_KEY` and `WELT_WORKSPACE` if they override the default you selected
with `workspaces use`. Use explicit per-command selection for concurrent agents;
the shared active profile is a convenience for interactive use. CLI origins must
be HTTPS or explicit localhost HTTP. Redirects and cross-origin requests never
receive bearer credentials.

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

Use IDs returned by earlier commands. `models list` reports actual availability,
qualified tasks/configurations and limits; commands do not activate held models.
Table JSON can be supplied using `--input -` from stdin. Prediction input contains
`columns` and `rows`; use `--probabilities` explicitly. Fit returns a durable job ID
and replay key immediately. `jobs wait` times out locally while accepted work
continues. Reconnect with its ID. Reuse the original idempotency key and identical
payload after an uncertain submission; inspect jobs before submitting again.
Replay keys are printed to stderr before submission, including generated ones.
Writes are not automatically retried; bounded transient retries apply to GETs.

## File uploads and causal discovery

```sh
welt uploads create train.csv --target outcome --schema schema.json
welt uploads get upload_id
welt uploads file train.csv --target outcome --schema schema.json --upload-id upload_id
welt discover --dataset ds_train --model arrow --seed 19 --constraints constraints.json --idempotency-key graph-run-000001
welt jobs result job_graph
welt jobs scores job_graph
```

Use `uploads create` first when recovery matters: keep the upload ID before
sending bytes. `uploads file` verifies file checksum/size/format/schema/target
against a resumed session, skips acknowledged chunks and completes validation.
`uploads chunk ID INDEX FILE` and `uploads complete ID` expose the lower-level
contract. CSV/Parquet support depends on the deployed server configuration and
qualified parser/envelope; do not assume every deployment enables Parquet.
Schema example: `{"columns":["x","outcome"],"types":["number","string"]}`.
Types follow the API file schema (`number`, `boolean`, `string`); use `string` for
categorical columns. Constraints retain required/forbidden edges and temporal
order; inspect model capabilities before using them. Causal graph scores are
method-defined scores, not normalized posterior probabilities. Python provides
validated graph objects and optional local Ergodic conversion.

## Other commands and authority

`datasets list/get`, `predictors list/get`, `jobs list/get`, `research list/get/download`
and `pricing` cover resource inspection. Research search is paginated; pass its
`next_cursor` as `--cursor`, and preserve exact version/checksum provenance for
downloads. `research download ID PATH --version HASH --max-bytes LIMIT` uses
verified downloads to a new local file; the configured server may meter downloads.
`jobs cancel ID --yes`, `datasets delete ID --yes` and
`causal-results delete ID --yes` are explicit mutations. Dataset deletion refuses
retained dependencies; the CLI never cascades. Primary result deletion retains
job/usage metadata. Never delete real customer state without an exact user request.

Account changes, workspace creation, invitations/member administration, key
administration and billing need browser account authority. Use
`welt console account|workspace|members|keys|billing|create [--no-browser]` to open
the corresponding control; signed-in browser state determines its active
workspace, independently of CLI profiles. `workspaces list` is local saved
connections, not an account-wide membership query. Payments require explicit
completion of browser Checkout/Portal. `pricing` only reads public offers.

## Output, errors and exit codes

Successful commands emit one JSON value on stdout. Login links/codes, submission
replay keys and errors go to stderr; secrets never appear. A script should parse
stdout only when exit status is zero. `capabilities` is a machine-readable command
inventory. Error JSON has `error.code`, `message`, `status_code`, `request_id`,
`job_id`, `retryable`, `retry_after`, `idempotency_key` where available.

- 0: success (a submitted job is still pending; inspect it separately).
- 1: API, local file, credential, validation or wait failure.
- 2: command syntax error (argparse usage text on stderr).
- 130: interrupted locally; server job state is unchanged.

Batch wait returns a manifest for `succeeded` or `partially_completed`; inspect
failed ranges and payload errors rather than counting missing rows as predictions.
An entirely failed/cancelled job raises its terminal error. Manifests remain
inspectable after payload retention expires; payload retrieval returns explicit
expiry errors. Use the server's `expires_at` and version fields.

## Hidden model runs and forecasting

```sh
welt models get chronos-2
welt run --input forecast.json --idempotency-key forecast-00000001
welt jobs wait job_returned --timeout 600
welt model-runs delete job_returned --yes
```

`forecast.json` uses the task payload in [Hidden model runs](model-runs.html).
`models list` stays curated; inspect a hidden model using its exact ID. Licence
scope is enforced by the server's workspace policy. `run` returns a durable job;
`jobs wait` returns its native task-specific JSON result. These bounded runs create
no reusable fitted predictor. Deleting a terminal run explicitly purges both its
private input and result, while job, replay and usage metadata remain.
