# Connect your workspace

## Terminal

Install the SDK, run this command, then approve the named workspace in your browser:

```sh
welt login
```

The browser shows the requested read/write permissions and 30-day key lifetime
before you approve. Read access is required; write access is optional. A read-only
connection can inspect/download data but fitting and prediction need write access.
When browser approval issues a key, the command saves it
privately. Existing explicit/environment keys are reused and are not copied to
disk; an existing process-connected key can be saved with explicit `save=True`.
The key is returned to the waiting SDK once; it is never shown
in the browser or command output. `welt login` saves it privately for later Python
processes. To revoke it, use your workspace's API keys page.

## Notebook or remote Python

```python
from welt import Client

with Client() as client:
    client.connect()
```

Open the printed verification link on your own computer. The short verification
code is safe to compare with the browser; never share a device secret or API key.
There is no localhost callback and no helper file. `connect()` retains access in
this Python process so subsequent `Classifier()`/`Regressor()`/`Client()` calls
use the approved origin and credentials. It never changes environment variables.

To retain the connection after the process exits, opt in explicitly:

```python
with Client() as client:
    client.connect(save=True)
```

Use `open_browser=False` or terminal `welt login --no-browser` when the process
has no local browser. The verification link is always printed. Interrupting the
wait stops local polling; the outstanding browser code expires after ten minutes.
Declined, expired, consumed and unavailable connections raise actionable errors.
Do not blindly repeat a fit when an unrelated waiting operation times out.

## Credential choice and storage

An explicit `api_key` wins, then `WELT_API_KEY`, then the most recent connection
in this Python process for the exact API origin, then a saved origin credential.
The origin follows explicit `base_url`, `WELT_BASE_URL`, then the hosted default.
A localhost key or another service's key is never loaded for the hosted origin.
Constructors do not sign in, open browsers, create keys or write credentials.
`connect()` verifies an existing credential using workspace metadata and reuses it
without another key; invalid/expired credentials can be reconnected explicitly.
Insufficient permissions require an appropriate key rather than silently replacing it.

POSIX systems store credentials under `~/.config/welt` with directory mode0700 and
file mode0600. `WELT_CONFIG_DIR` can select another private directory. Saved files
contain a secret: keep the directory outside your repository and backups intended
for sharing. Symlinks and unsafe permissions are rejected. Expired saved keys are
ignored; their server key remains revocable in the console. Windows users should
use process-only `connect()` or configured `WELT_API_KEY`; secure persisted Windows
storage is not yet supported. `welt login --no-save` validates access without
saving it; the command's process exits afterward, so use it only as a connection check.

Keep keys out of notebook cells, Git, logs and screenshots. Manual API-key setup
remains available in Account. Native model availability and permissions still
apply after successful connection; authentication does not qualify every model.

Process-only connection credentials are not serialized into child processes.
For process-based parallel cross-validation, configure `WELT_API_KEY` in workers
or explicitly save a private origin credential first. Never pickle a secret.
