# Connect your workspace

Connect the SDK to your Welt workspace from a terminal or from Python. You approve
access in your browser, so you don't need to copy a key.

## Connect from a terminal {#terminal}

Install the SDK, run this command, then approve the named workspace in your browser:

```sh
welt login
```

Before you approve, the browser shows the permissions requested (read and write) and
the key's 30-day lifetime. Read access is required; write access is optional. With
read access only, you can inspect and download data, but fitting and prediction need
write access.

The key goes back to the waiting SDK once. It is never shown in the browser or in
the command output. `welt login` saves it privately for later Python processes. To
revoke it, use your workspace's API keys page.

<details markdown="1"><summary>What if I already have a key configured?</summary>

The command saves a key only when browser approval issues one. Keys you already pass
explicitly or set in the environment are reused and are not copied to disk. A key
already connected in the current process is saved only if you ask with `save=True`.

</details>

## Connect from a notebook or remote Python {#notebook-or-remote-python}

```python
from welt import Client

with Client() as client:
    client.connect()
```

Open the printed verification link on your own computer. The short verification code
is safe to compare with the one in the browser. Never share a device secret or an
API key.

There is no localhost callback and no helper file. `connect()` keeps access in this
Python process, so later `Classifier()`, `Regressor()` and `Client()` calls use the
approved origin and credentials. It never changes environment variables.

To keep the connection after the process exits, opt in explicitly:

```python
with Client() as client:
    client.connect(save=True)
```

!!! tip "No browser on this machine?"
    Use `open_browser=False`, or `welt login --no-browser` in a terminal. The
    verification link is always printed.

If you interrupt the wait, local polling stops; the outstanding browser code expires
after ten minutes. Declined, expired, consumed (already used) and unavailable
connections raise errors that tell you what to do next.

!!! caution "Don't repeat a fit because something else timed out"
    If an unrelated waiting operation times out, don't blindly repeat a fit. See
    [waiting and recovery](async-and-errors.md).

## Which credential the SDK uses {#credential-choice-and-storage}

The SDK uses the first credential it finds, in this order:

1. An explicit `api_key`.
2. `WELT_API_KEY`.
3. The most recent connection in this Python process for the exact API origin.
4. A saved credential for that origin.

The origin (the API address) comes from an explicit `base_url`, then `WELT_BASE_URL`,
then the hosted default. A localhost key or another service's key is never loaded
for the hosted origin.

Constructors never sign in, open a browser, create keys or write credentials.
`connect()` checks an existing credential using workspace metadata and reuses it
without creating another key. If a credential is invalid or expired, you can
reconnect explicitly. If it lacks permissions, you need a key with the right
permissions; the SDK doesn't silently replace it.

## Where credentials are stored

On POSIX systems, credentials are stored under `~/.config/welt`, with directory
mode 0700 and file mode 0600. Set `WELT_CONFIG_DIR` to choose another private
directory. The SDK rejects symlinks and unsafe permissions. It ignores expired saved
keys, but their server key stays revocable in the console.

!!! caution "Saved files contain a secret"
    Keep the directory outside your repository and outside backups you share. Keep
    keys out of notebook cells, Git, logs and screenshots.

!!! note "Windows"
    Secure saved storage is not yet supported on Windows. Use process-only
    `connect()` or set `WELT_API_KEY`.

`welt login --no-save` checks access without saving it. The command's process then
exits, so use it only as a connection check.

Manual API-key setup is still available in Account. After you connect, native model
availability and permissions still apply: authentication does not qualify every model.

<details markdown="1"><summary>Running cross-validation in parallel processes?</summary>

Process-only connection credentials are not serialized into child processes. For
process-based parallel cross-validation, set `WELT_API_KEY` in the workers, or
explicitly save a private origin credential first. Never pickle a secret.

</details>

<a id="multiple-workspaces-source-candidate"></a>

## Multiple workspaces

Name the workspace you want when you connect or when you create a client:

```python
from welt import Client, AsyncClient

client = Client().connect(workspace="acme-research")
# Explicitly persist a revocable workspace credential only when desired:
client.connect(workspace="acme-research", save=True)

with Client(workspace="acme-research") as selected:
    print(selected.workspace())
# Async resources use the same saved selection; browser connection itself is sync.
async_client = AsyncClient(workspace="acme-research")
```

`workspace` takes an exact workspace ID or slug. Login requests that workspace and
rejects approval of any other.

Constructors select an existing credential without starting login. If there is no
credential, they fail with guidance on how to log in. A manual or environment key
still takes precedence, and it must match the selected workspace (tenant) before
private operations.

## Next steps

<div class="cards" markdown="1">

- [Your first prediction](start.md): train and evaluate a model on a real table.
- [CLI commands and agent workflows](cli.md): select profiles and recover access.
- [Wait and recover](async-and-errors.md): handle timeouts without repeating work.

</div>
