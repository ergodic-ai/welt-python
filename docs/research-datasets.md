# Research datasets

Research datasets are a catalogue of the original p10k research snapshot, separate
from your workspace's private execution datasets. Listing or downloading research
content does not upload it, fit a model or change its targets or splits.
Catalogue metadata, downloadable content and model qualification are different
things: inspect availability, license notices, schema, recorded targets/splits
and provenance before using a dataset. Missing source information stays missing.

```python
import os
from pathlib import Path
from welt import Client

with Client(base_url="https://welt.ergodic.dev",
            api_key=os.environ["WELT_API_KEY"]) as client:
    page = client.research_datasets(q="iris", availability="available", limit=10)
    for item in page["items"]:
        print(item["id"], item["name"], item["availability"])

    # Choose an ID from the real catalogue, rather than inventing a source URL.
    chosen = page["items"][0]["id"]
    info = client.research_dataset(chosen)
    print(info["license"], info["schema"], info["targets"], info["splits"])
    content = info["content"]
    path = client.download_research_dataset(
        chosen, Path("research.parquet"), version=content["version"])
```

A page preserves `next_cursor`, `total`, `catalogue_version` and `facets`. Request
the next page by passing `cursor=page["next_cursor"]`; stop when it is `None`.
Filters are `q`, `source`, `availability` and `license`, with `limit` from 1 to 50.
`research_dataset(id)` returns full metadata. `content=None` means there are no
eligible downloadable bytes; a `blocked_reason` explains the recorded boundary.
Do not interpret metadata presence or a source retrieval signal as redistribution
permission or a model-compatible training recipe.

Downloads request only the service's canonical content endpoint, with an explicit
version pinned to its SHA-256. The SDK checks exact byte size and digest, rejects
redirects and content encodings, and streams at most 64 KiB per write. The default
64 MiB file limit covers each file in the canonical snapshot; callers may choose
`max_bytes` up to 2 GiB after inspecting the size. This is a per-file safety limit,
not permission to fetch arbitrary URLs or an automatic bulk download.

The caller chooses the destination. Catalogue filenames and URLs never choose a
local path. A download creates a private temporary file, publishes complete
verified bytes atomically, and refuses an existing file or symlink; interrupted,
corrupt or cancelled downloads leave no destination. An existing destination
raises `FileExistsError`; choose a new path rather than deleting data as cleanup.
Inspect/save `info` alongside a research experiment to retain its license,
version and provenance. No output or dataset belongs in Git by default.

```python
from welt import AsyncClient

async def fetch(dataset_id, path):
    async with AsyncClient(base_url="https://welt.ergodic.dev") as client:
        info = await client.research_dataset(dataset_id)
        return await client.download_research_dataset(
            dataset_id, path, version=info["content"]["version"])
```

Both clients use the same current workspace-key authorization and typed errors.
GET metadata has bounded retries. An interrupted content stream raises a sanitized
`TransportError`; rerun the download into a new destination after checking the
catalogue. Partial bytes are never resumed or accepted implicitly.

The downloaded format is Parquet. For local inspection install the existing
`parquet` extra and use `pandas.read_parquet(path)`. Full research files can exceed
a serving model's row/feature/class bounds; inspect `Client.models()` and use an
explicitly declared preparation/split recipe before uploading. There is no silent
sampling, target inference, scientific benchmark claim or automatic upload here.

SDK 0.6 adds this research surface to the released 0.3 interfaces. Resumable CSV
0.4 and durable batch 0.5 remain separate unreleased candidates; neither is included
in this package. Actual endpoint availability follows the independently reviewed
service release. See [first request](onboarding.md) and [capabilities](capabilities.md).
