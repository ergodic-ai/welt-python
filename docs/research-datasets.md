# Research datasets

Research datasets are a catalogue of the original p10k research snapshot, separate
from your workspace's private execution datasets. Listing or downloading research
content does not upload it, fit a model or change its targets or splits.
Catalogue metadata, downloadable content and model qualification are different
things: inspect availability, license notices, schema, recorded targets/splits
and provenance before using a dataset. Missing source information stays missing.

```python
from pathlib import Path
from welt import Client

with Client() as client:  # Connect once using welt login or Client.connect().
    page = client.research_datasets(q="iris", availability="available", limit=10)
    for item in page["items"]:
        print(item["id"], item["name"], item["availability"])

    # Search is discovery; choose this exact reviewed asset deliberately.
    chosen = "asset-d839644b6de36fc5cafdda18c5daeca3"  # Reviewed Iris asset.
    info = client.research_dataset(chosen)
    print(info["license"], info["schema"], info["targets"], info["splits"])
    content = info["content"]
    print(content["media_type"], content["filename"])
    suffix = {"application/vnd.apache.parquet": ".parquet",
              "application/zip": ".zip"}[content["media_type"]]
    from tempfile import mkdtemp
    path = client.download_research_dataset(
        chosen, Path(mkdtemp(prefix="welt-research-")) / ("research" + suffix),
        version="d839644b6de36fc5cafdda18c5daeca3284a5ce8bb6169aff2fcb3fbb4a12021")

import pandas as pd
data = pd.read_parquet(path)
print(data.head())
print(data.shape)
data.info()
```

Continue with [Start](start.md) to split this exact table, fit a classifier, predict
held-out rows and measure their quality. Listing/download itself does not model.

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

Inspect `content.media_type` and its `filename` hint before opening a download.
Ordinary tables are `application/vnd.apache.parquet`; install the existing
`parquet` extra and use `pandas.read_parquet(path)`. Some source-qualified datasets
are `application/zip` notice-bearing distributions containing exactly
`dataset.parquet`, `LICENSE.txt` and `ATTRIBUTION.txt`. Explicitly extract those
three members together into a new experiment directory; use fixed local member
names, refuse unexpected members or existing destinations, and retain both notice
files with the table when sharing. The SDK downloads verified binary bytes and
does not silently extract archives or discard notices. The pinned Iris onboarding
dataset remains an ordinary Parquet download.

`content.version` and `content.sha256` identify the complete downloaded bytes.
For a ZIP these are the archive checksum, while
`info["provenance"]["canonical_sha256"]` identifies the unchanged canonical
`dataset.parquet` inside it. Keep the full detail metadata and distinguish these
two checksums when checking the extracted table. Downloadability remains a
dataset-specific serving decision; it does not imply that every catalogue entry
has eligible content.

Full research files can exceed
a serving model's row/feature/class bounds; inspect `Client.models()` and use an
explicitly declared preparation/split recipe before uploading. There is no silent
sampling, target inference, scientific benchmark claim or automatic upload here.

SDK 0.6 adds this research surface to the released 0.3 interfaces. Resumable CSV
0.4 and durable batch 0.5 remain separate unreleased candidates; neither is included
in this package. Actual endpoint availability follows the independently reviewed
service release. See [first request](onboarding.md) and [capabilities](capabilities.md).
