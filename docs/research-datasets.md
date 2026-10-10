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

## Keep metadata with the file

Pass `metadata=True` to also write the catalogue detail you downloaded against
beside the data, as `<path>.metadata.json` (for example `iris.parquet.metadata.json`).
The result is then a frozen `ResearchDownload` with `path`, `metadata_path`,
`version`, `sha256` and `metadata` (the full detail dictionary) instead of a plain path.

```python
from pathlib import Path
from tempfile import mkdtemp
from welt import Client

dataset_id = "asset-d839644b6de36fc5cafdda18c5daeca3"
version = "d839644b6de36fc5cafdda18c5daeca3284a5ce8bb6169aff2fcb3fbb4a12021"
with Client() as client:
    saved = client.download_research_dataset(
        dataset_id, Path(mkdtemp(prefix="welt-research-")) / "iris.parquet",
        version=version, metadata=True)
print(saved.path, saved.metadata_path, saved.version)
print(saved.metadata.get("display_name") or saved.metadata["name"])
for column in saved.metadata["schema"]:
    print(column["name"], column.get("description"))
```

Both destinations are checked before any request, and an existing file or symlink
at either raises `FileExistsError` with nothing written. The metadata file is
published create-only after the verified data file. The detail is written exactly
as served; optional fields such as `display_name`, `summary`, `tags`, column
`description`/`description_source` and `enrichment` may be absent, and future
fields are kept. A requested `version` may be the current `content.version` or
any earlier version listed in the detail's `previous_content`; the SDK then requests
exactly that version and verifies size and SHA-256 against that version's own content
descriptor. Any other version raises `InvalidInputError` (`research_version_mismatch`)
before anything is written. The sidecar always holds the current detail, so compare
`saved.version` with `metadata["content"]["version"]` when you pinned an earlier one.
The default `metadata=False` behavior and its `pathlib.Path` return are unchanged.

## Load straight into a DataFrame

Install the `research` extra (`pip install "welt-client[research]"`, which provides
pandas and pyarrow), then load a table without choosing a path:

```python
from welt import Client

dataset_id = "asset-d839644b6de36fc5cafdda18c5daeca3"
version = "d839644b6de36fc5cafdda18c5daeca3284a5ce8bb6169aff2fcb3fbb4a12021"
with Client() as client:
    data, meta = client.load_research_dataset(dataset_id, version=version, metadata=True)
print(meta.name, meta.version)
print(meta.column_descriptions)
print(data.head())
```

`load_research_dataset` uses the same verified download into a private temporary
directory, reads it with pandas, and removes the temporary file. It returns a
DataFrame, or `(DataFrame, ResearchMetadata)` with `metadata=True`.
`ResearchMetadata` carries `dataset_id`, `version`, `sha256`, `media_type`,
`detail`, `license_text` and `attribution_text`. `version=None` uses the current
`content.version` from the detail; pin a version for reproducible experiments.
Pinned versions keep working after the catalogue publishes a newer one (for example
a restored-header table), as long as the detail lists them in `previous_content`;
`meta.version` and `meta.media_type` describe the version you actually loaded. Content
whose declared size exceeds `max_bytes` (default 64 MiB) is refused before
download. Without pandas/pyarrow the call raises `OptionalDependencyError`, an
`ImportError` naming `welt-client[research]`, before any request.

For a notice-bearing `application/zip`, the loader accepts only exactly
`dataset.parquet`, `LICENSE.txt` and `ATTRIBUTION.txt`, refusing other, duplicate,
nested, absolute or traversal names, encrypted or oversized members, and a table
that does not match its recorded hash (`WeltError` code `invalid_research_archive`).
For the current version that is `provenance.delivery.table_sha256` when recorded (a
restored-header archive, whose table differs from the canonical bytes), otherwise
`provenance.canonical_sha256`; earlier listed versions carry the canonical table and
are checked against `provenance.canonical_sha256`. It reads the table in memory and
emits a `ResearchNoticeWarning`: the licence and attribution apply to the table.
Use `metadata=True` to read `meta.license_text` and `meta.attribution_text`, and
retain them whenever you share the data. `AsyncClient` provides both methods with
the same contract.

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
files with the table when sharing. `download_research_dataset` writes verified
binary bytes and does not silently extract archives or discard notices;
`load_research_dataset` validates the archive in memory and exposes both notices
(see [Load straight into a DataFrame](#load-straight-into-a-dataframe)). The pinned Iris onboarding
dataset remains an ordinary Parquet download. Its pinned version is the canonical
table with the target column `__target__#0`; if a restored-header Iris version
(original column names) becomes current, the pinned example keeps downloading the
same canonical bytes through `previous_content`, so its `target = "__target__#0"`
stays correct. Use the current `content.version` and the detail's `targets` to
work with restored names instead.

`content.version` and `content.sha256` identify the complete downloaded bytes.
For a ZIP these are the archive checksum, while
`info["provenance"]["canonical_sha256"]` identifies the unchanged canonical
`dataset.parquet` inside it, or, for a restored-header archive,
`info["provenance"]["delivery"]["table_sha256"]` identifies its restored table. Keep the full detail metadata and distinguish these
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
