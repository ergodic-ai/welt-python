# Research datasets

Find a table, download it, or load it directly into a DataFrame. These are research
examples; downloading a table does not upload it into your private workspace.

## Find a table

```python
from welt import Client

with Client() as client:
    page = client.research_datasets(q="iris", availability="available", limit=10)
    for item in page["items"]:
        print(item["id"], item["name"], item["rows"], item["columns"])
```

Use the dataset ID to inspect its columns and recorded targets:

```python
dataset_id = "asset-d839644b6de36fc5cafdda18c5daeca3"
with Client() as client:
    info = client.research_dataset(dataset_id)
print([column["name"] for column in info["schema"]])
print(info["targets"])
```

For this Iris table the outcome column is `target`. For Yacht Hydrodynamics it is
`Residuary.resistance`. Use the names in the downloaded table and its metadata.

## Load a DataFrame

Install the reader with `pip install "welt-client[parquet]"`, then load a table:

```python
with Client() as client:
    data = client.load_research_dataset(dataset_id)
print(data.head())
print(data.shape)
```

The loader downloads the current published content, verifies it, reads it with
pandas and removes its temporary file. It handles supported Parquet and archive
content. Pass `metadata=True` when you also want the table's column descriptions:

```python
with Client() as client:
    data, meta = client.load_research_dataset(dataset_id, metadata=True)
print(meta.name)
print(meta.column_descriptions)
```

Continue with [your first prediction](start.md) to split Iris into training and
test rows, fit a classifier and evaluate its predictions.

## Save a file

```python
from pathlib import Path
from tempfile import mkdtemp

path = Path(mkdtemp(prefix="welt-research-")) / "iris.parquet"
with Client() as client:
    client.download_research_dataset(dataset_id, path)
print(path)
```

With no version argument, the SDK downloads the current published content.
It verifies the byte size and SHA-256 before publishing the file. An existing
file or symlink raises `FileExistsError`; choose a new path to save another copy.
Interrupted downloads leave no partial destination.

Pass `metadata=True` to also save `<path>.metadata.json` and return a
`ResearchDownload` containing both paths and the dataset metadata:

```python
with Client() as client:
    saved = client.download_research_dataset(
        dataset_id, Path(mkdtemp(prefix="welt-research-")) / "iris.parquet",
        metadata=True)
print(saved.path, saved.metadata_path)
```

`download_research_dataset` saves the served file as-is. Inspect
`info["content"]["media_type"]` when choosing how to open it; use
`load_research_dataset` to read supported archives directly into a DataFrame.
The default download size limit is 128 MiB; pass `max_bytes` for a larger table,
up to 2 GiB. A dataset whose `content` is `None` has no downloadable file.

## Async downloads

```python
from welt import AsyncClient

async def fetch(dataset_id, path):
    async with AsyncClient() as client:
        return await client.download_research_dataset(dataset_id, path)
```

Both clients use your workspace credentials and the same download checks.
A listing returns `next_cursor`; pass it as `cursor` to get the next page.
Search filters include `q`, `source` and `availability`, with `limit` from 1 to 50.

Before fitting, inspect the [model's limits](capabilities.md). Keep test rows
separate from training rows and select a target explicitly. The [notebooks](notebooks.md)
show complete classification, regression, async and causal workflows.
