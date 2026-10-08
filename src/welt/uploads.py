"""Bounded local file reads shared by sync and async resumable upload clients."""
import hashlib
from pathlib import Path

from .errors import InvalidInputError


def declaration(path, target=None, schema=None):
    path = Path(path)
    format = {".csv": "csv", ".parquet": "parquet"}.get(path.suffix.lower())
    if format is None:
        raise InvalidInputError("File must be CSV or Parquet.", code="invalid_input")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while body := source.read(1024 * 1024):
            size += len(body)
            digest.update(body)
    if not size:
        raise InvalidInputError("Cannot upload an empty file.", code="invalid_input")
    return dict(format=format, size_bytes=size, sha256=digest.hexdigest(), target=target, schema=schema)


def matches(info, declared, upload_id=None):
    if not isinstance(info, dict):
        raise InvalidInputError("Server returned invalid upload metadata.", code="invalid_response")
    if (not isinstance(info.get("id"), str) or not 0 < len(info["id"]) <= 128
        or upload_id is not None and info["id"] != str(upload_id)):
        raise InvalidInputError("Server returned a different upload identity.", code="invalid_response")
    if any(info.get(k) != v for k, v in declared.items()):
        raise InvalidInputError("Local file or upload schema differs from the original declaration.", code="upload_source_changed")
    if (type(info.get("chunk_bytes")) is not int or not 0 < info["chunk_bytes"] <= 10 * 1024 * 1024
        or not isinstance(info.get("received_indices"), list)
        or any(type(i) is not int or i < 0 for i in info["received_indices"])):
        raise InvalidInputError("Server returned invalid upload chunk metadata.", code="invalid_response")
    indices = info["received_indices"]
    count = (declared["size_bytes"] + info["chunk_bytes"] - 1) // info["chunk_bytes"]
    if len(set(indices)) != len(indices) or any(i >= count for i in indices):
        raise InvalidInputError("Server returned invalid received chunk indices.", code="invalid_response")


def chunks(path, info, declared):
    digest = hashlib.sha256()
    size = 0
    seen = set(info["received_indices"])
    with Path(path).open("rb") as source:
        index = 0
        while body := source.read(info["chunk_bytes"]):
            digest.update(body)
            size += len(body)
            if size > declared["size_bytes"]:
                raise InvalidInputError("File changed during upload.", code="upload_source_changed")
            if index not in seen:
                yield index, body, hashlib.sha256(body).hexdigest()
            index += 1
    if size != declared["size_bytes"] or digest.hexdigest() != declared["sha256"]:
        raise InvalidInputError("File changed during upload.", code="upload_source_changed")


def annotate(error, upload_id):
    # An interrupted caller can persist just the opaque ID and resume later.
    error.upload_id = upload_id
    return error
