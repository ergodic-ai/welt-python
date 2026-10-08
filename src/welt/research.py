"""Verified research-file downloads, separate from private execution datasets."""
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile

from .errors import InvalidInputError, WeltError

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024


def research_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("research dataset ID must be a nonempty catalogue identifier.")
    return value


def search_params(q, source, availability, license, limit, cursor):
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("limit must be an integer from 1 to 50.")
    result = {"limit": limit}
    for key, value, bound in (("q", q, 200), ("source", source, 100),
                               ("availability", availability, 40), ("license", license, 1000),
                               ("cursor", cursor, 512)):
        if value is not None:
            if not isinstance(value, str) or not value or len(value) > bound:
                raise ValueError(f"{key} must be a nonempty string of at most {bound} characters.")
            result[key] = value
    return result


def invalid_download():
    return WeltError("Research download did not match its pinned metadata.", code="invalid_research_download")


def download_metadata(detail, dataset_id, version, max_bytes):
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_DOWNLOAD_BYTES:
        raise ValueError("max_bytes must be from 1 byte to 2 GiB.")
    if not isinstance(detail, dict) or detail.get("id") != dataset_id:
        raise invalid_download()
    content = detail.get("content")
    if content is None:
        raise InvalidInputError("This research dataset has no downloadable content.", code="research_content_unavailable")
    if not isinstance(content, dict):
        raise invalid_download()
    digest, pinned, size = content.get("sha256"), content.get("version"), content.get("size_bytes")
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(pinned, str) or not re.fullmatch(r"[0-9a-f]{64}", pinned)
            or digest != pinned or type(size) is not int or not 1 <= size <= MAX_DOWNLOAD_BYTES):
        raise invalid_download()
    if version is not None and (not isinstance(version, str) or version != pinned):
        raise InvalidInputError("Requested research version differs from the current pinned content.", code="research_version_mismatch")
    if size > max_bytes:
        raise InvalidInputError("Research content exceeds max_bytes; inspect its size before choosing a larger limit.", code="research_download_limit")
    return pinned, digest, size


class Download:
    def __init__(self, handle, digest, size):
        self.handle, self.digest, self.size = handle, digest, size
        self.hasher, self.received = sha256(), 0

    def headers(self, response):
        if response.status_code != 200 or response.headers.get("content-encoding", "identity").lower() != "identity":
            raise invalid_download()
        value = response.headers.get("content-length")
        if value is not None and (len(value) > 20 or not value.isascii() or not value.isdigit() or int(value) != self.size):
            raise invalid_download()

    def write(self, chunk):
        self.received += len(chunk)
        if self.received > self.size:
            raise invalid_download()
        self.handle.write(chunk)
        self.hasher.update(chunk)

    def finish(self):
        if self.received != self.size or self.hasher.hexdigest() != self.digest:
            raise invalid_download()
        self.handle.flush()
        os.fsync(self.handle.fileno())


@contextmanager
def destination(path, digest, size):
    """Publish complete verified bytes atomically, refusing existing destinations."""
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Research download destination already exists.")
    # Caller chooses the path. Never derive it from untrusted catalogue filenames.
    fd, temporary = tempfile.mkstemp(prefix=".welt-research-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer = Download(handle, digest, size)
            yield writer
            writer.finish()
        os.link(temporary, target)  # Exclusive publication also covers a concurrent creator.
    finally:
        os.unlink(temporary)
