"""Verified research-file downloads, separate from private execution datasets."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import tempfile
import warnings
import zipfile

from .errors import InvalidInputError, OptionalDependencyError, WeltError

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


def checked_content(content):
    """Validate one content descriptor; return (version, sha256, size_bytes)."""
    if not isinstance(content, dict):
        raise invalid_download()
    digest, pinned, size = content.get("sha256"), content.get("version"), content.get("size_bytes")
    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(pinned, str) or not re.fullmatch(r"[0-9a-f]{64}", pinned)
            or digest != pinned or type(size) is not int or not 1 <= size <= MAX_DOWNLOAD_BYTES):
        raise invalid_download()
    return pinned, digest, size


def selected_content(detail, version):
    """Return (content descriptor, is_current) for ``version``.

    ``None`` or the current version selects ``content``; any version listed in
    ``previous_content`` selects that earlier immutable descriptor. Others are refused.
    """
    content = detail.get("content")
    if content is None:
        raise InvalidInputError("This research dataset has no downloadable content.", code="research_content_unavailable")
    pinned, _, _ = checked_content(content)
    if version is None or (isinstance(version, str) and version == pinned):
        return content, True
    previous = detail.get("previous_content") or []
    if not isinstance(previous, list):
        raise invalid_download()
    if isinstance(version, str):
        for old in previous:
            if isinstance(old, dict) and old.get("version") == version:
                return old, False
    raise InvalidInputError("Requested research version is neither the current content nor a listed "
                            "previous version.", code="research_version_mismatch")


def download_metadata(detail, dataset_id, version, max_bytes):
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_DOWNLOAD_BYTES:
        raise ValueError("max_bytes must be from 1 byte to 2 GiB.")
    if not isinstance(detail, dict) or detail.get("id") != dataset_id:
        raise invalid_download()
    content, _ = selected_content(detail, version)
    pinned, digest, size = checked_content(content)
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


# API-078: optional local metadata sidecars and DataFrame loading.
NOTICE_MEMBERS = ("dataset.parquet", "LICENSE.txt", "ATTRIBUTION.txt")
MAX_NOTICE_BYTES = 1024 * 1024
PARQUET = "application/vnd.apache.parquet"
ZIP = "application/zip"


class ResearchNoticeWarning(UserWarning):
    """A loaded research table is distributed with licence/attribution notices."""


@dataclass(frozen=True)
class ResearchDownload:
    """Verified research file plus the catalogue detail written beside it.

    ``version``/``sha256`` identify the complete downloaded bytes. ``metadata`` is
    the full research detail dictionary, also written to ``metadata_path``.
    """
    path: Path
    metadata_path: Path
    version: str
    sha256: str
    metadata: dict = field(repr=False)


@dataclass(frozen=True)
class ResearchMetadata:
    """Catalogue detail for a loaded research table.

    ``version``/``sha256`` identify the complete downloaded bytes (the archive for a
    ZIP). ``license_text``/``attribution_text`` are the notice members of a ZIP
    distribution and are ``None`` for an ordinary Parquet download.
    """
    dataset_id: str
    version: str
    sha256: str
    media_type: str
    detail: dict = field(repr=False)
    license_text: str | None = field(default=None, repr=False)
    attribution_text: str | None = field(default=None, repr=False)

    @property
    def name(self):
        """Curated display name when present, otherwise the catalogue name."""
        return self.detail.get("display_name") or self.detail.get("name")

    @property
    def column_descriptions(self):
        """Column name -> description for schema entries that carry one."""
        result = {}
        for column in self.detail.get("schema") or []:
            if (isinstance(column, dict) and isinstance(column.get("name"), str)
                    and isinstance(column.get("description"), str)):
                result[column["name"]] = column["description"]
        return result


def metadata_destination(path):
    return Path(str(Path(path)) + ".metadata.json")


def refuse_existing(*paths):
    for target in paths:
        target = Path(target)
        if target.exists() or target.is_symlink():
            raise FileExistsError("Research download destination already exists.")


def publish_metadata(path, detail):
    """Create-only, atomic, private publication of the detail JSON."""
    target = Path(path)
    body = (json.dumps(detail, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".welt-research-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        os.unlink(temporary)
    return target


def require_table_extra():
    """Import pandas/pyarrow lazily; name the extra that provides them."""
    try:
        import pandas
        import pyarrow  # noqa: F401  (pandas Parquet engine)
    except ImportError:
        raise OptionalDependencyError(
            "load_research_dataset needs pandas and pyarrow; install "
            "'welt-client[research]'.", code="optional_dependency_unavailable") from None
    return pandas


def invalid_archive():
    return WeltError("Research ZIP is not the expected notice-bearing distribution.",
                     code="invalid_research_archive")


def read_notice_zip(path, max_bytes, expected_table=None):
    """Validate exactly three fixed members; return (parquet bytes, licence, attribution)."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(NOTICE_MEMBERS) or set(names) != set(NOTICE_MEMBERS):
                raise invalid_archive()
            contents = {}
            for member in members:
                bound = max_bytes if member.filename == NOTICE_MEMBERS[0] else MAX_NOTICE_BYTES
                if member.is_dir() or member.flag_bits & 0x1 or not 0 <= member.file_size <= bound:
                    raise invalid_archive()
                with archive.open(member) as handle:  # CRC-checked; reads declared size only.
                    data = handle.read(bound + 1)
                if len(data) != member.file_size:
                    raise invalid_archive()
                contents[member.filename] = data
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, OSError, EOFError):
        raise invalid_archive() from None
    table = contents[NOTICE_MEMBERS[0]]
    if expected_table is not None and sha256(table).hexdigest() != expected_table:
        raise invalid_archive()
    try:
        return (table, contents["LICENSE.txt"].decode("utf-8"),
                contents["ATTRIBUTION.txt"].decode("utf-8"))
    except UnicodeDecodeError:
        raise invalid_archive() from None


def table_sha256(detail, current):
    """Expected SHA-256 of ``dataset.parquet`` inside a notice ZIP.

    The current restored archive binds its restored table in
    ``provenance.delivery.table_sha256``; canonical archives (and earlier versions,
    which carry the canonical table) bind ``provenance.canonical_sha256``.
    """
    provenance = detail.get("provenance")
    if not isinstance(provenance, dict):
        return None
    delivery = provenance.get("delivery")
    if current and isinstance(delivery, dict) and delivery.get("table_sha256") is not None:
        candidate = delivery["table_sha256"]
        if not (isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{64}", candidate)):
            raise invalid_archive()
        return candidate
    candidate = provenance.get("canonical_sha256")
    return candidate if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{64}", candidate) else None


def read_table(pandas, path, detail, dataset_id, pinned, digest, max_bytes):
    """Read a verified local download; return (DataFrame, ResearchMetadata)."""
    content, current = selected_content(detail, pinned)
    media_type = content.get("media_type")
    licence = attribution = None
    if media_type == PARQUET:
        frame = pandas.read_parquet(path)
    elif media_type == ZIP:
        table, licence, attribution = read_notice_zip(path, max_bytes, table_sha256(detail, current))
        frame = pandas.read_parquet(io.BytesIO(table))
        warnings.warn(
            f"Research dataset {dataset_id} is distributed with LICENSE.txt and "
            "ATTRIBUTION.txt; their terms apply to this table. Use metadata=True to "
            "read license_text and attribution_text, and retain both when sharing.",
            ResearchNoticeWarning, stacklevel=3)
    else:
        raise InvalidInputError("This research content type cannot be loaded as a table; "
                                "use download_research_dataset.",
                                code="research_media_type_unsupported")
    return frame, ResearchMetadata(dataset_id, pinned, digest, media_type, detail,
                                   licence, attribution)
