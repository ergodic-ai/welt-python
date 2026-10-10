"""Catalogue/download contract controls; synthetic bytes only, no remote datasets."""
import asyncio
from hashlib import sha256
import json
import os

import httpx
import pytest

from welt import AsyncClient, Client
from welt.errors import InvalidInputError, PermissionDeniedError, TransportError, WeltError

BODY = b'controlled-only research bytes\n'
DIGEST = sha256(BODY).hexdigest()
DETAIL = dict(id='p10k-example', content=dict(version=DIGEST, sha256=DIGEST,
    size_bytes=len(BODY), media_type='application/vnd.apache.parquet', filename='example.parquet'),
    license='controlled-test', targets=[], splits=[])


def service(detail=DETAIL, body=BODY, status=200, headers=None):
    seen = []
    def handle(request):
        seen.append(request)
        assert request.headers['Authorization'] == 'Bearer test-only'
        if request.url.path == '/v1/research-datasets':
            return httpx.Response(200, json=dict(items=[detail], next_cursor='next', total=1))
        if request.url.path.endswith('/content'):
            assert request.url.params['version'] == detail['content']['version']
            assert request.headers['Accept-Encoding'] == 'identity'
            return httpx.Response(status, headers=headers or {}, stream=httpx.ByteStream(body))
        assert request.url.path == '/v1/research-datasets/p10k-example'
        return httpx.Response(200, json=detail)
    return httpx.MockTransport(handle), seen


def test_search_paging_detail_and_private_resources_stay_separate():
    transport, seen = service()
    with Client(api_key='test-only', transport=transport) as c:
        assert c.research_datasets(q='a & b', source='p10k', availability='available',
                                  license='CC0', limit=1, cursor='previous')['next_cursor'] == 'next'
        assert c.research_dataset('p10k-example') == DETAIL
    assert dict(seen[0].url.params) == dict(q='a & b', source='p10k', availability='available',
                                           license='CC0', limit='1', cursor='previous')


@pytest.mark.parametrize('kwargs', [dict(limit=True), dict(limit=0), dict(limit=51),
    dict(q='x'*201), dict(cursor='x'*513), dict(source=3), dict(license=''), dict(license='x'*1001)])
def test_search_rejects_invalid_bounds_before_http(kwargs):
    with Client(transport=httpx.MockTransport(lambda r: pytest.fail('no HTTP'))) as c:
        with pytest.raises(ValueError): c.research_datasets(**kwargs)


@pytest.mark.parametrize('identifier', ['', '..', '../private', 'x/y', 'https://evil', 'x'*129, None])
def test_ids_cannot_escape_catalogue(identifier):
    with Client(transport=httpx.MockTransport(lambda r: pytest.fail('no HTTP'))) as c:
        with pytest.raises(ValueError): c.research_dataset(identifier)


def test_sync_download_pins_bytes_size_and_creates_private_file(tmp_path):
    transport, seen = service()
    path = tmp_path/'chosen-name.parquet'
    with Client(api_key='test-only', transport=transport) as c:
        assert c.download_research_dataset('p10k-example', path, version=DIGEST) == path
    assert path.read_bytes() == BODY
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert len(seen) == 2
    assert not list(tmp_path.glob('.welt-*'))


@pytest.mark.parametrize('body,headers,status', [
    (BODY+b'extra', {}, 200), (BODY[:-1], {}, 200), (b'x'*len(BODY), {}, 200),
    (BODY, {'Content-Length':'0'}, 200), (BODY, {'Content-Length':'9'*10000}, 200),
    (BODY, {'Content-Encoding':'gzip'}, 200),
    (b'', {'Location':'https://external.invalid/private'}, 302)])
def test_invalid_download_never_publishes_or_follows_redirect(tmp_path, body, headers, status):
    transport, seen = service(body=body, headers=headers, status=status)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError): c.download_research_dataset('p10k-example', tmp_path/'data')
    assert len(seen) == 2
    assert list(tmp_path.iterdir()) == []


def test_unavailable_content_and_version_or_size_refuse_before_stream(tmp_path):
    for detail, kwargs in [(dict(id='p10k-example', content=None), {}),
                            (DETAIL, dict(version='0'*64)), (DETAIL, dict(max_bytes=1))]:
        transport, seen = service(detail)
        with Client(api_key='test-only', transport=transport) as c:
            with pytest.raises(InvalidInputError): c.download_research_dataset('p10k-example', tmp_path/'data', **kwargs)
        assert len(seen) == 1
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', [dict(size_bytes=True), dict(sha256='oops'),
    dict(version='https://external'), dict(size_bytes=2**32)])
def test_malformed_metadata_refused(tmp_path, change):
    detail = {**DETAIL, 'content':{**DETAIL['content'], **change}}
    transport, seen = service(detail)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError): c.download_research_dataset('p10k-example', tmp_path/'data')
    assert len(seen) == 1


def test_existing_file_and_symlink_not_overwritten(tmp_path):
    path=tmp_path/'keep'; path.write_bytes(b'existing')
    link=tmp_path/'link'; link.symlink_to(path)
    transport, seen = service()
    with Client(api_key='test-only', transport=transport) as c:
        for target in (path, link):
            with pytest.raises(FileExistsError): c.download_research_dataset('p10k-example', target)
    assert path.read_bytes() == b'existing' and link.is_symlink()
    assert all(not r.url.path.endswith('/content') for r in seen)


def test_concurrent_destination_creator_wins_without_overwrite(tmp_path, monkeypatch):
    from welt import research
    original = research.os.link
    def race(source, target):
        target.write_bytes(b'other writer')
        original(source, target)
    monkeypatch.setattr(research.os, 'link', race)
    transport, _ = service()
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(FileExistsError): c.download_research_dataset('p10k-example', tmp_path/'data')
    assert (tmp_path/'data').read_bytes() == b'other writer'
    assert not list(tmp_path.glob('.welt-*'))


def test_bounded_error_body_keeps_typed_error_and_no_file(tmp_path):
    body=json.dumps({'error':dict(code='dataset_unavailable', message='Unavailable', retryable=False)}).encode()
    transport, _=service(body=body, status=403)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(PermissionDeniedError): c.download_research_dataset('p10k-example', tmp_path/'data')
    transport, _=service(body=b'x'*70000, status=403)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError, match='pinned metadata'): c.download_research_dataset('p10k-example', tmp_path/'data')
    assert not list(tmp_path.iterdir())


def test_interrupted_stream_preserves_transport_type_and_cleans(tmp_path):
    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield BODY[:4]
            raise httpx.ReadError('private transport detail')
    def handle(request):
        return (httpx.Response(200, stream=Broken()) if request.url.path.endswith('/content')
                else httpx.Response(200, json=DETAIL))
    with Client(transport=httpx.MockTransport(handle), max_retries=0) as c:
        with pytest.raises(TransportError) as error: c.download_research_dataset('p10k-example', tmp_path/'data')
    assert 'private transport detail' not in str(error.value)
    assert not list(tmp_path.iterdir())


def test_async_search_detail_download_and_cancel_cleanup(tmp_path):
    async def run():
        transport, _ = service()
        async with AsyncClient(api_key='test-only', transport=transport) as c:
            assert (await c.research_datasets(limit=1))['total'] == 1
            assert await c.download_research_dataset('p10k-example', tmp_path/'good') == tmp_path/'good'
        class Cancelled(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield BODY[:3]
                raise asyncio.CancelledError()
        async def handle(request):
            return (httpx.Response(200, stream=Cancelled()) if request.url.path.endswith('/content')
                    else httpx.Response(200, json=DETAIL))
        async with AsyncClient(transport=httpx.MockTransport(handle)) as c:
            with pytest.raises(asyncio.CancelledError):
                await c.download_research_dataset('p10k-example', tmp_path/'cancelled')
    asyncio.run(run())
    assert (tmp_path/'good').read_bytes() == BODY
    assert not (tmp_path/'cancelled').exists()
    assert not list(tmp_path.glob('.welt-*'))


def test_catalogue_urls_and_filename_are_not_used(tmp_path):
    detail={**DETAIL, 'content':{**DETAIL['content'], 'filename':'../../other',
        'url':'https://external.invalid/secret'}, 'download_url':'https://external.invalid'}
    transport, seen=service(detail)
    with Client(api_key='test-only', transport=transport) as c:
        # Even an altered underlying HTTP client must not forward credentials.
        c.http.follow_redirects = True
        c.download_research_dataset('p10k-example', tmp_path/'chosen')
    assert (tmp_path/'chosen').read_bytes() == BODY
    assert all(r.url.host == 'welt.ergodic.dev' for r in seen)


def test_version_must_equal_content_checksum(tmp_path):
    detail={**DETAIL, 'content':{**DETAIL['content'], 'version':'0'*64}}
    transport, seen=service(detail)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError): c.download_research_dataset('p10k-example', tmp_path/'data')
    assert len(seen) == 1


# API-078: metadata sidecars and DataFrame loading (controlled synthetic tables only).
import dataclasses
import io
import warnings
import zipfile

import pandas as pd

from welt import ResearchDownload, ResearchMetadata, ResearchNoticeWarning
from welt.errors import OptionalDependencyError

FRAME = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'label': ['a', 'b', 'a']})


def parquet_bytes(frame=FRAME):
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def zip_bytes(members, compression=zipfile.ZIP_STORED):
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # Duplicate names are deliberate in malformed fixtures.
        with zipfile.ZipFile(buffer, 'w', compression=compression) as archive:
            for name, data in members:
                archive.writestr(name, data)
    return buffer.getvalue()


TABLE = parquet_bytes()
NOTICES = [('dataset.parquet', TABLE), ('LICENSE.txt', b'Controlled licence text\n'),
           ('ATTRIBUTION.txt', b'Controlled attribution\n')]


def detail_for(body, media_type='application/vnd.apache.parquet', **extra):
    digest = sha256(body).hexdigest()
    return {**DETAIL, 'content': dict(version=digest, sha256=digest, size_bytes=len(body),
            media_type=media_type, filename='dataset'), **extra}


def test_default_download_return_and_files_unchanged(tmp_path):
    transport, seen = service()
    with Client(api_key='test-only', transport=transport) as c:
        result = c.download_research_dataset('p10k-example', tmp_path/'data.parquet', metadata=False)
    assert result == tmp_path/'data.parquet' and type(result) is type(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['data.parquet']
    assert len(seen) == 2


def test_metadata_sidecar_written_after_verified_data(tmp_path):
    detail = {**DETAIL, 'display_name': 'Example', 'summary': 'Short.', 'tags': {'domain': ['test']},
              'schema': [dict(name='x', dtype='float', index=0, description='Measured x',
                              description_source='source')],
              'enrichment': None, 'future_field': {'tolerated': True}}
    transport, seen = service(detail)
    path = tmp_path/'iris.parquet'
    with Client(api_key='test-only', transport=transport) as c:
        result = c.download_research_dataset('p10k-example', path, version=DIGEST, metadata=True)
    assert isinstance(result, ResearchDownload)
    assert result.path == path and result.metadata_path == tmp_path/'iris.parquet.metadata.json'
    assert result.version == DIGEST and result.sha256 == DIGEST and result.metadata == detail
    assert path.read_bytes() == BODY
    assert json.loads(result.metadata_path.read_text()) == detail
    assert os.stat(result.metadata_path).st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob('.welt-*')) and len(seen) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.version = 'other'


@pytest.mark.parametrize('existing', ['data.parquet', 'data.parquet.metadata.json'])
def test_existing_metadata_or_data_destination_refused_before_any_request(tmp_path, existing):
    (tmp_path/existing).write_bytes(b'keep')
    transport, seen = service()
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(FileExistsError):
            c.download_research_dataset('p10k-example', tmp_path/'data.parquet', metadata=True)
    assert seen == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [existing]
    assert (tmp_path/existing).read_bytes() == b'keep'


def test_metadata_symlink_destination_refused(tmp_path):
    (tmp_path/'data.metadata.json').symlink_to(tmp_path/'missing')
    transport, seen = service()
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(FileExistsError):
            c.download_research_dataset('p10k-example', tmp_path/'data', metadata=True)
    assert seen == [] and not (tmp_path/'data').exists()


def test_metadata_version_mismatch_writes_nothing(tmp_path):
    transport, seen = service()
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(InvalidInputError) as error:
            c.download_research_dataset('p10k-example', tmp_path/'data', version='0'*64, metadata=True)
    assert error.value.code == 'research_version_mismatch'
    assert len(seen) == 1 and list(tmp_path.iterdir()) == []


def test_invalid_stream_with_metadata_publishes_neither_file(tmp_path):
    transport, _ = service(body=BODY[:-1])
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError):
            c.download_research_dataset('p10k-example', tmp_path/'data', metadata=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_metadata_flag_must_be_boolean(tmp_path, value):
    with Client(transport=httpx.MockTransport(lambda r: pytest.fail('no HTTP'))) as c:
        with pytest.raises(ValueError):
            c.download_research_dataset('p10k-example', tmp_path/'data', metadata=value)
        with pytest.raises(ValueError):
            c.load_research_dataset('p10k-example', metadata=value)


def test_load_parquet_uses_current_version_and_leaves_no_file(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    detail = detail_for(TABLE, schema=[dict(name='x', dtype='float', index=0, description='Measured x')],
                        display_name='Example table')
    transport, seen = service(detail, TABLE)
    with Client(api_key='test-only', transport=transport) as c:
        frame = c.load_research_dataset('p10k-example')
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # Ordinary Parquet carries no notice warning.
            frame2, info = c.load_research_dataset(
                'p10k-example', version=detail['content']['version'], metadata=True)
    pd.testing.assert_frame_equal(frame, FRAME)
    pd.testing.assert_frame_equal(frame2, FRAME)
    assert isinstance(info, ResearchMetadata)
    assert (info.dataset_id, info.version, info.sha256) == (
        'p10k-example', detail['content']['version'], detail['content']['sha256'])
    assert info.media_type == 'application/vnd.apache.parquet' and info.detail == detail
    assert info.license_text is None and info.attribution_text is None
    assert info.name == 'Example table' and info.column_descriptions == {'x': 'Measured x'}
    assert len(seen) == 4 and list(tmp_path.iterdir()) == []


def test_load_notice_zip_exposes_notices_and_warns():
    body = zip_bytes(NOTICES)
    detail = detail_for(body, 'application/zip', provenance=dict(canonical_sha256=sha256(TABLE).hexdigest()))
    transport, _ = service(detail, body)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.warns(ResearchNoticeWarning, match='LICENSE.txt and ATTRIBUTION.txt'):
            frame, info = c.load_research_dataset('p10k-example', metadata=True)
        with pytest.warns(ResearchNoticeWarning):
            plain = c.load_research_dataset('p10k-example')
    pd.testing.assert_frame_equal(frame, FRAME)
    pd.testing.assert_frame_equal(plain, FRAME)
    assert info.media_type == 'application/zip' and info.sha256 == sha256(body).hexdigest()
    assert info.license_text == 'Controlled licence text\n'
    assert info.attribution_text == 'Controlled attribution\n'


@pytest.mark.parametrize('members,provenance', [
    (NOTICES[:2], None),
    (NOTICES + [('extra.txt', b'x')], None),
    (NOTICES + [('LICENSE.txt', b'second')], None),
    ([('../dataset.parquet', TABLE)] + NOTICES[1:], None),
    ([('/dataset.parquet', TABLE)] + NOTICES[1:], None),
    ([('data/dataset.parquet', TABLE)] + NOTICES[1:], None),
    (NOTICES[:2] + [('ATTRIBUTION.txt', b'x' * (1024 * 1024 + 1))], None),
    (NOTICES[:2] + [('ATTRIBUTION.txt', b'\xff\xfe')], None),
    (NOTICES, dict(canonical_sha256='0' * 64)),
    (None, None),
])
def test_malformed_notice_zip_refused(members, provenance):
    body = b'not a zip archive' if members is None else zip_bytes(members)
    extra = {} if provenance is None else dict(provenance=provenance)
    transport, _ = service(detail_for(body, 'application/zip', **extra), body)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError) as error:
            c.load_research_dataset('p10k-example')
    assert error.value.code == 'invalid_research_archive'


def test_zip_table_member_bounded_by_max_bytes():
    # A compressible member that expands beyond max_bytes is refused before reading.
    body = zip_bytes([('dataset.parquet', b'\0' * 200_000)] + NOTICES[1:], zipfile.ZIP_DEFLATED)
    assert len(body) < 200_000
    transport, _ = service(detail_for(body, 'application/zip'), body)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError) as error:
            c.load_research_dataset('p10k-example', max_bytes=len(body))
    assert error.value.code == 'invalid_research_archive'


def test_unsupported_media_type_refused():
    transport, _ = service(detail_for(BODY, 'text/csv'), BODY)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(InvalidInputError) as error:
            c.load_research_dataset('p10k-example')
    assert error.value.code == 'research_media_type_unsupported'


@pytest.mark.parametrize('module', ['pyarrow', 'pandas'])
def test_missing_table_dependencies_name_the_extra_before_http(monkeypatch, module):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == module or name.startswith(module + '.'):
            raise ImportError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    with Client(transport=httpx.MockTransport(lambda r: pytest.fail('no HTTP'))) as c:
        with pytest.raises(OptionalDependencyError, match=r'welt-client\[research\]') as error:
            c.load_research_dataset('p10k-example')
    assert isinstance(error.value, ImportError)


def test_load_max_bytes_and_version_refuse_before_download():
    transport, seen = service(detail_for(TABLE), TABLE)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(InvalidInputError) as error:
            c.load_research_dataset('p10k-example', max_bytes=len(TABLE) - 1)
        with pytest.raises(InvalidInputError) as mismatch:
            c.load_research_dataset('p10k-example', version='0' * 64)
    assert error.value.code == 'research_download_limit'
    assert mismatch.value.code == 'research_version_mismatch'
    assert len(seen) == 2 and not any(r.url.path.endswith('/content') for r in seen)


def test_load_unavailable_content_typed_error():
    transport, seen = service(dict(id='p10k-example', content=None))
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(InvalidInputError) as error:
            c.load_research_dataset('p10k-example')
    assert error.value.code == 'research_content_unavailable' and len(seen) == 1


def test_async_metadata_and_load_parity(tmp_path):
    body = zip_bytes(NOTICES)
    zipped = detail_for(body, 'application/zip')
    async def run():
        transport, seen = service()
        async with AsyncClient(api_key='test-only', transport=transport) as c:
            plain = await c.download_research_dataset('p10k-example', tmp_path/'plain')
            result = await c.download_research_dataset('p10k-example', tmp_path/'meta', metadata=True)
            count = len(seen)
            with pytest.raises(FileExistsError):
                await c.download_research_dataset('p10k-example', tmp_path/'meta', metadata=True)
            assert len(seen) == count
            with pytest.raises(InvalidInputError):
                await c.download_research_dataset('p10k-example', tmp_path/'other', version='0'*64,
                                                  metadata=True)
        transport, _ = service(zipped, body)
        async with AsyncClient(api_key='test-only', transport=transport) as c:
            with pytest.warns(ResearchNoticeWarning):
                frame, info = await c.load_research_dataset('p10k-example', metadata=True)
            with pytest.raises(InvalidInputError):
                await c.load_research_dataset('p10k-example', max_bytes=1)
        return plain, result, frame, info
    plain, result, frame, info = asyncio.run(run())
    assert plain == tmp_path/'plain'
    assert isinstance(result, ResearchDownload) and result.metadata == DETAIL
    assert json.loads((tmp_path/'meta.metadata.json').read_text()) == DETAIL
    pd.testing.assert_frame_equal(frame, FRAME)
    assert info.license_text == 'Controlled licence text\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['meta', 'meta.metadata.json', 'plain']


# Review R1/R2: restored notice archives and listed previous versions.
RESTORED_FRAME = pd.DataFrame({'label': ['a', 'b', 'a'], 'x': [1.0, 2.0, 3.0]})
RESTORED_TABLE = parquet_bytes(RESTORED_FRAME)
RESTORED_ZIP = zip_bytes([('dataset.parquet', RESTORED_TABLE)] + NOTICES[1:])
CANONICAL_ZIP = zip_bytes(NOTICES)


def descriptor(body, media_type):
    digest = sha256(body).hexdigest()
    return dict(version=digest, sha256=digest, size_bytes=len(body), media_type=media_type,
                filename='dataset.zip' if media_type == 'application/zip' else 'dataset.parquet')


def restored_detail(table_sha=None):
    """Current restored ZIP; the canonical ZIP and a canonical Parquet stay listed."""
    return {**DETAIL, 'content': descriptor(RESTORED_ZIP, 'application/zip'),
            'previous_content': [descriptor(CANONICAL_ZIP, 'application/zip'),
                                 descriptor(TABLE, 'application/vnd.apache.parquet')],
            'provenance': dict(canonical_sha256=sha256(TABLE).hexdigest(),
                               delivery=dict(format='zip', table_sha256=table_sha or sha256(RESTORED_TABLE).hexdigest()),
                               header_restoration=dict(restored_sha256=sha256(RESTORED_TABLE).hexdigest()))}


def versioned_service(detail, bodies):
    """Serve each exact version's bytes; unknown versions get the API's 409."""
    seen = []
    def handle(request):
        seen.append(request)
        if request.url.path.endswith('/content'):
            body = bodies.get(request.url.params['version'])
            if body is None:
                return httpx.Response(409, json={'error': dict(code='dataset_version_mismatch',
                                                               message='Mismatch', retryable=False)})
            return httpx.Response(200, stream=httpx.ByteStream(body))
        return httpx.Response(200, json=detail)
    return httpx.MockTransport(handle), seen


BODIES = {sha256(b).hexdigest(): b for b in (RESTORED_ZIP, CANONICAL_ZIP, TABLE)}


def content_versions(seen):
    return [r.url.params['version'] for r in seen if r.url.path.endswith('/content')]


def test_restored_zip_verified_against_delivery_table_hash():
    transport, _ = versioned_service(restored_detail(), BODIES)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.warns(ResearchNoticeWarning):
            frame, info = c.load_research_dataset('p10k-example', metadata=True)
    pd.testing.assert_frame_equal(frame, RESTORED_FRAME)
    assert info.version == sha256(RESTORED_ZIP).hexdigest() and info.media_type == 'application/zip'


@pytest.mark.parametrize('table_sha', ['0' * 64, 'not-a-digest'])
def test_restored_zip_with_wrong_delivery_hash_refused(table_sha):
    transport, _ = versioned_service(restored_detail(table_sha), BODIES)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError) as error:
            c.load_research_dataset('p10k-example')
    assert error.value.code == 'invalid_research_archive'


@pytest.mark.parametrize('body,media_type,expected', [
    (CANONICAL_ZIP, 'application/zip', FRAME),  # Previous canonical ZIP: canonical_sha256.
    (TABLE, 'application/vnd.apache.parquet', FRAME),
])
def test_load_previous_version_uses_its_own_descriptor(body, media_type, expected):
    version = sha256(body).hexdigest()
    transport, seen = versioned_service(restored_detail(), BODIES)
    with Client(api_key='test-only', transport=transport) as c:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ResearchNoticeWarning)
            frame, info = c.load_research_dataset('p10k-example', version=version, metadata=True)
    pd.testing.assert_frame_equal(frame, expected)
    assert (info.version, info.sha256, info.media_type) == (version, version, media_type)
    assert content_versions(seen) == [version]


@pytest.mark.parametrize('body', [RESTORED_ZIP, CANONICAL_ZIP, TABLE])
def test_download_current_or_previous_version(tmp_path, body):
    version = sha256(body).hexdigest()
    transport, seen = versioned_service(restored_detail(), BODIES)
    with Client(api_key='test-only', transport=transport) as c:
        saved = c.download_research_dataset('p10k-example', tmp_path/'data', version=version, metadata=True)
    assert saved.path.read_bytes() == body and saved.version == saved.sha256 == version
    assert content_versions(seen) == [version]


def test_previous_version_size_and_sha_checked_against_its_descriptor(tmp_path):
    version = sha256(TABLE).hexdigest()
    transport, _ = versioned_service(restored_detail(), {version: TABLE + b'x'})
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError, match='pinned metadata'):
            c.download_research_dataset('p10k-example', tmp_path/'data', version=version)
        with pytest.raises(InvalidInputError) as limit:
            c.download_research_dataset('p10k-example', tmp_path/'data', version=version,
                                        max_bytes=len(TABLE) - 1)
    assert limit.value.code == 'research_download_limit'
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('previous', [[dict(version='0' * 64)], 'not-a-list'])
def test_malformed_previous_descriptor_refused(tmp_path, previous):
    detail = {**restored_detail(), 'previous_content': previous}
    transport, seen = versioned_service(detail, BODIES)
    with Client(api_key='test-only', transport=transport) as c:
        with pytest.raises(WeltError):
            c.download_research_dataset('p10k-example', tmp_path/'data', version='0' * 64)
    assert content_versions(seen) == [] and list(tmp_path.iterdir()) == []


def test_unknown_version_refused_before_download(tmp_path):
    transport, seen = versioned_service(restored_detail(), BODIES)
    with Client(api_key='test-only', transport=transport) as c:
        for call in (lambda: c.download_research_dataset('p10k-example', tmp_path/'data', version='1' * 64),
                     lambda: c.load_research_dataset('p10k-example', version='1' * 64)):
            with pytest.raises(InvalidInputError) as error:
                call()
            assert error.value.code == 'research_version_mismatch'
    assert content_versions(seen) == [] and list(tmp_path.iterdir()) == []


def test_async_restored_and_previous_version_parity(tmp_path):
    previous = sha256(CANONICAL_ZIP).hexdigest()
    async def run():
        transport, seen = versioned_service(restored_detail(), BODIES)
        async with AsyncClient(api_key='test-only', transport=transport) as c:
            with pytest.warns(ResearchNoticeWarning):
                current = await c.load_research_dataset('p10k-example')
            with pytest.warns(ResearchNoticeWarning):
                old, info = await c.load_research_dataset('p10k-example', version=previous, metadata=True)
            saved = await c.download_research_dataset('p10k-example', tmp_path/'old.zip', version=previous)
            with pytest.raises(InvalidInputError) as error:
                await c.download_research_dataset('p10k-example', tmp_path/'unknown', version='1' * 64)
        transport, _ = versioned_service(restored_detail('0' * 64), BODIES)
        async with AsyncClient(api_key='test-only', transport=transport) as c:
            with pytest.raises(WeltError) as invalid:
                await c.load_research_dataset('p10k-example')
        return current, old, info, saved, error.value, invalid.value, content_versions(seen)
    current, old, info, saved, error, invalid, versions = asyncio.run(run())
    pd.testing.assert_frame_equal(current, RESTORED_FRAME)
    pd.testing.assert_frame_equal(old, FRAME)
    assert info.version == previous and saved.read_bytes() == CANONICAL_ZIP
    assert error.code == 'research_version_mismatch' and invalid.code == 'invalid_research_archive'
    assert versions == [sha256(RESTORED_ZIP).hexdigest(), previous, previous]
    assert sorted(p.name for p in tmp_path.iterdir()) == ['old.zip']
