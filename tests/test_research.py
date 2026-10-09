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
