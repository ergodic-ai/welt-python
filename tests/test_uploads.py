"""Bounded resumable source identity, interruptions and async cancellation."""
import asyncio
import hashlib
import json
import threading

import httpx
import pytest

from welt import Client, AsyncClient
from welt.errors import InvalidInputError, TransportError


class Service:
    def __init__(self):
        self.info = None
        self.chunks = {}
        self.requests = []
        self.break_index = None
        self.on_create = None

    def call(self, request):
        self.requests.append((request.method, request.url.path))
        if request.method == 'POST' and request.url.path == '/v1/uploads':
            declaration = json.loads(request.content)
            self.info = dict(declaration, id='upload-one', workspace_id='ws', status='uploading',
                dataset_id=None, received_indices=[], chunk_bytes=8, expires_at='2099-01-01T00:00:00+00:00')
            if self.on_create: self.on_create()
            return httpx.Response(200, json=self.info)
        if request.method == 'GET':
            return httpx.Response(200, json=self.info)
        if request.method == 'PUT':
            index = int(request.url.path.rsplit('/',1)[1])
            assert request.headers['X-Chunk-SHA256'] == hashlib.sha256(request.content).hexdigest()
            assert len(request.content) <= 8
            self.chunks[index] = request.content
            self.info['received_indices'] = sorted(self.chunks)
            if index == self.break_index:
                self.break_index = None
                raise httpx.ReadError('response interrupted', request=request)
            return httpx.Response(200, json=self.info)
        assert request.url.path.endswith('/complete')
        whole = b''.join(self.chunks[i] for i in sorted(self.chunks))
        assert hashlib.sha256(whole).hexdigest() == self.info['sha256']
        self.info.update(status='complete', dataset_id='ds-one')
        return httpx.Response(200, json=dict(id='ds-one', columns=['a','b']))


@pytest.fixture
def file(tmp_path):
    path = tmp_path / 'example.csv'
    path.write_bytes(b'a,b\n1,2\n3,4\n5,6\n')
    return path


def test_sync_resume_after_ambiguous_chunk_acknowledgement(file):
    service = Service();service.break_index = 1
    with Client(api_key='test', transport=httpx.MockTransport(service.call)) as client:
        with pytest.raises(TransportError) as error:
            client.upload_file(file)
        assert error.value.upload_id == 'upload-one'
        assert service.info['received_indices'] == [0,1]
        assert not any(path.endswith('/complete') for _, path in service.requests)
        service.requests.clear()
        assert client.resume_upload(error.value.upload_id, file)['id'] == 'ds-one'
        assert not any(method == 'PUT' for method, _ in service.requests)


@pytest.mark.parametrize('field,value', [('sha256','f'*64),('size_bytes',9),('target','wrong'),
    ('schema',{'columns':['a'], 'types':['string']}),('format','parquet'),('id','another-upload'),
    ('received_indices',[0,0]),('received_indices',[100]),('received_indices',[True])])
def test_resume_declaration_or_identity_drift_rejected_before_mutation(file, field, value):
    from welt.uploads import declaration
    service = Service();service.info = dict(declaration(file),id='upload-one',chunk_bytes=8,received_indices=[])
    service.info[field] = value
    with Client(api_key='test', transport=httpx.MockTransport(service.call)) as client:
        with pytest.raises(InvalidInputError): client.resume_upload('upload-one',file)
    assert all(method == 'GET' for method,_ in service.requests)


def test_file_changes_after_scan_prevents_completion(file):
    service = Service()
    service.on_create = lambda: file.write_bytes(b'a,b\n9,9\n9,9\n9,9\n')
    with Client(api_key='test', transport=httpx.MockTransport(service.call)) as client:
        with pytest.raises(InvalidInputError) as error: client.upload_file(file)
    assert error.value.code == 'upload_source_changed'
    assert error.value.upload_id == 'upload-one'
    assert not any(path.endswith('/complete') for _, path in service.requests)


def test_completed_upload_rejects_changed_local_file(file):
    service = Service()
    with Client(api_key='test', transport=httpx.MockTransport(service.call)) as client:
        client.upload_file(file)
        file.write_bytes(b'a,b\n8,8\n8,8\n8,8\n')
        service.requests.clear()
        with pytest.raises(InvalidInputError): client.resume_upload('upload-one',file)
    assert service.requests == [('GET','/v1/uploads/upload-one')]


@pytest.mark.parametrize('metadata', [None, [], {}, {'chunk_bytes':8}])
def test_malformed_upload_metadata_stays_typed(file, metadata):
    def respond(request): return httpx.Response(200,content=json.dumps(metadata),headers={'Content-Type':'application/json'})
    with Client(api_key='test',transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(InvalidInputError) as error: client.upload_file(file)
    assert error.value.code == 'invalid_response'


def test_async_resumable_equivalence(file):
    async def run():
        service = Service();service.break_index = 0
        async with AsyncClient(api_key='test',transport=httpx.MockTransport(service.call)) as client:
            with pytest.raises(TransportError) as error: await client.upload_file(file)
            assert error.value.upload_id == 'upload-one'
            assert (await client.resume_upload('upload-one',file))['id'] == 'ds-one'
    asyncio.run(run())


@pytest.mark.parametrize('read_error', [False, True])
def test_async_cancel_waits_for_bounded_read_then_closes_preserving_id(file, monkeypatch, read_error):
    from welt import uploads
    started = threading.Event();release = threading.Event();closed = threading.Event()
    original = uploads.chunks
    def blocked(*args):
        try:
            started.set()
            assert release.wait(2)
            if read_error: raise OSError('read failed after cancellation')
            yield from original(*args)
        finally:
            closed.set()
    monkeypatch.setattr(uploads,'chunks',blocked)
    async def run():
        service = Service()
        async with AsyncClient(api_key='test',transport=httpx.MockTransport(service.call)) as client:
            task = asyncio.create_task(client.upload_file(file))
            assert await asyncio.to_thread(started.wait,2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError) as error: await task
            assert error.value.upload_id == 'upload-one'
            assert closed.is_set()
            assert service.info['id'] == 'upload-one'
            assert not any(path.endswith('/complete') for _, path in service.requests)
    asyncio.run(run())
