"""Declared sync/async upload and batch contracts without live data or model execution."""
import asyncio
import hashlib
import json

import httpx
import pytest
from welt import Client, AsyncClient


@pytest.mark.parametrize('asynchronous', [False, True])
def test_file_resume_skips_owned_chunks_and_rejects_changed_input(tmp_path, asynchronous):
    path = tmp_path/'table.csv'; data = b'x,y\n1,2\n3,4\n'; path.write_bytes(data)
    spec = dict(format='csv', size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), target='y', schema=None)
    row = dict(spec, id='up-a', status='uploading', chunk_bytes=4, received_indices=[0], workspace_id='ws-a')
    calls = []
    def handle(r):
        calls.append(r)
        if r.method == 'GET': return httpx.Response(200, json=row)
        if '/chunks/' in r.url.path:
            index = int(r.url.path.rsplit('/',1)[1]); assert r.content == data[index*4:index*4+4]
            assert r.headers['X-Chunk-SHA256'] == hashlib.sha256(r.content).hexdigest()
            return httpx.Response(200, json=row)
        return httpx.Response(200, json={'id':'ds-a'})
    async def run():
        async with AsyncClient(base_url='https://welt.test',api_key='key',transport=httpx.MockTransport(handle)) as c:
            assert (await c.upload_file(path, target='y', upload_id='up-a'))['id'] == 'ds-a'
            with pytest.raises(ValueError,match='metadata'): await c.upload_file(path, target='other', upload_id='up-a')
    if asynchronous: asyncio.run(run())
    else:
        with Client(base_url='https://welt.test',api_key='key',transport=httpx.MockTransport(handle)) as c:
            assert c.upload_file(path, target='y', upload_id='up-a')['id'] == 'ds-a'
            with pytest.raises(ValueError,match='metadata'): c.upload_file(path, target='other', upload_id='up-a')
    assert [r.url.path for r in calls] == ['/v1/uploads/up-a','/v1/uploads/up-a/chunks/1','/v1/uploads/up-a/chunks/2','/v1/uploads/up-a/complete','/v1/uploads/up-a']


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('status', ['succeeded','partially_completed'])
def test_batch_submission_wait_and_payload_use_same_auth_origin(asynchronous,status):
    calls=[]
    manifest=dict(job_id='job-b',row_count=3,successful_ranges=[[0,2]],failed_ranges=[[2,3]],result_reference='batch-result-a',expires_at='2099-01-01T00:00:00Z',model_version='pinned-v1')
    def handle(r):
        calls.append(r)
        assert r.headers['Authorization']=='Bearer key'
        if r.method=='POST':
            assert r.headers['Idempotency-Key']=='replay-a-12345678'
            assert json.loads(r.content)==dict(predictor_id='pred-a',dataset_id='ds-a',probabilities=True)
            return httpx.Response(202,json={'id':'job-b'})
        if r.url.path.endswith('/payload'):
            assert r.url.params['result_reference']=='batch-result-a'
            return httpx.Response(200,json=dict(job_id='job-b',predictions=[1,0,None],errors=[dict(start=2,end=3)]))
        if r.url.path.endswith('/result'):return httpx.Response(200,json=manifest)
        return httpx.Response(200,json=dict(id='job-b',status=status,operation='batch_predict'))
    async def run():
        async with AsyncClient(base_url='https://welt.test',api_key='key',transport=httpx.MockTransport(handle)) as c:
            job=await c.submit_batch_predict('pred-a','ds-a',probabilities=True,idempotency_key='replay-a-12345678')
            assert await job.result()==manifest
            assert (await c.batch_payload(job.id))['predictions']==[1,0,None]
    if asynchronous:asyncio.run(run())
    else:
        with Client(base_url='https://welt.test',api_key='key',transport=httpx.MockTransport(handle)) as c:
            job=c.submit_batch_predict('pred-a','ds-a',probabilities=True,idempotency_key='replay-a-12345678')
            assert job.result()==manifest
            assert c.batch_payload(job.id)['predictions']==[1,0,None]
    assert calls[-1].url.host=='welt.test'
