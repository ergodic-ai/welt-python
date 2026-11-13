import asyncio
import copy
from decimal import Decimal
import json
from pathlib import Path

import httpx
import pytest

from welt import Client,AsyncClient
from welt.errors import JobTimeoutError,WeltError
from welt.upstream import UpstreamResult

FIXTURE=json.loads((Path(__file__).parent/'fixtures/upstream-result.json').read_text())
RESULT_ID=FIXTURE['preview']['id']
JOB_ID='dret_'+'b'*32


class Service:
    def __init__(self):
        self.calls=[]
        self.artifact=copy.deepcopy(FIXTURE['result'])
        self.status='succeeded'
        self.body_override=None
        self.headers={}
        self.limits={}

    def __call__(self,request):
        assert request.headers['Authorization']=='Bearer synthetic-key'
        self.calls.append(request)
        path=request.url.path
        if path=='/v1/data/datasets':
            return httpx.Response(200,json=[dict(id='world_bank:SP.POP.TOTL')])
        if path=='/v1/data/datasets/world_bank:SP.POP.TOTL':
            return httpx.Response(200,json=dict(id='world_bank:SP.POP.TOTL',schema_revision='c'*64,limits=self.limits))
        if path=='/v1/data/retrievals' or path=='/v1/data/retrievals/'+JOB_ID:
            return httpx.Response(202 if request.method=='POST' else 200,
                json=dict(id=JOB_ID,status=self.status,result_id=RESULT_ID if self.status=='succeeded' else None))
        if path=='/v1/data/results/'+RESULT_ID:
            return httpx.Response(200,json=FIXTURE['preview'])
        if path=='/v1/data/results/'+RESULT_ID+'/download':
            body=self.body_override if self.body_override is not None else json.dumps(self.artifact).encode()
            return httpx.Response(200,headers={'Content-Type':'application/json','Content-Length':str(len(body)),**self.headers},
                                  stream=httpx.ByteStream(body))
        raise AssertionError(path)


def client(service):
    return Client(base_url='https://api.test',api_key='synthetic-key',transport=httpx.MockTransport(service),max_retries=0)


def test_complete_sync_journey_and_decimal_dataframe(tmp_path):
    service=Service()
    with client(service) as c:
        assert c.upstream.search()[0]['id']=='world_bank:SP.POP.TOTL'
        result=c.upstream.fetch('world_bank:SP.POP.TOTL',idempotency_key='same')
        posted=next(r for r in service.calls if r.method=='POST')
        assert posted.headers['Idempotency-Key']=='same'
        assert json.loads(posted.content)['schema_revision']=='c'*64
        assert result.rows[0][3]=='123.000000000000001'
        frame=result.to_pandas()
        assert frame.iloc[0]['value']==Decimal('123.000000000000001')
        assert frame.iloc[0]['year']=='2020'
        assert frame.attrs['welt_upstream']['snapshot_id']==result.snapshot_id
        path=c.upstream.download(RESULT_ID,tmp_path/'result.json')
        assert json.loads(path.read_text())==FIXTURE['result']
        before=path.read_bytes()
        with pytest.raises(FileExistsError): result.save(path)
        assert path.read_bytes()==before


def test_complete_async_journey(tmp_path):
    async def run():
        service=Service()
        async with AsyncClient(base_url='https://api.test',api_key='synthetic-key',transport=httpx.MockTransport(service)) as c:
            assert (await c.upstream.search())[0]['id']=='world_bank:SP.POP.TOTL'
            result=await c.upstream.fetch('world_bank:SP.POP.TOTL')
            assert result.rows==UpstreamResult(RESULT_ID,FIXTURE['result'],FIXTURE['preview']).rows
            path=await c.upstream.download(RESULT_ID,tmp_path/'async.json')
            assert path.exists()
    asyncio.run(run())


@pytest.mark.parametrize('change',['row','receipt','provenance','length','compression','duplicate','oversize'])
def test_download_refuses_corruption_or_limits(change,tmp_path):
    service=Service()
    if change=='row': service.artifact['snapshot']['rows'][0][3]='0'
    if change=='receipt': service.artifact['snapshot']['receipt']['complete']=False
    if change=='provenance': service.artifact['attribution']='different'
    if change=='length': service.headers['Content-Length']='1'
    if change=='compression': service.headers['Content-Encoding']='gzip'
    if change=='duplicate': service.body_override=b'{"a":1,"a":2}'
    with client(service) as c:
        with pytest.raises(WeltError):
            c.upstream.download(RESULT_ID,tmp_path/'bad.json',max_bytes=1 if change=='oversize' else 10*1024*1024)
    assert not (tmp_path/'bad.json').exists()


def test_wait_timeout_does_not_resubmit():
    service=Service();service.status='running'
    with client(service) as c:
        with pytest.raises(JobTimeoutError) as error:
            c.upstream.wait(JOB_ID,timeout=.01,poll_interval=.001)
    assert error.value.code=='upstream_wait_timeout'
    assert '.upstream.wait' in str(error.value) and 'fit' not in str(error.value)
    assert all(r.method=='GET' for r in service.calls)


@pytest.mark.parametrize('bad',[RESULT_ID+'/download','../other','https://evil.test/result'])
def test_resource_path_refusal(bad):
    service=Service()
    with client(service) as c:
        with pytest.raises(ValueError): c.upstream.preview(bad)
    assert not service.calls


def test_invalid_fetch_budget_and_existing_file_do_not_make_requests(tmp_path):
    service=Service()
    existing=tmp_path/'existing'
    existing.write_text('preserve')
    with client(service) as c:
        for options in ({'timeout':-1},{'max_bytes':0},{'poll_interval':0}):
            with pytest.raises(ValueError): c.upstream.fetch('world_bank:SP.POP.TOTL',**options)
        with pytest.raises(FileExistsError): c.upstream.download(RESULT_ID,existing)
    assert not service.calls and existing.read_text()=='preserve'


def test_async_invalid_fetch_budget_and_existing_file_do_not_make_requests(tmp_path):
    async def run():
        service=Service()
        existing=tmp_path/'existing-async'
        existing.write_text('preserve')
        async with AsyncClient(base_url='https://api.test',api_key='synthetic-key',transport=httpx.MockTransport(service)) as c:
            for options in ({'timeout':-1},{'max_bytes':0},{'poll_interval':0}):
                with pytest.raises(ValueError):
                    await c.upstream.fetch('world_bank:SP.POP.TOTL',**options)
            with pytest.raises(FileExistsError):
                await c.upstream.download(RESULT_ID,existing)
        assert not service.calls and existing.read_text()=='preserve'
    asyncio.run(run())


@pytest.mark.parametrize('asynchronous',[False,True])
def test_submission_uses_dataset_ceilings_preserves_explicit_limits_and_revision(asynchronous):
    service=Service()
    service.limits=dict(max_rows=52,max_bytes=262144,max_pages=3,timeout_seconds=30)
    async def run():
        async with AsyncClient(base_url='https://api.test',api_key='synthetic-key',transport=httpx.MockTransport(service)) as c:
            await c.upstream.submit('world_bank:SP.POP.TOTL',schema_revision='d'*64,limits={'max_rows':1})
    if asynchronous:asyncio.run(run())
    else:
        with client(service) as c:
            c.upstream.submit('world_bank:SP.POP.TOTL',schema_revision='d'*64,limits={'max_rows':1})
    posted=next(r for r in service.calls if r.method=='POST')
    body=json.loads(posted.content)
    assert body['limits']==dict(service.limits,max_rows=1)
    assert body['schema_revision']=='d'*64
