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


def connection(source='fred',revoked=False):
    return dict(id='cred_'+'a'*32,source_id=source,version=1,created_at='2026-11-13T00:00:00Z',
        updated_at='2026-11-13T00:00:00Z',revoked_at='2026-11-13T00:01:00Z' if revoked else None)


@pytest.mark.parametrize('asynchronous',[False,True])
@pytest.mark.parametrize('source,token',[('fred','b'*32),('census','c'*40),('open_meteo','s'*128),('eia','e'*128)])
def test_source_connection_management_and_no_key_retention(asynchronous,source,token):
    from welt.credentials import Credential
    requests=[]
    def service(request):
        requests.append(request)
        if request.method=='GET':return httpx.Response(200,json=[connection(source)])
        if request.method=='PUT':assert json.loads(request.content)=={'api_key':token}
        return httpx.Response(200,json=connection(source,revoked=request.method=='POST'))
    async def run():
        async with AsyncClient(base_url='https://api.test',api_key='platform-key',transport=httpx.MockTransport(service)) as c:
            assert await c.upstream.connections()==[connection(source)]
            assert await c.upstream.connect(source,api_key=Credential(token))==connection(source)
            assert (await c.upstream.revoke_connection(source))['revoked_at']
            assert token not in repr(c.upstream.__dict__)
    if asynchronous:asyncio.run(run())
    else:
        with Client(base_url='https://api.test',api_key='platform-key',transport=httpx.MockTransport(service)) as c:
            assert c.upstream.connections()==[connection(source)]
            assert c.upstream.connect(source,api_key=Credential(token))==connection(source)
            assert c.upstream.revoke_connection(source)['revoked_at']
            assert token not in repr(c.upstream.__dict__)
    assert [(r.method,r.url.path) for r in requests]==[
        ('GET','/v1/data/connections'),('PUT','/v1/data/connections/'+source),('POST','/v1/data/connections/'+source+'/revoke')]


def test_bad_connection_inputs_fail_before_requests():
    def service(request):pytest.fail('Invalid secret/source sent to server')
    with client(service) as c:
        for source,key in [('fred','private-invalid-secret'),('census','b'*32),('../fred','b'*32),('other','b'*32),('eia','bad key'),('eia','x'*129),('eia',''),('open_meteo','bad key'),('open_meteo','x'*129),('open_meteo','\n')]:
            with pytest.raises(ValueError) as exc:c.upstream.connect(source,api_key=key)
            assert not key or key not in str(exc.value)
        with pytest.raises(ValueError):c.upstream.revoke_connection('../fred')


@pytest.mark.parametrize('asynchronous',[False,True])
def test_source_secret_reflection_is_sanitized_in_errors_and_metadata(asynchronous):
    key='b'*32
    def service(request):
        return httpx.Response(422,json={'error':{'message':'bad '+key,'code':key,'request_id':key,
            'details':{'reflected':key}}})
    async def run():
        async with AsyncClient(base_url='https://api.test',api_key='platform-key',transport=httpx.MockTransport(service)) as c:
            with pytest.raises(WeltError) as exc:await c.upstream.connect('fred',api_key=key)
            assert not key or key not in str(exc.value)+repr(exc.value.__dict__)
    if asynchronous:asyncio.run(run())
    else:
        with client(service) as c:
            with pytest.raises(WeltError) as exc:c.upstream.connect('fred',api_key=key)
            assert not key or key not in str(exc.value)+repr(exc.value.__dict__)
    with client(lambda r:httpx.Response(200,json=dict(connection(),api_key=key))) as c:
        with pytest.raises(WeltError) as exc:c.upstream.connect('fred',api_key=key)
        assert not key or key not in str(exc.value)


@pytest.mark.parametrize('asynchronous',[False,True])
@pytest.mark.parametrize('operation',['connections','connect','revoke_connection'])
def test_connection_methods_refuse_redirect_even_when_client_follows(asynchronous,operation):
    requests=[]
    def service(request):
        requests.append(request)
        return httpx.Response(307,headers={'Location':'https://foreign.test/capture'},json=connection())
    args=() if operation=='connections' else ('fred',)
    kwargs={'api_key':'b'*32} if operation=='connect' else {}
    async def run():
        async with AsyncClient(base_url='https://api.test',api_key='platform-key',transport=httpx.MockTransport(service)) as c:
            c.http.follow_redirects=True
            with pytest.raises(WeltError):await getattr(c.upstream,operation)(*args,**kwargs)
    if asynchronous:asyncio.run(run())
    else:
        with client(service) as c:
            c.http.follow_redirects=True
            with pytest.raises(WeltError):getattr(c.upstream,operation)(*args,**kwargs)
    assert len(requests)==1 and requests[0].url.host=='api.test'


@pytest.mark.parametrize('asynchronous',[False,True])
def test_lists_all_supported_connections(asynchronous):
    expected=[connection(source) for source in ('fred','census','open_meteo','eia')]
    def service(request):
        assert request.method=='GET' and request.url.path=='/v1/data/connections'
        return httpx.Response(200,json=expected)
    async def run():
        async with AsyncClient(base_url='https://api.test',api_key='platform-key',transport=httpx.MockTransport(service)) as c:
            assert await c.upstream.connections()==expected
    if asynchronous:asyncio.run(run())
    else:
        with Client(base_url='https://api.test',api_key='platform-key',transport=httpx.MockTransport(service)) as c:
            assert c.upstream.connections()==expected


@pytest.mark.parametrize('asynchronous',[False,True])
@pytest.mark.parametrize('key',['o','0','a','2026','open_meteo'])
def test_short_open_meteo_key_can_overlap_valid_metadata(asynchronous,key):
    def service(request):
        assert json.loads(request.content)=={'api_key':key}
        return httpx.Response(200,json=connection('open_meteo'))
    async def run():
        async with AsyncClient(base_url='https://api.test',api_key='platform-key',transport=httpx.MockTransport(service)) as c:
            assert await c.upstream.connect('open_meteo',api_key=key)==connection('open_meteo')
    if asynchronous:asyncio.run(run())
    else:
        with Client(base_url='https://api.test',api_key='platform-key',transport=httpx.MockTransport(service)) as c:
            assert c.upstream.connect('open_meteo',api_key=key)==connection('open_meteo')


def test_credential_reflected_as_entire_opaque_id_is_refused():
    key='b'*32
    metadata=dict(connection('open_meteo'),id='cred_'+key)
    with client(lambda r:httpx.Response(200,json=metadata)) as c:
        with pytest.raises(WeltError) as error:c.upstream.connect('open_meteo',api_key=key)
        assert key not in str(error.value)


@pytest.mark.parametrize('stamp',['2026-11-13o00:00:00Z','2026-W46-500:00:00Z','2026-11-13T00:00:00+99:00','2026-11-13T00:00:00+00:99'])
def test_connection_metadata_refuses_noncanonical_timestamp_text(stamp):
    metadata=dict(connection('open_meteo'),created_at=stamp)
    with client(lambda r:httpx.Response(200,json=metadata)) as c:
        with pytest.raises(WeltError):c.upstream.connect('open_meteo',api_key='short')
