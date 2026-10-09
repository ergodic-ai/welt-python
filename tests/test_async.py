"""Async parity and durable execution semantics; controlled HTTP only."""
import asyncio
import json

import httpx
import pytest

from welt import (AsyncClient, Client, ExecutionError, JobCancelledError,
                  JobTimeoutError, PredictionPendingError, RateLimitError, TransportError)


def test_sync_async_supported_resource_parity_and_cleanup():
    calls = []
    def service(request):
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path in ('/v1/models', '/v1/datasets', '/v1/predictors', '/v1/jobs', '/v1/usage') and request.method=='GET':
            return httpx.Response(200,json={'items':[{'id':'fixture'}]})
        if path=='/v1/datasets': return httpx.Response(201,json={'id':'ds'})
        if path=='/v1/datasets/ds': return httpx.Response(200,json={'id':'ds'})
        if path=='/v1/fits':
            assert request.headers['Idempotency-Key']=='caller-replay-key'
            return httpx.Response(202,json={'id':'job'})
        if path=='/v1/jobs/job': return httpx.Response(200,json={'id':'job','operation':'fit','status':'succeeded','predictor_id':'pred'})
        if path=='/v1/predictors/pred': return httpx.Response(200,json={'id':'pred','model_version':'fixture-pinned-v1'})
        if path.endswith('/predict'): return httpx.Response(200,json={'predictions':[1], 'probabilities':[[0.,1.]]})
        if path.endswith('/cancel'): return httpx.Response(200,json={'id':'job','status':'cancelled'})
        raise AssertionError(path)
    transport=httpx.MockTransport(service)
    def sync():
        with Client(api_key="fixture-only", transport=transport) as client:
            output=[client.models(),client.datasets(),client.predictors(),client.jobs(),client.usage()]
            output.append(client.upload(columns=['x','target'],rows=[[1,1]],target='target'))
            output.append(client.dataset('ds'))
            job=client.submit_fit('ds',model='fixture',task='classification',idempotency_key='caller-replay-key')
            output.append(job.result());output.append(client.predict('pred',columns=['x'],rows=[[1]],probabilities=True))
            output.append(client.job(job.id).cancel())
        assert client.http.is_closed
        return output
    async def asynchronous():
        async with AsyncClient(api_key="fixture-only", transport=transport) as client:
            output=[await client.models(),await client.datasets(),await client.predictors(),await client.jobs(),await client.usage()]
            output.append(await client.upload(columns=['x','target'],rows=[[1,1]],target='target'))
            output.append(await client.dataset('ds'))
            job=await client.submit_fit('ds',model='fixture',task='classification',idempotency_key='caller-replay-key')
            output.append(await job.result());output.append(await client.predict('pred',columns=['x'],rows=[[1]],probabilities=True))
            output.append(await client.job(job.id).cancel())
        assert client.http.is_closed
        return output
    expected=sync(); first_calls=list(calls);calls.clear()
    assert asyncio.run(asynchronous())==expected
    assert calls==first_calls


@pytest.mark.parametrize('asynchronous',[False,True])
def test_local_wait_timeout_has_job_identity_and_never_cancels(asynchronous):
    calls=[]
    def service(request):
        calls.append(request.method)
        return httpx.Response(200,json={'id':'durable','operation':'fit','status':'queued'})
    transport=httpx.MockTransport(service)
    async def run():
        async with AsyncClient(api_key="fixture-only", transport=transport) as client:
            await client.job('durable').result(timeout=0)
    with pytest.raises(JobTimeoutError) as error:
        if asynchronous:asyncio.run(run())
        else:
            with Client(api_key="fixture-only", transport=transport) as client:client.job('durable').result(timeout=0)
    assert error.value.job_id=='durable' and isinstance(error.value,TimeoutError)
    assert calls==[]


def test_cancelled_local_async_await_does_not_cancel_server_job():
    calls=[]
    async def service(request):
        calls.append(request.method)
        return httpx.Response(200,json={'id':'durable','operation':'fit','status':'running'})
    async def run():
        async with AsyncClient(api_key="fixture-only", transport=httpx.MockTransport(service)) as client:
            task=asyncio.create_task(client.job('durable').result(poll_interval=30))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                # A cancellation race must not leave the local task polling.
                async with asyncio.timeout(.1):
                    await task
            assert (await client.job('durable').inspect())['status']=='running'
        assert client.http.is_closed
    asyncio.run(run());assert calls==['GET','GET']


@pytest.mark.parametrize('asynchronous',[False,True])
@pytest.mark.parametrize('status,error_type',[('failed',ExecutionError),('cancelled',JobCancelledError)])
def test_terminal_job_errors_preserve_identity(asynchronous,status,error_type):
    transport=httpx.MockTransport(lambda request:httpx.Response(200,json={'id':'job','status':status,
        'operation':'fit','error':{'code':'execution_failed','message':'Fixture failed.','request_id':'req','retryable':False}}))
    async def run():
        async with AsyncClient(api_key="fixture-only", transport=transport) as client:await client.job('job').result()
    with pytest.raises(error_type) as error:
        if asynchronous:asyncio.run(run())
        else:
            with Client(api_key="fixture-only", transport=transport) as client:client.job('job').result()
    assert error.value.job_id=='job' and error.value.request_id=='req'


@pytest.mark.parametrize('asynchronous',[False,True])
def test_read_retry_after_is_bounded_and_mutations_never_retry(asynchronous,monkeypatch):
    calls=[];sleeps=[]
    def service(request):
        calls.append(request.method)
        if len(calls)==1:
            return httpx.Response(429,headers={'Retry-After':'0'},json={'error':{'code':'rate_limit','message':'Wait.','retryable':True}})
        return httpx.Response(200,json={'items':[]})
    monkeypatch.setattr('welt.client.time.sleep',sleeps.append)
    async def sleep(value):sleeps.append(value)
    monkeypatch.setattr('welt.client.asyncio.sleep',sleep)
    async def run(method):
        async with AsyncClient(api_key="fixture-only", transport=httpx.MockTransport(service)) as client:return await client.request(method,'/v1/models')
    def run_sync(method):
        with Client(api_key="fixture-only", transport=httpx.MockTransport(service)) as client:return client.request(method,'/v1/models')
    invoke=lambda method:asyncio.run(run(method)) if asynchronous else run_sync(method)
    assert invoke('GET')=={'items':[]} and calls==['GET','GET'] and sleeps==[0]
    calls.clear();sleeps.clear()
    with pytest.raises(RateLimitError):invoke('POST')
    assert calls==['POST'] and not sleeps


def test_long_retry_after_is_not_ignored_or_retried_early():
    calls=[]
    def service(request):
        calls.append(request.method)
        return httpx.Response(429,headers={'Retry-After':'90'},json={'error':{'code':'rate_limit','message':'Wait.'}})
    with Client(api_key="fixture-only", transport=httpx.MockTransport(service),max_retry_delay=5) as client:
        with pytest.raises(RateLimitError) as error:client.models()
    assert calls==['GET'] and error.value.retry_after==90


@pytest.mark.parametrize('asynchronous',[False,True])
def test_transport_error_redacts_request_and_secret(asynchronous):
    def service(request):raise httpx.ConnectError('upstream invented-secret-details',request=request)
    transport=httpx.MockTransport(service)
    async def run():
        async with AsyncClient(api_key='invented-secret',transport=transport,max_retries=0) as client:await client.models()
    with pytest.raises(TransportError) as error:
        if asynchronous:asyncio.run(run())
        else:
            with Client(api_key='invented-secret',transport=transport,max_retries=0) as client:client.models()
    assert 'invented-secret' not in str(error.value) and error.value.retryable


def test_prediction_pending_and_204_are_supported():
    def service(request):
        if request.method=='POST':return httpx.Response(504,json={'error':{'code':'prediction_timeout','message':'Continue polling.','job_id':'pred-job','request_id':'req','retryable':True}})
        return httpx.Response(204)
    with Client(api_key="fixture-only", transport=httpx.MockTransport(service)) as client:
        with pytest.raises(PredictionPendingError) as error:client.predict('pred',columns=['x'],rows=[[1]])
        assert error.value.job_id=='pred-job' and error.value.status_code==504
        assert client.request('GET','/fixture') is None
