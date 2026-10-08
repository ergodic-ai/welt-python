"""Controlled terminal manifest/payload semantics; no hosted batch qualification."""
import asyncio
from copy import deepcopy
import dataclasses
import json

import httpx
import pytest

from welt import (AsyncClient, Client, BatchResult, BatchPayload, InvalidBatchResultError,
                  JobTimeoutError, JobCancelledError, ResultExpiredError)


def job(status='partially_completed'):
    return dict(id='job_batch',status=status,stage='complete',operation='batch_predict',
        operation_id='operation_batch',model_id='tabicl-v2',model_version='fixed-v1',
        dataset_id='query_dataset',predictor_id='predictor_original',task='regression',
        configuration='default',resolved_configuration={'recipe':'fixed'},seed=42,
        result=None,error=None,created_at='2026-10-08T01:00:00+00:00',
        updated_at='2026-10-08T01:00:01+00:00',expires_at='2026-10-15T01:00:01+00:00')


def manifest(status='partially_completed'):
    if status=='succeeded': successful,failed=[[0,2],[2,4]],[]
    elif status=='failed':successful,failed=[],[[0,2],[2,4]]
    else:successful,failed=[[0,2]],[[2,4]]
    return dict(job_id='job_batch',row_count=4,successful_ranges=successful,failed_ranges=failed,
        result_reference='batch_result_original',expires_at='2026-10-15T01:00:01+00:00',model_version='fixed-v1')


def numerical(status='partially_completed'):
    m=manifest(status)
    failed=[dict(range=bounds,code='cancelled' if status=='cancelled' else 'execution_failed',
                 retryable=False,outcome='cancelled' if status=='cancelled' else 'failed') for bounds in m['failed_ranges']]
    predictions=[0,1,2,3] if status=='succeeded' else [None]*4 if status=='failed' else [0,1,None,None]
    return dict(**{key:m[key] for key in ('job_id','result_reference','row_count','model_version')},
        predictor_id='predictor_original',operation_id='operation_batch',batch_protocol='welt-batch-contiguous-v1',
        predictions=predictions,probabilities=None,classes=None,errors=failed)


def service(status='partially_completed', *, expiry=False):
    requests=[]
    def handle(request):
        requests.append(request)
        path=request.url.path
        if path=='/v1/batch-predictions':
            assert json.loads(request.content)==dict(predictor_id='predictor_original',dataset_id='query_dataset',probabilities=False)
            assert 16<=len(request.headers['Idempotency-Key'])<=128
            return httpx.Response(202,json=job(status))
        if path.endswith('/result/payload'):
            assert request.url.params['result_reference']=='batch_result_original'
            if expiry:return httpx.Response(410,json={'error':{'code':'result_expired','message':'Expired','retryable':False}})
            return httpx.Response(200,json=numerical(status))
        if path.endswith('/result'):return httpx.Response(200,json=manifest(status))
        return httpx.Response(200,json=job(status))
    return httpx.MockTransport(handle),requests


@pytest.mark.parametrize('status',['succeeded','partially_completed','failed','cancelled'])
def test_sync_blocking_and_reconnect_return_manifest_no_download(status):
    transport,requests=service(status)
    with Client('https://test',transport=transport) as client:
        result=client.batch('query_dataset',predictor_id='predictor_original',timeout=0)
        assert isinstance(result,BatchResult) and result.status==status
        assert result.to_dict()==manifest(status)
        assert client.job(result.job_id).result(timeout=0).to_dict()==result.to_dict()
    assert not any(request.url.path.endswith('/payload') for request in requests)


@pytest.mark.parametrize('status',['succeeded','partially_completed','failed','cancelled'])
def test_async_terminal_manifest_and_explicit_payload(status):
    async def run():
        transport,requests=service(status)
        async with AsyncClient('https://test',transport=transport) as client:
            result=await client.batch('query_dataset',predictor_id='predictor_original',timeout=0)
            assert result.status==status and result.to_dict()==manifest(status)
            assert not any(request.url.path.endswith('/payload') for request in requests)
            value=await client.batch_payload(result.job_id,result_reference=result.result_reference)
            assert isinstance(value,BatchPayload) and value.to_dict()==numerical(status)
    asyncio.run(run())


def test_reference_mismatch_refuses_download_and_expiry_keeps_manifest():
    transport,requests=service(expiry=True)
    with Client('https://test',transport=transport) as client:
        result=client.batch_result('job_batch')
        with pytest.raises(InvalidBatchResultError):client.batch_payload(result.job_id,result_reference='foreign_reference')
        assert not any(r.url.path.endswith('/payload') for r in requests)
        with pytest.raises(ResultExpiredError):client.batch_payload(result.job_id)
        assert client.batch_result(result.job_id).to_dict()==result.to_dict()


def test_result_immutability_and_detached_payload():
    result=BatchResult(manifest(),job())
    with pytest.raises(dataclasses.FrozenInstanceError):result.status='succeeded'
    with pytest.raises(TypeError):result.failed_ranges[0][0]=0
    copy=result.to_dict();copy['successful_ranges'][0][0]=77
    assert result.successful_ranges==((0,2),)
    value=BatchPayload(numerical(),result,job())
    with pytest.raises(TypeError):value.errors[0]['code']='changed'
    copy=value.to_dict();copy['predictions'][0]=77
    assert value.predictions[0]==0


def test_cancelled_after_all_commits_stays_cancelled():
    j=job('cancelled');m=manifest('succeeded')
    result=BatchResult(m,j)
    assert result.status=='cancelled' and result.failed_ranges==()
    assert BatchPayload(numerical('succeeded'),result,j).predictions==(0,1,2,3)


@pytest.mark.parametrize('field,value', [('job_id','other'),('model_version','other'),
    ('row_count',True),('successful_ranges',[[0,3]]),('failed_ranges',[[1,4]]),('expires_at','not-time')])
def test_manifest_binding_drift_is_typed(field,value):
    m=manifest();m[field]=value
    with pytest.raises(InvalidBatchResultError):BatchResult(m,job())


@pytest.mark.parametrize('status',['succeeded','failed'])
def test_terminal_wire_rejects_range_larger_than_protocol(status):
    m=manifest(status);m['row_count']=101
    m['successful_ranges']=[[0,101]] if status=='succeeded' else []
    m['failed_ranges']=[[0,101]] if status=='failed' else []
    def handle(request):
        return httpx.Response(200,json=m if request.url.path.endswith('/result') else job(status))
    with Client('https://test',transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(InvalidBatchResultError):client.batch_result('job_batch')


@pytest.mark.parametrize('task',['causal_discovery',None])
def test_terminal_wire_rejects_non_tabular_job_task(task):
    def handle(request):
        return httpx.Response(200,json=manifest() if request.url.path.endswith('/result') else dict(job(),task=task))
    with Client('https://test',transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(InvalidBatchResultError):client.batch_result('job_batch')


@pytest.mark.parametrize('field,value',[('operation_id','other'),('result_reference','other'),
    ('predictions',[0,1,2,None]),('errors',[]),('row_count',True),('batch_protocol','other')])
def test_payload_binding_and_failed_mask_drift_is_typed(field,value):
    p=numerical();p[field]=value
    with pytest.raises(InvalidBatchResultError):BatchPayload(p,BatchResult(manifest(),job()),job())


def test_local_timeout_keeps_job_identity_and_does_not_cancel():
    transport,requests=service('running')
    with Client('https://test',transport=transport) as client:
        with pytest.raises(JobTimeoutError) as error:client.job('job_batch').result(timeout=0)
        assert error.value.job_id=='job_batch'
    assert not any(request.method=='POST' for request in requests)


def test_async_wait_cancellation_does_not_cancel_server():
    async def run():
        seen=asyncio.Event();requests=[]
        async def handle(request):
            requests.append(request)
            seen.set();return httpx.Response(200,json=job('running'))
        async with AsyncClient('https://test',transport=httpx.MockTransport(handle)) as client:
            task=asyncio.create_task(client.job('job_batch').result(poll_interval=30))
            await seen.wait();await asyncio.sleep(0);task.cancel()
            with pytest.raises(asyncio.CancelledError) as error:await task
            assert error.value.job_id=='job_batch'
            assert not any(r.method=='POST' for r in requests)
    asyncio.run(run())


def test_other_job_cancellation_stays_error():
    def handle(_):return httpx.Response(200,json=dict(job('cancelled'),operation='fit'))
    with Client('https://test',transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(JobCancelledError):client.job('job_batch').result(timeout=0)


def test_acknowledged_async_blocking_cancel_keeps_job_id():
    async def run():
        seen=asyncio.Event();requests=[]
        async def handle(request):
            requests.append(request)
            if request.method=='POST':return httpx.Response(202,json=job('queued'))
            seen.set();return httpx.Response(200,json=job('running'))
        async with AsyncClient('https://test',transport=httpx.MockTransport(handle)) as client:
            task=asyncio.create_task(client.batch('query_dataset',predictor_id='predictor_original'))
            await seen.wait();await asyncio.sleep(0);task.cancel()
            with pytest.raises(asyncio.CancelledError) as error:await task
            assert error.value.job_id=='job_batch'
            assert not any(r.url.path.endswith('/cancel') for r in requests)
    asyncio.run(run())
