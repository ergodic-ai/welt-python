"""Explicit model lookup and single-run sync/async/CLI protocol parity."""
import asyncio
import json
from types import SimpleNamespace

import httpx

from welt import Client,AsyncClient
from welt.cli import _parser,_run


def service(request):
    path=request.url.path
    if path=='/v1/models':return httpx.Response(200,json={'items':[{'id':'tabicl-v2'}]})
    if path=='/v1/models/chronos-2':return httpx.Response(200,json={'id':'chronos-2','visibility':'hidden'})
    if path=='/v1/model-runs':
        assert request.method=='POST' and request.headers['Idempotency-Key']=='frozen-key'
        assert json.loads(request.content)=={'model_id':'chronos-2','task':'univariate_forecasting'}
        return httpx.Response(202,json={'id':'run-job'})
    if path=='/v1/jobs/run-job':return httpx.Response(200,json={'id':'run-job','status':'succeeded','operation':'model_run','predictor_id':None})
    if path=='/v1/jobs/run-job/result':return httpx.Response(200,json={'predictions':[1.2],'task':'univariate_forecasting'})
    if path=='/v1/model-runs/run-job':
        assert request.method=='DELETE';return httpx.Response(200,json={'deleted':True})
    raise AssertionError('unexpected route '+path)


def test_hidden_run_sdk_sync_async_parity_and_job_result_uses_result_route():
    payload={'model_id':'chronos-2','task':'univariate_forecasting'}
    with Client(api_key='fixture',transport=httpx.MockTransport(service)) as client:
        expected=[client.model('chronos-2'),client.model('tabicl-v2')]
        job=client.submit_model_run(payload,idempotency_key='frozen-key')
        expected += [job.result(),client.delete_model_run(job.id)]
    async def run():
        async with AsyncClient(api_key='fixture',transport=httpx.MockTransport(service)) as client:
            result=[await client.model('chronos-2'),await client.model('tabicl-v2')]
            job=await client.submit_model_run(payload,idempotency_key='frozen-key')
            result += [await job.result(),await client.delete_model_run(job.id)]
            return result
    assert asyncio.run(run())==expected


def test_cli_run_lookup_and_explicit_delete(tmp_path,capsys):
    payload=tmp_path/'request.json';payload.write_text(json.dumps({'model_id':'chronos-2','task':'univariate_forecasting'}))
    with Client(api_key='fixture',transport=httpx.MockTransport(service)) as client:
        parser=_parser()
        result=_run(parser.parse_args(['run','--input',str(payload),'--idempotency-key','frozen-key']),client)
        assert result=={'job_id':'run-job','idempotency_key':'frozen-key'}
        assert 'frozen-key' in capsys.readouterr().err
        assert _run(parser.parse_args(['models','get','chronos-2']),client)['visibility']=='hidden'
        assert _run(parser.parse_args(['model-runs','delete','run-job','--yes']),client)['deleted']
