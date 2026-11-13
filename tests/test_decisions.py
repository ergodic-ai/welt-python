import asyncio
import json

import httpx
import pandas as pd
import pytest

from welt import Client, AsyncClient, DecisionResult
from welt.decision import decision_request


RESULT = dict(task='decision',protocol='decision-contrasts_v1',population=[
    dict(variable='A',baseline=False,value=True,effect=2.,rank=1)],individual=[],skipped=[])


def handler(request):
    if request.url.path == '/v1/decisions':
        payload=json.loads(request.content)
        assert payload['context']['table']['columns']==['A','Y']
        assert payload['interventions']=={'A':{}}
        assert request.headers['Idempotency-Key']=='same'
        return httpx.Response(202,json={'id':'job_decision'})
    if request.url.path.endswith('/result'):
        return httpx.Response(200,json=RESULT)
    return httpx.Response(200,json={'id':'job_decision','status':'succeeded','operation':'decide'})


def kwargs():
    return dict(target='Y',target_kind='continuous',discovery_model='arrow',
                estimation_model='kumo-medium',interventions={'A':{}},idempotency_key='same')


def test_sync_dataframe_submission_and_typed_result():
    with Client(api_key='key',transport=httpx.MockTransport(handler)) as client:
        frame=pd.DataFrame({'A':[False,True],'Y':[1.,3.]})
        answer=client.decisions.create(frame,**kwargs())
        assert isinstance(answer,DecisionResult)
        assert answer.population[0]['effect']==2
        value=answer.population;value.clear();assert answer.population
        assert client.decisions.job('job_decision').result().to_dict()==RESULT


def test_async_same_contract():
    async def run():
        async with AsyncClient(api_key='key',transport=httpx.MockTransport(handler)) as client:
            answer=await client.decisions.create(pd.DataFrame({'A':[False,True],'Y':[1.,3.]}),**kwargs())
            assert answer.population[0]['effect']==2
    asyncio.run(run())


def test_invalid_result_and_request_fail():
    with pytest.raises(ValueError):
        DecisionResult.from_dict(dict(RESULT,protocol='unexpected'))
    with pytest.raises(ValueError):
        decision_request('ds_x',target='Y',target_kind='continuous',discovery_model='arrow',
                         estimation_model='kumo-medium',interventions={'Y':{}})
