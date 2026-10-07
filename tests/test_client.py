"""Independent controlled HTTP fixture; no backend/model imports or live requests."""
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from welt import Classifier, Client, Regressor
from welt.errors import AuthenticationError
from welt.estimators import _RemoteEstimator


class ServiceFixture:
    """Protocol responses generated in memory, solely to verify client behavior."""
    def __init__(self):
        self.datasets, self.predictors, self.jobs = {}, {}, {}

    def __call__(self, request):
        if request.headers.get('Authorization') != 'Bearer fixture-only':
            return httpx.Response(401, json={'error':{'code':'authentication_required',
                'message':'Authentication required.','request_id':'request-fixture','retryable':False}})
        path=request.url.path
        data=json.loads(request.content) if request.content else {}
        if path=='/v1/datasets' and request.method=='POST':
            record={'id':f'ds{len(self.datasets)}',**data};self.datasets[record['id']]=record
            return httpx.Response(201,json=record)
        if path=='/v1/fits':
            ds=self.datasets[data['dataset_id']];target_index=ds['columns'].index(ds['target'])
            X=np.asarray([[v for i,v in enumerate(row) if i!=target_index] for row in ds['rows']],dtype=float)
            y=np.asarray([row[target_index] for row in ds['rows']]);classes=np.unique(y)
            pid=f'pred{len(self.predictors)}';jid=f'job{len(self.jobs)}'
            record=dict(id=pid,dataset_id=ds['id'],model_id=data['model_id'],model_version='fixture-v1',
                configuration=data['configuration'],seed=data['seed'],task=data['task'],
                columns=[c for c in ds['columns'] if c!=ds['target']],classes=classes.tolist(),
                centroids=[X[y==c].mean(axis=0).tolist() for c in classes],mean=float(y.mean()))
            self.predictors[pid]=record
            self.jobs[jid]=dict(id=jid,status='succeeded',operation='fit',predictor_id=pid)
            return httpx.Response(202,json=self.jobs[jid])
        if path.startswith('/v1/jobs/'):
            return httpx.Response(200,json=self.jobs[path.rsplit('/',1)[1]])
        if path.startswith('/v1/predictors/'):
            pid=path.split('/')[3];record=self.predictors[pid]
            if path.endswith('/predict'):
                if set(data['columns'])!=set(record['columns']):
                    return httpx.Response(422,json={'error':{'code':'schema_mismatch','message':'Columns differ.','request_id':'request-fixture','retryable':False}})
                order=[data['columns'].index(c) for c in record['columns']]
                rows=np.asarray(data['rows'])[:,order]
                if record['task']=='regression':predictions=[record['mean']]*len(rows);probs=None
                else:
                    selected=np.linalg.norm(rows[:,None]-record['centroids'],axis=2).argmin(axis=1)
                    predictions=np.asarray(record['classes'])[selected].tolist()
                    probs=np.eye(len(record['classes']))[selected].tolist() if data.get('probabilities') else None
                return httpx.Response(200,json={'predictions':predictions,'probabilities':probs})
            return httpx.Response(200,json=record)
        raise AssertionError(f'unexpected fixture route {path}')


@pytest.fixture
def connected(monkeypatch):
    fixture=ServiceFixture();transport=httpx.MockTransport(fixture)
    monkeypatch.setattr(_RemoteEstimator,'_client',lambda self:Client(api_key='fixture-only',transport=transport))
    yield fixture,transport


def test_clone_fit_named_order_and_probability_class_order(connected):
    fixture,_=connected;model=Classifier(random_state=12)
    assert is_classifier(model) and is_regressor(Regressor())
    assert clone(model).get_params()==model.get_params()
    with pytest.raises(NotFittedError):model.predict([[1,2]])
    X=pd.DataFrame({'a':[0,1,8,9],'b':[1,2,9,10]})
    assert model.fit(X,[0,0,1,1]) is model
    assert model.predict(X[['b','a']]).tolist()==[0,0,1,1]
    assert model.predict_proba(X).shape==(4,2)
    assert model.classes_.tolist()==[0,1]
    assert not hasattr(clone(model),'predictor_id_')
    with pytest.raises(ValueError):model.predict([[1]])


def test_pipeline_cv_isolates_fit_identity(connected):
    fixture,_=connected;X=np.asarray([[i,i+1] for i in range(16)]);y=np.asarray([0]*8+[1]*8)
    scores=cross_val_score(make_pipeline(StandardScaler(),Classifier()),X,y,cv=2)
    assert scores.shape==(2,) and np.isfinite(scores).all()
    assert len(fixture.predictors)==2
    assert len({p['dataset_id'] for p in fixture.predictors.values()})==2


def test_regression_validation_and_typed_auth_error(connected):
    _,transport=connected
    model=Regressor().fit([[0],[1]],[1.,3.]);assert model.predict([[2]]).tolist()==[2.]
    with pytest.raises(ValueError):model.fit([[0]],[np.nan])
    with Client(api_key='wrong-fixture',transport=transport) as client:
        with pytest.raises(AuthenticationError) as error:client.datasets()
    assert error.value.request_id=='request-fixture'


def test_explicit_upload_job_and_predictor_reuse(connected):
    _,transport=connected
    with Client(api_key='fixture-only',transport=transport) as client:
        dataset=client.upload(columns=['x','target'],rows=[[0,0],[1,1]],target='target')
        job=client.submit_fit(dataset['id'],model='fixture-model',task='classification')
        result=client.job(job.id).result()
        assert result['dataset_id']==dataset['id']
        assert client.predictor(result['id'])['model_version']=='fixture-v1'
