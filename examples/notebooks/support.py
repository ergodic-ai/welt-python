"""Synthetic notebook transport; live execution requires explicit environment opt-in."""
import json
import os
from urllib.parse import urlsplit

import httpx
import numpy as np
from welt import Client, AsyncClient


class SyntheticService:
    """SDK plumbing fixture, not a foundation model or scientific benchmark."""
    def __init__(self):
        self.data, self.fitted, self.job_records, self.events = {}, {}, {}, []
        self.regression_coefficients = {}

    def usage(self, operation, kind, status, dataset, predictor=None, rows=0, task="classification"):
        version = "synthetic-regression-contract-v1" if task == "regression" else "synthetic-contract-v1"
        self.events.append(dict(id=f'event-{len(self.events)}', operation_id=operation,
            kind=kind, status=status, dataset_id=dataset, predictor_id=predictor,
            model_id='tabicl-v2', model_version=version, task=task,
            configuration='default', rows=rows, columns=4, attempt=1,
            duration_ms=None, created_at='2026-01-01T00:00:00Z'))

    def __call__(self, request):
        assert request.url.host == 'notebook.invalid', 'Fixture must not make live requests.'
        path, method = request.url.path, request.method
        value = json.loads(request.content) if request.content else {}
        if path == '/v1/models':
            return httpx.Response(200, json={'items':[dict(id='tabicl-v2', status='available',
                tasks=['classification', 'regression'], version='synthetic-contract-v1',
                task_profiles=[dict(task=task, status='available', execution_available=True,
                    version='synthetic-regression-contract-v1' if task=='regression' else 'synthetic-contract-v1',
                    limits=dict(max_rows=500,max_columns=20,max_classes=0 if task=='regression' else 10,max_predict_rows=100))
                    for task in ('classification','regression')],
                limits=dict(max_rows=500,max_columns=20,max_classes=10,max_predict_rows=100))]})
        if path == '/v1/datasets' and method == 'POST':
            item=dict(id=f'dataset-{len(self.data)}', **value);self.data[item['id']]=item
            return httpx.Response(201,json=item)
        if path == '/v1/fits':
            ds=self.data[value['dataset_id']];pid=f'predictor-{len(self.fitted)}';jid=f'job-{len(self.job_records)}'
            task=value['task']
            classes=sorted({row[ds['columns'].index(ds['target'])] for row in ds['rows']}) if task=='classification' else None
            if task=='regression':
                table=np.asarray(ds['rows'],dtype=float);target_index=ds['columns'].index(ds['target'])
                features=np.delete(table,target_index,axis=1)
                self.regression_coefficients[pid]=np.linalg.lstsq(np.column_stack((np.ones(len(features)),features)),table[:,target_index],rcond=None)[0]
            item=dict(id=pid,dataset_id=ds['id'],model_id=value['model_id'],task=value['task'],
                model_version='synthetic-regression-contract-v1' if task=='regression' else 'synthetic-contract-v1',configuration=value['configuration'],
                seed=value['seed'],classes=classes,columns=[c for c in ds['columns'] if c!=ds['target']])
            self.fitted[pid]=item;self.job_records[jid]=dict(id=jid,status='succeeded',operation='fit',predictor_id=pid)
            for status in ('queued','running','succeeded'):self.usage(jid,'fit',status,ds['id'],pid,len(ds['rows']),task)
            return httpx.Response(202,json=self.job_records[jid])
        if path.startswith('/v1/jobs/'):
            return httpx.Response(200,json=self.job_records[path.rsplit('/',1)[1]])
        if path.startswith('/v1/predictors/'):
            item=self.fitted[path.split('/')[3]]
            if path.endswith('/predict'):
                order=[value['columns'].index(c) for c in item['columns']]
                X=np.asarray(value['rows'])[:,order]
                if item['task']=='regression':
                    if value.get('probabilities'):
                        return httpx.Response(422,json=dict(error=dict(code='unsupported_capability',message='Regression has no class probabilities.',details=None)))
                    labels=np.column_stack((np.ones(len(X)),X)) @ self.regression_coefficients[item['id']]
                    probs=None
                else:
                    labels=np.where(X[:,0]+0.5*X[:,1]>0,'positive','negative')
                    probs=[[0.1,0.9] if label=='positive' else [0.9,0.1] for label in labels]
                operation=f'query-{len(self.events)}'
                for status in ('running','succeeded'):self.usage(operation,'predict',status,item['dataset_id'],item['id'],len(X),item['task'])
                return httpx.Response(200,json=dict(predictions=labels.tolist(),
                    probabilities=probs if value.get('probabilities') else None,classes=item['classes'],
                    predictor_id=item['id'],model_id=item['model_id'],model_version=item['model_version'],
                    operation_id=operation,cold=False))
            return httpx.Response(200,json=item)
        resources={'/v1/datasets':self.data,'/v1/predictors':self.fitted,'/v1/jobs':self.job_records}
        if path in resources:return httpx.Response(200,json={'items':list(resources[path].values())})
        if path=='/v1/usage':return httpx.Response(200,json={'items':self.events})
        raise AssertionError(f'Unsupported notebook fixture operation: {method} {path}')


def notebook_options():
    mode=os.environ.get('WELT_NOTEBOOK_MODE','fixture')
    if mode=='fixture':
        return dict(base_url='https://notebook.invalid',api_key='not-a-key',transport=httpx.MockTransport(SyntheticService()))
    if mode!='live':raise ValueError('WELT_NOTEBOOK_MODE must be fixture or live.')
    base=os.environ.get('WELT_BASE_URL','');parts=urlsplit(base)
    if parts.scheme!='https' or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Live mode requires an explicit HTTPS WELT_BASE_URL without embedded credentials.')
    if not os.environ.get('WELT_API_KEY'):raise ValueError('Live mode requires a locally configured WELT_API_KEY.')
    return dict(base_url=base)


def notebook_client():
    return Client(**notebook_options())


def notebook_async_client():
    return AsyncClient(**notebook_options())


def synthetic_table():
    rng=np.random.default_rng(9);X=rng.normal(size=(160,4))
    labels=np.where(X[:,0]+0.5*X[:,1]>0,'positive','negative')
    return ['feature_a','feature_b','feature_c','feature_d'],X[:128],labels[:128],X[128:]


def prepare(client):
    columns,train,labels,query=synthetic_table()
    entry=next(m for m in client.models() if m['id']=='tabicl-v2')
    if entry.get('status')!='available' or 'classification' not in entry.get('tasks',[]):
        raise RuntimeError('The requested classifier is unavailable; no substitution is permitted.')
    dataset=client.upload(columns=columns+['label'],rows=[r.tolist()+[y] for r,y in zip(train,labels)],target='label',name='Synthetic SDK notebook')
    predictor=client.submit_fit(dataset['id'],model='tabicl-v2',task='classification',seed=9).result(timeout=120)
    return columns,query,predictor


def regression_table():
    """Exact planted qualification fixture: seed42, support128/query32, features4."""
    rng=np.random.default_rng(42)
    train=rng.normal(size=(128,4));query=rng.normal(size=(32,4))
    weights=rng.normal(size=4);target=train@weights+rng.normal(scale=0.05,size=128)
    return ['feature_a','feature_b','feature_c','feature_d'],train,target,query


def prepare_regression(client):
    columns,train,target,query=regression_table()
    entry=next(m for m in client.models() if m['id']=='tabicl-v2')
    profile=next((p for p in entry.get('task_profiles',[]) if p['task']=='regression'),None)
    if not profile or profile.get('status')!='available' or not profile.get('execution_available'):
        raise RuntimeError('Qualified regression worker unavailable; no task/model substitution permitted.')
    dataset=client.upload(columns=columns+['target'],rows=[r.tolist()+[float(y)] for r,y in zip(train,target)],target='target',name='Synthetic regression SDK notebook')
    predictor=client.submit_fit(dataset['id'],model='tabicl-v2',task='regression',seed=42).result(timeout=120)
    if predictor['model_version']!=profile['version']:
        raise RuntimeError('Prepared regression version differs from the selected available profile.')
    return columns,query,predictor
