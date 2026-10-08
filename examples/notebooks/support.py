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


class BatchSyntheticService(SyntheticService):
    """Partial batch wire fixture; no native calls, database CAS or resource proof."""
    def __init__(self):
        super().__init__()
        self.batch_results, self.batch_payloads = {}, {}
        self.payload_reads = 0
        self.expired = False

    def usage(self, *args, **kwargs):
        super().usage(*args, **kwargs)
        self.events[-1].update(resolved_configuration={}, device=None)

    def __call__(self, request):
        assert request.url.host == 'notebook.invalid', 'Fixture must not make live requests.'
        path, method = request.url.path, request.method
        if path == '/v1/batch-predictions' and method == 'POST':
            value = json.loads(request.content)
            dataset = self.data[value['dataset_id']]
            predictor = self.fitted[value['predictor_id']]
            assert dataset['target'] is None and len(dataset['rows']) == 5
            assert predictor['task'] == 'classification'
            jid = f'batch-job-{len(self.batch_results)}'
            op = f'batch-operation-{len(self.batch_results)}'
            reference = f'batch-result-{len(self.batch_results)}'
            result = dict(job_id=jid, row_count=5, successful_ranges=[[0,2],[4,5]],
                failed_ranges=[[2,4]], result_reference=reference,
                expires_at='2099-01-01T00:00:00+00:00', model_version=predictor['model_version'])
            ordered = [dataset['columns'].index(c) for c in predictor['columns']]
            rows = [[row[i] for i in ordered] for row in dataset['rows']]
            labels = ['positive' if row[0]+.5*row[1]>0 else 'negative' for row in rows]
            predictions = [labels[0],labels[1],None,None,labels[4]]
            vectors = [[.1,.9] if label=='positive' else [.9,.1] for label in labels]
            probabilities = [vectors[0],vectors[1],None,None,vectors[4]] if value['probabilities'] else None
            self.batch_results[jid] = result
            self.batch_payloads[jid] = dict(job_id=jid,result_reference=reference,row_count=5,
                model_version=predictor['model_version'],predictor_id=predictor['id'],operation_id=op,
                batch_protocol='welt-batch-contiguous-v1',predictions=predictions,
                probabilities=probabilities,classes=predictor['classes'],
                errors=[dict(range=[2,4],code='execution_failed',retryable=False,outcome='failed')])
            self.job_records[jid] = dict(id=jid,status='partially_completed',stage='partially_completed',
                operation='batch_predict',operation_id=op,model_id=predictor['model_id'],
                model_version=predictor['model_version'],dataset_id=dataset['id'],
                predictor_id=predictor['id'],task='classification',configuration='default',
                resolved_configuration={},seed=predictor['seed'],result=result,error=None,
                created_at='2026-01-01T00:00:00+00:00',updated_at='2026-01-01T00:00:00+00:00',
                expires_at=result['expires_at'])
            for status in ('queued','partially_completed'):
                self.usage(op,'batch_predict',status,dataset['id'],predictor['id'],5)
            return httpx.Response(202,json=self.job_records[jid])
        if path.startswith('/v1/jobs/batch-job-'):
            jid = path.split('/')[3]
            if path.endswith('/result/payload'):
                assert request.url.params['result_reference'] == self.batch_results[jid]['result_reference']
                self.payload_reads += 1
                if self.expired:
                    return httpx.Response(410,json=dict(error=dict(code='result_expired',
                        message='Synthetic fixture clock has crossed payload expiry.',retryable=False,
                        request_id='synthetic-batch-expiry',job_id=jid)))
                return httpx.Response(200,json=self.batch_payloads[jid])
            if path.endswith('/result'):
                return httpx.Response(200,json=self.batch_results[jid])
            return httpx.Response(200,json=self.job_records[jid])
        return super().__call__(request)


def notebook_batch_client(service=None):
    if os.environ.get('WELT_NOTEBOOK_MODE','fixture') != 'fixture':
        raise RuntimeError('Batch has no qualified live task; this example is controlled transport only.')
    options=notebook_options()
    options['transport']=httpx.MockTransport(service if service is not None else BatchSyntheticService())
    return Client(**options)


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


class CausalSyntheticService(SyntheticService):
    """Native-result wire fixture only; no discovery model or causal truth claim."""
    def __init__(self):
        super().__init__()
        self.causal_results, self.causal_scores = {}, {}

    def __call__(self, request):
        assert request.url.host == 'notebook.invalid', 'Fixture must not make live requests.'
        path, method = request.url.path, request.method
        value = json.loads(request.content) if request.content else {}
        if path == '/v1/models':
            return httpx.Response(200, json={'items': [dict(id='arrow', status='available',
                tasks=['causal_discovery'], task_profiles=[dict(task='causal_discovery',
                status='available', execution_available=True, version='synthetic-arrow-contract-v1',
                limits=dict(min_rows=2,max_rows=500,min_features=2,max_features=20))])]})
        if path == '/v1/discoveries':
            dataset=self.data[value['dataset_id']]
            assert dataset['target'] is None and value['model_id']=='arrow'
            jid=f'job-{len(self.job_records)}';rid=f'causal-{len(self.causal_results)}'
            names=dataset['columns']
            edges=[dict(u=names[0],v=names[1],u_mark='tail',v_mark='arrow'),
                   dict(u=names[1],v=names[2],u_mark='tail',v_mark='arrow')]
            result=dict(id=rid,workspace_id='workspace-fixture',dataset_id=dataset['id'],
                model_version='synthetic-arrow-contract-v1',configuration_version='synthetic-native-dag-v1',
                variables=names,native_graph_type='dag',edge_marks=edges,
                score_semantics='native_directed_edge_probability',score_reference=f'scores-{rid}',
                decoder=dict(name='synthetic wire fixture; no model decoder'),
                assumptions=['Synthetic graph fixture, not learned discovery'],
                constraints_applied=value['constraints'],diagnostics=dict(job_id=jid,seed=value['seed']))
            self.causal_results[jid]=result
            self.causal_scores[jid]=dict(result_id=rid,score_reference=result['score_reference'],
                variables=names,score_axes='source_row_target_column',
                score_semantics='native_directed_edge_probability',
                values=[[0,.8,.2,.1],[.1,0,.7,.1],[.2,.1,0,.1],[.1,.1,.1,0]])
            operation_id=f'discovery-operation-{len(self.causal_results)}'
            self.job_records[jid]=dict(id=jid,status='succeeded',operation='discover',
                predictor_id=None,operation_id=operation_id)
            for status in ('running','succeeded'):
                self.events.append(dict(id=f'event-{len(self.events)}',operation_id=operation_id,kind='discover',
                    status=status,dataset_id=dataset['id'],predictor_id=None,model_id='arrow',
                    model_version=result['model_version'],task='causal_discovery',rows=len(dataset['rows']),
                    columns=len(names),attempt=1,created_at='2026-01-01T00:00:00Z'))
            return httpx.Response(202,json=self.job_records[jid])
        if path.startswith('/v1/jobs/') and path.endswith('/result/scores'):
            return httpx.Response(200,json=self.causal_scores[path.split('/')[3]])
        if path.startswith('/v1/jobs/') and path.endswith('/result'):
            return httpx.Response(200,json=self.causal_results[path.split('/')[3]])
        return super().__call__(request)


def notebook_causal_client():
    options=notebook_options()
    if os.environ.get('WELT_NOTEBOOK_MODE','fixture')=='fixture':
        options['transport']=httpx.MockTransport(CausalSyntheticService())
    return Client(**options)


def causal_table():
    """Arrow frozen chain/isolate panel input: seed42, full128x4 observations."""
    rng=np.random.default_rng(42)
    values=rng.exponential(size=(128,4))-1
    values[:,1]=1.5*values[:,0]+.25*values[:,1]
    values[:,2]=1.1*values[:,1]+.25*values[:,2]
    return ['z_treatment','m_mediator','a_outcome','isolated'],values


class UploadSyntheticService(SyntheticService):
    """Tiny CSV protocol fixture; memory use here is not server capacity evidence."""
    def __init__(self, interrupt_once=True):
        super().__init__()
        self.uploads = {}
        self.file_tables = {}
        self.interrupt_once = interrupt_once

    def __call__(self, request):
        import csv
        import hashlib
        import io
        assert request.url.host == 'notebook.invalid', 'Fixture must not make live requests.'
        path, method = request.url.path, request.method
        if path == '/v1/uploads' and method == 'POST':
            declared = json.loads(request.content)
            uid = f'upload-{len(self.uploads)}'
            self.uploads[uid] = dict(info=dict(id=uid, workspace_id='workspace-fixture', **declared, chunk_bytes=16,
                received_indices=[], status='uploading', dataset_id=None,
                expires_at='2099-01-01T00:00:00Z'), chunks={})
            return httpx.Response(200, json=self.uploads[uid]['info'])
        if path.startswith('/v1/uploads/'):
            parts = path.split('/'); uid = parts[3]
            state = self.uploads[uid]; info = state['info']
            if method == 'GET':
                return httpx.Response(200, json=info)
            if len(parts) == 6 and parts[4] == 'chunks' and method == 'PUT':
                index = int(parts[5]); body = request.content
                assert hashlib.sha256(body).hexdigest() == request.headers['x-chunk-sha256']
                expected = min(info['chunk_bytes'], info['size_bytes'] - index * info['chunk_bytes'])
                assert len(body) == expected and expected > 0
                if index in state['chunks']:
                    assert state['chunks'][index] == body
                state['chunks'][index] = body
                info['received_indices'] = sorted(state['chunks'])
                if self.interrupt_once:
                    self.interrupt_once = False
                    raise httpx.ReadError('Synthetic lost chunk acknowledgement.', request=request)
                return httpx.Response(200, json=info)
            if path.endswith('/complete') and method == 'POST':
                if info['dataset_id']:
                    return httpx.Response(200, json=self.data[info['dataset_id']])
                body = b''.join(state['chunks'][i] for i in range(len(state['chunks'])))
                assert len(body) == info['size_bytes'] and hashlib.sha256(body).hexdigest() == info['sha256']
                table = list(csv.reader(io.StringIO(body.decode('utf-8'))))
                columns = table[0]; types = info['schema']['types']
                assert columns == info['schema']['columns']
                def cell(value, kind):
                    if value == '': return None
                    if kind == 'number': return float(value)
                    if kind == 'boolean':
                        assert value in ('true', 'false')
                        return value == 'true'
                    return value
                rows = [[cell(v, t) for v, t in zip(row, types)] for row in table[1:]]
                item = dict(id=f'dataset-{len(self.data)}', name='Uploaded CSV', columns=columns,
                    types=['numeric' if t == 'number' else 'categorical' for t in types],
                    rows=len(rows), target=info['target'], created_at='2026-01-01T00:00:00Z')
                self.file_tables[item['id']] = rows
                self.data[item['id']] = item
                info.update(status='complete', dataset_id=item['id'])
                return httpx.Response(200, json=item)
        if path.startswith('/v1/datasets/') and method == 'GET':
            return httpx.Response(200, json=self.data[path.rsplit('/', 1)[1]])
        return super().__call__(request)


def notebook_file_client():
    options = notebook_options()
    if os.environ.get('WELT_NOTEBOOK_MODE', 'fixture') == 'fixture':
        options['transport'] = httpx.MockTransport(UploadSyntheticService())
    return Client(**options)


def notebook_async_file_client():
    options = notebook_options()
    if os.environ.get('WELT_NOTEBOOK_MODE', 'fixture') == 'fixture':
        options['transport'] = httpx.MockTransport(UploadSyntheticService(interrupt_once=False))
    return AsyncClient(**options)
