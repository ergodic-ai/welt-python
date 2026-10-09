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
                    labels=np.full(len(X), item["classes"][0])
                    probs=[[1.0] + [0.0]*(len(item["classes"])-1) for _ in labels]
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




class ResearchFixture(CausalSyntheticService):
    """Test wire responses with authentic pinned canonical research bytes.

    Outputs are deliberately constant test plumbing, not model-quality evidence.
    No data is stored in Git; caller supplies a reviewed temporary fixture directory.
    """
    def __init__(self, directory):
        super().__init__()
        from pathlib import Path
        from hashlib import sha256
        self.research = {}
        for name, digest in (("iris", "d839644b6de36fc5cafdda18c5daeca3284a5ce8bb6169aff2fcb3fbb4a12021"),
                             ("yacht", "bc7dddd884c5e798c1b2abfc559a36ab79a6d09c4e2ea3e9bbd95c06539c7342")):
            root = Path(directory)
            detail = json.loads((root / (name + ".json")).read_text())
            body = (root / (name + ".parquet")).read_bytes()
            assert sha256(body).hexdigest() == digest
            assert detail["content"]["version"] == detail["content"]["sha256"] == digest
            assert detail["content"]["size_bytes"] == len(body)
            self.research[detail["id"]] = (detail, body)

    def __call__(self, request):
        path = request.url.path
        if path == "/v1/research-datasets":
            return httpx.Response(200, json=dict(items=[item[0] for item in self.research.values()],
                total=2, next_cursor=None, catalogue_version="test-pinned-research", facets={}))
        if path.startswith("/v1/research-datasets/"):
            detail, body = self.research[path.split("/")[3]]
            if path.endswith("/content"):
                assert request.url.params["version"] == detail["content"]["version"]
                return httpx.Response(200, stream=httpx.ByteStream(body))
            return httpx.Response(200, json=detail)
        if path == "/v1/models":
            prediction = super(CausalSyntheticService, self).__call__(request).json()["items"]
            discovery = super().__call__(request).json()["items"]
            prediction += [dict(prediction[0], id=name) for name in ("kumo-medium", "tabdpt-1.3", "mitra-v2")]
            return httpx.Response(200, json={"items": prediction + discovery})
        return super().__call__(request)


def patch_clients(directory):
    """Only the test runner calls this; public examples do not import this module."""
    from contextlib import ExitStack
    from unittest.mock import patch
    fixture = ResearchFixture(directory)
    stack = ExitStack()
    for kind in (Client, AsyncClient):
        original = kind.__init__
        def initialize(self, *args, _original=original, **kwargs):
            kwargs.update(base_url="https://notebook.invalid", api_key="test-only-key",
                          transport=httpx.MockTransport(fixture))
            _original(self, *args, **kwargs)
        stack.enter_context(patch.object(kind, "__init__", initialize))
    return stack
