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
        self.evaluations = {}

    def evaluation(self, value):
        """Constant evaluation wire shapes on the stored rows; no cross-validation is computed."""
        ds = self.data[value['dataset_id']]
        target = ds['target']
        labels = [row[ds['columns'].index(target)] for row in ds['rows']]
        task, k = value['task'], value['folds'] or 5
        classification = task == 'classification'
        classes = sorted(set(labels)) if classification else None
        metrics = (['roc_auc', 'log_loss', 'brier', 'accuracy', 'balanced_accuracy', 'f1_macro', 'ece']
                   if classification else ['rmse', 'mae', 'r2', 'mape', 'median_ae'])
        version = 'synthetic-contract-v1' if classification else 'synthetic-regression-contract-v1'
        models = value['models'] or ['tabicl-v2']
        identity = f'evaluation-{len(self.evaluations)}'
        entry = lambda model_id, name, score: dict(model_id=model_id, name=name, mean=dict.fromkeys(metrics, score),
                                                   sd=dict.fromkeys(metrics, 0.0), pooled=dict.fromkeys(metrics, score))
        summary = dict(primary_metric=metrics[0], metrics=metrics, directions={}, comparison=None,
                       leaderboard=[entry(m, m, 0.5) for m in models], baseline=entry('baseline', 'Baseline', 0.5),
                       warnings=[dict(code='synthetic', message='Synthetic fixture scores; not model evidence.')])
        resource = dict(id=identity, status='queued', stage='queued', operation_id='op-' + identity,
            dataset_id=ds['id'], dataset_name=ds.get('name'), task=task, target=target, rows=len(labels),
            dataset_rows=len(labels), feature_columns=[c for c in ds['columns'] if c != target], folds=k,
            split=value['split'], group_column=None, fold_column=None, seed=value['seed'], sample_rows=None,
            primary_metric=metrics[0], positive_class=None, classes=classes, class_counts=None, fold_sha256='0' * 64,
            protocol_version='evaluation-protocol-v1', metrics_version='metrics-v1', report_version='report-v1',
            model_selection='explicit' if value['models'] else 'evaluation-defaults-v1',
            models=[dict(model_id=m, name=m, model_version=version, configuration='default', resolved_configuration={},
                         job_id=f'{identity}-{m}', status='queued', error=None) for m in models],
            skipped=[], estimated_credits=5, credits=None, runtime={}, summary=None, report_available=False,
            predictions_expires_at=None, error=None, created_at='2026-01-01T00:00:00Z',
            updated_at='2026-01-01T00:00:00Z', completed_at=None)
        folds = [i % k for i in range(len(labels))]
        outputs = {m: dict(predictions=[classes[0] if classification else 0.0] * len(labels),
                           probabilities=[[1.0] + [0.0] * (len(classes) - 1)] * len(labels) if classification else None)
                   for m in models + ['baseline']}
        finished = dict(resource, status='succeeded', stage='complete', summary=summary, report_available=True,
                        credits=5, predictions_expires_at='2026-01-08T00:00:00Z', completed_at='2026-01-01T00:00:00Z',
                        models=[dict(m, status='succeeded') for m in resource['models']])
        self.evaluations[identity] = dict(resource=resource, finished=finished, links=[], predictions=dict(
            evaluation_id=identity, task=task, classes=classes, row_index=list(range(len(labels))), fold=folds,
            actual=labels, models=outputs, expires_at='2026-01-08T00:00:00Z'), folds=dict(
            evaluation_id=identity, metrics=metrics, items=[dict(model_id=m, fold=f, rows=folds.count(f),
                values=dict.fromkeys(metrics, 0.5)) for m in models + ['baseline'] for f in range(k)]))
        return resource

    def evaluation_route(self, path, method):
        if path == '/v1/evaluations' and method == 'GET':
            return httpx.Response(200, json={'items': [e['finished'] for e in self.evaluations.values()]})
        parts = path.split('/')
        record = self.evaluations[parts[3]]
        tail = '/'.join(parts[4:])
        if tail == '':
            return httpx.Response(200, json=record['finished'])
        if tail in ('folds', 'predictions'):
            return httpx.Response(200, json=record[tail])
        if tail == 'report':
            return httpx.Response(200, text='<!doctype html><html><head><meta charset="utf-8"><title>Synthetic report'
                                  '</title></head><body><div class="welt-report"><p>Synthetic fixture report; not '
                                  'model evidence.</p></div></body></html>', headers={'content-type': 'text/html'})
        if tail == 'report-links' and method == 'POST':
            link = dict(id=f'link-{len(record["links"])}', evaluation_id=parts[3], created_at='2026-01-01T00:00:00Z',
                        expires_at='2026-01-02T00:00:00Z', revoked=False, created_by=None)
            record['links'].append(link)
            return httpx.Response(201, json=dict(link, url='https://notebook.invalid/reports/synthetic'))
        raise AssertionError(f'Unsupported notebook fixture operation: {method} {path}')

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
        if path == '/v1/evaluations' and method == 'POST':
            return httpx.Response(202, json=self.evaluation(value))
        if path.startswith('/v1/evaluations'):
            return self.evaluation_route(path, method)
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


FORECAST_LEVELS = [v / 100 for v in (1, *range(5, 100, 5), 99)]
_SECONDS = {'week': 604800, 'day': 86400, 'hour': 3600, 'minute': 60, 'second': 1}


class ForecastSyntheticService:
    """DEC-085 forecasting wire fixture (POST /v1/forecasts), not Chronos-2 or model evidence.

    Points repeat each series' last seven observed values (a seasonal-naive rule) and the
    quantiles spread around them by the series' spread, so tutorials can compute metrics
    and plots offline. These numbers say nothing about Chronos-2 forecast quality.
    """
    version, configuration = 'synthetic-chronos-2-v1', 'synthetic-chronos-2-forecast-v1'
    # The seven corners API-160 measured and accepted (context x variates x series x horizon).
    envelopes = [dict(max_context=c, max_variates=v, max_series=n, max_horizon=h) for c, v, n, h in (
        (8000, 50, 1, 1024), (8000, 5, 8, 1024), (4000, 10, 8, 1024), (2000, 20, 8, 1024),
        (2000, 5, 32, 256), (1000, 50, 8, 256), (512, 20, 32, 256))]

    def __init__(self):
        self.jobs, self.posts = {}, []

    def handles(self, path):
        return (path in ('/v1/models/chronos-2', '/v1/forecasts')
                or path.startswith('/v1/jobs/forecast-'))

    def metadata(self):
        return dict(id='chronos-2', family='forecasting', status='available', tasks=['forecasting'],
                    operations=['forecast'], version=self.version,
                    configuration={'version': self.configuration}, supports_version_expectations=True,
                    forecast_envelopes=self.envelopes, min_context=16, quantile_levels=FORECAST_LEVELS,
                    default_quantile_levels=[0.1, 0.5, 0.9],
                    limits=dict(min_context=16, **self.envelopes[0]))

    @staticmethod
    def _frequency(request, stamps):
        if 'frequency' in request:
            return request['frequency']
        seconds = int((stamps[1] - stamps[0]).total_seconds())
        unit = next(u for u, size in _SECONDS.items() if seconds % size == 0)
        return {'unit': unit, 'every': seconds // _SECONDS[unit]}

    def result(self, request):
        from statistics import NormalDist
        import pandas as pd
        table = request['history']['table']
        columns, rows = table['columns'], table['rows']
        t = columns.index(request['time_column'])
        i = columns.index(request['id_column']) if 'id_column' in request else None
        groups = {}
        for row in rows:
            groups.setdefault(None if i is None else str(row[i]), []).append(row)
        frequency, series = None, []
        for key, members in groups.items():
            members.sort(key=lambda row: pd.Timestamp(row[t]))
            stamps = [pd.Timestamp(row[t]).tz_convert('UTC') if pd.Timestamp(row[t]).tzinfo
                      else pd.Timestamp(row[t]).tz_localize('UTC') for row in members]
            frequency = frequency or self._frequency(request, stamps)
            unit, every = frequency['unit'], frequency['every']
            if unit in _SECONDS:
                future = [stamps[-1] + pd.Timedelta(seconds=_SECONDS[unit] * every * k)
                          for k in range(1, request['horizon'] + 1)]
            else:
                months = every * dict(month=1, quarter=3, year=12)[unit]
                future = [stamps[-1] + pd.DateOffset(months=months * k) for k in range(1, request['horizon'] + 1)]
            for target in request['targets']:
                values = [row[columns.index(target)] for row in members]
                known = [float(v) for v in values if v is not None]
                level = sum(known) / len(known)
                recent = [level if v is None else float(v) for v in values[-7:]]
                spread = (sum((v - level) ** 2 for v in known) / len(known)) ** 0.5 or 1.0
                points = [recent[k % len(recent)] for k in range(request['horizon'])]
                quantiles = [[p + 0.5 * spread * NormalDist().inv_cdf(q) for q in request['quantile_levels']]
                             for p in points]
                series.append(dict(id=key, target=target, context_length=len(members),
                                   timestamps=[s.isoformat().replace('+00:00', 'Z') for s in future],
                                   predictions=points, quantiles=quantiles))
        return dict(task='forecasting', model_id=request['model_id'], model_version=self.version,
                    configuration=self.configuration, horizon=request['horizon'], frequency=frequency,
                    quantile_levels=request['quantile_levels'], point_semantics='median',
                    calibrated_intervals=False, cross_learning=request['cross_learning'], rearranged_steps=0,
                    columns=dict(time=request['time_column'], id=request.get('id_column'),
                                 targets=request['targets'], past_covariates=request['past_covariates'],
                                 future_covariates=request['future_covariates'],
                                 categorical_covariates=request['categorical_covariates']),
                    series=series)

    def __call__(self, request):
        assert request.url.host == 'notebook.invalid', 'Fixture must not make live requests.'
        path, method = request.url.path, request.method
        if path == '/v1/workspace':
            return httpx.Response(200, json={'id': 'workspace-fixture'})
        if path == '/v1/models':
            return httpx.Response(200, json={'items': [self.metadata()]})
        if path == '/v1/models/chronos-2':
            return httpx.Response(200, json=self.metadata())
        if path == '/v1/forecasts' and method == 'POST':
            body = json.loads(request.content)
            headers = request.headers
            assert 'dataset_id' not in body['history'], 'Fixture tables are sent inline.'
            if (headers.get('X-Welt-Model-Version') != self.version
                    or headers.get('X-Welt-Configuration-Version') != self.configuration):
                return httpx.Response(409, json={'error': {'code': 'model_version_mismatch',
                                                           'message': 'Fit again.'}})
            self.posts.append(body)
            for job, previous, key in self.jobs.values():
                if key == headers['Idempotency-Key']:
                    if previous != body:
                        return httpx.Response(409, json={'error': {'code': 'idempotency_conflict',
                                                                   'message': 'Key reused.'}})
                    return httpx.Response(202, json=job)
            name = f'forecast-{len(self.jobs) + 1}'
            job = dict(id=name, operation='forecast', task='forecasting', model_id=body['model_id'],
                       model_version=self.version, resolved_configuration={'version': self.configuration},
                       status='succeeded', predictor_id=None)
            self.jobs[name] = (job, body, headers['Idempotency-Key'])
            return httpx.Response(202, json=job)
        if path.startswith('/v1/jobs/forecast-'):
            job, body, _ = self.jobs[path.split('/')[3]]
            if path.endswith('/result'):
                return httpx.Response(200, json=self.result(body))
            if path.endswith('/cancel'):
                return httpx.Response(200, json=dict(job, status='cancelled'))
            return httpx.Response(200, json=job)
        raise AssertionError(f'Unsupported forecasting fixture operation: {method} {path}')


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
        for name, digest in (("iris", "49abb2b00bc1453978ff6d80ab3a71adbdda3a43d68f204e9251580bf6ef44cb"),
                             ("yacht", "3f69b3143b7c34a1f0af4088dfd9d1561dc8300542b7ba8dc97ea33caddfbff2")):
            root = Path(directory)
            detail = json.loads((root / (name + ".json")).read_text())
            body = (root / (name + ".parquet")).read_bytes()
            assert sha256(body).hexdigest() == digest
            assert detail["content"]["version"] == detail["content"]["sha256"] == digest
            assert detail["content"]["size_bytes"] == len(body)
            self.research[detail["id"]] = (detail, body)
        self.forecasting = ForecastSyntheticService()

    def __call__(self, request):
        path = request.url.path
        if self.forecasting.handles(path):
            return self.forecasting(request)
        if path == "/v1/workspace":
            return httpx.Response(200, json={"id": "workspace-fixture"})
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
