"""Forecast context privacy, time semantics, durable jobs and hostile response controls."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import pickle

import httpx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from welt import (Client, Forecast, ForecastJob, ForecastResult, InvalidForecastResultError,
                  AuthenticationError, ConflictError, JobTimeoutError, WeltError)


class Service:
    def __init__(self):
        self.requests, self.jobs, self.payloads = [], {}, []
        self.workspace, self.version, self.configuration = 'workspace-a', 'model-v1', 'recipe-v1'
        self.supports = True
        self.result_change = lambda value: None
        self.job_change = lambda value: None
        self.status = 'succeeded'

    def handle(self, request):
        self.requests.append(request)
        path = request.url.path
        if path == '/v1/workspace':
            return httpx.Response(200, json={'id': self.workspace, 'slug': 'example'})
        if path == '/v1/models':
            return httpx.Response(200, json={'items': []})
        if path == '/v1/models/chronos-2':
            return httpx.Response(200, json=dict(id='chronos-2', version=self.version,
                configuration={'version': self.configuration}, operations=['model_run'],
                tasks=['univariate_forecasting'], supports_version_expectations=self.supports,
                limits={'min_context': 16, 'max_context': 512, 'max_horizon': 64}))
        if path == '/v1/model-runs':
            payload = json.loads(request.content)
            self.payloads.append(payload)
            if (request.headers.get('X-Welt-Model-Version') != self.version or
                    request.headers.get('X-Welt-Configuration-Version') != self.configuration):
                return httpx.Response(409, json={'error': {'code': 'model_version_mismatch', 'message': 'Refit.'}})
            name = 'job-' + str(len(self.jobs) + 1)
            job = dict(id=name, model_id='chronos-2', model_version=self.version,
                       resolved_configuration={'version': self.configuration}, operation='model_run',
                       task='univariate_forecasting', status=self.status, predictor_id=None)
            self.jobs[name] = (job, payload)
            return httpx.Response(202, json=job)
        if path.endswith('/cancel'):
            self.jobs[path.split('/')[-2]][0]['status'] = 'cancelled'
            return httpx.Response(200, json=self.jobs[path.split('/')[-2]][0])
        if path.startswith('/v1/jobs/'):
            name = path.split('/')[3]
            job, payload = self.jobs[name]
            if path.endswith('/result'):
                horizon, series = payload['horizon'], payload['series']
                levels = payload['quantile_levels']
                start = datetime.fromisoformat(series['start'])
                points = [float(i + 1) for i in range(horizon)]
                result = dict(task='univariate_forecasting', model_id='chronos-2',
                    model_version=job['model_version'], configuration=job['resolved_configuration']['version'],
                    context_length=len(series['values']), horizon=horizon,
                    frequency_seconds=series['frequency_seconds'],
                    timestamps=[(start + timedelta(seconds=series['frequency_seconds'] *
                                (len(series['values']) + i))).isoformat() for i in range(horizon)],
                    predictions=points, point_semantics='median', quantile_levels=levels,
                    quantiles=[[v + level - 0.5 for level in levels] for v in points],
                    calibrated_intervals=False)
                self.result_change(result)
                return httpx.Response(200, json=result)
            value = deepcopy(job)
            self.job_change(value)
            return httpx.Response(200, json=value)
        raise AssertionError('unexpected route ' + path)


@pytest.fixture
def service(monkeypatch):
    svc = Service()
    clients = []
    def factory(self):
        client = Client(base_url='https://fixture.example', api_key=self.api_key or 'controlled-key',
                        transport=httpx.MockTransport(svc.handle))
        clients.append(client)
        return client
    monkeypatch.setattr(Forecast, '_client', factory)
    svc.clients = clients
    return svc


def history():
    return pd.DataFrame({'timestamp': pd.date_range('2026-01-01', periods=16, freq='h', tz='UTC'),
                         'sales': np.arange(16, dtype=float)})


def fitted():
    return Forecast().fit(history(), time='timestamp', target='sales')


def test_fit_is_private_owned_and_predictions_have_exact_time_and_provenance(service):
    frame = history()
    model = Forecast()
    assert model.fit(frame, time='timestamp', target='sales') is model
    assert not service.payloads and all(r.method == 'GET' for r in service.requests)
    assert model.context_length_ == 16 and model.frequency_seconds_ == 3600
    assert model.cutoff_ == datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
    frame.loc[0, 'sales'] = 99999
    result = model.predict_details(horizon=3, idempotency_key='request-one')
    assert isinstance(result, ForecastResult) and result.job_id == 'job-1'
    assert service.payloads[0]['series']['values'][0] == 0
    assert len(service.payloads) == 1 and not any('expected' in k for k in service.payloads[0])
    assert result.timestamps.tolist() == list(pd.date_range('2026-01-01T16:00Z', periods=3, freq='h'))
    assert result.model_version == 'model-v1' and result.configuration == 'recipe-v1'
    assert result.quantile_levels == (.1, .5, .9)
    np.testing.assert_array_equal(result.predictions, [1., 2., 3.])
    assert result.quantiles.shape == (3, 3)
    assert list(result.to_frame()) == ['median', 'q_0.1', 'q_0.5', 'q_0.9']
    a = result.predictions; a[0] = 800
    d = result.to_dict(); d['quantiles'][0][0] = -800
    assert result.predictions[0] == 1 and result.quantiles[0, 0] == pytest.approx(.6)
    assert service.clients[-1].http.is_closed
    assert model.predict(2).shape == (2,)
    assert model.predict_quantiles(2, quantile_levels=[.2, .8]).shape == (2, 2)
    assert len(service.payloads) == 3


def test_series_and_array_inputs_preserve_instants_and_explicit_spacing(service):
    indexed = pd.Series(np.arange(16), index=history()['timestamp'].dt.tz_convert('Europe/Paris'))
    model = Forecast().fit(indexed)
    result = model.predict_details(1)
    assert result.timestamps[0] == pd.Timestamp('2026-01-01T16:00Z')
    model.fit(np.arange(16), start='2026-01-01T01:00:00+01:00', frequency_seconds=3600)
    assert model.predict_details(1).timestamps[0] == result.timestamps[0]


@pytest.mark.parametrize('change',[
    lambda f: f.iloc[::-1],
    lambda f: f.assign(timestamp=f['timestamp'].mask(f.index == 3, f['timestamp'][2])),
    lambda f: f.assign(timestamp=f['timestamp'].mask(f.index == 3, f['timestamp'][3] + pd.Timedelta(minutes=1))),
    lambda f: f.assign(timestamp=f['timestamp'].dt.tz_localize(None)),
    lambda f: f.assign(sales=f['sales'].mask(f.index == 1, np.nan)),
    lambda f: f.assign(sales=f['sales'].astype(str)),
    lambda f: f.assign(sales=True),
    lambda f: f.assign(sales=float('inf')),
    lambda f: f.assign(sales=1e39),
    lambda f: pd.concat([f] * 33, ignore_index=True),
])
def test_invalid_history_never_reaches_transport(service, change):
    with pytest.raises(ValueError):
        Forecast().fit(change(history()), time='timestamp', target='sales')
    assert not service.requests


def test_context_bounds_come_from_metadata_and_never_submit(service):
    # 0.14 (DEC-085 §3): the 16-observation minimum is the server's published min_context,
    # read at fit, not a client constant; only metadata reads precede the refusal.
    with pytest.raises(ValueError, match='16 to 512'):
        Forecast().fit(history().iloc[:15], time='timestamp', target='sales')
    assert not service.payloads and all(r.method == 'GET' for r in service.requests)


@pytest.mark.parametrize('kwargs',[
    {}, {'start': '2026-01-01', 'frequency_seconds': 1},
    {'start': '2026-01-01T00:00Z', 'frequency_seconds': True},
    {'start': '2026-01-01T00:00Z', 'frequency_seconds': 1.5},
    {'start': '2026-01-01T00:00Z', 'frequency_seconds': 0},
])
def test_arrays_require_exact_time_arguments_before_network(service, kwargs):
    with pytest.raises(ValueError): Forecast().fit(np.arange(16), **kwargs)
    assert not service.requests


@pytest.mark.parametrize('horizon,levels',[(0,[.1,.5,.9]),(65,[.1,.5,.9]),(True,[.1]),
 (1.5,[.1]),(2,[.9,.1]),(2,[.5,.5]),(2,[.25]),(2,[True]),(2,None)])
def test_invalid_prediction_arguments_do_not_submit(service, horizon, levels):
    model = fitted(); before = len(service.requests)
    with pytest.raises(ValueError): model.predict_details(horizon, quantile_levels=levels)
    assert not service.payloads and len(service.requests) == before


def test_fit_failure_and_parameter_changes_clear_context_clone_has_no_history(service):
    model = fitted()
    duplicate = clone(model)
    with pytest.raises(NotFittedError): duplicate.predict(2)
    assert 'sales' not in repr(model) and 'controlled-key' not in repr(model)
    assert 'explicit-secret' not in repr(Forecast(api_key='explicit-secret'))
    assert b'explicit-secret' not in pickle.dumps(Forecast(api_key='explicit-secret'))
    model.set_params(model='other')
    with pytest.raises(NotFittedError): model.predict(2)
    model = fitted(); model.model = 'other'
    with pytest.raises(ValueError, match='parameters changed'): model.predict(2)
    model = fitted()
    with pytest.raises(ValueError): model.fit(history().iloc[:2], time='timestamp', target='sales')
    with pytest.raises(NotFittedError): model.predict(2)


def test_fit_requires_server_guard_advertisement(service):
    service.supports = False
    with pytest.raises(WeltError) as error: fitted()
    assert error.value.code == 'unsupported_capability' and not service.payloads


def test_changed_model_or_workspace_refuses_private_submission(service):
    model = fitted(); service.version = 'model-v2'
    with pytest.raises(ConflictError): model.predict(2)
    assert not service.jobs
    service.version = 'model-v1'; service.payloads.clear()
    service.workspace = 'workspace-b'
    with pytest.raises(AuthenticationError): model.predict(2)
    assert not service.payloads


def test_explicit_job_timeout_does_not_resubmit_or_cancel_and_reconnect_is_typed(service):
    service.status = 'queued'
    model = fitted()
    with model.submit_predict(2, idempotency_key='durable-one') as job:
        assert isinstance(job, ForecastJob)
        assert job.inspect()['status'] == 'queued'
        with pytest.raises(JobTimeoutError) as error: job.result(timeout=.001)
        assert error.value.job_id == job.id and len(service.jobs) == 1
        assert not any(r.url.path.endswith('/cancel') for r in service.requests)
        service.jobs[job.id][0]['status'] = 'succeeded'
        reopened = job.client.forecast_job(job.id)
        assert isinstance(reopened.result(), ForecastResult)
        assert len(service.payloads) == 1
    assert job.client.http.is_closed


@pytest.mark.parametrize('change',[
    lambda r: r.update(model_version='wrong-version'),
    lambda r: r.update(configuration='wrong-recipe'),
    lambda r: r.update(model_id='wrong-model'),
    lambda r: r.update(predictions=[1.]),
    lambda r: r.update(point_semantics='mean'),
    lambda r: r.update(calibrated_intervals=True),
    lambda r: r.update(quantile_levels=[.9,.5,.1]),
    lambda r: r.update(quantiles=[[3.,2.,1.]] * 2),
    lambda r: r.update(quantiles=[[.6,99.,1.4]] * 2),
    lambda r: r.update(timestamps=['2026-01-02T16:00:00Z','2026-01-02T17:00:00Z']),
    lambda r: r.update(frequency_seconds=60),
    lambda r: r.update(context_length=17),
    lambda r: r.update(extra='unadvertised'),
])
def test_wrong_native_results_fail_without_repair(service, change):
    model = fitted(); service.result_change = change
    with pytest.raises(InvalidForecastResultError): model.predict_details(2)
    assert len(service.payloads) == 1 and service.clients[-1].http.is_closed


def test_wrong_job_identity_refuses_result_fetch(service):
    model = fitted(); service.job_change = lambda j: j.update(task='regression')
    with model.submit_predict(2) as job:
        with pytest.raises(InvalidForecastResultError): job.result()
    assert not any(r.url.path.endswith('/result') for r in service.requests)


def test_changed_effective_key_or_origin_is_refused_before_private_data(service, monkeypatch):
    model = fitted()
    before = len(service.requests)
    def changed_key(self):
        return Client(base_url='https://fixture.example', api_key='another-key',
                      transport=httpx.MockTransport(service.handle))
    monkeypatch.setattr(Forecast, '_client', changed_key)
    with pytest.raises(AuthenticationError): model.predict(2)
    assert len(service.requests) == before and not service.payloads


def test_guard_headers_async_parity_and_typed_reconnect():
    import asyncio
    from welt import AsyncClient, AsyncForecastJob
    svc = Service()
    payload = dict(model_id='chronos-2', task='univariate_forecasting',
        series=dict(values=list(range(16)), start='2026-01-01T00:00:00Z', frequency_seconds=3600),
        horizon=2, quantile_levels=[.1,.5,.9])
    async def run():
        async with AsyncClient(api_key='fixture', transport=httpx.MockTransport(svc.handle)) as client:
            raw = await client.submit_model_run(payload, idempotency_key='async-example',
                          expected_model_version='model-v1', expected_configuration_version='recipe-v1')
            job = client.forecast_job(raw.id)
            assert isinstance(job, AsyncForecastJob)
            assert (await job.inspect())['task'] == 'univariate_forecasting'
            result = await job.result()
            assert isinstance(result, ForecastResult) and result.quantiles.shape == (2,3)
            assert (await job.cancel())['status'] == 'cancelled'
    asyncio.run(run())
    assert len(svc.payloads) == 1
    post = next(r for r in svc.requests if r.method == 'POST' and r.url.path == '/v1/model-runs')
    assert post.headers['X-Welt-Model-Version'] == 'model-v1'
    assert post.headers['X-Welt-Configuration-Version'] == 'recipe-v1'


def test_paired_version_header_validation_is_local_sync_and_async():
    import asyncio
    from welt import AsyncClient
    requests = []
    def handle(r):
        requests.append(r)
        raise AssertionError('must not send malformed headers')
    with Client(api_key='fixture', transport=httpx.MockTransport(handle)) as c:
        with pytest.raises(ValueError): c.submit_model_run({}, expected_model_version='v1')
        with pytest.raises(ValueError): c.submit_model_run({}, expected_model_version='v1\nunsafe', expected_configuration_version='c1')
    async def run():
        async with AsyncClient(api_key='fixture', transport=httpx.MockTransport(handle)) as c:
            with pytest.raises(ValueError): await c.submit_model_run({}, expected_configuration_version='c1')
    asyncio.run(run())
    assert not requests


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True])
def test_nonfinite_or_boolean_native_result_values_are_rejected_without_leaking(value, service):
    result = fitted().predict_details(1)
    payload = result.to_dict(); payload['predictions'] = [value]
    with pytest.raises(InvalidForecastResultError) as error:
        ForecastResult(payload, job_id=result.job_id)
    assert 'no repair' in str(error.value)


@pytest.mark.parametrize('change', [lambda j: j.update(task='regression'),
                                   lambda j: j.update(model_version='unexpected-version')])
def test_typed_cancellation_validates_before_mutation(service, change):
    with fitted().submit_predict(2) as job:
        service.job_change = change
        with pytest.raises(InvalidForecastResultError): job.cancel()
    assert not any(r.url.path.endswith('/cancel') for r in service.requests)


def test_reconnected_job_borrows_shared_client(service):
    with fitted().submit_predict(2) as submitted:
        with Client(api_key='fixture', transport=httpx.MockTransport(service.handle)) as client:
            with client.forecast_job(submitted.id) as reconnected:
                assert reconnected.result().predictions.shape == (2,)
            assert not client.http.is_closed
            assert client.workspace()['id'] == 'workspace-a'
    assert submitted.client.http.is_closed


def test_async_reconnected_job_borrows_client_and_refuses_nonforecast_cancel():
    import asyncio
    from welt import AsyncClient
    svc = Service()
    svc.jobs['job-owned'] = (dict(id='job-owned', operation='model_run', task='regression',
        model_id='chronos-2', model_version='model-v1', resolved_configuration={'version':'recipe-v1'},
        status='queued', predictor_id=None), {})
    async def run():
        async with AsyncClient(api_key='fixture', transport=httpx.MockTransport(svc.handle)) as client:
            async with client.forecast_job('job-owned') as job:
                with pytest.raises(InvalidForecastResultError): await job.cancel()
            assert not client.http.is_closed
            assert (await client.workspace())['id'] == 'workspace-a'
    asyncio.run(run())
    assert not any(r.url.path.endswith('/cancel') for r in svc.requests)


def test_published_forecasting_example_uses_user_file_and_current_sdk(service, tmp_path, monkeypatch, capsys):
    import runpy
    import sys
    from pathlib import Path
    path = tmp_path/'history.csv'
    history().to_csv(path, index=False)
    example = Path(__file__).resolve().parents[1]/'examples/forecasting.py'
    monkeypatch.setattr(sys, 'argv', [str(example), str(path), '--time', 'timestamp', '--target', 'sales', '--horizon', '2'])
    runpy.run_path(str(example), run_name='__main__')
    assert len(service.payloads) == 1
    assert 'Job:' in capsys.readouterr().out


def test_published_forecasting_guide_executes_success_and_owned_context(service, tmp_path, monkeypatch, capsys):
    import re
    from pathlib import Path
    history().rename(columns={'sales':'value'}).to_csv(tmp_path/'history.csv', index=False)
    monkeypatch.chdir(tmp_path)
    guide = (Path(__file__).resolve().parents[1]/'docs/forecasting.md').read_text()
    namespace = {}
    for program in re.findall(r'```python\n(.*?)```', guide, re.S):
        exec(compile(program, 'Forecast guide', 'exec'), namespace)
    assert len(service.payloads) == 5
    assert namespace['saved_job_id'] is None
    assert isinstance(namespace['result'], ForecastResult)
    assert namespace['job'].client.http.is_closed
    capsys.readouterr()  # Controlled values do not qualify prediction quality.
