"""DEC-085 forecasting (POST /v1/forecasts): panels, covariates, envelopes, transport and results."""
import asyncio
from copy import deepcopy
import json

import httpx
import numpy as np
import pandas as pd
import pytest

import welt.forecast as forecast_module
import pickle

from welt import (AsyncClient, AsyncForecastJob, AuthenticationError, Client, ConflictError, Forecast,
                  ForecastResult, ForecastSeries, InvalidForecastResultError, InvalidInputError, WeltError)
from welt.cli import _parser, _run

LEVELS = [v / 100 for v in (1, *range(5, 100, 5), 99)]


def _steps(last, frequency, horizon):
    """Independent oracle for future timestamps (pandas offsets, not SDK arithmetic)."""
    unit, every = frequency['unit'], frequency['every']
    seconds = dict(second=1, minute=60, hour=3600, day=86400, week=604800)
    if unit in seconds:
        return [last + pd.Timedelta(seconds=seconds[unit] * every * k) for k in range(1, horizon + 1)]
    months = every * dict(month=1, quarter=3, year=12)[unit]
    if last.is_month_end and last.day > 28:
        return [last + pd.offsets.MonthEnd(months * k) for k in range(1, horizon + 1)]
    return [last + pd.DateOffset(months=months * k) for k in range(1, horizon + 1)]


class Service:
    """A fake Welt API implementing the DEC-085 contract from its own reading of the request."""

    def __init__(self):
        self.requests, self.jobs, self.posts, self.datasets = [], {}, [], {}
        self.version, self.configuration = 'chronos-2-v1', 'chronos-2-forecast-v1'
        self.envelopes = [dict(max_context=512, max_variates=1, max_series=1, max_horizon=64),
                          dict(max_context=2000, max_variates=10, max_series=20, max_horizon=256)]
        self.min_context = 16
        self.tasks, self.operations = ['forecasting', 'univariate_forecasting'], ['forecast', 'model_run']
        self.inferred = {'unit': 'day', 'every': 1}
        self.status = 'succeeded'
        self.rearranged = 0
        self.result_change = lambda value: None
        self.catalogue = []
        self.fail_post = False

    def metadata(self):
        return dict(id='chronos-2', version=self.version, configuration={'version': self.configuration},
                    operations=self.operations, tasks=self.tasks, supports_version_expectations=True,
                    limits={'min_context': 16, 'max_context': 512, 'max_horizon': 64},
                    forecast_envelopes=self.envelopes, min_context=self.min_context, quantile_levels=LEVELS)

    def table(self, reference):
        if 'dataset_id' in reference:
            return self.datasets[reference['dataset_id']]
        return reference['table']

    def result(self, request):
        history = self.table(request['history'])
        columns, rows = history['columns'], history['rows']
        t, i = columns.index(request['time_column']), (columns.index(request['id_column'])
                                                     if 'id_column' in request else None)
        order, last, length = [], {}, {}
        for row in rows:
            key = None if i is None else str(row[i])
            if key not in length:
                order.append(key)
            stamp = pd.Timestamp(row[t]).tz_convert('UTC')
            last[key] = max(last.get(key, stamp), stamp)
            length[key] = length.get(key, 0) + 1
        frequency = request.get('frequency', self.inferred)
        levels, horizon = request['quantile_levels'], request['horizon']
        series = []
        for n, key in enumerate(order):
            for target in request['targets']:
                points = [float(n * 100 + k + 1) for k in range(horizon)]
                series.append(dict(id=key, target=target, context_length=length[key],
                                   timestamps=[s.isoformat().replace('+00:00', 'Z')
                                               for s in _steps(last[key], frequency, horizon)],
                                   predictions=points,
                                   quantiles=[[p + level - 0.5 for level in levels] for p in points]))
        return dict(task='forecasting', model_id=request['model_id'], model_version=self.version,
                    configuration=self.configuration, horizon=horizon, frequency=frequency,
                    quantile_levels=levels, point_semantics='median', calibrated_intervals=False,
                    cross_learning=request['cross_learning'], rearranged_steps=self.rearranged,
                    columns=dict(time=request['time_column'], id=request.get('id_column'),
                                 targets=request['targets'], past_covariates=request['past_covariates'],
                                 future_covariates=request['future_covariates'],
                                 categorical_covariates=request['categorical_covariates']),
                    series=series)

    def handle(self, request):
        self.requests.append(request)
        path = request.url.path
        if path == '/v1/workspace':
            return httpx.Response(200, json={'id': 'workspace-a', 'slug': 'example'})
        if path == '/v1/models':
            return httpx.Response(200, json={'items': self.catalogue})
        if path == '/v1/models/chronos-2':
            return httpx.Response(200, json=self.metadata())
        if path == '/v1/datasets' and request.method == 'POST':
            body = json.loads(request.content)
            name = 'ds_' + str(len(self.datasets) + 1)
            self.datasets[name] = dict(columns=body['columns'], rows=body['rows'])
            return httpx.Response(201, json={'id': name, 'name': body['name']})
        if path == '/v1/forecasts':
            if self.fail_post:
                raise httpx.ConnectError('dropped')
            body = json.loads(request.content)
            self.posts.append((body, dict(request.headers)))
            if (request.headers.get('X-Welt-Model-Version') != self.version
                    or request.headers.get('X-Welt-Configuration-Version') != self.configuration):
                return httpx.Response(409, json={'error': {'code': 'model_version_mismatch', 'message': 'Refit.'}})
            for job, (_, previous_body, key) in self.jobs.items():
                if key == request.headers['Idempotency-Key']:
                    if previous_body != body:
                        return httpx.Response(409, json={'error': {'code': 'idempotency_conflict', 'message': 'x'}})
                    return httpx.Response(202, json=self.jobs[job][0])
            name = 'fc-' + str(len(self.jobs) + 1)
            job = dict(id=name, operation='forecast', task='forecasting', model_id='chronos-2',
                       model_version=self.version, resolved_configuration={'version': self.configuration},
                       status=self.status, predictor_id=None)
            self.jobs[name] = (job, body, request.headers['Idempotency-Key'])
            return httpx.Response(202, json=job)
        if path.startswith('/v1/forecasts/') and request.method == 'DELETE':
            name = path.split('/')[3]
            if name not in self.jobs:
                return httpx.Response(404, json={'error': {'code': 'not_found', 'message': 'Resource not found.'}})
            self.jobs[name][0]['purged'] = True
            return httpx.Response(200, json={'id': name, 'deleted': True})
        if path.startswith('/v1/jobs/'):
            name = path.split('/')[3]
            job, body, _ = self.jobs[name]
            if path.endswith('/cancel'):
                job['status'] = 'cancelled'
                return httpx.Response(200, json=job)
            if path.endswith('/result'):
                value = self.result(body)
                self.result_change(value)
                return httpx.Response(200, json=value)
            return httpx.Response(200, json=deepcopy(job))
        raise AssertionError('unexpected route ' + path)


@pytest.fixture
def service(monkeypatch):
    svc = Service()
    svc.clients = []

    def factory(self):
        client = Client(base_url='https://fixture.example', api_key=self.api_key or 'controlled-key',
                        transport=httpx.MockTransport(svc.handle))
        svc.clients.append(client)
        return client
    monkeypatch.setattr(Forecast, '_client', factory)
    return svc


def panel(n=20, ids=('b', 'a')):
    frames = []
    for k, sid in enumerate(ids):
        frames.append(pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=n + k, freq='D', tz='UTC'),
            'store': sid,
            'sales': np.arange(n + k, dtype=float) + 10 * k,
            'visits': np.arange(n + k, dtype=float),
            'temperature': np.linspace(0, 1, n + k),
            'promo': np.arange(n + k) % 2,
            'holiday': ['no'] * (n + k),
        }))
    return pd.concat(frames, ignore_index=True)


def future(frame, horizon, ids=None):
    rows = []
    for sid, group in frame.groupby('store', sort=False):
        if ids is not None and sid not in ids:
            continue
        for k, stamp in enumerate(pd.date_range(group['date'].max() + pd.Timedelta(days=1), periods=horizon,
                                                freq='D')):
            rows.append(dict(date=stamp, store=sid, promo=k % 2, holiday='yes' if k == 0 else 'no'))
    return pd.DataFrame(rows)


def fitted_panel(**kwargs):
    return Forecast().fit(panel(), time='date', id='store', target=['sales', 'visits'],
                          past_covariates=['temperature'], future_covariates=['promo', 'holiday'],
                          categorical=['holiday'], **kwargs)


def test_panel_multi_target_covariates_inline_request_and_long_frame(service):
    frame = panel()
    frame.loc[3, 'sales'] = np.nan          # native missing-value handling for targets
    frame.loc[4, 'temperature'] = None      # and past covariates
    model = Forecast().fit(frame, time='date', id='store', target=['sales', 'visits'],
                           past_covariates=['temperature'], future_covariates=['promo', 'holiday'],
                           categorical=['holiday'])
    assert not service.posts and all(r.method == 'GET' for r in service.requests)
    assert model.task_ == 'forecasting' and model.n_series_ == 2 and model.series_ids_ == ('b', 'a')
    assert model.context_length_ == 21 and model.frequency_ == {'unit': 'day', 'every': 1}
    frame.loc[0, 'sales'] = 1e6             # fit owns a copy
    result = model.forecast(3, future=future(panel(), 3), idempotency_key='panel-one')
    body, headers = service.posts[0]
    assert headers['idempotency-key'] == 'panel-one'
    assert headers['x-welt-model-version'] == 'chronos-2-v1'
    assert headers['x-welt-configuration-version'] == 'chronos-2-forecast-v1'
    assert {k: body[k] for k in ('model_id', 'configuration', 'time_column', 'id_column', 'targets',
                                 'past_covariates', 'future_covariates', 'categorical_covariates',
                                 'horizon', 'quantile_levels', 'cross_learning')} == dict(
        model_id='chronos-2', configuration='default', time_column='date', id_column='store',
        targets=['sales', 'visits'], past_covariates=['temperature'], future_covariates=['promo', 'holiday'],
        categorical_covariates=['holiday'], horizon=3, quantile_levels=[.1, .5, .9], cross_learning=False)
    assert 'frequency' not in body       # inferred by the server when not given
    history = body['history']['table']
    assert history['columns'] == ['date', 'store', 'sales', 'visits', 'temperature', 'promo', 'holiday']
    assert history['rows'][0] == ['2026-01-01T00:00:00+00:00', 'b', 0.0, 0.0, 0.0, 0, 'no']
    assert history['rows'][3][2] is None and history['rows'][4][4] is None
    assert body['future']['table']['columns'] == ['date', 'store', 'promo', 'holiday']
    assert len(body['future']['table']['rows']) == 6
    assert isinstance(result, ForecastResult) and result.task == 'forecasting'
    assert [(s.id, s.target) for s in result.series] == [('b', 'sales'), ('b', 'visits'),
                                                         ('a', 'sales'), ('a', 'visits')]
    assert all(isinstance(s, ForecastSeries) for s in result.series)
    assert [s.context_length for s in result.series] == [20, 20, 21, 21]
    assert result.rearranged_steps == 0 and result.frequency == {'unit': 'day', 'every': 1}
    assert result.model_version == 'chronos-2-v1' and result.configuration == 'chronos-2-forecast-v1'
    table = result.to_frame()
    assert list(table) == ['id', 'target', 'timestamp', 'median', 'q_0.1', 'q_0.5', 'q_0.9']
    assert len(table) == 12 and str(table['timestamp'].dt.tz) == 'UTC'
    assert table['timestamp'].iloc[0] == pd.Timestamp('2026-01-21T00:00Z')
    assert table['timestamp'].iloc[6] == pd.Timestamp('2026-01-22T00:00Z')
    assert table.loc[0, 'median'] == 1.0 and table.loc[0, 'q_0.1'] == pytest.approx(.6)
    with pytest.raises(ValueError, match='several series'):
        result.predictions
    with pytest.raises(ValueError, match='forecast'):
        model.predict(3, future(panel(), 3))
    assert service.clients[-1].http.is_closed and len(service.posts) == 1


def test_integer_ids_render_as_strings_and_cross_learning_is_sent(service):
    frame = panel(ids=(7, 3)).drop(columns=['visits'])
    model = Forecast().fit(frame, time='date', id='store', target='sales', past_covariates=['temperature'])
    result = model.forecast(2, cross_learning=True, quantile_levels=[.05, .25, .5, .99])
    assert service.posts[0][0]['cross_learning'] is True and 'future' not in service.posts[0][0]
    assert [s.id for s in result.series] == ['7', '3'] and result.cross_learning is True
    assert result.quantile_levels == (.05, .25, .5, .99)
    assert list(result.to_frame())[-4:] == ['q_0.05', 'q_0.25', 'q_0.5', 'q_0.99']


@pytest.mark.parametrize('alias,wire', [
    ('D', {'unit': 'day', 'every': 1}), ('d', {'unit': 'day', 'every': 1}),
    ({'unit': 'day', 'every': 1}, {'unit': 'day', 'every': 1}),
    ({'unit': 'hour', 'every': 24}, {'unit': 'day', 'every': 1}),
])
def test_daily_frequency_aliases_are_sent_explicitly(service, alias, wire):
    Forecast().fit(panel(), time='date', id='store', target='sales', frequency=alias).forecast(2)
    assert service.posts[0][0]['frequency'] == wire


@pytest.mark.parametrize('alias,canonical', [
    ('s', ('s', 1)), ('S', ('s', 1)), ('30s', ('s', 30)), ('min', ('s', 60)), ('15min', ('s', 900)),
    ('T', ('s', 60)), ('h', ('s', 3600)), ('H', ('s', 3600)), ('D', ('s', 86400)), ('W', ('s', 604800)),
    ('2W', ('s', 1209600)), ('M', ('m', 1)), ('MS', ('m', 1)), ('ME', ('m', 1)), ('Q', ('m', 3)),
    ('QS', ('m', 3)), ('Y', ('m', 12)), ('YS', ('m', 12)), ('A', ('m', 12)), ('month', ('m', 1)),
    ({'unit': 'week', 'every': 2}, ('s', 1209600)), ({'unit': 'quarter', 'every': 2}, ('m', 6)),
])
def test_frequency_mapping_table(alias, canonical):
    assert forecast_module._frequency(alias) == canonical


@pytest.mark.parametrize('canonical,wire', [
    (('s', 604800), {'unit': 'week', 'every': 1}), (('s', 900), {'unit': 'minute', 'every': 15}),
    (('s', 90), {'unit': 'second', 'every': 90}), (('m', 12), {'unit': 'year', 'every': 1}),
    (('m', 6), {'unit': 'quarter', 'every': 2}), (('m', 5), {'unit': 'month', 'every': 5}),
])
def test_wire_frequency_uses_largest_whole_unit(canonical, wire):
    assert forecast_module._wire_frequency(canonical) == wire


@pytest.mark.parametrize('alias', ['m', 'ms', 'X', '0D', 'W-SUN', '', 3600, {'unit': 'fortnight', 'every': 1},
                                   {'unit': 'day', 'every': 0}, {'unit': 'day'}, {'unit': 'day', 'every': True}])
def test_invalid_frequency_is_rejected_before_network(service, alias):
    with pytest.raises(ValueError):
        Forecast().fit(panel(), time='date', id='store', target='sales', frequency=alias)
    assert not service.requests


@pytest.mark.parametrize('dates,alias,wire,expected', [
    (pd.date_range('2024-01-31', periods=20, freq='ME', tz='UTC'), 'M', {'unit': 'month', 'every': 1},
     ['2025-09-30', '2025-10-31', '2025-11-30']),
    (pd.date_range('2024-01-01', periods=20, freq='MS', tz='UTC'), 'MS', {'unit': 'month', 'every': 1},
     ['2025-09-01', '2025-10-01', '2025-11-01']),
    (pd.date_range('2020-02-01', periods=20, freq='3MS', tz='UTC') + pd.Timedelta(days=14), 'Q',
     {'unit': 'quarter', 'every': 1}, ['2025-02-15', '2025-05-15', '2025-08-15']),
    (pd.date_range('2000-12-31', periods=20, freq='YE', tz='UTC'), 'Y', {'unit': 'year', 'every': 1},
     ['2020-12-31', '2021-12-31', '2022-12-31']),
])
def test_calendar_frequencies_validate_future_and_result_timestamps(service, dates, alias, wire, expected):
    frame = pd.DataFrame({'date': dates, 'sales': np.arange(20.)})
    inferred = Forecast().fit(frame, time='date', target=['sales'])
    assert inferred.frequency_ == wire and inferred.frequency_seconds_ is None
    service.inferred = wire
    result = inferred.forecast(3)
    assert 'frequency' not in service.posts[-1][0]
    assert [t.strftime('%Y-%m-%d') for t in result.timestamps] == expected
    explicit = Forecast().fit(frame, time='date', target='sales', frequency=alias)
    np.testing.assert_array_equal(explicit.predict(3), [1., 2., 3.])
    assert service.posts[-1][0]['frequency'] == wire


@pytest.mark.parametrize('dates', [
    pd.date_range('2024-01-30', periods=20, freq='MS', tz='UTC') + pd.Timedelta(days=29),  # day 30
    pd.date_range('2024-01-01', periods=20, freq='MS', tz='UTC').delete(5),                # gap
])
def test_irregular_calendar_history_is_rejected_before_network(service, dates):
    frame = pd.DataFrame({'date': dates, 'sales': np.arange(len(dates), dtype=float)})
    with pytest.raises(ValueError):
        Forecast().fit(frame, time='date', target=['sales'])
    with pytest.raises(ValueError):
        Forecast().fit(frame, time='date', target=['sales'], frequency='M')
    assert not service.requests


def test_envelopes_and_min_context_come_from_metadata(service):
    service.envelopes = [dict(max_context=40, max_variates=5, max_series=2, max_horizon=8),
                         dict(max_context=25, max_variates=2, max_series=3, max_horizon=30)]
    model = fitted_panel()                                   # 21 context, 5 variates, 2 series
    with pytest.raises(InvalidInputError, match='horizon 9') as error:
        model.forecast(9, future=future(panel(), 9))
    assert error.value.code == 'workload_limit' and 'context ≤ 40' in str(error.value)
    assert not service.posts
    assert len(model.forecast(8, future=future(panel(), 8)).series) == 4
    narrow = Forecast().fit(panel(), time='date', id='store', target='sales')   # 1 variate
    assert narrow.forecast(30).horizon == 30                 # admitted by the second corner only
    with pytest.raises(InvalidInputError):
        Forecast().fit(panel(n=41), time='date', id='store', target=['sales', 'visits'])
    with pytest.raises(InvalidInputError):
        Forecast().fit(panel(ids=('a', 'b', 'c', 'd')), time='date', id='store', target='sales')
    service.min_context = 21
    with pytest.raises(InvalidInputError, match='at least 21'):
        Forecast().fit(panel(), time='date', id='store', target='sales')
    assert len(service.posts) == 2


def test_forecasting_server_unlocks_longer_univariate_context_and_finer_levels(service):
    history = pd.DataFrame({'t': pd.date_range('2026-01-01', periods=1500, freq='h', tz='UTC'),
                            'y': np.arange(1500.)})
    model = Forecast().fit(history, time='t', target='y')
    assert model.task_ == 'forecasting' and model.context_length_ == 1500
    points = model.predict(100)
    assert points.shape == (100,) and model.predict_quantiles(4, quantile_levels=[.01, .5, .99]).shape == (4, 3)
    body = service.posts[0][0]
    assert body['frequency'] == {'unit': 'hour', 'every': 1} and body['targets'] == ['y']
    assert 'id_column' not in body and body['history']['table']['columns'] == ['t', 'y']
    details = model.predict_details(2)
    # F4: a univariate fit keeps the 0.13 result shape whichever route served it.
    assert details.task == 'univariate_forecasting' and details.series[0].id is None
    frame = details.to_frame()
    assert list(frame) == ['median', 'q_0.1', 'q_0.5', 'q_0.9'] and frame.index.name == 'timestamp'
    assert frame.index[0] == pd.Timestamp('2026-03-04T12:00Z')
    assert set(details.to_dict()) == {'task', 'model_id', 'model_version', 'configuration', 'context_length',
                                      'horizon', 'frequency_seconds', 'timestamps', 'predictions',
                                      'point_semantics', 'quantile_levels', 'quantiles', 'calibrated_intervals'}
    assert details.to_dict()['context_length'] == 1500 and details.to_dict()['frequency_seconds'] == 3600
    assert details.configuration == 'chronos-2-forecast-v1' and details.rearranged_steps == 0
    assert model.forecast(2).task == 'univariate_forecasting'
    array = Forecast().fit(np.arange(20.), start='2026-01-01T00:00:00Z', frequency_seconds=86400)
    array.predict(1)
    assert service.posts[-1][0]['history']['table']['columns'] == ['timestamp', 'value']
    assert service.posts[-1][0]['frequency'] == {'unit': 'day', 'every': 1}
    assert not any(r.url.path == '/v1/model-runs' for r in service.requests)


def test_requests_above_the_body_limit_are_refused_before_any_request(service, monkeypatch):
    model = fitted_panel()
    before = len(service.requests)
    monkeypatch.setattr(forecast_module, 'MAX_REQUEST_BYTES', 2000)
    with pytest.raises(InvalidInputError, match='body limit of 2,000 bytes') as error:
        model.forecast(2, future=future(panel(), 2))
    assert error.value.code == 'workload_limit' and isinstance(error.value, ValueError)
    assert len(service.requests) == before and not service.posts and not service.datasets
    assert forecast_module.MAX_REQUEST_BYTES == 2000 and not hasattr(forecast_module, 'INLINE_TABLE_BYTES')
    monkeypatch.setattr(forecast_module, 'MAX_REQUEST_BYTES', 10 * 1024 * 1024)
    with model.submit_forecast(2, future=future(panel(), 2)) as job:
        assert 'table' in service.posts[0][0]['history'] and 'table' in service.posts[0][0]['future']
        assert not hasattr(job, 'datasets') and len(job.result().series) == 4
    assert not service.datasets and not any(r.url.path == '/v1/datasets' for r in service.requests)


def _stored(service):
    """Store the canonical history/future tables of one inline request as workspace datasets."""
    fitted_panel().forecast(2, future=future(panel(), 2))
    body = service.posts.pop()[0]
    service.datasets.update(ds_hist=body['history']['table'], ds_fut=body['future']['table'])
    return body


def test_dataset_ids_are_sent_by_reference_and_validated_by_the_server(service):
    inline = _stored(service)
    before = len(service.requests)
    model = Forecast().fit('ds_hist', time='date', id='store', target=['sales', 'visits'],
                           past_covariates=['temperature'], future_covariates=['promo', 'holiday'],
                           categorical=['holiday'])
    assert model.n_series_ is None and model.context_length_ is None and model.cutoff_ is None
    assert model.frequency_ is None and all(r.method == 'GET' for r in service.requests[before:])
    result = model.forecast(2, future='ds_fut', idempotency_key='by-reference')
    body = service.posts[-1][0]
    assert body['history'] == {'dataset_id': 'ds_hist'} and body['future'] == {'dataset_id': 'ds_fut'}
    assert {k: v for k, v in body.items() if k not in ('history', 'future')} == {
        k: v for k, v in inline.items() if k not in ('history', 'future')}
    assert [(s.id, s.target) for s in result.series] == [('b', 'sales'), ('b', 'visits'),
                                                         ('a', 'sales'), ('a', 'visits')]
    again = model.forecast(2, future='ds_fut', idempotency_key='by-reference')   # same body: same job
    assert again.job_id == result.job_id and len(service.jobs) == 2
    mixed = model.forecast(2, future=future(panel(), 2))          # a DataFrame future travels inline
    assert 'table' in service.posts[-1][0]['future'] and len(mixed.series) == 4
    with pytest.raises(ValueError, match='forecast'):
        Forecast().fit({'id': 'ds_hist'}, time='date', target='sales').predict(2)
    assert service.posts[-1][0]['history'] == {'dataset_id': 'ds_hist'}
    daily = Forecast().fit('ds_hist', time='date', id='store', target='sales', frequency='D')
    daily.forecast(1)
    assert service.posts[-1][0]['frequency'] == {'unit': 'day', 'every': 1}


@pytest.mark.parametrize('history,kwargs', [
    ('history.csv', dict(time='date', target='sales')),
    ('not a dataset id', dict(time='date', target='sales')),
    ('ds_hist', dict(target='sales')),
    ('ds_hist', dict(time='date', target='sales', start='2026-01-01T00:00Z')),
    ('ds_hist', dict(time='date', target='sales', categorical=['sales'])),
])
def test_invalid_dataset_references_are_refused_before_network(service, history, kwargs):
    with pytest.raises(ValueError):
        Forecast().fit(history, **kwargs)
    assert not service.requests


def test_dataset_history_needs_the_forecasting_task(service):
    service.tasks, service.operations = ['univariate_forecasting'], ['model_run']
    with pytest.raises(InvalidInputError) as error:
        Forecast().fit('ds_hist', time='date', target='sales')
    assert error.value.code == 'unsupported_capability'


def test_version_guard_conflict_and_idempotent_replay(service):
    model = fitted_panel()
    first = model.submit_forecast(2, future=future(panel(), 2), idempotency_key='same-key')
    again = model.submit_forecast(2, future=future(panel(), 2), idempotency_key='same-key')
    assert first.id == again.id and len(service.jobs) == 1
    first.close(); again.close()
    with pytest.raises(ConflictError):
        model.submit_forecast(3, future=future(panel(), 3), idempotency_key='same-key')
    service.version = 'chronos-2-v2'
    with pytest.raises(ConflictError):
        model.forecast(2, future=future(panel(), 2))
    assert len(service.jobs) == 1
    with pytest.raises(ValueError):
        model.forecast(2, future=future(panel(), 2), idempotency_key='bad key\n')


def _swap_series(r):
    r['series'][0], r['series'][2] = r['series'][2], r['series'][0]


def _shift_times(r):
    r['series'][1]['timestamps'] = [t.replace('2026-01-21', '2026-01-20') for t in r['series'][1]['timestamps']]


@pytest.mark.parametrize('change', [
    lambda r: r['series'][0]['quantiles'][0].reverse(),                 # crossed quantiles
    lambda r: r['series'][0]['quantiles'][1].__setitem__(1, 99.),       # median differs from q_0.5
    lambda r: r['series'][0]['predictions'].__setitem__(0, 'inf'),
    lambda r: r['series'][0]['predictions'].__setitem__(0, None),
    lambda r: r['series'][0]['predictions'].__setitem__(0, True),
    _swap_series,
    lambda r: r['series'].pop(),
    lambda r: r['series'][0].update(id='z'),
    lambda r: r['series'][0].update(target='visits'),
    lambda r: r['series'][0].update(context_length=19),
    _shift_times,
    lambda r: r['series'][0].update(timestamps=[t.replace('Z', '+01:00') for t in r['series'][0]['timestamps']]),
    lambda r: r.update(horizon=3),
    lambda r: r.update(quantile_levels=[.1, .5, .95]),
    lambda r: r.update(cross_learning=True),
    lambda r: r.update(frequency={'unit': 'hour', 'every': 24 * 7}),
    lambda r: r.update(frequency='D'),
    lambda r: r.update(rearranged_steps=-1),
    lambda r: r.update(rearranged_steps=9),
    lambda r: r.update(rearranged_steps=True),
    lambda r: r.update(point_semantics='mean'),
    lambda r: r.update(calibrated_intervals=True),
    lambda r: r.update(model_version='other'),
    lambda r: r.update(configuration='chronos-2-research-v1'),
    lambda r: r.update(task='univariate_forecasting'),
    lambda r: r['columns'].update(past_covariates=[]),
    lambda r: r['columns'].update(targets=['visits', 'sales']),
    lambda r: r.pop('rearranged_steps'),
])
def test_wrong_forecasting_results_fail_without_repair(service, change):
    model = fitted_panel()
    service.result_change = change
    with pytest.raises(InvalidForecastResultError):
        model.forecast(2, future=future(panel(), 2))
    assert service.clients[-1].http.is_closed


@pytest.mark.parametrize('value', [float('inf'), float('nan'), True])
def test_nonfinite_forecasting_values_are_rejected_on_construction(service, value):
    result = fitted_panel().forecast(2, future=future(panel(), 2))
    payload = result.to_dict()
    payload['series'][1]['quantiles'][0][2] = value
    with pytest.raises(InvalidForecastResultError, match='no repair'):
        ForecastResult(payload, job_id=result.job_id)
    assert ForecastResult(result.to_dict(), job_id=result.job_id).to_dict() == result.to_dict()


def test_rearranged_steps_are_reported_when_quantiles_are_sorted(service):
    service.rearranged = 3
    result = fitted_panel().forecast(2, future=future(panel(), 2))
    assert result.rearranged_steps == 3 and result.to_dict()['rearranged_steps'] == 3


def test_result_is_immutable_and_conversions_are_owned(service):
    result = fitted_panel().forecast(2, future=future(panel(), 2))
    series = result.series[0]
    series.predictions[0] = -1
    result.to_dict()['series'][0]['predictions'][0] = -1
    table = result.to_frame()
    table.loc[0, 'median'] = -1
    assert result.series[0].predictions[0] == 1.0
    with pytest.raises(AttributeError):
        result.job_id = 'other'
    with pytest.raises(AttributeError):
        series.id = 'other'


def _set(frame, column, index, value):
    frame = frame.astype({column: object})
    frame.loc[index, column] = value
    return frame


@pytest.mark.parametrize('change,kwargs', [
    (lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True), {}),                 # duplicate (id, time)
    (lambda f: f.drop(index=5), {}),                                                # gap
    (lambda f: _set(f, 'temperature', 0, True), {}),                                # boolean covariate
    (lambda f: _set(f, 'sales', 0, 'text'), {}),
    (lambda f: _set(f, 'sales', 0, float('inf')), {}),
    (lambda f: _set(f, 'promo', 0, None), {}),                                      # incomplete future covariate
    (lambda f: _set(f, 'holiday', 0, None), {}),                                    # missing category
    (lambda f: _set(f, 'holiday', 0, 1.5), {}),
    (lambda f: f.assign(sales=np.where(f['store'] == 'a', np.nan, f['sales'])), {}),  # target entirely missing
    (lambda f: _set(f, 'store', 0, 5).assign(store=lambda g: g['store'].where(g.index != 1, '5')), {}),
    (lambda f: f, {'categorical': ['sales']}),
    (lambda f: f, {'past_covariates': ['promo']}),                                  # overlapping roles
    (lambda f: f.drop(columns=['temperature']), {}),
    (lambda f: f.assign(date=np.where(f['store'] == 'a', f['date'] + pd.Timedelta(hours=1),
                                      f['date'])), {'frequency': 'h'}),
    (lambda f: pd.concat([f[f['store'] == 'b'], f[f['store'] == 'a'].assign(
        date=pd.date_range('2026-01-01', periods=21, freq='2D', tz='UTC'))], ignore_index=True), {}),
    (lambda f: f.iloc[:1], {}),
    (lambda f: f, {'start': '2026-01-01T00:00Z'}),
    (lambda f: f, {'frequency': 'D', 'frequency_seconds': 86400}),
])
def test_invalid_long_history_never_reaches_transport(service, change, kwargs):
    arguments = dict(time='date', id='store', target=['sales', 'visits'], past_covariates=['temperature'],
                     future_covariates=['promo', 'holiday'], categorical=['holiday'])
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        Forecast().fit(change(panel()), **arguments)
    assert not service.requests


@pytest.mark.parametrize('change', [
    lambda f: f.iloc[1:],                                                   # missing step
    lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True),               # duplicate step
    lambda f: f.assign(date=f['date'] + pd.Timedelta(days=1)),              # wrong timestamps
    lambda f: f.assign(store=f['store'].replace({'a': 'z'})),               # unknown series
    lambda f: f.assign(sales=1.0),                                          # target in future
    lambda f: f.drop(columns=['holiday']),
    lambda f: f.assign(promo=np.where(f.index == 0, np.nan, f['promo'])),   # incomplete
    lambda f: None,
])
def test_invalid_future_never_submits(service, change):
    model = fitted_panel()
    before = len(service.requests)
    with pytest.raises(ValueError):
        model.forecast(2, future=change(future(panel(), 2)))
    assert len(service.requests) == before and not service.posts


def test_future_is_refused_without_future_covariates(service):
    model = Forecast().fit(panel(), time='date', id='store', target='sales')
    with pytest.raises(ValueError, match='future_covariates'):
        model.forecast(2, future=future(panel(), 2))
    with pytest.raises(ValueError, match='future_covariates'):
        Forecast().fit(panel().query("store == 'a'"), time='date', target='sales').predict(2, future(panel(), 2))
    assert not service.posts


def test_single_series_with_future_covariate_uses_predict(service):
    frame = panel().query("store == 'a'").drop(columns=['store'])
    model = Forecast().fit(frame, time='date', target='sales', future_covariates=['promo'])
    known = future(panel(), 3, ids=['a']).drop(columns=['store', 'holiday'])
    assert model.predict(3, known).shape == (3,)
    assert service.posts[0][0]['future']['table']['columns'] == ['date', 'promo']


def test_legacy_server_keeps_model_runs_and_refuses_new_capabilities(service):
    service.tasks, service.operations = ['univariate_forecasting'], ['model_run']
    with pytest.raises(WeltError) as error:
        fitted_panel()
    assert error.value.code == 'unsupported_capability'
    monthly = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=20, freq='MS', tz='UTC'),
                            'sales': np.arange(20.)})
    with pytest.raises(WeltError, match='calendar'):
        Forecast().fit(monthly, time='date', target='sales')
    series = panel().query("store == 'a'")
    model = Forecast().fit(series, time='date', target='sales')
    assert model.task_ == 'univariate_forecasting' and model.configuration_version_ == 'chronos-2-forecast-v1'
    with pytest.raises(WeltError) as error:
        model.forecast(2)
    assert error.value.code == 'unsupported_capability'
    with pytest.raises(ValueError):
        model.predict_quantiles(2, quantile_levels=[.25])      # legacy levels: 0.1 … 0.9 only
    assert not service.posts


def test_public_catalogue_entry_is_completed_by_the_model_detail(service):
    entry = {k: v for k, v in service.metadata().items()
             if k not in ('supports_version_expectations', 'configuration')}
    service.catalogue = [entry]
    assert fitted_panel().task_ == 'forecasting'
    assert any(r.url.path == '/v1/models/chronos-2' for r in service.requests)


def test_task_profile_metadata_and_invalid_envelopes(service, monkeypatch):
    meta = service.metadata()
    profile = dict(task='forecasting', operations=['forecast'], configuration={'version': 'profile-config'},
                   limits={'forecast_envelopes': service.envelopes, 'min_context': 16})
    public = {k: v for k, v in meta.items() if k not in ('forecast_envelopes', 'min_context')}
    public.update(operations=['model_run'], task_profiles=[profile])
    monkeypatch.setattr(service, 'metadata', lambda: public)
    assert fitted_panel().configuration_version_ == 'profile-config'
    profile['limits'] = {}
    with pytest.raises(WeltError) as error:
        fitted_panel()
    assert error.value.code == 'invalid_response'
    profile['limits'] = {'forecast_envelopes': [dict(max_context=0, max_variates=1, max_series=1,
                                                     max_horizon=1)], 'min_context': 16}
    with pytest.raises(WeltError):
        fitted_panel()
    assert not service.posts


def test_typed_reconnect_timeout_and_cancel(service):
    service.status = 'queued'
    model = fitted_panel()
    with model.submit_forecast(2, future=future(panel(), 2)) as job:
        assert job.inspect()['operation'] == 'forecast'
        with pytest.raises(Exception) as error:
            job.result(timeout=.001)
        assert error.value.job_id == job.id
        service.jobs[job.id][0]['status'] = 'succeeded'
        with Client(api_key='fixture', transport=httpx.MockTransport(service.handle)) as client:
            reopened = client.forecast_job(job.id)
            assert reopened.result().task == 'forecasting'
            raw = client.job(job.id).result()
            assert raw['task'] == 'forecasting'
            assert reopened.cancel()['status'] == 'cancelled'
    assert len(service.posts) == 1


def test_async_submit_forecast_and_typed_reconnect():
    svc = Service()
    request = dict(model_id='chronos-2', configuration='default', time_column='t', targets=['y'],
                   past_covariates=[], future_covariates=[], categorical_covariates=[], horizon=2,
                   quantile_levels=[.1, .5, .9], cross_learning=False, frequency={'unit': 'hour', 'every': 1},
                   history={'table': {'columns': ['t', 'y'], 'rows': [
                       [f'2026-01-01T{h:02d}:00:00Z', float(h)] for h in range(16)]}})

    async def run():
        async with AsyncClient(api_key='fixture', transport=httpx.MockTransport(svc.handle)) as client:
            raw = await client.submit_forecast(request, idempotency_key='async-forecast',
                                               expected_model_version='chronos-2-v1',
                                               expected_configuration_version='chronos-2-forecast-v1')
            job = client.forecast_job(raw.id)
            assert isinstance(job, AsyncForecastJob)
            assert (await job.inspect())['task'] == 'forecasting'
            result = await job.result()
            assert result.task == 'forecasting' and result.predictions.shape == (2,)
            assert result.timestamps[0] == pd.Timestamp('2026-01-01T16:00Z')
            assert (await raw.result())['series'][0]['id'] is None
            assert (await job.cancel())['status'] == 'cancelled'
            with pytest.raises(ValueError):
                await client.submit_forecast(request, expected_model_version='only-one')
    asyncio.run(run())
    body, headers = svc.posts[0]
    assert body == request and headers['idempotency-key'] == 'async-forecast'
    assert headers['x-welt-model-version'] == 'chronos-2-v1'
    assert headers['x-welt-configuration-version'] == 'chronos-2-forecast-v1'
    assert len(svc.posts) == 1


def test_sync_submit_forecast_without_guards_and_cli(tmp_path, capsys):
    svc = Service()
    svc.version, svc.configuration = None, None
    request = dict(model_id='chronos-2', time_column='t', targets=['y'], past_covariates=[], future_covariates=[],
                   categorical_covariates=[], horizon=1, quantile_levels=[.5], cross_learning=False,
                   history={'table': {'columns': ['t', 'y'], 'rows': [['2026-01-01T00:00:00Z', 1.0]]}})
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(request))
    with Client(api_key='fixture', transport=httpx.MockTransport(svc.handle)) as client:
        result = _run(_parser().parse_args(['forecast', '--input', str(path), '--idempotency-key', 'cli-key']), client)
        assert result == {'job_id': 'fc-1', 'idempotency_key': 'cli-key'}
        assert 'cli-key' in capsys.readouterr().err
    body, headers = svc.posts[0]
    assert body == request and 'x-welt-model-version' not in headers


def test_injected_client_is_borrowed_shared_by_clones_and_never_closed():
    from sklearn.base import clone
    svc = Service()
    with Client(base_url='https://fixture.example', api_key='injected-key',
                transport=httpx.MockTransport(svc.handle)) as client:
        model = Forecast(client=client).fit(panel(), time='date', id='store', target=['sales', 'visits'],
                                            past_covariates=['temperature'], future_covariates=['promo', 'holiday'],
                                            categorical=['holiday'])
        result = model.forecast(2, future=future(panel(), 2), idempotency_key='injected-one')
        assert len(result.series) == 4 and not client.http.is_closed
        with model.submit_forecast(2, future=future(panel(), 2)) as job:
            assert job.client is client
        assert not client.http.is_closed
        twin = clone(model)
        assert twin.client is client and not hasattr(twin, 'task_')
        single = Forecast(client=client).fit(panel().query("store == 'a'"), time='date', target='sales')
        assert single.predict(2).shape == (2,) and not client.http.is_closed
        assert 'injected-key' not in repr(model)
        assert svc.posts[0][1]['authorization'] == 'Bearer injected-key'
    assert client.http.is_closed


@pytest.mark.parametrize('kwargs', [dict(api_key='other'), dict(base_url='https://other.example'),
                                    dict(workspace='ws_other')])
def test_injected_client_excludes_connection_parameters(kwargs):
    requests = []
    with Client(api_key='fixture', transport=httpx.MockTransport(lambda r: requests.append(r))) as client:
        with pytest.raises(ValueError, match='not both'):
            Forecast(client=client, **kwargs).fit(panel(), time='date', id='store', target='sales')
    assert not requests


def test_injected_client_must_be_an_open_sync_client():
    async_client = AsyncClient(api_key='fixture', transport=httpx.MockTransport(lambda r: None))
    with pytest.raises(ValueError, match='synchronous'):
        Forecast(client=async_client).fit(panel(), time='date', id='store', target='sales')
    closed = Client(api_key='fixture', transport=httpx.MockTransport(lambda r: None))
    closed.close()
    with pytest.raises(ValueError, match='closed'):
        Forecast(client=closed).fit(panel(), time='date', id='store', target='sales')


def test_replacing_the_injected_client_is_checked_against_the_fitted_connection(service):
    svc = Service()
    def client(key):
        return Client(api_key=key, base_url='https://fixture.example', transport=httpx.MockTransport(svc.handle))
    with client('fixture') as first, client('fixture') as second, client('another-key') as third:
        model = Forecast(client=first).fit(panel(), time='date', id='store', target='sales')
        model.client = second                     # same origin, credential and workspace
        assert len(model.forecast(2).series) == 2
        model.client = third
        with pytest.raises(AuthenticationError):
            model.forecast(2)
        model = Forecast(client=first).fit(panel(), time='date', id='store', target='sales')
        model.client = None
        with pytest.raises(ValueError, match='parameters changed'):
            model.forecast(2)
    assert len(svc.posts) == 1


def test_missing_past_categorical_cells_are_sent_as_null(service):
    frame = panel().astype({'holiday': object})
    frame.loc[[0, 5], 'holiday'] = [None, np.nan]
    model = Forecast().fit(frame, time='date', id='store', target='sales', past_covariates=['holiday'],
                           categorical=['holiday'])
    model.forecast(1)
    rows = service.posts[0][0]['history']['table']['rows']
    assert rows[0][3] is None and rows[5][3] is None and rows[1][3] == 'no'
    with pytest.raises(ValueError, match='complete'):            # future categories stay complete
        Forecast().fit(frame, time='date', id='store', target='sales', future_covariates=['holiday'],
                       categorical=['holiday'])


def test_new_capability_and_limit_refusals_are_value_errors(service):
    service.tasks, service.operations = ['univariate_forecasting'], ['model_run']
    monthly = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=20, freq='MS', tz='UTC'),
                            'sales': np.arange(20.)})
    for history, kwargs in ((monthly, dict(time='date', target='sales')),
                            (panel(), dict(time='date', id='store', target='sales'))):
        with pytest.raises(ValueError) as error:
            Forecast().fit(history, **kwargs)
        assert isinstance(error.value, InvalidInputError) and error.value.code == 'unsupported_capability'
    model = Forecast().fit(panel().query("store == 'a'"), time='date', target='sales')
    with pytest.raises(ValueError) as error:
        model.forecast(2)
    assert error.value.code == 'unsupported_capability'
    with pytest.raises(ValueError) as error:                      # 513 > the published 512
        Forecast().fit(np.arange(513.), start='2026-01-01T00:00Z', frequency_seconds=60)
    assert error.value.code == 'workload_limit'
    before = len(service.requests)
    with pytest.raises(ValueError, match='16'):                   # the 0.13 floor stays offline
        Forecast().fit(np.arange(15.), start='2026-01-01T00:00Z', frequency_seconds=60)
    assert len(service.requests) == before and not service.posts


def test_unknown_result_fields_are_ignored_and_known_ones_required(service):
    def extend(r):
        r.update(warnings=['note'], native_grid={'levels': 21})
        r['columns']['extra_role'] = []
        for s in r['series']:
            s['native_context_length'] = s['context_length']
    service.result_change = extend
    result = fitted_panel().forecast(2, future=future(panel(), 2))
    assert 'warnings' not in result.to_dict() and 'extra_role' not in result.columns
    assert 'native_context_length' not in result.to_dict()['series'][0] and len(result.series) == 4
    for missing in ('rearranged_steps', 'columns'):
        payload = result.to_dict()
        payload.pop(missing)
        with pytest.raises(InvalidForecastResultError):
            ForecastResult(payload, job_id=result.job_id)
    payload = result.to_dict()
    payload['series'][0].pop('context_length')
    with pytest.raises(InvalidForecastResultError):
        ForecastResult(payload, job_id=result.job_id)


def test_fitted_forecast_pickles_without_client_or_credential_and_needs_reinjection(service):
    svc = Service()
    with Client(api_key='secret-injected-key', base_url='https://fixture.example',
                transport=httpx.MockTransport(svc.handle)) as client:
        model = Forecast(client=client).fit(panel(), time='date', id='store', target='sales')
        data = pickle.dumps(model)
        assert b'secret-injected-key' not in data and model.client is client
        restored = pickle.loads(data)
        assert restored.client is None and restored.n_series_ == 2
        with pytest.raises(ValueError, match='set estimator.client'):
            restored.forecast(2)
        restored.client = client
        assert len(restored.forecast(2).series) == 2
    plain = pickle.loads(pickle.dumps(fitted_panel()))       # owned clients: credential re-read at call time
    assert len(plain.forecast(2, future=future(panel(), 2)).series) == 4


def test_delete_forecast_sync_async_and_cli(capsys):
    svc = Service()
    request = dict(model_id='chronos-2', time_column='t', targets=['y'], past_covariates=[], future_covariates=[],
                   categorical_covariates=[], horizon=1, quantile_levels=[.5], cross_learning=False,
                   history={'table': {'columns': ['t', 'y'], 'rows': [['2026-01-01T00:00:00Z', 1.0]]}})
    svc.version = svc.configuration = None
    with Client(api_key='fixture', transport=httpx.MockTransport(svc.handle)) as client:
        job = client.submit_forecast(request, idempotency_key='to-delete')
        assert client.delete_forecast(job.id) == {'id': job.id, 'deleted': True}
        assert _run(_parser().parse_args(['forecasts', 'delete', job.id, '--yes']), client)['deleted']
        with pytest.raises(SystemExit):
            _parser().parse_args(['forecasts', 'delete', job.id])     # --yes is required
    capsys.readouterr()

    async def run():
        async with AsyncClient(api_key='fixture', transport=httpx.MockTransport(svc.handle)) as client:
            return await client.delete_forecast(job.id)
    assert asyncio.run(run())['deleted'] and svc.jobs[job.id][0]['purged']
    deletes = [r for r in svc.requests if r.method == 'DELETE']
    assert len(deletes) == 3 and all(r.url.path == f'/v1/forecasts/{job.id}' for r in deletes)
