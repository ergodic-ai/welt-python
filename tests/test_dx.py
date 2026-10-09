"""API-067: first-result ergonomics without live accounts or network calls."""
import asyncio
import json
import time

import httpx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from test_client import ServiceFixture
from welt import (AsyncClient, AuthenticationError, Classifier, Client,
                  JobTimeoutError, PermissionDeniedError, Regressor, WeltError)
from welt.estimators import _RemoteEstimator


@pytest.mark.parametrize('client_type', [Client, AsyncClient])
def test_origin_precedence_and_explicit_localhost(monkeypatch, client_type):
    monkeypatch.delenv('WELT_BASE_URL', raising=False)
    clients = [client_type()]
    monkeypatch.setenv('WELT_BASE_URL', 'https://configured.invalid')
    clients.extend([client_type(), client_type(base_url='http://localhost:8080')])
    assert [c.base_url for c in clients] == [
        'https://welt.ergodic.dev', 'https://configured.invalid', 'http://localhost:8080']
    for c in clients:
        if client_type is Client: c.close()
        else: asyncio.run(c.aclose())


@pytest.mark.parametrize('asynchronous', [False, True])
def test_keyless_metadata_and_private_preflight(monkeypatch, asynchronous):
    monkeypatch.delenv('WELT_API_KEY', raising=False)
    calls = []
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={'items': [], 'id': 'p10k-example'})
    async def run():
        async with AsyncClient(transport=httpx.MockTransport(handler)) as c:
            await c.models(); await c.research_datasets(); await c.research_dataset('p10k-example')
            for invoke in [lambda: c.datasets(), lambda: c.upload(columns=['x'], rows=[[1]]),
                           lambda: c.submit_fit('d', model='tabicl-v2', task='classification'),
                           lambda: c.request('POST', '/v1/models')]:
                with pytest.raises(AuthenticationError, match='WELT_API_KEY') as error:
                    await invoke()
                assert error.value.code == 'missing_api_key'
    if asynchronous: asyncio.run(run())
    else:
        with Client(transport=httpx.MockTransport(handler)) as c:
            c.models(); c.research_datasets(); c.research_dataset('p10k-example')
            for invoke in [c.datasets, lambda: c.upload(columns=['x'], rows=[[1]]),
                           lambda: c.submit_fit('d', model='tabicl-v2', task='classification'),
                           lambda: c.request('POST', '/v1/models')]:
                with pytest.raises(AuthenticationError, match='WELT_API_KEY'): invoke()
    assert calls == ['/v1/models', '/v1/research-datasets', '/v1/research-datasets/p10k-example']


@pytest.mark.parametrize('status,code,error_type,expected', [
    (401, 'invalid_api_key', AuthenticationError, 'invalid or revoked'),
    (401, 'invalid_input', AuthenticationError, 'invalid or revoked'),
    (401, 'api_key_expired', AuthenticationError, 'expired'),
    (403, 'permission_denied', PermissionDeniedError, 'required read/write'),
])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_auth_errors_actionable_and_sanitized(status, code, error_type, expected, asynchronous):
    key = 'invented-secret-for-redaction'
    def handler(request):
        return httpx.Response(status, json={'error': {'code': code, 'message': key,
            'request_id': 'request-safe', 'job_id': key, 'retryable': False}})
    async def run():
        async with AsyncClient(api_key=key, transport=httpx.MockTransport(handler)) as c:
            await c.datasets()
    with pytest.raises(error_type) as error:
        if asynchronous: asyncio.run(run())
        else:
            with Client(api_key=key, transport=httpx.MockTransport(handler)) as c: c.datasets()
    assert expected in str(error.value) and 'request-safe' in str(error.value)
    assert key not in str(error.value) and error.value.job_id is None


def test_diagnostic_ids_reject_untrusted_control_text_and_bound_length():
    e = WeltError('Safe.', request_id='safe_req', job_id="bad\nsecret")
    assert str(e) == 'Safe. [request safe_req]'
    assert e.job_id is None
    assert 'x' * 129 not in str(WeltError('Safe.', request_id='x' * 129))


@pytest.fixture
def connected(monkeypatch):
    fixture = ServiceFixture()
    monkeypatch.setattr(_RemoteEstimator, '_client',
                        lambda self: Client(api_key='fixture-only', transport=httpx.MockTransport(fixture)))
    return fixture


@pytest.mark.parametrize('estimator,labels', [(Classifier, [0, 0, 1, 1]), (Regressor, [0., 1., 8., 9.])])
def test_target_column_matches_separate_labels_and_reopen(connected, monkeypatch, estimator, labels):
    train = pd.DataFrame({'a': [0, 1, 8, 9], 'label': labels, 'b': [1, 2, 9, 10]})
    before = train.copy(deep=True)
    model = estimator(random_state=17).fit(train, target='label')
    separate = estimator(random_state=17).fit(train[['a', 'b']], labels)
    pd.testing.assert_frame_equal(train, before)
    np.testing.assert_array_equal(model.predict(train[['b', 'a']]), separate.predict(train[['a', 'b']]))
    assert model.feature_names_in_.tolist() == ['a', 'b'] and model.n_features_in_ == 2
    assert model.model_version_ == 'fixture-v1' and model.metadata()['seed'] == 17
    assert not hasattr(clone(model), 'predictor_id_')
    for dataset in connected.datasets.values():
        assert 'label' not in dataset['columns']
        assert dataset['columns'] == ['a', 'b', dataset['target']]
        assert [r[-1] for r in dataset['rows']] == labels
    job = estimator().submit_fit(train, target='label')
    try: assert job.result()['columns'] == ['a', 'b']
    finally: job.client.close()
    monkeypatch.setattr('welt.estimators.Client', lambda **kwargs:
                        Client(api_key='fixture-only', transport=httpx.MockTransport(connected)))
    reopened = estimator.from_predictor(model.predictor_id_)
    np.testing.assert_array_equal(reopened.predict(train[['b','a']]), model.predict(train[['a','b']]))
    assert reopened.model_version_ == model.model_version_
    if estimator is Classifier:
        np.testing.assert_array_equal(reopened.classes_, model.classes_)
        np.testing.assert_array_equal(reopened.predict_proba(train[['b','a']]), model.predict_proba(train[['a','b']]))
        assert model.classes_.tolist() == [0, 1]
        np.testing.assert_array_equal(model.predict_proba(train[['b','a']]), separate.predict_proba(train[['a','b']]))


@pytest.mark.parametrize('method', ['fit', 'submit_fit'])
@pytest.mark.parametrize('X,y,target,match', [
    (pd.DataFrame({'x': [1], 'label': [0]}), [0], 'label', 'exactly one'),
    (pd.DataFrame({'x': [1]}), None, None, 'exactly one'),
    (pd.DataFrame({'x': [1]}), None, 'label', 'missing'),
    (pd.DataFrame([[1, 0]], columns=['label', 'label']), None, 'label', 'unique'),
    (pd.DataFrame({'': [1], 'label': [0]}), None, 'label', 'nonempty'),
    (pd.DataFrame({'x': [1], 'label': [0]}), None, '', 'nonempty'),
    (pd.DataFrame({'x': [1], 'label': [None]}), None, 'label', 'missing'),
    (pd.DataFrame({'label': [0]}), None, 'label', 'features'),
    (np.array([[1, 0]]), None, 'label', 'DataFrame'),
    (pd.DataFrame({'x': [pd.Timestamp('2020-01-01')], 'label': [0]}), None, 'label', 'scalar'),
])
def test_target_validation_precedes_upload(monkeypatch, method, X, y, target, match):
    monkeypatch.setattr(_RemoteEstimator, '_client', lambda self: pytest.fail('invalid input reached network'))
    with pytest.raises(ValueError, match=match): getattr(Classifier(), method)(X, y, target=target)


def test_nullable_features_and_missing_target_handling(connected):
    train = pd.DataFrame({'x': pd.Series([1, None, 8, 9], dtype='Int64'), 'label': [0, 0, 1, 1]})
    job = Classifier().submit_fit(train, target='label')
    try:
        dataset = next(iter(connected.datasets.values()))
        assert dataset['rows'][1][0] is None
    finally: job.client.close()


def test_local_schema_error_has_bounded_names_and_no_request(connected):
    model = Classifier().fit(pd.DataFrame({'a': [0, 1], 'b': [1, 2]}), [0, 1])
    with pytest.raises(ValueError, match="Missing:.*b.*Extra:.*label"):
        model.predict(pd.DataFrame({'a': [2], 'label': [0]}))
    assert len(connected.datasets) == len(connected.jobs) == 1


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('stage', ['inspect', 'retry', 'result'])
def test_deadline_propagates_to_reads_retries_and_results(monkeypatch, asynchronous, stage):
    clock = [100.0]
    monkeypatch.setattr('welt.client.time.monotonic', lambda: clock[0])
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path, request.extensions['timeout']))
        assert all(0 < t <= .101 for t in request.extensions['timeout'].values())
        if stage == 'inspect':
            clock[0] += .2
        elif stage == 'retry':
            return httpx.Response(503, headers={'Retry-After': '1'}, json={'error': {'code': 'busy'}})
        elif request.url.path.endswith('/p'):
            clock[0] += .2
            return httpx.Response(200, json={'id': 'p'})
        return httpx.Response(200, json={'id': 'durable', 'status': 'succeeded', 'operation': 'fit', 'predictor_id': 'p'})
    def sleep(delay): clock[0] += delay
    monkeypatch.setattr('welt.client.time.sleep', sleep)
    async def async_sleep(delay): clock[0] += delay
    monkeypatch.setattr('welt.client.asyncio.sleep', async_sleep)
    async def run():
        async with AsyncClient(api_key='fixture', transport=httpx.MockTransport(handler)) as c:
            await c.job('durable').result(timeout=.1)
    with pytest.raises(JobTimeoutError, match=r"Client\(\).job\('durable'\).result") as error:
        if asynchronous: asyncio.run(run())
        else:
            with Client(api_key='fixture', transport=httpx.MockTransport(handler)) as c:
                c.job('durable').result(timeout=.1)
    assert error.value.job_id == 'durable'
    assert all(method == 'GET' for method, _, _ in calls)
    assert len(calls) == (2 if stage == 'result' else 1)


def test_async_slow_poll_is_locally_bounded_and_reconnects_same_job():
    calls = []
    async def handler(request):
        calls.append(request.url.path)
        if len(calls) == 1: await asyncio.sleep(.3)
        return httpx.Response(200, json={'id': 'durable', 'status': 'succeeded', 'operation': 'fit', 'predictor_id': 'p'}) if request.url.path.endswith('durable') else httpx.Response(200, json={'id': 'p'})
    async def run():
        async with AsyncClient(api_key='fixture', transport=httpx.MockTransport(handler)) as c:
            start = time.monotonic()
            with pytest.raises(JobTimeoutError): await c.job('durable').result(timeout=.02)
            assert time.monotonic() - start < .15
            assert (await c.job('durable').result())['id'] == 'p'
    asyncio.run(run())
    assert calls == ['/v1/jobs/durable', '/v1/jobs/durable', '/v1/predictors/p']


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('job_failure', [False, True])
def test_server_echoed_credentials_are_removed_from_error_attributes(asynchronous, job_failure):
    key = 'invented-secret-for-redaction'
    def handler(request):
        fields = dict(message=f'Failed {key}', code=key, job_id=key, request_id=key, retryable=key)
        if job_failure:
            return httpx.Response(200, json=dict(id=key, status='failed', operation='fit', error=fields))
        return httpx.Response(422, json={'error': fields})
    async def run():
        async with AsyncClient(api_key=key, transport=httpx.MockTransport(handler)) as c:
            if job_failure: await c.job('durable').result()
            else: await c.datasets()
    with pytest.raises(WeltError) as error:
        if asynchronous: asyncio.run(run())
        else:
            with Client(api_key=key, transport=httpx.MockTransport(handler)) as c:
                if job_failure: c.job('durable').result()
                else: c.datasets()
    assert key not in str(error.value)
    assert key not in repr(vars(error.value))
    assert key not in repr(error.value)
    assert error.value.code == 'unknown' and error.value.retryable is False


@pytest.mark.parametrize('status,error_type', [(401, AuthenticationError), (403, PermissionDeniedError)])
def test_unstructured_auth_errors_keep_setup_guidance(status, error_type):
    with Client(api_key='fixture', transport=httpx.MockTransport(lambda r:
                httpx.Response(status, text='untrusted server text'))) as c:
        with pytest.raises(error_type) as error: c.datasets()
    assert error.value.status_code == status
    assert 'untrusted server text' not in str(error.value)
    assert 'key' in str(error.value)


def test_missing_key_fit_sends_no_data(monkeypatch):
    monkeypatch.delenv('WELT_API_KEY', raising=False)
    calls = []
    monkeypatch.setattr(_RemoteEstimator, '_client', lambda self:
                        Client(transport=httpx.MockTransport(lambda r: calls.append(r))))
    with pytest.raises(AuthenticationError, match='WELT_API_KEY'):
        Classifier().fit(pd.DataFrame({'x': [0, 1], 'label': [0, 1]}), target='label')
    assert calls == []


def test_explicit_target_and_internal_label_name_collision(connected):
    train = pd.DataFrame({'__welt_target__': [0, 1], 'label': [0, 1]})
    model = Classifier().fit(train, target='label')
    dataset = next(iter(connected.datasets.values()))
    assert dataset['target'] == '__welt_target___'
    assert model.feature_names_in_.tolist() == ['__welt_target__']
    assert model.predict(train[['__welt_target__']]).tolist() == [0, 1]


def test_nullable_numeric_regression_target(connected):
    train = pd.DataFrame({'x': [0, 1], 'label': pd.Series([1, 3], dtype='Int64')})
    assert Regressor().fit(train, target='label').predict([[2]]).tolist() == [2.]


@pytest.mark.parametrize('asynchronous', [False, True])
def test_server_job_identifier_cannot_echo_credential_in_reconnect(asynchronous):
    key = 'invented-secret-for-redaction'
    def handler(request): return httpx.Response(202, json={'id': key})
    async def run():
        async with AsyncClient(api_key=key, transport=httpx.MockTransport(handler)) as c:
            await c.submit_fit('d', model='tabicl-v2', task='classification')
    with pytest.raises(WeltError) as error:
        if asynchronous: asyncio.run(run())
        else:
            with Client(api_key=key, transport=httpx.MockTransport(handler)) as c:
                c.submit_fit('d', model='tabicl-v2', task='classification')
    assert key not in str(error.value) + repr(vars(error.value))
    assert error.value.code == 'invalid_response'


@pytest.mark.parametrize('asynchronous', [False, True])
def test_request_timeout_context_does_not_echo_configured_key(asynchronous):
    from welt.client import _deadline
    key = 'invented-secret-for-redaction'
    async def run():
        async with AsyncClient(api_key=key, transport=httpx.MockTransport(lambda r: pytest.fail('no read'))) as c:
            await c.request('GET', '/v1/jobs/opaque', _deadline=_deadline(0, .25), _job_id=key)
    with pytest.raises(JobTimeoutError) as error:
        if asynchronous: asyncio.run(run())
        else:
            with Client(api_key=key, transport=httpx.MockTransport(lambda r: pytest.fail('no read'))) as c:
                c.request('GET', '/v1/jobs/opaque', _deadline=_deadline(0, .25), _job_id=key)
    assert key not in str(error.value) + repr(vars(error.value))
    assert error.value.job_id is None
