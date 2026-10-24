"""Backtest client-only controls; native/API integration lives in welt-api API-194."""

import asyncio
from copy import deepcopy
import json

import httpx
import numpy as np
import pandas as pd
import pytest

from welt import AsyncClient, Backtest, Client, JobTimeoutError, TransportError, WeltError
from welt.backtest import request


def payload():
    return dict(id="bt_fixture", status="succeeded", index_kind="position", frequency=None,
        calendar_anchor=None, targets=["value"], series_ids=[None], quantile_levels=[.1, .5, .9],
        models=[dict(model_id="chronos-2", model_version="model-v1", configuration="config-v1",
                     jobs=[dict(job_id="job_fixture", operation_id="op_fixture", fold=0)])],
        folds=1, horizon=1, step=1, context_length=None, seasonal_period=1, final_horizon=None,
        primary_metric="mae", protocol_version="backtest-protocol-v1", metrics_version="backtest-metrics-v1",
        report_version="backtest-report-v1", split_sha256="a" * 64,
        completed_jobs=1, total_jobs=1, summary=dict(leaderboard=[]))


def predictions():
    return dict(backtest_id="bt_fixture", index_kind="position", quantile_levels=[.1, .5, .9], final=[],
        items=[dict(model_id="chronos-2", fold=0, id=None, target="value", lead=1, position=20,
                    index=20, cutoff=19, train_start=0, train_stop=20, actual=2., prediction=2.,
                    quantiles=[1., 2., 3.])])


def test_array_and_dataframe_request_is_owned_and_only_sends_roles():
    frame = pd.DataFrame(dict(y=np.arange(40), private=["not-sent"] * 40))
    body = request(frame, target="y", models=["chronos-2"], horizon=4)
    assert body["history"]["table"]["columns"] == ["y"] and body["time_column"] is None
    frame.loc[0, "y"] = 999
    assert body["history"]["table"]["rows"][0] == [0]
    assert request(np.arange(40), models=["chronos-2"], horizon=4)["targets"] == ["value"]


@pytest.mark.parametrize("change", [dict(index=0), dict(cutoff=0), dict(position=0),
    dict(quantiles=[3., 2., 1.]), dict(prediction=100.), dict(quantiles=[1., 2.]),
    dict(lead=2), dict(fold=1), dict(target="unrelated"), dict(id="foreign"), dict(train_start=-1),
    dict(position=20.5), dict(train_stop=20.5), dict(model_id="baseline:naive")])
def test_invalid_prediction_invariants_refused(change):
    output = predictions()
    output["items"][0].update(change)
    def handler(req):
        return httpx.Response(200, json=output)
    with Client(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        handle = Backtest(client, payload())
        with pytest.raises(WeltError, match="Invalid backtest prediction"):
            handle.predictions()


@pytest.mark.parametrize("change", [dict(total_jobs=0), dict(horizon=True), dict(status="unknown"),
    dict(models=[]), dict(completed_jobs=2), dict(frequency={"unit": "hour", "every": 1})])
def test_intrinsic_invalid_metadata_refused(change):
    response = payload()
    response.update(change)
    with Client(api_key="test-key") as client:
        with pytest.raises(WeltError, match="Invalid backtest response"):
            Backtest(client, response)


@pytest.mark.parametrize("change", [dict(horizon=999), dict(folds=2), dict(step=999),
    dict(primary_metric="rmse"), dict(index_kind="datetime", frequency={"unit": "hour", "every": 1})])
def test_submission_metadata_is_bound_to_request(change):
    response = payload()
    response.update(change)
    def handler(req):
        return httpx.Response(202, json=response)
    with Client(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(WeltError):
            client.submit_backtest(np.arange(20), models=["chronos-2"], horizon=1, folds=1)


def test_sync_and_async_invalid_timeout_never_submit():
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(202, json=payload())
    with Client(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            client.backtest(np.arange(20), models=["chronos-2"], horizon=1, folds=1, timeout=-1)
    async def run():
        async with AsyncClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await client.backtest(np.arange(20), models=["chronos-2"], horizon=1, folds=1, timeout=-1)
    asyncio.run(run())
    assert calls == []


def test_timeout_names_backtest_reconnect_route_and_does_not_cancel():
    calls = []
    response = payload()
    response.update(status="queued", completed_jobs=0)
    def handler(req):
        calls.append((req.method, req.url.path))
        return httpx.Response(200, json=response)
    with Client(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JobTimeoutError) as error:
            Backtest(client, response).wait(timeout=.001)
    assert error.value.job_id == "bt_fixture" and "reopen_backtest" in str(error.value)
    assert "resubmit the fit" not in str(error.value) and all(method == "GET" for method, _ in calls)


def test_uncertain_submission_retains_same_safe_replay_key():
    def handler(req):
        raise httpx.ConnectError("network unavailable", request=req)
    with Client(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TransportError) as error:
            client.submit_backtest(np.arange(20), models=["chronos-2"], horizon=1, folds=1,
                                   idempotency_key="retained-safe-key")
        assert "retained-safe-key" in str(error.value)
        with pytest.raises(ValueError):
            client.submit_backtest(np.arange(20), models=["chronos-2"], horizon=1,
                                   idempotency_key="test-key")


def test_nanosecond_index_is_refused_without_truncation():
    frame = pd.DataFrame(dict(date=pd.date_range("2026-01-01T00:00:00.000000001Z", periods=20, freq="h"),
                             y=np.arange(20)))
    with pytest.raises(ValueError):
        request(frame, models=["chronos-2"], horizon=1, time="date", target="y")


def test_full_native_levels_and_month_end_request_are_preserved():
    series = pd.Series(np.arange(20), index=pd.date_range("2024-01-31", periods=20, freq="ME", tz="UTC"))
    body = request(series, models=["chronos-2"], horizon=1, quantile_levels=[.01, .5, .99], frequency="ME")
    assert body["quantile_levels"] == [.01, .5, .99]
    assert body["frequency"] == dict(unit="month", every=1)
