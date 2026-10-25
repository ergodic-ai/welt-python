import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from welt import AsyncClient, Backtest, BacktestWarning, Client, WeltError
from welt.backtest import request
from welt.backtest_benchmarks import IDS, validate


def resource():
    return dict(
        id="bt_fixture",
        status="succeeded",
        index_kind="position",
        frequency=None,
        calendar_anchor=None,
        targets=["value"],
        series_ids=[None],
        quantile_levels=[0.1, 0.5, 0.9],
        models=[
            dict(
                model_id="chronos-2",
                model_version="model-v1",
                configuration="config-v1",
                jobs=[dict(job_id="job_fixture", operation_id="op_fixture", fold=0)],
            )
        ],
        folds=1,
        horizon=1,
        step=1,
        context_length=None,
        seasonal_period=1,
        final_horizon=None,
        primary_metric="mae",
        protocol_version="backtest-protocol-v1",
        metrics_version="backtest-metrics-v1",
        report_version="backtest-report-v1",
        split_sha256="a" * 64,
        completed_jobs=1,
        total_jobs=1,
        summary=dict(leaderboard=[]),
        benchmark_suite="fast-v1",
        benchmarks=None,
        predictions_expires_at="2099-01-01T00:00:00+00:00",
    )


def output():
    r = resource()
    models = []
    units = []
    rows = []
    metrics = []
    for model in sorted(IDS):
        name = model.split(":")[1]
        version = (
            "skaters-0.16.0"
            if name == "laplace"
            else "statsforecast-2.1.1"
            if name in ("auto-theta", "auto-ets", "mfles")
            else "sklearn-1.9.1"
            if name in ("ridge", "elastic-net")
            else "welt-drift-v1"
        )
        models.append(
            dict(
                model_id=model,
                model_version=version,
                configuration="fast-v1:" + name,
                status="succeeded",
                completed_units=1,
                total_units=1,
                point_semantics="median" if name == "laplace" else "point",
                covariates="fixture",
            )
        )
        units.append(dict(model_id=model, key='[0,null,"value"]', status="succeeded"))
        rows.append(
            dict(
                model_id=model,
                fold=0,
                id=None,
                target="value",
                lead=1,
                position=20,
                index=20,
                cutoff=19,
                train_start=0,
                train_stop=20,
                actual=2.0,
                prediction=2.0,
                quantiles=[1.0, 2.0, 3.0] if name == "laplace" else None,
            )
        )
        values = dict(
            mae=0.0,
            rmse=0.0,
            mase=0.0,
            pinball=1 / 15 if name == "laplace" else None,
            coverage=1.0 if name == "laplace" else None,
            interval_width=2.0 if name == "laplace" else None,
        )
        metrics.append(
            dict(
                dimension="overall",
                model_id=model,
                values=values,
                counts={k: 0 if v is None else 1 for k, v in values.items()},
                undefined={k: "Point-only model." for k, v in values.items() if v is None},
                prediction_events=1,
                observed_events=1,
                nominal_coverage=0.8 if name == "laplace" else None,
            )
        )
    return dict(
        backtest_id=r["id"],
        suite_version="fast-v1",
        split_sha256=r["split_sha256"],
        index_kind=r["index_kind"],
        quantile_levels=r["quantile_levels"],
        expires_at=r["predictions_expires_at"],
        metadata=dict(
            suite_version="fast-v1",
            status="succeeded",
            budget_seconds=20,
            elapsed_ms=123.0,
            models=models,
            warnings=[],
        ),
        units=units,
        items=rows,
        final=[],
        metrics=metrics,
        leaderboard=[{k: v for k, v in row.items() if k != "dimension"} for row in metrics],
        details=[
            dict(
                model_id="benchmark:elastic-net",
                key='[0,null,"value"]',
                leads=[
                    dict(
                        lead=1,
                        alpha=0.1,
                        l1_ratio=0.5,
                        training_rows=16,
                        feature_count=1,
                        selected_features=[dict(name="target.lag_1", coefficient=1.0)],
                    )
                ],
            )
        ],
    )


def test_opt_in_request_and_sync_wait_scores_predictions_features():
    assert (
        request(list(range(40)), models=["chronos-2"], horizon=1, benchmark_suite="fast-v1")[
            "benchmark_suite"
        ]
        == "fast-v1"
    )
    with pytest.raises(ValueError):
        request(list(range(40)), models=["chronos-2"], horizon=1, benchmark_suite="unknown")
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(
            200, json=output() if req.url.path.endswith("/benchmarks") else resource()
        )

    with Client(api_key="test-key", transport=httpx.MockTransport(handler)) as c:
        b = Backtest(c, resource()).wait(timeout=10)
        assert b.to_dict()["benchmarks"]["status"] == "succeeded"
        assert len(b.to_frame(include_benchmarks=True)) == 7
        assert len(b.benchmark_predictions()) == 7 and len(b.benchmark_scores()) == 7
        assert b.feature_selection().iloc[0]["feature"] == "target.lag_1"
        assert b.final_predictions if hasattr(b, "final_predictions") else True
    assert all(path.endswith("bt_fixture") or path.endswith("/benchmarks") for path in calls)


def test_async_wait_and_methods_have_same_semantics():
    async def run():
        def handler(req):
            return httpx.Response(
                200, json=output() if req.url.path.endswith("/benchmarks") else resource()
            )

        async with AsyncClient(api_key="test-key", transport=httpx.MockTransport(handler)) as c:
            b = await c.reopen_backtest("bt_fixture")
            await b.wait(timeout=10)
            assert len(await b.benchmark_predictions()) == 7
            assert len(await b.benchmark_scores()) == 7
            assert len(await b.feature_selection()) == 1
            assert len(b.to_frame(include_benchmarks=True)) == 7

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(backtest_id="bt_foreign"),
        lambda p: p["items"][0].update(position=20.5, index=20.5, train_stop=20.5, cutoff=19.5),
        lambda p: p["items"][0].update(position=21, index=21, train_stop=21, cutoff=20),
        lambda p: p["items"][0].update(actual=float("nan")),
        lambda p: p["items"][0].update(quantiles=[1.0, 2.0, 3.0]),
        lambda p: p["leaderboard"].append(deepcopy(p["leaderboard"][0])),
        lambda p: p["metrics"][0].update(values=[]),
        lambda p: p["metrics"][0].update(dimension="bogus"),
        lambda p: p["metrics"][0]["counts"].update(mae=999),
        lambda p: p["metadata"]["models"][0].update(model_version="foreign-v999"),
        lambda p: p["metadata"]["models"][0].update(status="pending"),
        lambda p: p["details"][0].update(model_id="foreign"),
        lambda p: p["details"][0]["leads"][0].update(lead=999),
        lambda p: p["details"][0]["leads"][0]["selected_features"][0].update(
            coefficient=float("inf")
        ),
        lambda p: p["details"][0]["leads"][0].update(alpha=-1),
        lambda p: p["units"].pop(),
    ],
)
def test_untrusted_benchmark_response_rejected(mutation):
    p = output()
    mutation(p)
    with pytest.raises(WeltError, match="Invalid backtest benchmark"):
        validate(p, resource())


def test_suite_cannot_change_during_refresh():
    with Client(api_key="test-key") as client:
        b = Backtest(client, resource())
        changed = resource()
        changed["benchmark_suite"] = None
        with pytest.raises(WeltError):
            b._update(changed)


def test_terminal_failure_is_explicit_and_native_result_survives():
    p = output()
    p.update(items=[], final=[], metrics=[], leaderboard=[], details=[], units=[])
    p["metadata"]["status"] = "failed"
    for m in p["metadata"]["models"]:
        m.update(status="failed", completed_units=0)
    with Client(
        api_key="test-key", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=p))
    ) as client:
        b = Backtest(client, resource())
        with pytest.warns(BacktestWarning):
            assert b.benchmarks()["metadata"]["status"] == "failed"
        assert b.status == "succeeded"
