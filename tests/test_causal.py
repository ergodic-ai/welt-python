"""Source-level causal SDK tests use HTTP fixtures, no learned model/quality claim."""

import asyncio
from copy import deepcopy
import sys
from types import ModuleType

import httpx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from welt import AsyncClient, CausalDiscovery, CausalResult, Client
from welt.causal import validated_scores
from welt.errors import (
    InvalidCausalResultError,
    OptionalDependencyError,
    UnsupportedGraphConversionError,
)


def result_payload():
    return dict(
        id="causal_fixture",
        workspace_id="workspace_fixture",
        dataset_id="dataset_fixture",
        model_version="arrow-fixture-version",
        configuration_version="fixture-config",
        variables=["z", "isolated", "a"],
        native_graph_type="dag",
        edge_marks=[dict(u="z", v="a", u_mark="tail", v_mark="arrow")],
        score_semantics="native_directed_edge_probability",
        score_reference="score_fixture",
        decoder={"name": "literal-native-fixture"},
        assumptions=["fixture only"],
        constraints_applied=dict(
            required_edges=[], forbidden_edges=[], temporal_order=[]
        ),
        diagnostics={"seed": 42, "job_id": "job_fixture"},
    )


def score_payload():
    return dict(
        result_id="causal_fixture",
        score_reference="score_fixture",
        variables=["z", "isolated", "a"],
        score_axes="source_row_target_column",
        score_semantics="native_directed_edge_probability",
        values=[[0, 0.8, 0.1], [0.9, 0, 0.8], [0.9, 0.2, 0]],
    )


class Fixture:
    def __init__(self):
        self.calls = []

    def handle(self, request):
        self.calls.append(request)
        path = request.url.path
        if path == "/v1/discoveries":
            assert request.method == "POST"
            return httpx.Response(202, json={"id": "job_fixture"})
        if path == "/v1/jobs/job_fixture":
            return httpx.Response(
                200,
                json=dict(
                    id="job_fixture",
                    status="succeeded",
                    operation="discover",
                    predictor_id=None,
                ),
            )
        if path == "/v1/jobs/job_fixture/result":
            return httpx.Response(200, json=result_payload())
        if path == "/v1/jobs/job_fixture/result/scores":
            return httpx.Response(200, json=score_payload())
        if path == "/v1/datasets":
            return httpx.Response(201, json={"id": "dataset_fixture"})
        raise AssertionError(f"Unexpected fixture path {path}")


def test_sync_explicit_submit_reopen_scores_and_blocking_convenience():
    fixture = Fixture()
    with Client(
        api_key="fixture-only", transport=httpx.MockTransport(fixture.handle)
    ) as client:
        job = client.submit_discover(
            "dataset_fixture", seed=42, idempotency_key="fixed-key"
        )
        result = client.job(job.id).result()
        assert isinstance(result, CausalResult) and result.variables == (
            "z",
            "isolated",
            "a",
        )
        assert client.causal_result(job.id).to_dict() == result.to_dict()
        scores = client.causal_scores(job.id)
        assert scores["values"][0][2] == 0.1 and result.edge_marks[0]["u"] == "z"
        assert client.discover("dataset_fixture").id == "causal_fixture"
    first = fixture.calls[0]
    assert first.headers["Idempotency-Key"] == "fixed-key"
    import json

    body = json.loads(first.content)
    assert body["seed"] == 42 and body["constraints"] == dict(
        required_edges=[], forbidden_edges=[], temporal_order=[]
    )
    assert all("predictors" not in r.url.path for r in fixture.calls)


def test_async_equivalent_submit_reopen_scores_and_blocking_convenience():
    async def exercise():
        fixture = Fixture()
        async with AsyncClient(
            api_key="fixture-only", transport=httpx.MockTransport(fixture.handle)
        ) as client:
            job = await client.submit_discover(
                "dataset_fixture", seed=42, idempotency_key="fixed-key"
            )
            result = await client.job(job.id).result()
            assert isinstance(result, CausalResult)
            assert (await client.causal_result(job.id)).to_dict() == result.to_dict()
            assert (await client.causal_scores(job.id))[
                "score_reference"
            ] == result.score_reference
            assert (await client.discover("dataset_fixture")).id == result.id
        assert fixture.calls[0].headers["Idempotency-Key"] == "fixed-key"

    asyncio.run(exercise())


def test_result_data_and_marks_are_immutable_without_hiding_ordered_isolates():
    payload = result_payload()
    result = CausalResult(payload)
    payload["variables"].reverse()
    payload["edge_marks"][0]["u"] = "mutated"
    exported = result.to_dict()
    exported["variables"].reverse()
    assert (
        result.variables == ("z", "isolated", "a") and result.edge_marks[0]["u"] == "z"
    )
    with pytest.raises(TypeError):
        result.edge_marks[0]["u"] = "mutated"
    assert "workspace_fixture" not in repr(result)
    assert (
        result.job_id == "job_fixture"
        and result.configuration_version == "fixture-config"
    )
    assert result.assumptions == ("fixture only",)
    with pytest.raises(TypeError):
        result.diagnostics["job_id"] = "other"


def test_tuple_edge_collection_also_freezes_nested_mutation_boundaries():
    payload = result_payload()
    payload["edge_marks"] = tuple(payload["edge_marks"])
    result = CausalResult(payload)
    payload["edge_marks"][0]["u"] = "mutated"
    assert result.edge_marks[0]["u"] == "z"
    with pytest.raises(TypeError):
        result.edge_marks[0]["u"] = "mutated"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_result_reopen_rejects_different_owned_job_identity(asynchronous):
    def response(request):
        return httpx.Response(200, json=result_payload())

    if asynchronous:

        async def exercise():
            async with AsyncClient(api_key="fixture-only", transport=httpx.MockTransport(response)) as client:
                with pytest.raises(InvalidCausalResultError):
                    await client.causal_result("other-job")

        asyncio.run(exercise())
    else:
        with Client(api_key="fixture-only", transport=httpx.MockTransport(response)) as client:
            with pytest.raises(InvalidCausalResultError):
                client.causal_result("other-job")


@pytest.mark.parametrize(
    "fault",
    [
        "scalar",
        "duplicate",
        "loop",
        "cycle",
        "unknown-mark",
        "missing-provenance",
        "malformed-edges",
    ],
)
def test_malformed_result_is_typed_and_never_projected(fault):
    payload = result_payload()
    if fault == "scalar":
        payload = None
    if fault == "duplicate":
        payload["variables"][1] = "z"
    if fault == "loop":
        payload["edge_marks"][0]["v"] = "z"
    if fault == "cycle":
        payload["edge_marks"].append(dict(u="a", v="z", u_mark="tail", v_mark="arrow"))
    if fault == "unknown-mark":
        payload["edge_marks"][0]["u_mark"] = "circle"
    if fault == "missing-provenance":
        del payload["id"]
    if fault == "malformed-edges":
        payload["edge_marks"] = None
    with pytest.raises(InvalidCausalResultError):
        CausalResult(payload)


@pytest.mark.parametrize(
    "fault",
    [
        "result-id",
        "reference",
        "variable-order",
        "axis",
        "shape",
        "nonfinite",
        "diagonal",
        "boolean",
        "huge-integer",
    ],
)
def test_scores_require_owned_result_axis_identity_and_native_probability_semantics(
    fault,
):
    payload = score_payload()
    if fault == "result-id":
        payload["result_id"] = "other"
    if fault == "reference":
        payload["score_reference"] = "other"
    if fault == "variable-order":
        payload["variables"].reverse()
    if fault == "axis":
        payload["score_axes"] = "target_row_source_column"
    if fault == "shape":
        payload["values"].pop()
    if fault == "nonfinite":
        payload["values"][0][1] = float("nan")
    if fault == "diagonal":
        payload["values"][0][0] = 0.5
    if fault == "boolean":
        payload["values"][0][0] = False
    if fault == "huge-integer":
        payload["values"][0][1] = 10**1000
    with pytest.raises(InvalidCausalResultError):
        validated_scores(CausalResult(result_payload()), payload)


def test_explicit_native_conversion_preserves_direction_order_and_isolate(monkeypatch):
    import welt.causal as causal

    observed = []

    class Graph:
        @classmethod
        def from_dict(cls, payload):
            observed.append(payload)
            graph = cls()
            graph.nodes = frozenset(payload["nodes"])
            return graph

    module = ModuleType("ergodic")
    module.MixedGraph = Graph
    monkeypatch.setitem(sys.modules, "ergodic", module)
    monkeypatch.setattr(causal, "version", lambda name: "0.1.1")
    result = CausalResult(result_payload())
    graph = result.to_ergodic()
    assert graph.nodes == frozenset(result.variables)
    assert observed == [
        dict(
            kind="dag",
            nodes=["z", "isolated", "a"],
            edges=[dict(u="z", v="a", u_mark="tail", v_mark="arrow")],
        )
    ]


def test_native_directed_cycle_preserved_but_dag_conversion_refused_before_optional_import(
    monkeypatch,
):
    payload = result_payload()
    payload["native_graph_type"] = "directed"
    payload["edge_marks"].append(dict(u="a", v="z", u_mark="tail", v_mark="arrow"))
    result = CausalResult(payload)
    assert len(result.edge_marks) == 2
    with pytest.raises(UnsupportedGraphConversionError):
        result.to_ergodic()


def test_unavailable_or_unrelated_ergodic_distribution_has_typed_optional_error(
    monkeypatch,
):
    import welt.causal as causal

    module = ModuleType("ergodic")
    monkeypatch.setitem(sys.modules, "ergodic", module)
    monkeypatch.setattr(causal, "version", lambda name: "1.0.0")
    with pytest.raises(OptionalDependencyError):
        CausalResult(result_payload()).to_ergodic()


def test_unrelated_distribution_is_refused_before_optional_module_import(monkeypatch):
    import builtins
    import welt.causal as causal

    original = builtins.__import__

    def observed(name, *args, **kwargs):
        if name == "ergodic":
            pytest.fail("Unrelated distribution must not import.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observed)
    monkeypatch.setattr(causal, "version", lambda name: "1.0.0")
    with pytest.raises(OptionalDependencyError):
        CausalResult(result_payload()).to_ergodic()


def test_unexpected_local_conversion_rejection_is_typed_and_sanitized(monkeypatch):
    import welt.causal as causal

    class Graph:
        @classmethod
        def from_dict(cls, payload):
            raise ValueError("private-customer-value")

    module = ModuleType("ergodic")
    module.MixedGraph = Graph
    monkeypatch.setitem(sys.modules, "ergodic", module)
    monkeypatch.setattr(causal, "version", lambda name: "0.1.1")
    with pytest.raises(InvalidCausalResultError) as error:
        CausalResult(result_payload()).to_ergodic()
    assert "private-customer-value" not in str(error.value)


@pytest.mark.parametrize("kind", ["dataframe", "array", "csv", "dataset-id"])
def test_discovery_facade_uploads_target_free_full_table_or_reuses_dataset_and_closes_blocking_client(
    tmp_path, monkeypatch, kind
):
    fixture = Fixture()
    client = Client(
        api_key="fixture-only", transport=httpx.MockTransport(fixture.handle)
    )
    monkeypatch.setattr(CausalDiscovery, "_client", lambda self: client)
    estimator = CausalDiscovery(random_state=42)
    assert clone(estimator).get_params() == estimator.get_params()
    data = pd.DataFrame([[1, 7, 3], [2, 7, 5]], columns=["z", "isolated", "a"])
    if kind == "array":
        data = data.to_numpy()
    if kind == "csv":
        path = tmp_path / "observations.csv"
        data.to_csv(path, index=False)
        data = path
    result = (
        estimator.discover(dataset_id="dataset_fixture")
        if kind == "dataset-id"
        else estimator.discover(data)
    )
    assert isinstance(result, CausalResult) and client.http.is_closed
    uploads = [r for r in fixture.calls if r.url.path == "/v1/datasets"]
    assert len(uploads) == (0 if kind == "dataset-id" else 1)
    if uploads:
        import json

        payload = json.loads(uploads[0].content)
        assert (
            payload["target"] is None
            and len(payload["columns"]) == 3
            and len(payload["rows"]) == 2
        )
        assert payload["columns"] == (
            ["x0", "x1", "x2"] if kind == "array" else ["z", "isolated", "a"]
        )


@pytest.mark.parametrize("seed", [True, -1, 2**32, "42"])
def test_invalid_seed_rejected_locally_without_upload_or_request(seed):
    with Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail("No request expected")
        )
    ) as client:
        with pytest.raises(ValueError):
            client.submit_discover("dataset_fixture", seed=seed)


@pytest.mark.parametrize('asynchronous', [False, True])
def test_explicit_owned_deletion_path_and_typed_primary_result_deleted(asynchronous):
    from welt import ResultDeletedError
    calls=[]
    def handle(request):
        calls.append((request.method, request.url.raw_path.decode()))
        if request.method=='DELETE':
            return httpx.Response(204)
        return httpx.Response(410, json={'error':dict(code='result_deleted',message='Discovery result was explicitly deleted.',retryable=False,request_id='req_fixture')})
    async def run():
        async with AsyncClient(base_url='https://fixture.invalid',api_key='fixture',transport=httpx.MockTransport(handle)) as client:
            assert await client.delete_causal_result('causal/owned') is None
            assert await client.delete_dataset('dataset/owned') is None
            with pytest.raises(ResultDeletedError) as caught:
                await client.causal_scores('job_fixture')
            assert caught.value.code=='result_deleted' and caught.value.status_code==410
    if asynchronous:
        asyncio.run(run())
    else:
        with Client(base_url='https://fixture.invalid',api_key='fixture',transport=httpx.MockTransport(handle)) as client:
            assert client.delete_causal_result('causal/owned') is None
            assert client.delete_dataset('dataset/owned') is None
            with pytest.raises(ResultDeletedError) as caught:
                client.causal_scores('job_fixture')
            assert caught.value.code=='result_deleted' and caught.value.status_code==410
    assert calls==[('DELETE','/v1/causal-results/causal%2Fowned'),('DELETE','/v1/datasets/dataset%2Fowned'),('GET','/v1/jobs/job_fixture/result')]


def test_dataset_provenance_accessor_is_immutable():
    result = CausalResult(result_payload())
    assert result.dataset_id == 'dataset_fixture'
    with pytest.raises(AttributeError):
        result.dataset_id = 'different_dataset'
