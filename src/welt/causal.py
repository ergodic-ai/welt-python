"""Owned discovery results and explicit local graph conversion.

No model runtime or optional causal dependency is imported during SDK import.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType

from sklearn.base import BaseEstimator

from .credentials import credential
from .errors import (
    InvalidCausalResultError,
    OptionalDependencyError,
    UnsupportedGraphConversionError,
)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _graph(payload):
    if not isinstance(payload, dict):
        raise ValueError("Result must be a structured causal object.")
    names = payload["variables"]
    if (
        not isinstance(names, (list, tuple))
        or not names
        or any(not isinstance(n, str) or not n or len(n) > 128 for n in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("Causal variables must retain unique ordered names.")
    if payload["native_graph_type"] not in ("dag", "directed"):
        raise ValueError("This SDK slice supports native directed graphs only.")
    edges, pairs = [], set()
    for value in payload["edge_marks"]:
        if not isinstance(value, dict) or set(value) != {"u", "v", "u_mark", "v_mark"}:
            raise ValueError("Edges must have explicit native endpoint marks.")
        u, v = value["u"], value["v"]
        marks = value["u_mark"], value["v_mark"]
        if (
            u not in names
            or v not in names
            or u == v
            or marks not in (("tail", "arrow"), ("arrow", "tail"))
        ):
            raise ValueError("Unsupported or invalid native directed edge.")
        pair = (u, v) if marks == ("tail", "arrow") else (v, u)
        if pair in pairs:
            raise ValueError("Duplicate directed edge.")
        pairs.add(pair)
        edges.append(deepcopy(value))
    if payload["native_graph_type"] == "dag":
        children = {n: [] for n in names}
        indegree = dict.fromkeys(names, 0)
        for source, target in pairs:
            children[source].append(target)
            indegree[target] += 1
        ready = [n for n in names if indegree[n] == 0]
        visited = 0
        while ready:
            source = ready.pop()
            visited += 1
            for target in children[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(names):
            raise ValueError("Native DAG contains a directed cycle.")
    return tuple(names), edges


@dataclass(frozen=True, init=False)
class CausalResult:
    """Immutable native result; score fetches use explicit client/job ownership."""

    _payload: object = field(repr=False)

    def __init__(self, payload):
        try:
            _graph(payload)
            for name in (
                "id",
                "workspace_id",
                "dataset_id",
                "model_version",
                "configuration_version",
                "score_reference",
            ):
                if not isinstance(payload[name], str) or not payload[name]:
                    raise ValueError("Missing result provenance.")
            if payload["score_semantics"] != "native_directed_edge_probability":
                raise ValueError("Unsupported causal score semantics.")
            if not isinstance(payload["diagnostics"], dict) or not isinstance(
                payload["decoder"], dict
            ):
                raise ValueError("Result must preserve decoder/provenance metadata.")
            job_id = payload["diagnostics"].get("job_id")
            if not isinstance(job_id, str) or not job_id or len(job_id) > 128:
                raise ValueError(
                    "Result must preserve its owned discovery job identity."
                )
            if not isinstance(payload["assumptions"], list) or any(
                not isinstance(a, str) for a in payload["assumptions"]
            ):
                raise ValueError("Assumptions must be explicit strings.")
            if not isinstance(payload["constraints_applied"], dict):
                raise ValueError("Constraints must be explicit.")
        except (KeyError, TypeError, ValueError):
            raise InvalidCausalResultError(
                "Invalid native causal result; no graph repair was applied.",
                code="invalid_causal_result",
            ) from None
        object.__setattr__(self, "_payload", _freeze(deepcopy(payload)))

    def to_dict(self):
        return _thaw(self._payload)

    @property
    def variables(self):
        return self._payload["variables"]

    @property
    def id(self):
        return self._payload["id"]

    @property
    def score_reference(self):
        return self._payload["score_reference"]

    @property
    def model_version(self):
        return self._payload["model_version"]

    @property
    def dataset_id(self):
        return self._payload["dataset_id"]

    @property
    def job_id(self):
        return self._payload["diagnostics"]["job_id"]

    @property
    def configuration_version(self):
        return self._payload["configuration_version"]

    @property
    def diagnostics(self):
        return self._payload["diagnostics"]

    @property
    def decoder(self):
        return self._payload["decoder"]

    @property
    def assumptions(self):
        return self._payload["assumptions"]

    @property
    def edge_marks(self):
        return self._payload["edge_marks"]

    @property
    def native_graph_type(self):
        return self._payload["native_graph_type"]

    def to_ergodic(self):
        """Convert a declared native DAG locally, preserving marks and all nodes."""
        if self.native_graph_type != "dag":
            raise UnsupportedGraphConversionError(
                "Only native DAG conversion is qualified; no projection was applied.",
                code="unsupported_graph_conversion",
            )
        try:
            if version("ergodic") != "0.1.1":
                raise OptionalDependencyError(
                    "Install the approved Ergodic co-release artifact; the public dependency is not yet available.",
                    code="optional_dependency_unavailable",
                )
            import ergodic

            if not hasattr(ergodic, "MixedGraph"):
                raise OptionalDependencyError(
                    "The installed Ergodic package lacks the approved graph API.",
                    code="optional_dependency_unavailable",
                )
        except (ImportError, PackageNotFoundError):
            raise OptionalDependencyError(
                "Install the approved Ergodic co-release artifact; Welt does not download it automatically.",
                code="optional_dependency_unavailable",
            ) from None
        payload = dict(
            kind="dag",
            nodes=list(self.variables),
            edges=[_thaw(edge) for edge in self.edge_marks],
        )
        try:
            graph = ergodic.MixedGraph.from_dict(payload)
        except (ValueError, TypeError, AttributeError):
            raise InvalidCausalResultError(
                "Ergodic rejected native DAG conversion; no graph repair was applied.",
                code="invalid_causal_result",
            ) from None
        # MixedGraph stores node sets; ensure actual conversion retains all names.
        if graph.nodes != frozenset(self.variables):
            raise InvalidCausalResultError(
                "Ergodic conversion changed native variables.",
                code="invalid_causal_result",
            )
        return graph


def validated_scores(result, payload):
    """Bounded original-axis scores; never decode or threshold another graph."""
    import numpy as np

    try:
        if (
            not isinstance(payload, dict)
            or payload["result_id"] != result.id
            or payload["score_reference"] != result.score_reference
        ):
            raise ValueError("Scores are not bound to this result.")
        if (
            tuple(payload["variables"]) != result.variables
            or payload["score_axes"] != "source_row_target_column"
            or payload["score_semantics"] != "native_directed_edge_probability"
        ):
            raise ValueError("Score semantics/order differ from the native result.")
        values = payload["values"]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for row in values
            for value in row
        ):
            raise ValueError("Scores must be numeric.")
        array = np.asarray(values, dtype=float)
        if (
            array.shape != (len(result.variables), len(result.variables))
            or not np.isfinite(array).all()
            or not ((array >= 0) & (array <= 1)).all()
            or not (array.diagonal() == 0).all()
        ):
            raise ValueError(
                "Scores require native square finite probability semantics."
            )
    except (KeyError, TypeError, ValueError, OverflowError):
        raise InvalidCausalResultError(
            "Invalid result-bound causal scores; no transpose or repair was applied.",
            code="invalid_causal_result",
        ) from None
    return deepcopy(payload)


class CausalDiscovery(BaseEstimator):
    """Dataset-based discovery facade with reusable sklearn-style parameters."""

    def __init__(
        self,
        model="arrow",
        configuration="default",
        base_url=None,
        api_key=None,
        random_state=0,
        timeout=600,
    ):
        self.model, self.configuration, self.base_url = model, configuration, base_url
        self.api_key = credential(api_key)
        self.random_state, self.timeout = random_state, timeout

    def _client(self):
        from .client import Client

        return Client(base_url=self.base_url, api_key=self.api_key)

    def set_params(self, **params):
        if "api_key" in params:
            params["api_key"] = credential(params["api_key"])
        return super().set_params(**params)

    def submit_discover(
        self,
        X=None,
        *,
        dataset_id=None,
        constraints=None,
        model_version=None,
        idempotency_key=None,
    ):
        if (X is None) == (dataset_id is None):
            raise ValueError("Supply exactly one observational table or dataset_id.")
        if (
            type(self.random_state) is not int
            or not 0 <= self.random_state <= 2**32 - 1
        ):
            raise ValueError("random_state must be an unsigned32-bit integer.")
        client = self._client()
        try:
            if dataset_id is None:
                from .estimators import table

                columns, rows, _ = table(X)
                dataset_id = client.upload(columns=columns, rows=rows, target=None)[
                    "id"
                ]
            return client.submit_discover(
                dataset_id,
                model=self.model,
                configuration=self.configuration,
                seed=self.random_state,
                constraints=constraints,
                model_version=model_version,
                idempotency_key=idempotency_key,
            )
        except BaseException:
            client.close()
            raise

    def discover(
        self,
        X=None,
        *,
        dataset_id=None,
        constraints=None,
        model_version=None,
        idempotency_key=None,
    ):
        job = self.submit_discover(
            X,
            dataset_id=dataset_id,
            constraints=constraints,
            model_version=model_version,
            idempotency_key=idempotency_key,
        )
        try:
            return job.result(timeout=self.timeout)
        finally:
            job.client.close()
