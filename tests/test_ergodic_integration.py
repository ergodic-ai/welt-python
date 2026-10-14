"""to_ergodic against the real ergodic package, when it is installed.

CI installs the oldest and latest supported ergodic in a separate job; the default
test environment has no ergodic, so these tests skip there. No network is used.
"""

from copy import deepcopy

import pytest

from welt import CausalResult
from welt.causal import SUPPORTED_ERGODIC_GRAPH_FORMATS
from welt.errors import OptionalDependencyError, UnsupportedGraphConversionError

ergodic = pytest.importorskip("ergodic")
SUPPORTED = getattr(ergodic, "GRAPH_FORMAT", None) in SUPPORTED_ERGODIC_GRAPH_FORMATS


def payload():
    return dict(
        id="causal_fixture",
        workspace_id="workspace_fixture",
        dataset_id="dataset_fixture",
        model_version="arrow-fixture-version",
        configuration_version="fixture-config",
        variables=["z", "isolated", "a", "b"],
        native_graph_type="dag",
        edge_marks=[
            dict(u="z", v="a", u_mark="tail", v_mark="arrow"),
            dict(u="b", v="a", u_mark="arrow", v_mark="tail"),
        ],
        score_semantics="native_directed_edge_probability",
        score_reference="score_fixture",
        decoder={"name": "literal-native-fixture"},
        assumptions=["fixture only"],
        constraints_applied=dict(
            required_edges=[], forbidden_edges=[], temporal_order=[]
        ),
        diagnostics={"seed": 42, "job_id": "job_fixture"},
    )


@pytest.mark.skipif(not SUPPORTED, reason="installed ergodic has no supported GRAPH_FORMAT")
def test_real_ergodic_receives_every_node_and_native_direction():
    result = CausalResult(payload())
    graph = result.to_ergodic()
    assert isinstance(graph, ergodic.MixedGraph)
    assert graph.nodes == frozenset(result.variables)
    assert graph.is_dag()
    matrix, order = graph.to_adjacency(order=list(result.variables))
    assert order == ["z", "isolated", "a", "b"]
    # z -> a, and the arrow-first edge {u: b, v: a, u_mark: arrow} is a -> b.
    assert matrix.tolist() == [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]
    written = graph.to_dict()
    assert written["format"] == ergodic.GRAPH_FORMAT
    assert written["kind"] == "dag"


@pytest.mark.skipif(not SUPPORTED, reason="installed ergodic has no supported GRAPH_FORMAT")
def test_real_ergodic_conversion_refuses_directed_output_without_projection():
    directed = deepcopy(payload())
    directed["native_graph_type"] = "directed"
    with pytest.raises(UnsupportedGraphConversionError):
        CausalResult(directed).to_ergodic()


@pytest.mark.skipif(SUPPORTED, reason="installed ergodic reads a supported format")
def test_real_ergodic_without_graph_format_is_refused_with_install_hint():
    """ergodic 0.1.x predates the graph format; conversion names the extra."""
    with pytest.raises(OptionalDependencyError) as error:
        CausalResult(payload()).to_ergodic()
    assert "welt-client[ergodic]" in str(error.value)
