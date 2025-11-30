import networkx as nx

from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

def reachable_from_evidence(model, evidence_nodes):
    """
    Returns all nodes reachable (in the undirected graph) from any evidence node,
    EXCLUDING the evidence nodes themselves.

    `model` can be either a bnlearn model dict or a raw pgmpy model.
    """
    # bnlearn-python wraps the pgmpy model under model["model"]
    G = model["model"].to_undirected() if isinstance(model, dict) else model.to_undirected()

    evidence_nodes = set(evidence_nodes)
    reachable = set()

    for e in evidence_nodes:
        if e in G:
            reachable |= nx.node_connected_component(G, e)

    # Remove the evidence variables themselves
    reachable -= evidence_nodes

    return sorted(reachable)


def validate_bn(model: BayesianModel) -> VariableElimination:
    # 1) Graph sanity: must be a DAG
    assert nx.is_directed_acyclic_graph(model), "BN graph is not a DAG"

    # 2) CPD + cardinality + normalization checks
    assert model.check_model(), "model.check_model() failed"

    # 3) Make sure inference can be constructed
    ve = VariableElimination(model)

    return ve