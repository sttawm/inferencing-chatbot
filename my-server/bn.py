import networkx as nx

def reachable_from_evidence(model, evidence_nodes):
    """
    Returns all nodes reachable from any evidence node
    in the *undirected* version of the BN graph.
    """
    G = model["model"].to_undirected() if "model" in model else model.to_undirected()

    reachable = set()
    for e in evidence_nodes:
        if e in G:
            reachable |= nx.node_connected_component(G, e)

    return sorted(reachable)