from womens_health import load_womens_health_bayes_net


def test_womens_health_bn_loads():
    bn = load_womens_health_bayes_net()

    assert bn.nodes, "Expected node list to be populated"
    assert bn.edges, "Expected edge list to be populated"
    assert bn.inference is not None
