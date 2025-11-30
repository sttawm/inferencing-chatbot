from womens_health import BN_VARIABLES_DICT

from womens_health import load_womens_health_bayes_net


def test_womens_health_bn_loads():
    bn = load_womens_health_bayes_net()

    assert bn.nodes, "Expected node list to be populated"
    assert bn.edges, "Expected edge list to be populated"
    assert bn.inference is not None

def test_to_prompt():
    user_query = "I have irregular periods and weight gain. What could be the cause?"
    dsl = to_compact_dsl(BN_VARIABLES_DICT)
