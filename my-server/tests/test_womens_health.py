import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fastapi.testclient import TestClient

from main import app, ChatRequest, Message, MessagePart, handle_bn_enhanced_request
from womens_health import load_womens_health_bayes_net


def test_womens_health_bn_loads():
    bn = load_womens_health_bayes_net()

    assert bn.nodes, "Expected node list to be populated"
    assert bn.edges, "Expected edge list to be populated"
    assert bn.inference is not None


@pytest.mark.integration
def test_bn_enhanced_flow_real_gemini():
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        pytest.skip("GOOGLE_CLOUD_PROJECT not set; real Gemini call skipped.")

    body = ChatRequest(
        messages=[
            Message(
                role="user",
                parts=[
                    MessagePart(
                        type="text",
                        text="I've been experiencing irregular periods and weight gain. What should I check?"
                    )
                ],
            )
        ],
        tools={},
        model="Gemini-2.5-Pro + BN",
    )

    reply = handle_bn_enhanced_request(body)

    assert "Updated probabilities" in reply
    assert "Assistant response" in reply


def test_bayes_graph_endpoint():
    client = TestClient(app)
    response = client.get("/bayes-graph")
    assert response.status_code == 200
    assert response.headers.get("content-type") == "image/png"
    assert response.content
