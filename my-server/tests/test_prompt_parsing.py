import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fastapi import HTTPException  # noqa: E402
from main import parse_update_payload  # noqa: E402
from womens_health import load_womens_health_bayes_net  # noqa: E402


def test_parse_update_payload_empty_object_returns_empty_dict():
    payload = parse_update_payload("{}")
    assert payload == {}


def test_empty_payload_produces_no_evidence_map():
    payload = parse_update_payload("{}")
    bn = load_womens_health_bayes_net()
    updates = {
        node: value
        for node, value in payload.items()
        if value is not None and node in bn.nodes
    }
    assert updates == {}


def test_markdown_fenced_json_parses():
    payload = parse_update_payload("```json\n{\"Age\": \"30s\"}\n```")
    assert payload == {"Age": "30s"}


def test_invalid_json_raises_http_error():
    bad = "{Age: 30s}"
    with pytest.raises(HTTPException) as excinfo:
        parse_update_payload(bad)
    assert "not valid JSON" in str(excinfo.value.detail)
