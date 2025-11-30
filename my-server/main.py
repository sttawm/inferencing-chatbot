import json
import logging
import os
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from womens_health import load_womens_health_bayes_net
from bn_helpers import reachable_from_evidence

import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

from prompt import make_prompt

# --------- Load env & init Vertex ---------
load_dotenv()

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-fastapi")

app = FastAPI(title="Gemini + BN Server")

BASELINE_MODEL_NAME = "Gemini-2.5-Pro (Baseline)"
BN_ENHANCED_MODEL_NAME = "Gemini-2.5-Pro + BN"
VERTEX_MODEL_ID = "gemini-2.5-pro"

BN = load_womens_health_bayes_net()


# --------- Pydantic models ---------

class MessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    data: Optional[Dict[str, str]] = None


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    parts: List[MessagePart]
    id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    tools: Dict[str, Dict] = Field(default_factory=dict)
    model: str
    temperature: float = 0.3
    max_output_tokens: int = 2048


class ChatResponse(BaseModel):
    reply: str


# --------- Helpers ---------

def render_conversation(messages: List[Message]) -> str:
    lines: List[str] = []
    for message in messages:
        text_parts = [part.text for part in message.parts if part.text]
        text = " ".join(text_parts).strip()
        if text:
            lines.append(f"{message.role.capitalize()}: {text}")
    return "\n".join(lines) if lines else "No prior conversation available."


def parse_update_payload(raw_text: str) -> Dict[str, Optional[bool]]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise HTTPException(
            status_code=500,
            detail="LLM response missing JSON payload for BN updates",
        )
    snippet = raw_text[start : end + 1]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"Unable to parse BN update JSON: {exc}"
        )

    normalized: Dict[str, Optional[bool]] = {}
    for key, value in payload.items():
        if isinstance(value, bool) or value is None:
            normalized[key] = value
        elif isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "t", "1"}:
                normalized[key] = True
            elif lowered in {"false", "f", "0"}:
                normalized[key] = False
            else:
                normalized[key] = None
        else:
            normalized[key] = None
    return normalized


def to_state_value(value: Optional[bool]) -> Optional[str]:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return value if isinstance(value, str) else None


# --------- Core Gemini call ---------

def call_gemini(req: ChatRequest) -> str:
    model = GenerativeModel(req.model)

    contents: list[Content] = []
    for message in req.messages:
        role = "user" if message.role in ("user", "system") else "model"
        text_chunks = [part.text for part in message.parts if part.text]
        if not text_chunks:
            continue
        contents.append(
            Content(
                role=role,
                parts=[Part.from_text("\n".join(text_chunks))],
            )
        )

    gen_config = GenerationConfig(
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
    )

    try:
        response = model.generate_content(
            contents,
            generation_config=gen_config,
        )
    except Exception as e:
        logger.exception("Gemini call failed")
        raise HTTPException(status_code=500, detail=str(e))

    text = (response.text or "").strip()
    if not text:
        raise HTTPException(status_code=500, detail="Empty response from Gemini")

    return text


# --------- FastAPI route ---------

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    logger.info(
        "Incoming request -> model=%s messages=%s tools=%s",
        body.model,
        len(body.messages),
        len(body.tools),
    )
    logger.info("Full request payload:\n%s", body.model_dump_json(indent=2))

    if body.model == BASELINE_MODEL_NAME:
        payload = body.model_dump()
        payload["model"] = VERTEX_MODEL_ID
        proxy_request = ChatRequest(**payload)
        reply = call_gemini(proxy_request)
    elif body.model == BN_ENHANCED_MODEL_NAME:
        reply = handle_bn_enhanced_request(body)
    else:
        raise HTTPException(statuscode=400, detail=f"Unknown model {body.model}")

    logger.info("Sending reply (%s chars): %s", len(reply), reply)
    return ChatResponse(reply=reply)


def handle_bn_enhanced_request(body: ChatRequest) -> str:
    conversation = render_conversation(body.messages)

    prompt = make_prompt(conversation)

    bn_prompt_request = ChatRequest(
        messages=[Message(role="user", parts=[MessagePart(type="text", text=prompt)])],
        tools={},
        model=VERTEX_MODEL_ID,
    )

    response_text = call_gemini(bn_prompt_request)
    updates_raw = parse_update_payload(response_text)
    updates = {
        node: value
        for node, value in updates_raw.items()
        if value is not None and node in BN.nodes
    }
    logger.info("Bayesian network updates: %s", updates if updates else "None")

    evidence_map: Dict[str, str] = {}
    for node, value in updates.items():
        state_value = to_state_value(value)
        if state_value:
            evidence_map[node] = state_value
    reachable_nodes = reachable_from_evidence(BN.model, list(evidence_map.keys()))

    probabilities: Dict[str, float] = {}

    if reachable_nodes:
        try:
            factors = BN.inference.query(
                variables=reachable_nodes,
                evidence=evidence_map or None,
                joint=False,
                show_progress=False,
            )

            if not isinstance(factors, dict):
                factors = {reachable_nodes[0]: factors}

            for node, factor in factors.items():
                state_names = factor.state_names[node]
                values = factor.values.tolist()
                try:
                    yes_index = state_names.index("yes")
                    probabilities[node] = float(values[yes_index])
                except ValueError:
                    logger.warning(
                        "Node %s missing 'yes' state in factor %s", node, state_names
                    )
        except Exception as exc:
            logger.exception("Failed to query probabilities: %s", exc)

    logger.info("Updated probabilities: %s", probabilities if probabilities else "None")

    updates_text = json.dumps(updates, indent=2) if updates else "None"
    probability_text = (
        json.dumps(probabilities, indent=2)
        if probabilities
        else "No probabilistic updates calculated."
    )
    bn_message = (
        "Probabilistic model insights:\n"
        f"Node updates:\n{updates_text}\n\n"
        f"Updated probabilities:\n{probability_text}\n\n"
        "Please incorporate these insights into your reply while still leveraging your "
        "broader knowledge."
    )

    analysis_messages = list(body.messages)
    analysis_messages.append(
        Message(role="user", parts=[MessagePart(type="text", text=bn_message)])
    )

    analysis_request = ChatRequest(
        messages=analysis_messages,
        tools=body.tools,
        model=VERTEX_MODEL_ID,
        temperature=body.temperature,
        max_output_tokens=body.max_output_tokens,
    )

    llm_reply = call_gemini(analysis_request)

    summary_lines = [
        "Bayesian network updates:",
        updates_text,
        "",
        "Updated probabilities:",
        probability_text,
        "",
        "Assistant response:",
        llm_reply,
    ]
    return "\n".join(summary_lines)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
