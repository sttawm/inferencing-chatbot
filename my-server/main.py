import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from graphviz import Digraph
from pydantic import BaseModel, Field

import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

from womens_health import load_womens_health_bayes_net
from bn_helpers import reachable_from_evidence

from prompt import make_prompt, make_probability_prompt

# --------- Load env & init Vertex ---------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

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


def parse_update_payload(raw_text: str) -> Dict[str, Optional[str]]:
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

    normalized: Dict[str, Optional[str]] = {}
    for key, value in payload.items():
        if value is None:
            normalized[key] = None
        elif isinstance(value, bool):
            normalized[key] = "true" if value else "false"
        else:
            normalized[key] = str(value).strip()
    return normalized


def compute_probabilities(
    variables: List[str], evidence_map: Dict[str, str]
) -> Dict[str, Dict[str, float]]:
    if not variables:
        raise HTTPException(
            status_code=500,
            detail="No BN variables available for probability query",
        )

    try:
        factors = BN.inference.query(
            variables=sorted(variables),
            evidence=evidence_map or None,
            joint=False,
            show_progress=False,
        )

        if not isinstance(factors, dict):
            # When querying a single variable, pgmpy returns a Factor instead of a dict.
            factors = {variables[0]: factors}
    except Exception as exc:
        logger.exception("Failed to query probabilities: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to query probabilities")

    probabilities: Dict[str, Dict[str, float]] = {}
    for node, factor in factors.items():
        state_names = factor.state_names[node]
        values = factor.values.tolist()
        probabilities[node] = {
            state: float(prob)
            for state, prob in zip(state_names, values)
        }

    if not probabilities:
        raise HTTPException(
            status_code=500,
            detail="BN probability query returned no results",
        )

    return probabilities


def format_probabilities(probabilities: Dict[str, Dict[str, float]]) -> str:
    if not probabilities:
        return "No probabilistic updates calculated."

    tables: list[str] = []
    for node in sorted(probabilities.keys()):
        state_probs = probabilities[node]
        header = f"**{node}**"
        rows = ["state | probability", "--- | ---"]
        rows.extend(f"{state} | {prob:.3f}" for state, prob in state_probs.items())
        tables.append("\n".join([header, *rows]))

    return "\n\n".join(tables)


# --------- Core Gemini call ---------

def _messages_to_contents(messages: List[Message]) -> List[Content]:
    contents: list[Content] = []
    for message in messages:
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
    return contents


def call_gemini(req: ChatRequest) -> str:
    model = GenerativeModel(req.model)
    contents = _messages_to_contents(req.messages)

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


def stream_gemini(req: ChatRequest) -> Iterator[str]:
    model = GenerativeModel(req.model)
    contents = _messages_to_contents(req.messages)

    gen_config = GenerationConfig(
        temperature=req.temperature,
        max_output_tokens=req.max_output_tokens,
    )

    try:
        responses = model.generate_content(
            contents,
            generation_config=gen_config,
            stream=True,
        )
    except Exception as e:
        logger.exception("Gemini streaming call failed")
        raise HTTPException(status_code=500, detail=str(e))

    collected: list[str] = []
    for response in responses:
        text = response.text or ""
        if text:
            collected.append(text)
            yield text

    if not collected:
        raise HTTPException(status_code=500, detail="Empty response from Gemini")

    logger.info("Streamed Gemini reply (%s chars)", len("".join(collected)))


# --------- FastAPI route ---------

@app.post("/chat")
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
        stream = stream_gemini(proxy_request)
    elif body.model == BN_ENHANCED_MODEL_NAME:
        stream = stream_bn_enhanced_response(body)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model {body.model}")

    return StreamingResponse(stream, media_type="text/plain")


def prepare_bn_analysis(body: ChatRequest) -> Tuple[str, ChatRequest, str, str]:
    conversation = render_conversation(body.messages)

    prompt = make_prompt(conversation)

    bn_prompt_request = ChatRequest(
        messages=[Message(role="user", parts=[MessagePart(type="text", text=prompt)])],
        tools={},
        model=VERTEX_MODEL_ID,
    )

    response_text = call_gemini(bn_prompt_request)
    logger.info("BN update request payload: %s", bn_prompt_request.model_dump())
    updates_raw = parse_update_payload(response_text)
    updates = {
        node: value
        for node, value in updates_raw.items()
        if value is not None and node in BN.nodes
    }
    logger.info("Bayesian network updates: %s", updates if updates else "None")

    evidence_map: Dict[str, str] = {}
    for node, value in updates.items():
        if value:
            evidence_map[node] = value
    logger.info("BN evidence map: %s", evidence_map if evidence_map else "None")

    reachable_nodes = reachable_from_evidence(BN.model, list(evidence_map.keys()))
    if not reachable_nodes:
        if evidence_map:
            logger.warning(
                "No reachable BN nodes found for evidence %s; defaulting to remaining nodes.",
                list(evidence_map.keys()),
            )
            reachable_nodes = [
                node for node in BN.nodes if node not in evidence_map
            ]
        else:
            logger.info("No BN evidence provided; querying all nodes.")
            reachable_nodes = list(BN.nodes)

    logger.info("BN nodes selected for probability query: %s", reachable_nodes)
    probabilities = compute_probabilities(reachable_nodes, evidence_map)
    logger.info("Updated probabilities: %s", probabilities if probabilities else "None")

    updates_text = json.dumps(updates, indent=2) if updates else "None"
    probability_text = format_probabilities(probabilities)
    bn_message = make_probability_prompt(
        conversation=conversation,
        updates_text=updates_text,
        probability_text=probability_text,
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

    prefix_text = "\n".join(
        [
            "Bayesian network updates:",
            updates_text,
            "",
            "Updated probabilities:",
            probability_text,
            "",
            "Assistant response:",
            "",
        ]
    )

    return prefix_text, analysis_request, updates_text, probability_text


def handle_bn_enhanced_request(body: ChatRequest) -> str:
    prefix_text, analysis_request, updates_text, probability_text = prepare_bn_analysis(body)
    logger.info("LLM probability request: %s", analysis_request.model_dump())
    llm_reply = call_gemini(analysis_request)
    logger.info("LLM response (with probabilities): %s", llm_reply)

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


def stream_bn_enhanced_response(body: ChatRequest) -> Iterator[str]:
    prefix_text, analysis_request, _, _ = prepare_bn_analysis(body)
    logger.info("LLM probability request (streaming): %s", analysis_request.model_dump())

    # Send the deterministic BN updates/probabilities first.
    yield prefix_text

    collected: list[str] = []
    for chunk in stream_gemini(analysis_request):
        collected.append(chunk)
        yield chunk

    logger.info("LLM BN-enhanced response streamed (%s chars)", len("".join(collected)))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/bayes-graph")
async def bayes_graph():
    dot = Digraph(name="womens_health")
    for node in BN.nodes:
        dot.node(node)
    for start, end in BN.edges:
        dot.edge(start, end)
    image = dot.pipe(format="png")
    return Response(content=image, media_type="image/png")
