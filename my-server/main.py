import json
import logging
import os
import time
import re
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from graphviz import Digraph
from pydantic import BaseModel, Field

from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling
import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

from womens_health import load_womens_health_bayes_net

from prompt import make_bn_extraction_prompt, make_probability_prompt

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
SAMPLER = BayesianModelSampling(BN.model)
FORWARD_SAMPLE_SIZE = 20000


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
    max_output_tokens: int = 4096


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
    text = raw_text.strip()

    # 1) Attempt direct JSON parse.
    candidates: list[str] = [text]

    # 2) Attempt fenced blocks, prefer the last one if multiple exist.
    fence_matches = re.findall(
        r"```(?:json)?\\s*({[\\s\\S]*?})\\s*```", raw_text, flags=re.IGNORECASE
    )
    if fence_matches:
        candidates.append(fence_matches[-1])

    # 3) Fallback: first-to-last brace slice.
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw_text[start : end + 1])

    payload = None
    last_error: Optional[Exception] = None
    for snippet in candidates:
        try:
            payload = json.loads(snippet)
            break
        except Exception as exc:
            last_error = exc

    if payload is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "LLM response was not valid JSON. "
                f"Raw response: {text or '<empty>'}. "
                f"Error: {last_error}"
            ),
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
) -> Tuple[Dict[str, Dict[str, float]], float]:
    return run_bn_inference(variables, evidence_map)


def run_bn_inference(
    variables: List[str], evidence_map: Dict[str, str]
) -> Tuple[Dict[str, Dict[str, float]], float]:
    if not variables:
        raise HTTPException(
            status_code=500,
            detail="No BN variables available for probability query",
        )

    start_time = time.perf_counter()
    logger.info("Sampling BN probabilities with evidence: %s", evidence_map or {})
    try:
        if evidence_map:
            evidence_states = [State(var, val) for var, val in evidence_map.items()]
            df = SAMPLER.likelihood_weighted_sample(
                evidence=evidence_states,
                size=FORWARD_SAMPLE_SIZE,
                show_progress=False,
            )
        else:
            df = SAMPLER.forward_sample(
                size=FORWARD_SAMPLE_SIZE,
                show_progress=False,
            )
    except Exception as exc:
        logger.exception("Failed to sample probabilities: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to sample probabilities")

    duration_ms = (time.perf_counter() - start_time) * 1000

    probabilities: Dict[str, Dict[str, float]] = {}
    for node in sorted(variables):
        if node not in df.columns:
            continue
        state_names = BN.model.get_cpds(node).state_names[node]
        if "_weight" in df:
            weighted = df.groupby(node)["_weight"].sum()
            total_weight = df["_weight"].sum()
            counts = weighted / total_weight if total_weight else weighted
        else:
            counts = df[node].value_counts(normalize=True)
        probabilities[node] = {
            state: float(counts.get(state, 0.0)) for state in state_names
        }

    if not probabilities:
        raise HTTPException(
            status_code=500,
            detail="BN probability query returned no results",
        )

    return probabilities, duration_ms


def run_exact_bn_inference(
    variables: List[str], evidence_map: Dict[str, str]
) -> Tuple[Dict[str, Dict[str, float]], float]:
    if not variables:
        raise HTTPException(
            status_code=500,
            detail="No BN variables available for probability query",
        )

    start_time = time.perf_counter()
    try:
        factors = BN.inference.query(
            variables=sorted(variables),
            evidence=evidence_map or None,
            joint=False,
            show_progress=False,
        )
        if not isinstance(factors, dict):
            factors = {variables[0]: factors}
    except Exception as exc:
        logger.exception("Failed exact BN inference: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to compute probabilities")

    duration_ms = (time.perf_counter() - start_time) * 1000
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

    return probabilities, duration_ms


def format_probabilities(probabilities: Dict[str, Dict[str, float]]) -> str:
    if not probabilities:
        return "No probabilistic updates calculated."

    tables: list[str] = []
    for node in sorted(probabilities.keys()):
        state_probs = probabilities[node]
        header = f"**{node}**"
        rows = ["state | probability", "--- | ---"]
        rows.extend(f"{state} | {prob:.2f}" for state, prob in state_probs.items())
        tables.append("\n".join([header, *rows]))

    return "\n\n".join(tables)


def compute_probability_deltas(
    baseline: Dict[str, Dict[str, float]],
    updated: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    deltas: Dict[str, Dict[str, float]] = {}
    for node in sorted(updated.keys()):
        baseline_states = baseline.get(node)
        if not baseline_states:
            continue
        delta_states: Dict[str, float] = {}
        for state, updated_value in updated[node].items():
            delta_states[state] = updated_value - baseline_states.get(state, 0.0)
        deltas[node] = delta_states
    return deltas


def format_probability_deltas(deltas: Dict[str, Dict[str, float]]) -> str:
    if not deltas:
        return "No probabilistic updates calculated."

    tables: list[str] = []
    sorted_nodes = sorted(
        deltas.keys(),
        key=lambda n: sum(abs(v) for v in deltas[n].values()),
        reverse=True,
    )
    for node in sorted_nodes:
        sorted_states = dict(
            sorted(deltas[node].items(), key=lambda x: x[1], reverse=True)
        )
        header = f"**{node}**"
        rows = ["state | delta", "--- | ---"]
        rows.extend(
            f"{state} | {delta:+.2f}" for state, delta in sorted_states.items()
        )
        tables.append("\n".join([header, *rows]))
    return "\n\n".join(tables)


def format_inference_timing(
    evidence_llm_ms: float,
    baseline_ms: float,
    evidence_ms: float,
    analysis_llm_ms: Optional[float],
) -> str:
    total_ms = evidence_llm_ms + baseline_ms + evidence_ms + (analysis_llm_ms or 0.0)
    lines = [
        f"LLM evidence extraction: {evidence_llm_ms:.2f} ms",
        f"BN inference (no evidence): {baseline_ms:.2f} ms",
        f"BN inference (with evidence): {evidence_ms:.2f} ms",
    ]
    if analysis_llm_ms is not None:
        lines.append(f"LLM answer (with probabilities): {analysis_llm_ms:.2f} ms")
    else:
        lines.append("LLM answer (with probabilities): streaming / not timed")
    lines.append(f"Total: {total_ms:.2f} ms")
    return "\n".join(lines)


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


def prepare_bn_analysis(body: ChatRequest) -> Tuple[ChatRequest, str, str, str, float, float, float]:
    conversation = render_conversation(body.messages)

    prompt = make_bn_extraction_prompt(conversation)

    bn_prompt_request = ChatRequest(
        messages=[Message(role="user", parts=[MessagePart(type="text", text=prompt)])],
        tools={},
        model=VERTEX_MODEL_ID,
        temperature=0.0,
    )

    logger.info("BN update request payload: %s", bn_prompt_request.model_dump())
    start_evidence_llm = time.perf_counter()
    response_text = call_gemini(bn_prompt_request)
    logger.info("LLM BN extraction raw response: %s", response_text)
    evidence_llm_ms = (time.perf_counter() - start_evidence_llm) * 1000

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

    query_nodes = [node for node in BN.nodes if node not in evidence_map]
    if not query_nodes:
        raise HTTPException(
            status_code=500,
            detail="All BN nodes provided as evidence; no nodes left to query.",
        )

    logger.info("BN nodes selected for probability query: %s", query_nodes)
    baseline_probs, baseline_ms = compute_probabilities(query_nodes, {})
    probabilities, evidence_ms = compute_probabilities(query_nodes, evidence_map)
    logger.info("Updated probabilities: %s", probabilities if probabilities else "None")

    deltas = compute_probability_deltas(baseline_probs, probabilities)
    logger.info("Probability deltas: %s", deltas if deltas else "None")
    logger.info(
        "BN inference timing -> baseline: %.2f ms, with evidence: %.2f ms",
        baseline_ms,
        evidence_ms,
    )

    updates_text = json.dumps(updates, indent=2) if updates else "None"
    probability_text = format_probabilities(probabilities)
    delta_text = format_probability_deltas(deltas)
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
        temperature=0.0,
        max_output_tokens=body.max_output_tokens,
    )

    return (
        analysis_request,
        updates_text,
        probability_text,
        delta_text,
        evidence_llm_ms,
        baseline_ms,
        evidence_ms,
    )


def handle_bn_enhanced_request(body: ChatRequest) -> str:
    (
        analysis_request,
        updates_text,
        probability_text,
        delta_text,
        evidence_llm_ms,
        baseline_ms,
        evidence_ms,
    ) = prepare_bn_analysis(body)
    logger.info("LLM probability request: %s", analysis_request.model_dump())
    start_analysis_llm = time.perf_counter()
    llm_reply = call_gemini(analysis_request)
    analysis_llm_ms = (time.perf_counter() - start_analysis_llm) * 1000
    logger.info("LLM response (with probabilities) raw: %s", llm_reply)

    inference_text = format_inference_timing(
        evidence_llm_ms=evidence_llm_ms,
        baseline_ms=baseline_ms,
        evidence_ms=evidence_ms,
        analysis_llm_ms=analysis_llm_ms,
    )

    summary_lines = [
        "Bayesian network updates:",
        updates_text,
        "",
        "Updated probabilities:",
        probability_text,
        "",
        "Probability deltas:",
        delta_text,
        "",
        "Inference timing:",
        inference_text,
        "",
        "Assistant response:",
        llm_reply,
    ]
    return "\n".join(summary_lines)


def stream_bn_enhanced_response(body: ChatRequest) -> Iterator[str]:
    (
        analysis_request,
        updates_text,
        probability_text,
        delta_text,
        evidence_llm_ms,
        baseline_ms,
        evidence_ms,
    ) = prepare_bn_analysis(body)
    logger.info("LLM probability request (streaming): %s", analysis_request.model_dump())

    inference_text = format_inference_timing(
        evidence_llm_ms=evidence_llm_ms,
        baseline_ms=baseline_ms,
        evidence_ms=evidence_ms,
        analysis_llm_ms=None,
    )

    prefix_text = "\n".join(
        [
            "Bayesian network updates:",
            updates_text,
            "",
            "Updated probabilities:",
            probability_text,
            "",
            "Probability deltas:",
            delta_text,
            "",
            "Inference timing:",
            inference_text,
            "",
            "Assistant response:",
            "",
        ]
    )

    # Send the deterministic BN updates/probabilities first.
    yield prefix_text

    collected: list[str] = []
    for chunk in stream_gemini(analysis_request):
        collected.append(chunk)
        yield chunk

    logger.info("LLM BN-enhanced response streamed (%s chars)", len("".join(collected)))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/bayes-graph")
async def bayes_graph():
    dot = Digraph(name="womens_health")
    for node in BN.nodes:
        dot.node(node)
    for start, end in BN.edges:
        dot.edge(start, end)
    image = dot.pipe(format="png")
    return Response(content=image, media_type="image/png")
