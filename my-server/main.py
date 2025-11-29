import json
import logging
import os
from io import StringIO
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pgmpy.readwrite import BIFReader

import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

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


# --------- Bayesian network setup ---------

BN_CANCER_BIF = """
network "cancer" { }

variable "Pollution" { type discrete [ 2 ] { "low" "high" }; }
variable "Smoking" { type discrete [ 2 ] { "true" "false" }; }
variable "Cancer" { type discrete [ 2 ] { "true" "false" }; }
variable "Xray" { type discrete [ 2 ] { "positive" "negative" }; }
variable "Dyspnoea" { type discrete [ 2 ] { "true" "false" }; }

probability ( "Pollution" ) {
    table 0.9, 0.1;
}

probability ( "Smoking" ) {
    table 0.5, 0.5;
}

probability ( "Cancer" | "Pollution", "Smoking" ) {
    ( "low", "true" ) 0.03, 0.97;
    ( "low", "false" ) 0.001, 0.999;
    ( "high", "true" ) 0.05, 0.95;
    ( "high", "false" ) 0.02, 0.98;
}

probability ( "Xray" | "Cancer" ) {
    ( "true" ) 0.9, 0.1;
    ( "false" ) 0.2, 0.8;
}

probability ( "Dyspnoea" | "Cancer" ) {
    ( "true" ) 0.65, 0.35;
    ( "false" ) 0.3, 0.7;
}
"""

BN_MODEL = BIFReader(string=BN_CANCER_BIF).get_model()
BN_NODES = list(BN_MODEL.nodes())
BN_EDGES = list(BN_MODEL.edges())


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


# --------- Core Gemini call ---------

def call_gemini(req: ChatRequest) -> str:
    model = GenerativeModel(req.model)

    # Convert messages to Vertex contents
    contents: list[Content] = []
    for message in req.messages:
        # Vertex uses "user" / "model" roles
        role = "user" if message.role in ("user", "system") else "model"
        text_chunks: List[str] = []
        for part in message.parts:
            if part.type == "text" and part.text:
                text_chunks.append(part.text)
            elif part.text:
                text_chunks.append(part.text)
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
    """
    Pass-through chat endpoint.

    body.messages: [{role, content}]
    body.model: e.g. "gemini-1.5-flash", "gemini-1.5-pro"
    """
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
        raise HTTPException(status_code=400, detail=f"Unknown model {body.model}")

    logger.info("Sending reply (%s chars): %s", len(reply), reply)
    return ChatResponse(reply=reply)


def handle_bn_enhanced_request(body: ChatRequest) -> str:
    conversation = render_conversation(body.messages)

    prompt = f"""
You are analyzing a Bayesian network with the following variables: {', '.join(BN_NODES)}

Conversation transcript:
{conversation}

Based on the conversation, determine the current boolean state for each variable.
Respond ONLY with a JSON object where keys are variable names and values are true, false, or null (if unknown).
""".strip()

    bn_prompt_request = ChatRequest(
        messages=[Message(role="user", parts=[MessagePart(type="text", text=prompt)])],
        tools={},
        model=VERTEX_MODEL_ID,
    )

    response_text = call_gemini(bn_prompt_request)
    updates = parse_update_payload(response_text)

    summary_lines = [
        "Inferred Bayesian network assignments:",
        json.dumps(updates, indent=2),
    ]
    return "\n".join(summary_lines)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
