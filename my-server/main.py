import logging
import os
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

app = FastAPI(title="Gemini Pass-Through API (Vertex)")


# --------- Pydantic models ---------

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gemini-2.5-pro"
    temperature: float = 0.3
    max_output_tokens: int = 1024


class ChatResponse(BaseModel):
    reply: str


# --------- Core Gemini call ---------

def call_gemini(req: ChatRequest) -> str:
    model = GenerativeModel(req.model)

    # Convert messages to Vertex contents
    contents: list[Content] = []
    for m in req.messages:
        # Vertex uses "user" / "model" roles
        role = "user" if m.role in ("user", "system") else "model"
        if not m.content.strip():
            continue
        contents.append(
            Content(
                role=role,
                parts=[Part.from_text(m.content)],
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
    payload = body.model_dump()
    logger.info(
        "Received chat request: model=%s messages=%s",
        payload.get("model"),
        len(payload.get("messages", [])),
    )
    logger.debug("Full payload: %s", payload)

    reply = call_gemini(body)
    logger.info("Sending Gemini reply (%s chars)", len(reply))
    return ChatResponse(reply=reply)
