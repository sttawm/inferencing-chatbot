"""
Generate conversations from JSONL input using Gemini.

Each input line should be a JSON object. For every line we:
  1) process_json(payload)      -> place to normalize/clean the input.
  2) get_prompt(processed)      -> YOU write this to build the LLM prompt.
  3) query_llm(prompt)          -> calls Gemini and returns its text.
  4) parse_conversation(text)   -> attempts to parse a list of {role, text} messages.

You only need to implement get_prompt; everything else is wired up.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import Content, GenerativeModel, Part

logger = logging.getLogger("conversation-generator")


def load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=False)

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Vertex AI")

    vertexai.init(project=project_id, location=location)


def process_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional place to normalize/clean the input JSON.
    Currently a pass-through.
    """
    return payload


def get_prompt(processed_payload: Dict[str, Any]) -> str:
    """
    TODO: Implement this.
    Build and return a prompt string that instructs the LLM to emit
    a conversation as a JSON list of messages like:
      [{"role": "patient", "text": "..."}, {"role": "clinician", "text": "..."}]
    """
    raise NotImplementedError("Fill in get_prompt(processed_payload) to build the LLM prompt.")


def query_llm(prompt: str, model_id: str = "gemini-2.5-pro") -> str:
    model = GenerativeModel(model_id)
    response = model.generate_content(
        [
            Content(
                role="user",
                parts=[Part.from_text(prompt)],
            )
        ],
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 2048,
        },
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("LLM returned an empty response")
    return text


def parse_conversation_response(text: str) -> List[Dict[str, str]]:
    """
    Tries to parse the LLM output as a JSON list of {role, text} dicts.
    Falls back to wrapping raw text in a single-message list.
    """
    try:
        data = json.loads(text)
        if isinstance(data, list):
            messages: List[Dict[str, str]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                msg_text = item.get("text")
                if role in {"patient", "clinician"} and isinstance(msg_text, str):
                    messages.append({"role": role, "text": msg_text})
            if messages:
                return messages
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON; returning raw text.")

    return [{"role": "assistant", "text": text}]


def process_file(input_path: Path, output_path: Optional[Path]) -> None:
    results: List[str] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Line %d is not valid JSON: %s", line_no, exc)
                continue

            processed = process_json(payload)
            prompt = get_prompt(processed)
            llm_text = query_llm(prompt)
            conversation = parse_conversation_response(llm_text)

            results.append(json.dumps({"input": payload, "conversation": conversation}, ensure_ascii=False))

    if output_path:
        output_path.write_text("\n".join(results), encoding="utf-8")
        logger.info("Wrote %d conversations to %s", len(results), output_path)
    else:
        for row in results:
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake conversations from JSONL using Gemini.")
    parser.add_argument("input", type=Path, help="Path to input JSONL file")
    parser.add_argument("-o", "--output", type=Path, help="Optional output path (JSONL). Defaults to stdout.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
