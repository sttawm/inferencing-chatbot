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
    Combine todos, messages, visit_notes, and qas into one list with a
    "type" field and sort chronologically (oldest first) by key "t".
    """
    combined: List[Dict[str, Any]] = []

    for key in ("todos", "messages", "visit_notes", "qas"):
        items = payload.get(key, [])
        if not isinstance(items, list):
            raise ValueError(f"Expected list for key '{key}', got {type(items)}")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict items in list for key '{key}', got {type(item)}")

            entry = dict(item)
            entry.setdefault("type", key)
            combined.append(entry)

    combined.sort(key=lambda item: item.get("t", 0))
    return combined


def get_prompt(processed_payload: Dict[str, Any]) -> str:
    return """
Convert the attached JSON to a conversation between an ObGyn clinician and a patient.

The attached JSON is an array of objects representing a chronologically ordered list of interactions between a single patient and an ObGyn clinic. Preserve the chronological ordering exactly.

Your job is to convert all objects into a list of conversational turns.

Each conversational turn in the output MUST be a JSON object of the form:
{ “user”: “patient” or “clinic”, “text”: “…” }

The final output MUST be a JSON array of these objects.

Do NOT include any other keys.
Do NOT include comments.
Respond ONLY with the JSON array.

After generating all turns, merge any consecutive messages from the same user into a single conversational turn by concatenating their text with a space between them.

⸻

COMPLETE INPUT SCHEMA EXAMPLE (PSEUDO-JSON WITH COMMENTS)

This is an illustrative example of the full incoming structure (not for output):

[
{
“type”: “questionnaire”,
“question”: “Smoking status”,
“answer”: “Occasional smoker”
},
{
“type”: “message”,
“user_role”: “patient”,
“text”: “I’m having mild cramps.”
},
{
“type”: “visit note”,
“checklists”: {
“pe”: [
{ “name”: “Abdom”, “value”: “Soft, tender LLQ” },
{ “name”: “Breast”, “value”: “Normal exam” }
],
“ros”: [
{ “name”: “GI”, “value”: “Intermittent cramping” },
{ “name”: “General”, “value”: “No fever, no chills” }
]
},
“bullets”: [
{
“category”: “Hpi”,
“text”: “Patient reports 3 days of intermittent cramping.”
},
{
“category”: “Plan”,
“text”: “”,
“note_document”: {
“summary”: “Recommend ibuprofen as needed and monitor symptoms.”
}
}
]
},
{
“type”: “todo”,
“title”: “Hydration”,
“category”: “read”,
“description”: “Hydration can help reduce cramping.”
}
]

⸻

HOW TO CONVERT EACH TYPE
	1.	Questionnaire objects

A questionnaire entry has “type”: “questionnaire”, plus “question” and “answer”.

Convert into two turns:
{ “user”: “clinic”, “text”: “<converted question?>” }
{ “user”: “patient”, “text”: “” }

Important rule: If the question field is a noun-phrase or statement (e.g. “Smoking status”, “Period frequency”, “Mood issues”), convert it into a real natural-sounding question, such as:
“Can you tell me about your smoking status?”
“How often do you get your period?”
“Have you been experiencing any mood issues?”

Preserve meaning faithfully.

⸻

	2.	Visit note objects

A visit note has the following structure:
“type”: “visit note”
“checklists”: { “pe”: […], “ros”: […] }
“bullets”: […]

Convert the entire visit note into a small number of clinic-authored conversational messages summarizing:
– What the patient reported (history, ROS, HPI)
– What the clinician found (physical exam, procedures, tests)
– The plan, recommendations, and follow-up

Checklist items:
Each PE or ROS element is an object containing “name” and “value”.
PE example names: General, Neck, Resp, CVS, Abdom, MSS, NS, Skin, Psych, Pelvic, Rectal, Eyes, HENT, Breast, GU.
ROS example names: General, Eyes, Resp, CVS, GI, GU, NS, HENT, MSS, Skin, Gyne, Psych, Diet, Exercise, Breast, Hemo, Endoc.

Bullets:
Each bullet contains “category” and “text”.
Some bullets may have empty or missing text but include a note_document.summary field.
If text is empty and note_document.summary exists, use the summary as the content.
If both text and summary exist, either may be used, but prefer the clearer one.

All visit-note turns must be attributed to “clinic”, unless a bullet explicitly contains direct patient quotes.

⸻

	3.	Todo objects

A todo object contains:
“type”: “todo”
“title”: “…”
“category”: “…”
“description”: “…”

Convert this into a single clinic message that briefly explains or recommends the task. Always attribute it to user “clinic”.

⸻

	4.	Message objects

A message object contains:
“type”: “message”
“user_role”: “patient” or “provider” or “careguide” or “admin”
“text”: “…”

Map user_role to user:
patient → “patient”
provider, careguide, admin → “clinic”

Keep text as the message content with minimal cleanup.

⸻

FINAL PROCESSING STEP (IMPORTANT)

After generating all turns, merge any consecutive turns by the same user into a single turn.
For example:
{ “user”: “patient”, “text”: “Hello” }
{ “user”: “patient”, “text”: “I have a question” }

Becomes:
{ “user”: “patient”, “text”: “Hello I have a question” }

This merging happens at the very end.

⸻

GENERAL RULES

Preserve chronological order.
Do not contradict the input.
You may add brief conversational transitions like “Thanks for sharing that,” but do not invent medical findings.
Output must be a JSON array containing only objects of the form:
{ “user”: “patient” or “clinic”, “text”: “…” }

Respond ONLY with the JSON array.

Attached JSON:""" + f"""
{json.dumps(processed_payload, ensure_ascii=False, indent=2)}
"""


def query_llm(prompt: str, payload: Dict[str, Any], model_id: str = "gemini-2.5-pro") -> str:
    logger.info("Full prompt being sent to LLM:\n%s", prompt)
    model = GenerativeModel(model_id)
    response = model.generate_content(
        [
            Content(
                role="user",
                parts=[
                    Part.from_text(prompt),
                    Part.from_text(json.dumps(payload, ensure_ascii=False)),
                ],
            )
        ],
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 50000,
        },
    )
    text = (response.text or "").strip()
    logger.info("Raw LLM response:\n%s", text)
    if not text:
        raise RuntimeError("LLM returned an empty response")
    return text


def parse_conversation_response(text: str) -> List[Dict[str, str]]:
    """
    Tries to parse the LLM output as a JSON list of {role, text} dicts.
    Falls back to wrapping raw text in a single-message list.
    """

    def _extract_first_json_array(s: str) -> str:
        start = s.find("[")
        if start == -1:
            return s
        depth = 0
        for idx in range(start, len(s)):
            if s[idx] == "[":
                depth += 1
            elif s[idx] == "]":
                depth -= 1
                if depth == 0:
                    return s[start : idx + 1]
        return s

    stripped = text.strip()
    if stripped.startswith("```"):
        # Attempt to extract fenced JSON
        start_brace = stripped.find("{")
        end_brace = stripped.rfind("}")
        start_bracket = stripped.find("[")
        end_bracket = stripped.rfind("]")

        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            stripped = stripped[start_brace : end_brace + 1].strip()
        elif start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
            stripped = stripped[start_bracket : end_bracket + 1].strip()

    def _parse_json(json_str: str) -> List[Dict[str, str]]:
        data = json.loads(json_str)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON list at the top level")
        messages: List[Dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Expected list items to be JSON objects")
            role = item.get("role")
            msg_text = item.get("text")
            if role in {"patient", "clinician"} and isinstance(msg_text, str):
                messages.append({"role": role, "text": msg_text})
            else:
                raise ValueError("Each message must have 'role' and 'text' fields")
        return messages

    try:
        return _parse_json(stripped)
    except json.JSONDecodeError:
        # Attempt to slice out first JSON array if extra text surrounds it
        candidate = _extract_first_json_array(stripped)
        try:
            return _parse_json(candidate.strip())
        except Exception as exc:
            logger.warning("Bracket-sliced parse failed: %s", exc)
        logger.warning("Failed to parse LLM response as JSON. Response was:\n%s", text)
        raise ValueError("Failed to parse LLM response as JSON.")
    except Exception as exc:
        logger.warning("Failed to parse LLM response as JSON. Response was:\n%s", text)
        raise ValueError(f"Failed to parse LLM response: {exc}")

def process_file(
    input_path: Path,
    output_dir: Path,
    limit: Optional[int],
    start_index: Optional[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Opening input file: %s", input_path)

    processed_counter = 0  # counts non-empty processed records seen (including skipped by start)
    output_counter = 0  # counts conversations actually generated

    raw_text = input_path.read_text(encoding="utf-8")
    stripped = raw_text.lstrip()

    if stripped.startswith("["):
        try:
            payloads = json.loads(raw_text)
            if not isinstance(payloads, list):
                raise ValueError("Top-level JSON is not a list")
            records = list(enumerate(payloads, start=1))
        except Exception as exc:
            logger.error("Input file is not valid JSON array: %s", exc)
            return
    else:
        records = []
        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                logger.error("Line %d is not valid JSON: %s", line_no, exc)
                continue

    for line_no, payload in records:
        processed = process_json(payload)

        if not processed:
            logger.info("Line %d skipped because messages array is empty.", line_no)
            continue

        if start_index is not None and processed_counter < start_index:
            processed_counter += 1
            continue

        prompt = get_prompt(processed)
        llm_text = query_llm(prompt, processed)
        conversation = parse_conversation_response(llm_text)

        processed_counter += 1
        if limit is not None and output_counter >= limit:
            break
        record = {
            "input": processed,
            "output": conversation,
        }
        out_path = output_dir / f"conversation_{processed_counter:04d}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        output_counter += 1
        logger.info("Wrote conversation %d to %s", output_counter, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake conversations from JSONL using Gemini.")
    parser.add_argument("input", type=Path, help="Path to input JSONL file")
    parser.add_argument(
        "-d",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write one conversation JSON file per input line.",
    )
    parser.add_argument("-n", "--limit", type=int, help="Process only the first N records.")
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        help="Skip to this zero-based line index before processing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_env()

    process_file(args.input, args.output_dir, args.limit, args.start)


if __name__ == "__main__":
    main()
