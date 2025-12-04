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

The attached JSON is an array of objects representing a chronologically ordered list of interactions between a single patient and an ObGyn clinic. Preserve the chronological ordering.

There are four types of interaction, given by the top-level “type” field of each object: “questionnaire”, “visit note”, “todo”, or “message”.

Your job is to convert all of these into a list of conversational turns.

Each conversational turn in the output MUST be a JSON object of the form:
{ “user”: “patient” or “clinic”, “text”: “…” }

The final output MUST be a JSON array of these objects, for example:
[
{ “user”: “clinic”, “text”: “…” },
{ “user”: “patient”, “text”: “…” }
]

Do NOT include any other keys. Do NOT include any explanations or comments. Respond ONLY with JSON.

⸻

DETAILS OF INPUT TYPES

⸻

1. Questionnaire objects

An object of type “questionnaire” looks like:
{
“type”: “questionnaire”,
“question”: “…”,
“answer”: “…”
}

Convert this into two conversational turns:
	•	A clinic turn that asks the question.
	•	A patient turn giving the answer.

Important rule:
If the “question” text is phrased as a statement (e.g. “Smoking status”, “Period frequency”, “Mood issues”), you MUST convert it into a natural-sounding question that captures the intended meaning (e.g. “Can you tell me about your smoking status?”, “How often do you get your period?”, “Have you been experiencing any mood issues?”).
Do NOT leave it as a statement. All questionnaire items must be rendered as genuine questions.
Keep the meaning as close as possible.

Use natural language but stay faithful to the intent of the original question and answer.

⸻

2. Visit note objects

An object of type “visit note” looks like:
{
“type”: “visit note”,
“checklists”: {
“pe”: { },
“ros”: { }
},
“bullets”: []
}

The elements of “pe” and “ros” are objects of the form:
{ “name”: “…”, “value”: “…” }

The possible names for “pe” are:
General, Neck, Resp, CVS, Abdom, MSS, NS, Skin, Psych, Pelvic, Rectal, Eyes, HENT, Breast, GU.

The possible names for “ros” are:
General, Eyes, Resp, CVS, GI, GU, NS, HENT, MSS, Skin, Gyne, Psych, Diet, Exercise, Breast, Hemo, Endoc.

The elements of “bullets” are of the form:
{ “category”: “…”, “text”: “…”, … }

The possible categories are:
Allergies, Assessment, Assessplan, Data, Family, Followup, Habits, Hpi, Instr, Med, Narrative, Objective, Orders, PE, Past, Plan, Problem, Procedure, ROS, Reason, Referenced, Social, Surgical, Test, Tx.

If a bullet has an empty or missing “text” but has “note_document.summary” (e.g.
{ “category”: “…”, “text”: “”, “note_document”: { “summary”: “…” } }),
then use “note_document.summary”. If both are present, pick either one.

Convert a “visit note” into a small number of conversational turns that summarize:
	•	What the patient reported (history, ROS).
	•	What the clinician found or did (PE, procedures, tests).
	•	The plan, recommendations, and follow-up.

Use clear conversational language, combining related checklist items and bullets into coherent sentences. Attribute these turns to:
{ “user”: “clinic”, “text”: “…” }
unless it is clearly something the patient said in their own words.

⸻

3. Todo objects

A “todo” object looks like:
{
“type”: “todo”,
“title”: “Low Libido”,
“category”: “read”,
“description”: “Low libido can have…”
}

Convert this into a clinic message to the patient that briefly explains the todo item and, if relevant, references the “description”. For example, it might sound like an instruction or recommendation from the clinic.

Output as:
{ “user”: “clinic”, “text”: “…” }

⸻

4. Message objects

A “message” object looks like:
{
“type”: “message”,
“user_role”: “patient” or “careguide” or “provider” or “admin”,
“text”: “…”
}

Map “user_role” to the “user” field as follows:
	•	If user_role == “patient”, then user = “patient”.
	•	For “careguide”, “provider”, or “admin”, set user = “clinic”.

Keep the “text” as the content of the turn, with minimal, neutral cleanup if needed.

⸻

GENERAL RULES
	•	Preserve the chronological order of the input array.
	•	Do not invent facts that conflict with the input, but you may add brief connecting phrases to make the conversation flow naturally.
	•	Keep the medical content accurate, but you may simplify clinical jargon into patient-friendly language.
	•	All output must be a single JSON array of objects of the form:
{ “user”: “patient” or “clinic”, “text”: “…” }
	•	Respond ONLY with this JSON array.

⸻

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
    stripped = text.strip()
    if stripped.startswith("```"):
        # Attempt to extract fenced JSON
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1].strip()

    try:
        data = json.loads(stripped)
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
        raise ValueError("Failed to parse LLM response as JSON. Response was:\n" + text)
        x
    return [{"role": "assistant", "text": text}]


def process_file(
    input_path: Path,
    output_dir: Path,
    limit: Optional[int],
    start_index: Optional[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Opening input file: %s", input_path)

    processed_counter = 0  # counts only non-empty message arrays
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(results) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Line %d is not valid JSON: %s", line_no, exc)
                continue

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
            record = {
                "input": payload,
                "conversation": conversation,
            }
            out_path = output_dir / f"conversation_{processed_counter:04d}.json"
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Wrote conversation %d to %s", processed_counter, out_path)


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
