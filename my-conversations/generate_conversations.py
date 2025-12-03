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
    return """
Convert the attached JSON to a conversation between an ObGyn clinician and a patient.

The attached JSON is an array of objects representing a chronologically ordered list of interactions between a single patient and an ObGyn clinic. Preserve the chronological ordering.

There are four types of interaction, given by the top-level "type" field of each object: "questionnaire", "visit note", "todo", or "message".

Your job is to convert all of these into a list of conversational turns.

Each conversational turn in the output MUST be a JSON object of the form:
{ "user": "patient" or "clinic", "text": "..." }

The final output MUST be a JSON array of these objects, for example:
[
  { "user": "clinic", "text": "..." },
  { "user": "patient", "text": "..." }
]

Do NOT include any other keys. Do NOT include any explanations or comments. Respond ONLY with JSON.

---

DETAILS OF INPUT TYPES

1. Questionnaire objects

An object of type "questionnaire" looks like:
{
  "type": "questionnaire",
  "question": "...",
  "answer": "..."
}

Convert this into two conversational turns:
- A clinic turn asking the question.
- A patient turn giving the answer.

Use natural language but keep the meaning of "question" and "answer" as close as possible.

2. Visit note objects

An object of type "visit note" looks like:
{
  "type": "visit note",
  "checklists": {
    "pe": { },
    "ros": { }
  },
  "bullets": []
}

The elements of "pe" and "ros" are objects of the form:
{ "name": "...", "value": "..." }

The possible names for "pe" are:
General, Neck, Resp, CVS, Abdom, MSS, NS, Skin, Psych, Pelvic, Rectal, Eyes, HENT, Breast, GU.

The possible names for "ros" are:
General, Eyes, Resp, CVS, GI, GU, NS, HENT, MSS, Skin, Gyne, Psych, Diet, Exercise, Breast, Hemo, Endoc.

The elements of "bullets" are of the form:
{ "category": "...", "text": "...", ... }

The possible categories are:
Allergies, Assessment, Assessplan, Data, Family, Followup, Habits, Hpi, Instr, Med, Narrative, Objective, Orders, PE, Past, Plan, Problem, Procedure, ROS, Reason, Referenced, Social, Surgical, Test, Tx.

If a bullet has an empty or missing "text" but has "note_document.summary" (e.g.
{ "category": "...", "text": "", "note_document": { "summary": "..." } }),
then use "note_document.summary" as the content. If both are present, pick either one.

Convert a "visit note" into a small number of conversational turns that summarize:
- What the patient reported (history, ROS).
- What the clinician found or did (PE, procedures, tests).
- The plan, recommendations, and follow-up.

Use clear conversational language, combining related checklist items and bullets into coherent sentences. Attribute these turns to:
{ "user": "clinic", "text": "..." }
unless it is clearly something the patient is saying in their own words.

3. Todo objects

A "todo" object looks like:
{
  "type": "todo",
  "title": "Low Libido",
  "category": "read",
  "description": "Low libido can have..."
}

Convert this into a clinic message to the patient that briefly explains the todo item and, if relevant, references the "description". For example, it might sound like an instruction or recommendation from the clinic.

Output as:
{ "user": "clinic", "text": "..." }

4. Message objects

A "message" object looks like:
{
  "type": "message",
  "user_role": "patient" or "careguide" or "provider" or "admin",
  "text": "..."
}

Map "user_role" to the "user" field as follows:
- If user_role == "patient", then user = "patient".
- For "careguide", "provider", or "admin", set user = "clinic".

Keep the "text" as the content of the turn, with minimal, neutral cleanup if needed.

---

GENERAL RULES

- Preserve the overall chronology of the input array.
- Do not invent facts that conflict with the input, but you may add brief connecting phrases to make the conversation flow naturally.
- Keep the medical content accurate, but you may simplify clinical jargon into patient-friendly language.
- Output must be a single JSON array of objects of the form:
  { "user": "patient" or "clinic", "text": "..." }
- Respond ONLY with this JSON array, with no additional explanation or formatting.

Attached JSON:
{json.dumps(processed_payload, ensure_ascii=False, indent=2)}
    """


def query_llm(prompt: str, payload: Dict[str, Any], model_id: str = "gemini-2.5-pro") -> str:
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
            "temperature": 0.2
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
            llm_text = query_llm(prompt, processed)
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
