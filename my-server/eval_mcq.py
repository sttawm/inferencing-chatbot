import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from main import (
    ChatRequest,
    Message,
    MessagePart,
    BASELINE_MODEL_NAME,
    BN_ENHANCED_MODEL_NAME,
    VERTEX_MODEL_ID,
    call_gemini,
    handle_bn_enhanced_request,
)

# Fill this in with your actual evaluation prompt.
# It should contain the placeholder "{question_block}" where the formatted question/options will be injected.
PROMPT_TEMPLATE = """
You will be given a multiple-choice question in the following format:

Question: "<question text>"

Choices:
(A) <option A>
(B) <option B>
(C) <option C>
(D) <option D>

Your task:
	1.	Read the entire block exactly as given.
	2.	Choose the single best answer: A, B, C, or D.
	3.	Respond only with a the letter in this form:

A

(Replace "A" with "B", "C", or "D" as appropriate.)

Strict rules:
	•	Do NOT output the answer text — only the letter.
	•	Do NOT provide explanations.
	•	Do NOT rewrite, summarize, or comment on the question.
	•	Do NOT output anything except the JSON object.

You will now receive the question block. 

{question_block}
""".strip()


def build_question_block(question_obj: Dict[str, Any], seed: int | None = None) -> Dict[str, Any]:
    question = question_obj["question"]
    correct = question_obj["correct_answer"]
    incorrect = question_obj.get("incorrect_answers", [])

    options = [{"text": correct, "label": "correct", "is_correct": True}]
    options.extend({"text": opt, "label": f"incorrect_{i}", "is_correct": False} for i, opt in enumerate(incorrect))

    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(options)

    letters = ["A", "B", "C", "D"]
    formatted_lines = [f"Question: {question}", ""]
    letter_map: Dict[str, Dict[str, Any]] = {}
    for letter, opt in zip(letters, options):
        formatted_lines.append(f"({letter}) {opt['text']}")
        letter_map[letter] = opt

    block = "\n".join(formatted_lines)
    correct_letter = next(letter for letter, opt in letter_map.items() if opt["is_correct"])
    return {"block": block, "correct_letter": correct_letter, "letter_map": letter_map}


def parse_letter(response_text: str) -> str:
    for ch in response_text:
        if ch.upper() in {"A", "B", "C", "D"}:
            return ch.upper()
        else:
            raise ValueError("Response is not a valid answer letter. Response was:\n" + response_text)
    raise ValueError("Response is empty. Response was:\n" + response_text)


def evaluate_questions(
    questions: List[Dict[str, Any]],
    seed: int | None = None,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    failures = 0
    total_attempted = 0
    for idx, q in enumerate(questions, start=1):
        if failures > 5:
            logging.error("Aborting: more than 5 failures encountered.")
            break
        attempt = 0
        while attempt < 3:
            attempt += 1
            try:
                total_attempted += 1
                formatted = build_question_block(q, seed=seed + idx if seed is not None else None)
                prompt = PROMPT_TEMPLATE.format(question_block=formatted["block"])

                logging.info("Q%d/%d (attempt %d) | Asking baseline: %s", idx, len(questions), attempt, q["question"])
                baseline_body = ChatRequest(
                    messages=[Message(role="user", parts=[MessagePart(type="text", text=prompt)])],
                    tools={},
                    model=VERTEX_MODEL_ID,
                    temperature=0.2,
                    max_output_tokens=2048,
                )
                baseline_reply = call_gemini(baseline_body)
                logging.info("Baseline full reply:\n%s", baseline_reply)
                baseline_letter = parse_letter(baseline_reply)
                if baseline_letter not in {"A", "B", "C", "D"}:
                    raise ValueError("Baseline reply did not contain a valid option letter")

                logging.info("Q%d/%d (attempt %d) | Asking BN model: %s", idx, len(questions), attempt, q["question"])
                bn_body = ChatRequest(
                    messages=[Message(role="user", parts=[MessagePart(type="text", text=prompt)])],
                    tools={},
                    model=BN_ENHANCED_MODEL_NAME,
                    temperature=0.2,
                    max_output_tokens=2048,
                )
                bn_reply_full = handle_bn_enhanced_request(bn_body)
                if "Assistant response:" in bn_reply_full:
                    bn_reply = bn_reply_full.split("Assistant response:", 1)[1].strip()
                else:
                    bn_reply = bn_reply_full.strip()
                logging.info("BN full reply (assistant section):\n%s", bn_reply)
                bn_letter = parse_letter(bn_reply)
                if bn_letter not in {"A", "B", "C", "D"}:
                    raise ValueError("BN reply did not contain a valid option letter")

                result = {
                    "question": q["question"],
                    "options": formatted["letter_map"],
                    "correct_letter": formatted["correct_letter"],
                    "baseline": {"reply": baseline_reply, "letter": baseline_letter},
                    "bn": {"reply": bn_reply, "letter": bn_letter},
                    "is_baseline_correct": baseline_letter == formatted["correct_letter"],
                    "is_bn_correct": bn_letter == formatted["correct_letter"],
                }
                results.append(result)

                logging.info(
                    "Q%d/%d done. Baseline: %s (%s). BN: %s (%s). Remaining: %d",
                    idx,
                    len(questions),
                    baseline_letter,
                    "correct" if result["is_baseline_correct"] else "wrong",
                    bn_letter,
                    "correct" if result["is_bn_correct"] else "wrong",
                    len(questions) - idx,
                )
                break
            except Exception as exc:
                logging.warning("Q%d/%d attempt %d failed: %s", idx, len(questions), attempt, exc)
                if attempt >= 3:
                    failures += 1
                    logging.error("Skipping question after 3 failed attempts: %s", q["question"])
                    break

        summary = {
            "total": len(results),
            "baseline_correct": sum(r["is_baseline_correct"] for r in results),
            "bn_correct": sum(r["is_bn_correct"] for r in results),
            "failures": failures,
            "results": results,
        }
    return summary


def load_questions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input file must be a JSON list of question objects")
    required_keys = {"question", "correct_answer", "incorrect_answers"}
    for q in data:
        if not required_keys.issubset(q):
            raise ValueError(f"Question missing required keys: {q}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs BN Gemini models on MCQ data.")
    parser.add_argument("in_domain", type=Path, help="Path to JSON file with in-domain MCQ questions.")
    parser.add_argument("out_domain", type=Path, help="Path to JSON file with out-of-domain MCQ questions.")
    parser.add_argument("-o", "--output", type=Path, help="Optional output JSON path.")
    parser.add_argument("--seed", type=int, help="Optional seed for deterministic shuffling.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Reduce chatter from the main app logger unless explicitly desired.
    logging.getLogger("gemini-fastapi").setLevel(logging.WARNING)
    in_questions = load_questions(args.in_domain)
    out_questions = load_questions(args.out_domain)

    logging.info("Evaluating in-domain set (%d questions)...", len(in_questions))
    in_summary = evaluate_questions(in_questions, seed=args.seed)

    logging.info("Evaluating out-of-domain set (%d questions)...", len(out_questions))
    out_summary = evaluate_questions(out_questions, seed=args.seed)

    summary = {
        "in_domain": in_summary,
        "out_domain": out_summary,
    }

    if args.output:
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote results to %s", args.output)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
