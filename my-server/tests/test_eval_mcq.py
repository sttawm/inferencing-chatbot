import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from eval_mcq import build_question_block


def test_build_question_block_formats_question_with_letters_and_correct_mapping():
    q = {
        "question": "What is the color of the sky?",
        "correct_answer": "Blue",
        "incorrect_answers": ["Green", "Red", "Yellow"],
    }

    formatted = build_question_block(q, seed=1)

    expected = "\n".join(
        [
            "Question: What is the color of the sky?",
            "",
            "(A) Yellow",
            "(B) Blue",
            "(C) Red",
            "(D) Green",
        ]
    )

    assert formatted["block"] == expected
    assert formatted["correct_letter"] == "B"
    assert formatted["letter_map"]["B"]["text"] == "Blue"
