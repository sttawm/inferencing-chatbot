from womens_health import BN_VARIABLES_DICT

def to_compact_dsl(var_dict: dict[str, list[str]]) -> str:
    """
    Convert BN variable definitions into a compact DSL:

    Metabolic_Imbalance ∈ {"false", "true"}
    Age ∈ {"child", "teen", "20s", ...}
    """
    lines = []
    for var, values in var_dict.items():
        val_list = ", ".join(f'"{v}"' for v in values)
        lines.append(f"{var} ∈ {{{val_list}}}")
    return "\n".join(lines)

def make_prompt(conversation: str) -> str:
    """
    Build an LLM prompt instructing the model to infer BN variable values
    from a conversation, using only the allowed categorical values.
    """
    dsl = to_compact_dsl(BN_VARIABLES_DICT)
    return f"""
You are analyzing a Bayesian network with the following variables and allowed values:

{dsl}

You must choose exactly ONE allowed value for any variable you decide to return.
If a variable cannot be inferred from the conversation, set its value to null.

Your response must be a valid JSON object:
- keys = variable names
- values = one of the allowed values, or null
- no extra text, no explanations

Conversation:
{conversation}

Respond ONLY with JSON.
""".strip()

def make_probability_prompt(
    conversation: str,
    updates_text: str,
    probability_text: str,
) -> str:
    """
    Build the follow-up prompt that combines the conversation,
    BN updates, and probabilities before asking Gemini to respond.
    """
    return (
        "You are assisting a user based on the following conversation and\n"
        "probabilistic insights derived from a Bayesian network.\n"
        "Use your own knowledge and the provided probabilities to craft a helpful response.\n\n"
        "Conversation:\n"
        f"{conversation}\n\n"
        "Node updates:\n"
        f"{updates_text}\n\n"
        "Updated probabilities:\n"
        f"{probability_text}\n\n"
        "Provide a thoughtful reply."
    )
