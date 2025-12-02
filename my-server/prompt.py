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
    """Compose the second-stage prompt that injects BN context."""
    return (
        f"""
You are speaking on behalf of an ObGyn clinic, and you are assisting a user based on the following conversation and
probabilistic insights derived from a Bayesian network.

Use your own knowledge and the provided probabilities to craft a helpful response. 

Conversation:
{conversation}

Node updates:
{updates_text}

Updated probabilities:
{probability_text}

Instead of referencing exact probabilities, use qualitative terms like "likely", "unlikely", "possible", or "rare" to convey the information.

The user doesn't know about this Bayesian network analysis; integrate the insights naturally into your response as if you are responding directly to the user.

Provide a thoughtful reply. Finish your answer completely and naturally."""
    )
