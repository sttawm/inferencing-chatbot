"""Women's health Bayesian network definition (formerly bayes_net)."""

from __future__ import annotations

from dataclasses import dataclass

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader

from bn_helpers import validate_bn

BN_VARIABLES_DICT: dict[str, list[str]] = {
    "Metabolic_Imbalance": ["false", "true"],
    "Adrenal_Imbalance": ["false", "true"],
    "PCOS": ["false", "true"],
    "Age": ["child", "teen", "20s", "30s", "40s", "50s", ">60"],
    "Irregular_Periods": ["false", "true"],
    "Painful_Periods": ["false", "true"],
    "Weight_Gain": ["false", "true"],
    "Facial_Hair": ["false", "true"],
    "Should_journal": ["false", "true"],
    "Should_get_mental_health_care": ["false", "true"],
    "Should_cut_foods_that_spike_blood_sugar": ["false", "true"],
    "Should_get_a_Continual_Glucose_Monitor": ["false", "true"],
    "Central_Obesity": ["false", "true"],
    "Sleep_Quality": ["good", "poor"],
    "High_Stress": ["false", "true"],
    "Cortisol_Level": ["low", "normal", "high"],
    "Adrenal_Androgens": ["normal", "high"],
    "Family_History_of_PCOS": ["false", "true"],
}

def dict_to_bif_variables(var_dict: dict[str, list[str]]) -> str:
    """
    Convert a {var: [values]} dict into BIF variable blocks.
    """
    blocks = []
    for var, values in var_dict.items():
        value_str = " ".join(f'"{v}"' for v in values)
        block = (
            f'variable "{var}" {{\n'
            f'    type discrete [ {len(values)} ] {{ {value_str} }};\n'
            f'}}\n'
        )
        blocks.append(block)
    return "\n".join(blocks)

BN_VARIABLES = dict_to_bif_variables(BN_VARIABLES_DICT)

BN_PROBABILITIES = """

// Kids / teens: Global estimates put metabolic syndrome at ~2–3% in children and ~4–5% in adolescents overall, higher in overweight/obese youth. 
// ~ https://link.springer.com/article/10.1186/s13098-020-00601-8
// 
// Adults: Large NHANES-based and other national studies show prevalence rising from about 10–20% in people 20–39 to ~30–35% in 40–59 and ~50–55% in ≥60. 
// ~ https://pmc.ncbi.nlm.nih.gov/articles/PMC7312413/?utm_source=chatgpt.com


probability ( "Age" ) {
    table 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1;
}

probability ( "Central_Obesity" | "Age" ) {
    ( "child" ) 0.95, 0.05;
    ( "teen" ) 0.90, 0.10;
    ( "20s" ) 0.80, 0.20;
    ( "30s" ) 0.70, 0.30;
    ( "40s" ) 0.60, 0.40;
    ( "50s" ) 0.55, 0.45;
    ( ">60" ) 0.50, 0.50;
}

probability ( "Metabolic_Imbalance" | "Central_Obesity" "Age" ) {

    // Central Obesity = false
    ( "false" "child" ) 0.985, 0.015;
    ( "false" "teen"  ) 0.970, 0.030;
    ( "false" "20s"   ) 0.920, 0.080;
    ( "false" "30s"   ) 0.860, 0.140;
    ( "false" "40s"   ) 0.780, 0.220;
    ( "false" "50s"   ) 0.720, 0.280;
    ( "false" ">60"   ) 0.600, 0.400;

    // Central Obesity = true
    ( "true"  "child" ) 0.900, 0.100;
    ( "true"  "teen"  ) 0.880, 0.120;
    ( "true"  "20s"   ) 0.750, 0.250;
    ( "true"  "30s"   ) 0.650, 0.350;
    ( "true"  "40s"   ) 0.550, 0.450;
    ( "true"  "50s"   ) 0.450, 0.550;
    ( "true"  ">60"   ) 0.300, 0.700;
}

probability ( "PCOS" | "Adrenal_Imbalance" "Metabolic_Imbalance" "Age" ) {
    // Age = child
    ( "false" "false" "child" ) 0.99, 0.01;
    ( "false" "true"  "child" ) 0.89, 0.11;
    ( "true"  "false" "child" ) 0.94, 0.06;
    ( "true"  "true"  "child" ) 0.79, 0.21;

    // Age = teen
    ( "false" "false" "teen" ) 0.85, 0.15;
    ( "false" "true"  "teen" ) 0.75, 0.25;
    ( "true"  "false" "teen" ) 0.80, 0.20;
    ( "true"  "true"  "teen" ) 0.65, 0.35;

    // Age = 20s
    ( "false" "false" "20s" ) 0.88, 0.12;
    ( "false" "true"  "20s" ) 0.78, 0.22;
    ( "true"  "false" "20s" ) 0.83, 0.17;
    ( "true"  "true"  "20s" ) 0.68, 0.32;

    // Age = 30s
    ( "false" "false" "30s" ) 0.92, 0.08;
    ( "false" "true"  "30s" ) 0.82, 0.18;
    ( "true"  "false" "30s" ) 0.87, 0.13;
    ( "true"  "true"  "30s" ) 0.72, 0.28;

    // Age = 40s
    ( "false" "false" "40s" ) 0.96, 0.04;
    ( "false" "true"  "40s" ) 0.86, 0.14;
    ( "true"  "false" "40s" ) 0.91, 0.09;
    ( "true"  "true"  "40s" ) 0.76, 0.24;

    // Age = 50s
    ( "false" "false" "50s" ) 0.98, 0.02;
    ( "false" "true"  "50s" ) 0.88, 0.12;
    ( "true"  "false" "50s" ) 0.93, 0.07;
    ( "true"  "true"  "50s" ) 0.78, 0.22;

    // Age > 60
    ( "false" "false" ">60" ) 0.99, 0.01;
    ( "false" "true"  ">60" ) 0.89, 0.11;
    ( "true"  "false" ">60" ) 0.94, 0.06;
    ( "true"  "true"  ">60" ) 0.79, 0.21;
}

probability ( "Irregular_Periods" | "PCOS" ) {
    ( "false" ) 0.80, 0.20;  // no PCOS
    ( "true"  ) 0.15, 0.85;  // PCOS
}

probability ( "Painful_Periods" | "PCOS" ) {
    ( "false" ) 0.30, 0.70;  // no PCOS
    ( "true"  ) 0.20, 0.80;  // PCOS
}

probability ( "Weight_Gain" | "PCOS" ) {
    ( "false" ) 0.50, 0.50;  // no PCOS
    ( "true"  ) 0.30, 0.70;  // PCOS
}

probability ( "Facial_Hair" | "PCOS" ) {
    ( "false" ) 0.93, 0.07;  // no PCOS
    ( "true"  ) 0.30, 0.70;  // PCOS
}

probability ( "Sleep_Quality" | "Age" ) {
    ( "child" ) 0.85, 0.15;
    ( "teen"  ) 0.75, 0.25;
    ( "20s"   ) 0.70, 0.30;
    ( "30s"   ) 0.65, 0.35;
    ( "40s"   ) 0.60, 0.40;
    ( "50s"   ) 0.55, 0.45;
    ( ">60"   ) 0.50, 0.50;
}

probability ( "High_Stress" | "Sleep_Quality" ) {
    ( "good" ) 0.85, 0.15;   // mostly low stress
    ( "poor" ) 0.50, 0.50;   // poor sleep → high stress more likely
}

probability ( "Cortisol_Level" | "High_Stress" ) {
    ( "false" ) 0.10, 0.80, 0.10;  // low, normal, high
    ( "true"  ) 0.05, 0.55, 0.40;  // stress shifts mass toward high cortisol
}

probability ( "Adrenal_Androgens" | "Cortisol_Level" ) {
    ( "low"    ) 0.95, 0.05;  // high androgens rare with low cortisol
    ( "normal" ) 0.90, 0.10;
    ( "high"   ) 0.75, 0.25;  // high cortisol raises chance of high adrenal androgens
}

probability ( "Adrenal_Imbalance" | "Cortisol_Level" "Adrenal_Androgens" ) {

    // Cortisol = low
    ( "low"    "normal" ) 0.9995, 0.0005;
    ( "low"    "high"   ) 0.9900, 0.0100;

    // Cortisol = normal
    ( "normal" "normal" ) 0.9990, 0.0010;
    ( "normal" "high"   ) 0.9800, 0.0200;

    // Cortisol = high
    ( "high"   "normal" ) 0.9950, 0.0050;
    ( "high"   "high"   ) 0.9500, 0.0500;
}

probability ( "Family_History_of_PCOS" | "PCOS" ) {
    ( "false" ) 0.90, 0.10;  // no PCOS → ~10% report positive family history
    ( "true"  ) 0.70, 0.30;  // PCOS   → ~30% family history
}

probability ( "Should_journal" | "High_Stress" ) {
    ( "false" ) 0.80, 0.20;
    ( "true"  ) 0.30, 0.70;
}

probability ( "Should_get_mental_health_care" | "High_Stress" ) {
    ( "false" ) 0.95, 0.05;
    ( "true"  ) 0.40, 0.60;
}

probability ( "Should_cut_foods_that_spike_blood_sugar" | "Metabolic_Imbalance" ) {
    ( "false" ) 0.70, 0.30;
    ( "true"  ) 0.20, 0.80;
}

probability ( "Should_get_a_Continual_Glucose_Monitor" | "Metabolic_Imbalance" ) {
    ( "false" ) 0.95, 0.05;
    ( "true"  ) 0.60, 0.40;
}
"""

BN_WOMENS_HEALTH = """
network "womens_health" { }
""" + BN_VARIABLES + BN_PROBABILITIES


@dataclass(frozen=True)
class WomensHealthBayesNet:
    model: any
    nodes: list[str]
    edges: list[tuple[str, str]]
    inference: VariableElimination


def load_womens_health_bayes_net() -> WomensHealthBayesNet:
    model = BIFReader(string=BN_WOMENS_HEALTH).get_model()
    nodes = list(model.nodes())
    edges = list(model.edges())
    inference = validate_bn(model)
    
    return WomensHealthBayesNet(model=model, nodes=nodes, edges=edges, inference=inference)
