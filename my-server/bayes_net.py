"""Bayesian network definition and helper functions for the cancer BN."""

from __future__ import annotations

from dataclasses import dataclass

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader

BN_CANCER_BIF = """
network "cancer" { }

variable "Metabolic Imbalance" { type discrete [ 2 ] { "false" "true" }; }
variable "Adrenal Imbalance" { type discrete [ 2 ] { "false" "true" }; }

variable "PCOS" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Age" { 
    type discrete [ 7 ] { "child" "teen" "20s" "30s" "40s" "50s" ">60" }; 
}

variable "Irregular Periods" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Painful Periods" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Weight Gain" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Facial Hair" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Should journal" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Should get mental health care" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Should cut foods that spike blood sugar" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Should get a Continual Glucose Monitor" { 
    type discrete [ 2 ] { "false" "true" }; 
}



probability ( "Age" ) {
    table 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1;
}

# Kids / teens: Global estimates put metabolic syndrome at ~2–3% in children and ~4–5% in adolescents overall, higher in overweight/obese youth. 
# ~ https://link.springer.com/article/10.1186/s13098-020-00601-8
# 
# Adults: Large NHANES-based and other national studies show prevalence rising from about 10–20% in people 20–39 to ~30–35% in 40–59 and ~50–55% in ≥60. 
# ~ https://pmc.ncbi.nlm.nih.gov/articles/PMC7312413/?utm_source=chatgpt.com

probability ( "Metabolic Imbalance" | "Age" ) {
    ( "child" ) 0.02, 0.98;
    ( "teen" ) 0.04, 0.96;
    ( "20s" ) 0.12, 0.88;
    ( "30s" ) 0.2, 0.8;
    ( "40s" ) 0.3, 0.7;
    ( "50s" ) 0.4, 0.6;
    ( ">60" ) 0.55, 0.45;
}

probability ( "Adrenal Imbalance" | "Age" ) {
    ( "child" ) 0.0003, 0.9997;
    ( "teen" ) 0.0005, 0.9995;
    ( "20s" ) 0.0007, 0.9993;
    ( "30s" ) 0.0010, 0.9990;
    ( "40s" ) 0.0015, 0.9985;
    ( "50s" ) 0.0020, 0.9980;
    ( ">60" ) 0.0030, 0.9970;
}

probability ( "PCOS" | "Adrenal Imbalance" "Metabolic Imbalance" "Age" ) {
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

probability ( "Irregular Periods" | "PCOS" ) {
    ( "false" ) 0.80, 0.20;  // no PCOS
    ( "true"  ) 0.15, 0.85;  // PCOS
}

probability ( "Painful Periods" | "PCOS" ) {
    ( "false" ) 0.30, 0.70;  // no PCOS
    ( "true"  ) 0.20, 0.80;  // PCOS
}

probability ( "Weight Gain" | "PCOS" ) {
    ( "false" ) 0.50, 0.50;  // no PCOS
    ( "true"  ) 0.30, 0.70;  // PCOS
}

probability ( "Facial Hair" | "PCOS" ) {
    ( "false" ) 0.93, 0.07;  // no PCOS
    ( "true"  ) 0.30, 0.70;  // PCOS
}
"""


@dataclass(frozen=True)
class CancerBayesNet:
    model: any
    nodes: list[str]
    edges: list[tuple[str, str]]
    inference: VariableElimination


def load_cancer_bayes_net() -> CancerBayesNet:
    model = BIFReader(string=BN_CANCER_BIF).get_model()
    nodes = list(model.nodes())
    edges = list(model.edges())
    inference = VariableElimination(model)
    return CancerBayesNet(model=model, nodes=nodes, edges=edges, inference=inference)
