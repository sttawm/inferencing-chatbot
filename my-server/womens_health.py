"""Women's health Bayesian network definition (formerly bayes_net)."""

from __future__ import annotations

from dataclasses import dataclass

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader

from bn_helpers import validate_bn

BN_WOMENS_HEALTH = """
network "womens_health" { }

variable "Metabolic Imbalance" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Adrenal Imbalance" { 
    type discrete [ 2 ] { "false" "true" }; 
}

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

variable "Central Obesity" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Sleep Quality" {
    type discrete [ 2 ] { "good" "poor" };
}

variable "High Stress" { 
    type discrete [ 2 ] { "false" "true" }; 
}

variable "Cortisol Level" {
    type discrete [ 3 ] { "low" "normal" "high" };
}

variable "Adrenal Androgens" {
    type discrete [ 2 ] { "normal" "high" };
}

variable "Family History of PCOS" {
    type discrete [ 2 ] { "false" "true" };
}

# Kids / teens: Global estimates put metabolic syndrome at ~2–3% in children and ~4–5% in adolescents overall, higher in overweight/obese youth. 
# ~ https://link.springer.com/article/10.1186/s13098-020-00601-8
# 
# Adults: Large NHANES-based and other national studies show prevalence rising from about 10–20% in people 20–39 to ~30–35% in 40–59 and ~50–55% in ≥60. 
# ~ https://pmc.ncbi.nlm.nih.gov/articles/PMC7312413/?utm_source=chatgpt.com

probability ( "Central Obesity" | "Age" ) {
    ( "child" ) 0.95, 0.05;
    ( "teen" ) 0.90, 0.10;
    ( "20s" ) 0.80, 0.20;
    ( "30s" ) 0.70, 0.30;
    ( "40s" ) 0.60, 0.40;
    ( "50s" ) 0.55, 0.45;
    ( ">60" ) 0.50, 0.50;
}

probability ( "Metabolic Imbalance" | "Central Obesity" "Age" ) {

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

probability ( "Sleep Quality" | "Age" ) {
    ( "child" ) 0.85, 0.15;
    ( "teen"  ) 0.75, 0.25;
    ( "20s"   ) 0.70, 0.30;
    ( "30s"   ) 0.65, 0.35;
    ( "40s"   ) 0.60, 0.40;
    ( "50s"   ) 0.55, 0.45;
    ( ">60"   ) 0.50, 0.50;
}

probability ( "High Stress" | "Sleep Quality" ) {
    ( "good" ) 0.85, 0.15;   // mostly low stress
    ( "poor" ) 0.50, 0.50;   // poor sleep → high stress more likely
}

probability ( "Cortisol Level" | "High Stress" ) {
    ( "false" ) 0.10, 0.80, 0.10;  // low, normal, high
    ( "true"  ) 0.05, 0.55, 0.40;  // stress shifts mass toward high cortisol
}

probability ( "Adrenal Androgens" | "Cortisol Level" ) {
    ( "low"    ) 0.95, 0.05;  // high androgens rare with low cortisol
    ( "normal" ) 0.90, 0.10;
    ( "high"   ) 0.75, 0.25;  // high cortisol raises chance of high adrenal androgens
}

probability ( "Adrenal Imbalance" | "Cortisol Level" "Adrenal Androgens" ) {

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

probability ( "Family History of PCOS" | "PCOS" ) {
    ( "false" ) 0.90, 0.10;  // no PCOS → ~10% report positive family history
    ( "true"  ) 0.70, 0.30;  // PCOS   → ~30% family history
}

probability ( "Should journal" | "High Stress" ) {
    ( "false" ) 0.80, 0.20;
    ( "true"  ) 0.30, 0.70;
}

probability ( "Should get mental health care" | "High Stress" ) {
    ( "false" ) 0.95, 0.05;
    ( "true"  ) 0.40, 0.60;
}

probability ( "Should cut foods that spike blood sugar" | "Metabolic Imbalance" ) {
    ( "false" ) 0.70, 0.30;
    ( "true"  ) 0.20, 0.80;
}

probability ( "Should get a Continual Glucose Monitor" | "Metabolic Imbalance" ) {
    ( "false" ) 0.95, 0.05;
    ( "true"  ) 0.60, 0.40;
}
"""


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
