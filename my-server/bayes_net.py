"""Bayesian network definition and helper functions for the cancer BN."""

from __future__ import annotations

from dataclasses import dataclass

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader

BN_CANCER_BIF = """
network "cancer" { }

variable "Pollution" { type discrete [ 2 ] { "low" "high" }; }
variable "Smoking" { type discrete [ 2 ] { "true" "false" }; }
variable "Cancer" { type discrete [ 2 ] { "true" "false" }; }
variable "Xray" { type discrete [ 2 ] { "positive" "negative" }; }
variable "Dyspnoea" { type discrete [ 2 ] { "true" "false" }; }

probability ( "Pollution" ) {
    table 0.9, 0.1;
}

probability ( "Smoking" ) {
    table 0.5, 0.5;
}

probability ( "Cancer" | "Pollution", "Smoking" ) {
    ( "low", "true" ) 0.03, 0.97;
    ( "low", "false" ) 0.001, 0.999;
    ( "high", "true" ) 0.05, 0.95;
    ( "high", "false" ) 0.02, 0.98;
}

probability ( "Xray" | "Cancer" ) {
    ( "true" ) 0.9, 0.1;
    ( "false" ) 0.2, 0.8;
}

probability ( "Dyspnoea" | "Cancer" ) {
    ( "true" ) 0.65, 0.35;
    ( "false" ) 0.3, 0.7;
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
