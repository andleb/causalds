"""Library-backed Rung-3 identifiability checks."""

import itertools as itt
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx

from .graph import SampledGraph

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_Y0_SRC = _REPO_ROOT / "src" / "y0" / "src"
if _LOCAL_Y0_SRC.is_dir():
    _local_y0_path = str(_LOCAL_Y0_SRC)
    if _local_y0_path not in sys.path:
        sys.path.insert(0, _local_y0_path)

_Y0_IMPORT_ERROR: Optional[Exception] = None

try:
    from y0.algorithm.identify import Identification, identify
    from y0.algorithm.identify.id_star import ConflictUnidentifiable, id_star
    from y0.algorithm.identify.idc_star import idc_star
    from y0.algorithm.identify.utils import Unidentifiable as Y0Unidentifiable
    from y0.dsl import (
        CounterfactualVariable,
        Fraction,
        One,
        Probability,
        Product,
        Sum,
        Variable,
        Zero,
    )
    from y0.graph import DEFAULT_TAG as Y0_DEFAULT_TAG
    from y0.graph import NxMixedGraph

    _HAS_Y0 = True
except Exception as import_error:
    Identification = None
    identify = None
    ConflictUnidentifiable = Exception
    id_star = None
    idc_star = None
    Y0Unidentifiable = Exception
    CounterfactualVariable = None
    Fraction = None
    One = None
    Probability = None
    Product = None
    Sum = None
    Variable = None
    Zero = None
    Y0_DEFAULT_TAG = "hidden"
    NxMixedGraph = None
    _HAS_Y0 = False
    _Y0_IMPORT_ERROR = import_error


COUNTERFACTUAL_ID_METHODS = frozenset({"id_star+id", "idc_star+id", "none", "unknown"})
_Y0_UNIDENTIFIABLE_ERRORS = (
    ConflictUnidentifiable,
    Y0Unidentifiable,
)


@dataclass
class CounterfactualIdentificationResult:
    """Identification result for a structured Rung-3 estimand."""

    estimand_kind: str
    identifiable: bool
    method: str
    query: str
    raw_estimand: Optional[str] = None
    observational_estimand: Optional[str] = None
    mediator_nodes: List[str] = field(default_factory=list)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "estimand_kind": self.estimand_kind,
            "identifiable": self.identifiable,
            "method": self.method,
            "query": self.query,
            "raw_estimand": self.raw_estimand,
            "observational_estimand": self.observational_estimand,
            "mediator_nodes": list(self.mediator_nodes),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "details": dict(self.details),
        }


def _latent_nodes_of(sg: SampledGraph) -> set[str]:
    latent_nodes = set(str(n) for n in (sg.latent_nodes or []))
    meta = sg.meta or {}
    latent_nodes.update(str(n) for n in meta.get("latent_nodes", []) or [])
    return latent_nodes


def sampled_graph_to_y0(sg: SampledGraph) -> NxMixedGraph:
    """Project the repo's explicit-latent DAG into y0's mixed-graph view."""
    if not _HAS_Y0:
        raise RuntimeError(_y0_unavailable_message())

    latent_nodes = _latent_nodes_of(sg)
    dag = nx.DiGraph()
    for node in sg.graph.nodes():
        dag.add_node(str(node), **{Y0_DEFAULT_TAG: str(node) in latent_nodes})
    dag.add_edges_from((str(u), str(v)) for u, v in sg.graph.edges())
    return NxMixedGraph.from_latent_variable_dag(dag, tag=Y0_DEFAULT_TAG)


def _is_observational_probability(expression: Probability) -> bool:
    variables = expression.children | expression.parents
    return not any(
        isinstance(variable, CounterfactualVariable) for variable in variables
    )


def _reduce_to_observational(graph: NxMixedGraph, expression):
    if isinstance(expression, Probability):
        if _is_observational_probability(expression):
            return expression
        return identify(Identification.from_expression(graph=graph, query=expression))
    if isinstance(expression, Product):
        return Product.safe(
            _reduce_to_observational(graph, part) for part in expression.expressions
        )
    if isinstance(expression, Sum):
        return Sum.safe(
            _reduce_to_observational(graph, expression.expression), expression.ranges
        )
    if isinstance(expression, Fraction):
        return _reduce_to_observational(
            graph, expression.numerator
        ) / _reduce_to_observational(graph, expression.denominator)
    if isinstance(expression, (One, Zero)):
        return expression
    raise TypeError(f"Unsupported y0 expression type: {type(expression)!r}")


def _variable(node: str) -> Variable:
    return Variable(str(node))


def _stringify_expression(expression) -> str:
    return expression.to_y0() if hasattr(expression, "to_y0") else str(expression)


def _failure_result(
    *,
    estimand_kind: str,
    query: str,
    error: Exception,
    mediator_nodes: Optional[List[str]] = None,
    method: str = "none",
    details: Optional[Dict[str, Any]] = None,
) -> CounterfactualIdentificationResult:
    return CounterfactualIdentificationResult(
        estimand_kind=estimand_kind,
        identifiable=False,
        method=method,
        query=query,
        mediator_nodes=list(mediator_nodes or []),
        error_type=type(error).__name__,
        error_message=str(error),
        details=dict(details or {}),
    )


def _has_singleton_ett_bidirected_child_obstruction(
    graph: NxMixedGraph,
    *,
    treatment: str,
    outcome: str,
) -> bool:
    """Check the graph-level singleton ETT non-identification obstruction."""
    treatment_var = _variable(treatment)
    outcome_var = _variable(outcome)
    directed_graph = graph.directed

    if treatment_var not in directed_graph or outcome_var not in directed_graph:
        return False

    ancestors = nx.ancestors(directed_graph, outcome_var) | {outcome_var}
    if treatment_var not in ancestors:
        return False

    children = [
        child
        for child in directed_graph.successors(treatment_var)
        if child in ancestors
    ]
    if not children:
        return False

    bidirected_graph = graph.undirected.subgraph(ancestors)
    if treatment_var not in bidirected_graph:
        return False

    return any(
        child in bidirected_graph
        and nx.has_path(bidirected_graph, treatment_var, child)
        for child in children
    )


def _y0_unavailable_message() -> str:
    message = (
        "y0 is required for Rung-3 counterfactual identification. "
        f"Expected repo-local y0 at {_LOCAL_Y0_SRC}."
    )
    if _Y0_IMPORT_ERROR is not None:
        message = f"{message} Import failed with: {_Y0_IMPORT_ERROR}"
    return message


def _compute_ett_identification(
    y0_graph: NxMixedGraph,
    *,
    sg: SampledGraph,
    treatment: str,
    outcome: str,
) -> CounterfactualIdentificationResult:
    treatment_var = _variable(treatment)
    outcome_var = _variable(outcome)
    query = f"P({outcome}_{{{treatment}=-}} | {treatment}=+)"
    try:
        raw_estimand = idc_star(
            y0_graph,
            outcomes={outcome_var @ -treatment_var: -outcome_var},
            conditions={treatment_var: +treatment_var},
        )
        reduced_estimand = _reduce_to_observational(y0_graph, raw_estimand)
        return CounterfactualIdentificationResult(
            estimand_kind="ett",
            identifiable=True,
            method="idc_star+id",
            query=query,
            raw_estimand=_stringify_expression(raw_estimand),
            observational_estimand=_stringify_expression(reduced_estimand),
        )
    except _Y0_UNIDENTIFIABLE_ERRORS as error:
        return _failure_result(estimand_kind="ett", query=query, error=error)
    except ZeroDivisionError as error:
        if _has_singleton_ett_bidirected_child_obstruction(
            y0_graph, treatment=treatment, outcome=outcome
        ):
            return _failure_result(
                estimand_kind="ett",
                query=query,
                error=error,
                details={
                    "status_source": "graph_rule",
                    "rule": (
                        "singleton_ett_bidirected_path_from_treatment_to_child_"
                        "in_ancestral_graph"
                    ),
                    "library_error_type": type(error).__name__,
                    "motif": str(getattr(sg, "motif", "")),
                },
            )
        raise


def _mediator_assignments(mediator_vars: Iterable[Variable]) -> List[tuple]:
    value_options = [(-mediator_var, +mediator_var) for mediator_var in mediator_vars]
    return list(itt.product(*value_options))


def _compute_natural_effect_identification(
    y0_graph: NxMixedGraph,
    *,
    treatment: str,
    outcome: str,
    mediators: List[str],
    estimand_kind: str,
) -> CounterfactualIdentificationResult:
    treatment_var = _variable(treatment)
    outcome_var = _variable(outcome)
    mediator_vars = [_variable(mediator) for mediator in mediators]
    target_treatment = +treatment_var if estimand_kind == "nde" else -treatment_var
    source_treatment = -treatment_var if estimand_kind == "nde" else +treatment_var
    direction = ("+", "-") if estimand_kind == "nde" else ("-", "+")
    query = f"P({outcome}_{{{treatment}={direction[0]}, {','.join(mediators)}_{{{treatment}={direction[1]}}}}})"

    try:
        raw_terms = []
        reduced_terms = []
        for assignment in _mediator_assignments(mediator_vars):
            event = {
                outcome_var @ tuple((target_treatment, *assignment)): -outcome_var,
            }
            for mediator_var, mediator_value in zip(mediator_vars, assignment):
                event[mediator_var @ source_treatment] = mediator_value
            raw_term = id_star(y0_graph, event)
            reduced_term = _reduce_to_observational(y0_graph, raw_term)
            raw_terms.append(_stringify_expression(raw_term))
            reduced_terms.append(_stringify_expression(reduced_term))

        return CounterfactualIdentificationResult(
            estimand_kind=estimand_kind,
            identifiable=True,
            method="id_star+id",
            query=query,
            raw_estimand=" + ".join(raw_terms),
            observational_estimand=" + ".join(reduced_terms),
            mediator_nodes=list(mediators),
            details={"n_assignments": len(raw_terms)},
        )
    except _Y0_UNIDENTIFIABLE_ERRORS as error:
        return _failure_result(
            estimand_kind=estimand_kind,
            query=query,
            error=error,
            mediator_nodes=list(mediators),
        )


def compute_counterfactual_identification(
    sg: SampledGraph,
    *,
    mediators: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute graph-level Rung-3 identifiability labels for ETT/NDE/NIE."""
    mediator_ids = [str(mediator) for mediator in (mediators or [])]

    if not _HAS_Y0:
        raise RuntimeError(_y0_unavailable_message())

    y0_graph = sampled_graph_to_y0(sg)
    results: Dict[str, Dict[str, Any]] = {}
    ett_result = _compute_ett_identification(
        y0_graph,
        sg=sg,
        treatment=str(sg.treatment),
        outcome=str(sg.outcome),
    )
    results["ett"] = ett_result.to_dict()

    if mediator_ids:
        for estimand_kind in ("nde", "nie"):
            result = _compute_natural_effect_identification(
                y0_graph,
                treatment=str(sg.treatment),
                outcome=str(sg.outcome),
                mediators=mediator_ids,
                estimand_kind=estimand_kind,
            )
            results[estimand_kind] = result.to_dict()

    return results


__all__ = [
    "COUNTERFACTUAL_ID_METHODS",
    "CounterfactualIdentificationResult",
    "compute_counterfactual_identification",
    "sampled_graph_to_y0",
]
