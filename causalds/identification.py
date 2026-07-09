"""Population-level causal-identification helpers."""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import networkx as nx

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_Y0_SRC = _REPO_ROOT / "src" / "y0" / "src"
if _LOCAL_Y0_SRC.is_dir():
    _local_y0_path = str(_LOCAL_Y0_SRC)
    if _local_y0_path not in sys.path:
        sys.path.insert(0, _local_y0_path)

_Y0_IMPORT_ERROR: Optional[Exception] = None

try:
    from y0.algorithm.identify import identify_outcomes
    from y0.dsl import Variable
    from y0.graph import DEFAULT_TAG as Y0_DEFAULT_TAG
    from y0.graph import NxMixedGraph

    _HAS_Y0 = True
except Exception as import_error:
    identify_outcomes = None
    Variable = None
    Y0_DEFAULT_TAG = "hidden"
    NxMixedGraph = None
    _HAS_Y0 = False
    _Y0_IMPORT_ERROR = import_error

_DOWHY_IMPORT_ERROR: Optional[Exception] = None

try:
    from dowhy.causal_identifier import identify_effect as _dowhy_identify_effect

    _HAS_DOWHY_FUNCTIONAL_ID = True
except Exception as import_error:
    _dowhy_identify_effect = None
    _HAS_DOWHY_FUNCTIONAL_ID = False
    _DOWHY_IMPORT_ERROR = import_error


@dataclass
class PopulationATEIdentification:
    """R2 population-ATE identifiability result."""

    identifiable: Optional[bool]
    engine: str
    raw_estimand: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifiable": self.identifiable,
            "engine": self.engine,
            "raw_estimand": self.raw_estimand,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "details": dict(self.details),
        }


def has_population_ate_identifier() -> bool:
    """Return whether an R2 population-ATE identification backend is available."""
    return bool(_HAS_Y0 or _HAS_DOWHY_FUNCTIONAL_ID)


def _observed_set_or_all(
    G: nx.DiGraph,
    observed_nodes: Optional[Iterable[str]],
) -> set[str]:
    if observed_nodes is None:
        return {str(node) for node in G.nodes()}
    return {str(node) for node in observed_nodes}


def _latent_set(
    G: nx.DiGraph,
    *,
    observed_nodes: Optional[Iterable[str]],
    latent_nodes: Optional[Iterable[str]],
) -> set[str]:
    observed = _observed_set_or_all(G, observed_nodes)
    explicit_latents = {str(node) for node in (latent_nodes or [])}
    return {
        str(node) for node in G.nodes() if str(node) not in observed
    } | explicit_latents


def graph_to_y0_mixed_graph(
    G: nx.DiGraph,
    *,
    observed_nodes: Optional[Iterable[str]] = None,
    latent_nodes: Optional[Iterable[str]] = None,
) -> Any:
    """Project an explicit-latent DAG into y0's mixed-graph representation."""
    if not _HAS_Y0:
        raise RuntimeError(_y0_unavailable_message())

    latent_set = _latent_set(
        G,
        observed_nodes=observed_nodes,
        latent_nodes=latent_nodes,
    )
    dag = nx.DiGraph()
    for node in G.nodes():
        dag.add_node(str(node), **{Y0_DEFAULT_TAG: str(node) in latent_set})
    dag.add_edges_from((str(u), str(v)) for u, v in G.edges())
    return NxMixedGraph.from_latent_variable_dag(dag, tag=Y0_DEFAULT_TAG)


def identify_population_ate(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    *,
    observed_nodes: Optional[Iterable[str]] = None,
    latent_nodes: Optional[Iterable[str]] = None,
) -> PopulationATEIdentification:
    """Identify the population ATE from the conceptual observational law.

    The target is the nonparametric population-level interventional
    distribution for the conceptual treatment and outcome. DoWhy's functional
    ID API is used first, but instrumental-variable estimands are filtered out.
    y0 is consulted only when DoWhy has no accepted non-IV estimand or errors.
    """
    treatment = str(treatment)
    outcome = str(outcome)
    graph_nodes = {str(node) for node in G.nodes()}
    if treatment not in graph_nodes:
        return PopulationATEIdentification(
            identifiable=False,
            engine="input_validation",
            error_type="missing_treatment",
            error_message=f"Treatment node {treatment!r} is not in the graph.",
        )
    if outcome not in graph_nodes:
        return PopulationATEIdentification(
            identifiable=False,
            engine="input_validation",
            error_type="missing_outcome",
            error_message=f"Outcome node {outcome!r} is not in the graph.",
        )

    observed = _observed_set_or_all(G, observed_nodes)
    if treatment not in observed or outcome not in observed:
        return PopulationATEIdentification(
            identifiable=False,
            engine="input_validation",
            error_type="target_not_observed",
            error_message=(
                "Population ATE target variables must be conceptual observed "
                "variables for this benchmark contract."
            ),
            details={
                "treatment_observed": treatment in observed,
                "outcome_observed": outcome in observed,
            },
        )

    if _HAS_DOWHY_FUNCTIONAL_ID:
        dowhy_result = _identify_population_ate_dowhy(
            G,
            treatment,
            outcome,
            observed_nodes=observed_nodes,
        )
        if dowhy_result.identifiable is True:
            return dowhy_result
        if _HAS_Y0:
            y0_result = _identify_population_ate_y0(
                G,
                treatment,
                outcome,
                observed_nodes=observed_nodes,
                latent_nodes=latent_nodes,
            )
            if y0_result.identifiable is not None:
                y0_result.details = {
                    **dict(y0_result.details),
                    "fallback_from": "dowhy.functional",
                    "dowhy_primary": dowhy_result.to_dict(),
                }
                return y0_result
            if dowhy_result.identifiable is False:
                dowhy_result.details = {
                    **dict(dowhy_result.details),
                    "fallback_backend": "y0.id",
                    "fallback_error_type": y0_result.error_type,
                    "fallback_error_message": y0_result.error_message,
                }
                return dowhy_result
            return PopulationATEIdentification(
                identifiable=None,
                engine="dowhy.functional+y0.id",
                error_type="all_backends_failed",
                error_message=(
                    "DoWhy did not produce a usable population-ATE result, "
                    "and y0 fallback also failed."
                ),
                details={
                    "dowhy_primary": dowhy_result.to_dict(),
                    "y0_fallback": y0_result.to_dict(),
                },
            )
        return dowhy_result

    if _HAS_Y0:
        return _identify_population_ate_y0(
            G,
            treatment,
            outcome,
            observed_nodes=observed_nodes,
            latent_nodes=latent_nodes,
        )

    return PopulationATEIdentification(
        identifiable=None,
        engine="unavailable",
        error_type="missing_backend",
        error_message=(
            "Neither y0 nor DoWhy's functional identification API is available."
        ),
        details={
            "y0_import_error": (
                None if _Y0_IMPORT_ERROR is None else repr(_Y0_IMPORT_ERROR)
            ),
            "dowhy_import_error": (
                None if _DOWHY_IMPORT_ERROR is None else repr(_DOWHY_IMPORT_ERROR)
            ),
        },
    )


def _identify_population_ate_y0(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    *,
    observed_nodes: Optional[Iterable[str]],
    latent_nodes: Optional[Iterable[str]],
) -> PopulationATEIdentification:
    try:
        y0_graph = graph_to_y0_mixed_graph(
            G,
            observed_nodes=observed_nodes,
            latent_nodes=latent_nodes,
        )
        estimand = identify_outcomes(
            y0_graph,
            treatments=Variable(treatment),
            outcomes=Variable(outcome),
            strict=False,
        )
        return PopulationATEIdentification(
            identifiable=estimand is not None,
            engine="y0.id",
            raw_estimand=(
                _stringify_expression(estimand) if estimand is not None else None
            ),
            details={"method_family": "id"},
        )
    except Exception as error:
        logger.debug("y0 population-ATE identification failed", exc_info=True)
        return PopulationATEIdentification(
            identifiable=None,
            engine="y0.id",
            error_type=type(error).__name__,
            error_message=str(error),
        )


def _identify_population_ate_dowhy(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    *,
    observed_nodes: Optional[Iterable[str]],
) -> PopulationATEIdentification:
    try:
        estimand = _dowhy_identify_effect(
            graph=G,
            action_nodes=[treatment],
            outcome_nodes=[outcome],
            observed_nodes=list(observed_nodes) if observed_nodes is not None else None,
        )
        if getattr(estimand, "no_directed_path", False):
            return PopulationATEIdentification(
                identifiable=True,
                engine="dowhy.functional",
                raw_estimand=str(estimand),
                details={"method_family": "trivial_zero"},
            )

        estimands = getattr(estimand, "estimands", None)
        non_iv_methods: Dict[str, Any] = {}
        if isinstance(estimands, dict):
            non_iv_methods = {
                str(key): value
                for key, value in estimands.items()
                if value is not None and str(key).lower() != "iv"
            }
            iv_present = estimands.get("iv") is not None
        else:
            iv_present = False

        return PopulationATEIdentification(
            identifiable=bool(non_iv_methods),
            engine="dowhy.functional",
            raw_estimand=str(estimand),
            details={
                "accepted_methods": sorted(non_iv_methods),
                "iv_candidate_present": bool(iv_present),
                "policy": "iv_does_not_identify_population_ate",
            },
        )
    except Exception as error:
        logger.debug("DoWhy population-ATE identification failed", exc_info=True)
        return PopulationATEIdentification(
            identifiable=None,
            engine="dowhy.functional",
            error_type=type(error).__name__,
            error_message=str(error),
        )


def _stringify_expression(expression: Any) -> str:
    if hasattr(expression, "to_y0"):
        return expression.to_y0()
    return str(expression)


def _y0_unavailable_message() -> str:
    if _Y0_IMPORT_ERROR is None:
        return "y0 is not available."
    return f"y0 is not available: {_Y0_IMPORT_ERROR!r}"


__all__ = [
    "PopulationATEIdentification",
    "graph_to_y0_mixed_graph",
    "has_population_ate_identifier",
    "identify_population_ate",
]
