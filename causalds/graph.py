"""
Graph sampler for small causal DAGs with motif support, random augmentation,
optional latent confounding, and population-ATE identifiability gating.

Key features
------------
- Native NetworkX graphs (nx.DiGraph) as the primary representation.
- CLadder-style motifs: confounding, collider, mediation/chain, IV, diamond,
    diamondcut, arrowhead, frontdoor.
- Random DAG sampler + forward-only augmentation up to N<=10 nodes.
- Optional latent injection (add hidden parent U -> {a,b}) and observed set control.
- Identifiability gate using DoWhy functional ID first, filtering out
  instrumental-variable estimands as population-ATE identification, with y0 ID
  as a fallback for cases DoWhy cannot identify or inspect.
- DOT export helper.

Notes
-----
* "Identifiable" here means the *query* (typically ATE of X on Y) is identifiable from
  the graph given the declared observed set, in the nonparametric sense.
* Front-door with true latent confounding becomes identifiable when its
  preconditions hold. Generic IV motifs are not treated as identifying the
  population ATE under this benchmark's default assumptions.
* You can flip on latents by either marking nodes as unobserved (via observed_nodes)
  or injecting new hidden parents with `add_latent_confounder`.
"""

import copy
import itertools as it
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
from networkx.algorithms.d_separation import is_d_separator, is_minimal_d_separator

from .identification import has_population_ate_identifier, identify_population_ate
from .utils import json_safe

logger = logging.getLogger(__name__)


# -----------------------------
# Motif Enum
# -----------------------------
class Motif(str, Enum):
    """Enumeration of available causal graph motifs."""

    CHAIN = "chain"
    MEDIATION = "mediation"
    CONFOUNDING = "confounding"
    FORK = "fork"
    COLLIDER = "collider"
    ARROWHEAD = "arrowhead"
    DIAMOND = "diamond"
    DIAMONDCUT = "diamondcut"
    FRONTDOOR = "frontdoor"
    IV = "iv"
    DOUBLE_NC = "double_nc"
    TRIANGLE = "triangle"

    def __str__(self) -> str:
        return self.value


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class SampledGraph:
    graph: nx.DiGraph
    treatment: str
    outcome: str
    motif: str
    observed_nodes: Optional[List[str]] = None  # for future ADMG support
    latent_nodes: Optional[List[str]] = None  # for future ADMG support
    needs_names: Optional[List[str]] = None  # nodes NOT renamed from CauseNet
    # extra info
    meta: Optional[Dict] = None
    _identifiable: Optional[bool] = None

    def to_dot(self) -> str:
        """Return a DOT string for visualization/persistence."""
        try:
            from networkx.drawing.nx_pydot import to_pydot

            return to_pydot(self.graph).to_string()
        except Exception:
            edges = ", ".join([f"{u}->{v}" for u, v in self.graph.edges()])
            return f"digraph G {{ {edges} }}"

    @property
    def is_identifiable(self) -> Optional[bool]:
        if self._identifiable is not None:
            return self._identifiable
        return is_identifiable(
            self.graph,
            self.treatment,
            self.outcome,
            observed_nodes=self.observed_nodes,
            latent_nodes=self.latent_nodes,
        )

    def copy(self) -> "SampledGraph":
        """Return a deep copy of this SampledGraph."""
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this sampled graph into a JSON-serializable payload."""
        return {
            "graph": {
                "nodes": [
                    {
                        "id": str(node_id),
                        "attrs": json_safe(dict(attrs)),
                    }
                    for node_id, attrs in self.graph.nodes(data=True)
                ],
                "edges": [
                    {
                        "source": str(source),
                        "target": str(target),
                        "attrs": json_safe(dict(attrs)),
                    }
                    for source, target, attrs in self.graph.edges(data=True)
                ],
            },
            "treatment": str(self.treatment),
            "outcome": str(self.outcome),
            "motif": str(self.motif),
            "observed_nodes": json_safe(self.observed_nodes),
            "latent_nodes": json_safe(self.latent_nodes),
            "needs_names": json_safe(self.needs_names),
            "meta": json_safe(self.meta),
            "_identifiable": self._identifiable,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SampledGraph":
        """Reconstruct a sampled graph from :meth:`to_dict` output."""
        payload = dict(payload or {})
        graph_payload = dict(payload.get("graph") or {})

        graph = nx.DiGraph()
        for node_record in graph_payload.get("nodes", []) or []:
            node_record = dict(node_record or {})
            graph.add_node(
                str(node_record.get("id")),
                **dict(node_record.get("attrs") or {}),
            )

        for edge_record in graph_payload.get("edges", []) or []:
            edge_record = dict(edge_record or {})
            graph.add_edge(
                str(edge_record.get("source")),
                str(edge_record.get("target")),
                **dict(edge_record.get("attrs") or {}),
            )

        return cls(
            graph=graph,
            treatment=str(payload.get("treatment", "")),
            outcome=str(payload.get("outcome", "")),
            motif=str(payload.get("motif", "")),
            observed_nodes=payload.get("observed_nodes"),
            latent_nodes=payload.get("latent_nodes"),
            needs_names=payload.get("needs_names"),
            meta=payload.get("meta"),
            _identifiable=payload.get("_identifiable"),
        )

    # NOTE: made this return self to work with copying
    def rename_nodes(self, mapping):
        """
        Hard-rename node IDs using the provided mapping {old_id -> new_id}.
        Updates treatment, outcome, observed_nodes. Returns the mapping actually applied.
        """
        if not mapping:
            logger.warning("rename_nodes: empty mapping provided")
            return {}, self
        # Only keep nodes that exist
        mapping = {u: v for u, v in mapping.items() if u in self.graph}
        logger.debug("rename_nodes: renaming %d nodes", len(mapping))

        # Ensure injective mapping (no two old ids target the same new id)
        # Track used new names to detect collisions
        applied: Dict[str, str] = {}
        used_new_names: set = set()

        for old, new in mapping.items():
            cand = str(new).strip()
            if not cand:
                cand = old  # fall back
            base = cand
            k = 2
            # Ensure new name is unique across all mappings
            while cand in used_new_names:
                cand = f"{base}_{k}"
                k += 1
            applied[old] = cand
            used_new_names.add(cand)

        # Relabel graph (copy=True is the safest semantics)
        import networkx as nx  # type: ignore

        newG = nx.relabel_nodes(self.graph, applied, copy=True)  #
        self.graph = newG

        # Update fields
        if self.treatment in applied:
            old_t = self.treatment
            self.treatment = applied[self.treatment]
            logger.debug("Updated treatment: %s -> %s", old_t, self.treatment)
        if self.outcome in applied:
            old_y = self.outcome
            self.outcome = applied[self.outcome]
            logger.debug("Updated outcome: %s -> %s", old_y, self.outcome)
        if self.observed_nodes:
            self.observed_nodes = [applied.get(n, n) for n in self.observed_nodes]
            # logger.debug("Updated observed nodes")
        if self.latent_nodes:
            self.latent_nodes = [applied.get(n, n) for n in self.latent_nodes]
        if hasattr(self, "node_types") and isinstance(
            getattr(self, "node_types"), dict
        ):
            self.node_types = {applied.get(n, n): t for n, t in self.node_types.items()}

        # Keep traceability (new -> old)
        self.meta = self.meta or {}
        if isinstance(self.meta.get("node_types"), dict):
            self.meta["node_types"] = {
                applied.get(n, n): t for n, t in self.meta["node_types"].items()
            }
        if isinstance(self.meta.get("latent_nodes"), list):
            self.meta["latent_nodes"] = [
                applied.get(n, n) for n in self.meta["latent_nodes"]
            ]
        if isinstance(self.meta.get("fixed_nodes"), list):
            self.meta["fixed_nodes"] = [
                applied.get(n, n) for n in self.meta["fixed_nodes"]
            ]
        if isinstance(self.meta.get("needs_names"), list):
            self.meta["needs_names"] = [
                applied.get(n, n) for n in self.meta["needs_names"]
            ]
        if isinstance(self.meta.get("fixed_name_assignments"), dict):
            self.meta["fixed_name_assignments"] = {
                applied.get(node_id, node_id): story_name
                for node_id, story_name in self.meta["fixed_name_assignments"].items()
            }
        if isinstance(self.meta.get("main_graph"), dict):
            main_graph = copy.deepcopy(self.meta["main_graph"])
            main_graph["node_ids"] = [
                applied.get(n, n) for n in (main_graph.get("node_ids", []) or [])
            ]
            self.meta["main_graph"] = main_graph
        if isinstance(self.meta.get("auxiliary_graph_grafts"), list):
            updated_grafts = []
            for graft in self.meta["auxiliary_graph_grafts"]:
                if not isinstance(graft, dict):
                    updated_grafts.append(graft)
                    continue
                record = copy.deepcopy(graft)
                anchor_node = record.get("anchor_node")
                if anchor_node is not None:
                    record["anchor_node"] = applied.get(anchor_node, anchor_node)
                record["auxiliary_graph_nodes"] = [
                    applied.get(n, n)
                    for n in (record.get("auxiliary_graph_nodes", []) or [])
                ]
                record["new_nodes"] = [
                    applied.get(n, n) for n in (record.get("new_nodes", []) or [])
                ]
                record["edges_added"] = [
                    (applied.get(u, u), applied.get(v, v))
                    for u, v in (record.get("edges_added", []) or [])
                ]
                record["latent_nodes_added"] = [
                    applied.get(n, n)
                    for n in (record.get("latent_nodes_added", []) or [])
                ]
                aux_relabel_map = record.get("auxiliary_relabel_map")
                if isinstance(aux_relabel_map, dict):
                    record["auxiliary_relabel_map"] = {
                        role_name: applied.get(node_id, node_id)
                        for role_name, node_id in aux_relabel_map.items()
                    }
                updated_grafts.append(record)
            self.meta["auxiliary_graph_grafts"] = updated_grafts
        if isinstance(self.meta.get("mapping_sequence"), list):
            updated_stages = []
            for stage in self.meta["mapping_sequence"]:
                if not isinstance(stage, dict):
                    updated_stages.append(stage)
                    continue
                record = copy.deepcopy(stage)
                record["node_ids"] = [
                    applied.get(n, n) for n in (record.get("node_ids", []) or [])
                ]
                record["new_node_ids"] = [
                    applied.get(n, n) for n in (record.get("new_node_ids", []) or [])
                ]
                record["fixed_nodes"] = [
                    applied.get(n, n) for n in (record.get("fixed_nodes", []) or [])
                ]
                record["needs_names"] = [
                    applied.get(n, n) for n in (record.get("needs_names", []) or [])
                ]
                anchor_node = record.get("anchor_node")
                if anchor_node is not None:
                    record["anchor_node"] = applied.get(anchor_node, anchor_node)
                updated_stages.append(record)
            self.meta["mapping_sequence"] = updated_stages
        back = self.meta.get("original_ids", {})
        back.update({new: old for old, new in applied.items()})
        self.meta["original_ids"] = back

        logger.info("rename_nodes: applied %d renames", len(applied))

        return applied, self


# -----------------------------
# D-Separation / cond. indep stuff
# -----------------------------
def powerset(s):
    """Yield all subsets of s as sets (including empty set)."""
    s = list(s)
    for k in range(len(s) + 1):
        for comb in it.combinations(s, k):
            yield set(comb)


def all_pairwise_cond_independencies(G):
    """
    Return all pairwise conditional independencies X ⟂ Y | Z
    implied by d-separation in a DAG G.

    Returns a list of (X, Y, frozenset(Z)), with X != Y.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("G must be a DAG for d-separation.")

    nodes = list(G.nodes())
    independencies = []

    for X, Y in it.combinations(nodes, 2):  # unordered pairs
        others = [v for v in nodes if v not in (X, Y)]
        for Z in powerset(others):
            if is_d_separator(G, {X}, {Y}, Z):
                independencies.append((X, Y, frozenset(Z)))

    return independencies


def minimal_pairwise_cond_independencies(G):
    nodes = list(G.nodes())
    independencies = []

    for X, Y in it.combinations(nodes, 2):
        others = [v for v in nodes if v not in (X, Y)]
        for Z in powerset(others):
            if is_minimal_d_separator(G, {X}, {Y}, Z):
                independencies.append((X, Y, frozenset(Z)))

    return independencies


# -----------------------------
# Utilities
# -----------------------------


def toposort(G: Any, strict: bool = True) -> List[str]:
    """
    Return a topological sort of nodes.
    Uses lexicographical_topological_sort for determinism if available.

    Args:
        G: The graph (networkx.DiGraph or similar).
        strict: If True, raises NetworkXUnfeasible (or similar) if G is not a DAG.
                If False, falls back to a deterministic sort (e.g. by node name) on failure.
    """
    if isinstance(G, nx.DiGraph):
        try:
            # Prefer deterministic sort
            if hasattr(nx, "lexicographical_topological_sort"):
                return list(nx.lexicographical_topological_sort(G))
            return list(nx.topological_sort(G))
        except Exception:
            if strict:
                raise
            # Fallback: deterministic by sorted node name
            return sorted(G.nodes())

    # Fallback for non-networkx objects or if G is not a DiGraph
    if hasattr(G, "nodes"):
        nodes = G.nodes()
        if callable(nodes):
            nodes = nodes()
        try:
            return sorted(list(nodes))
        except Exception:
            return list(nodes)

    return []


def toposort_node_subset(G: nx.DiGraph, nodes: List[str]) -> List[str]:
    """Return a subset of nodes in topological order."""
    node_set = set(nodes)
    topo = toposort(G, strict=False)
    return [node for node in topo if node in node_set]


def induced_edges(G: nx.DiGraph, nodes: List[str]) -> List[Tuple[str, str]]:
    """Return directed edges whose endpoints are both inside the given node subset."""
    node_set = set(nodes)
    return [(src, dst) for src, dst in G.edges() if src in node_set and dst in node_set]


def non_edge_pairs_undirected(G: nx.DiGraph, nodes: List[str]) -> List[Tuple[str, str]]:
    """Return unordered node pairs with no directed edge in either direction."""
    order = toposort_node_subset(G, nodes)
    pairs: List[Tuple[str, str]] = []
    for idx, src in enumerate(order):
        for dst in order[idx + 1 :]:
            if not G.has_edge(src, dst) and not G.has_edge(dst, src):
                pairs.append((src, dst))
    return pairs


def audit_edges(
    G: nx.DiGraph,
    observed_nodes: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str]]:
    """Return directed edges to audit, including those incident to latent nodes.

    Audit coverage should not be limited to the observed induced subgraph, because
    latent-edge semantics can still make a sampled graph nonsensical. This helper
    keeps a deterministic topological ordering while including every graph edge.
    """
    order = toposort(G, strict=False)
    rank = {node: idx for idx, node in enumerate(order)}
    return sorted(
        [(str(src), str(dst)) for src, dst in G.edges()],
        key=lambda edge: (
            rank.get(edge[0], len(rank)),
            rank.get(edge[1], len(rank)),
            edge[0],
            edge[1],
        ),
    )


def audit_non_edge_pairs_undirected(
    G: nx.DiGraph,
    observed_nodes: Optional[Iterable[str]] = None,
    include_latent_latent: bool = False,
) -> List[Tuple[str, str]]:
    """Return unordered non-edge pairs to audit.

    By default, this includes every non-edge with at least one observed endpoint
    and skips pure latent-latent pairs. That covers the important latent-related
    blind spots without introducing a large number of brittle latent-latent
    judgments.
    """
    observed = (
        {str(node) for node in observed_nodes}
        if observed_nodes is not None
        else {str(node) for node in G.nodes()}
    )
    order = [str(node) for node in toposort(G, strict=False)]
    pairs: List[Tuple[str, str]] = []
    for idx, src in enumerate(order):
        for dst in order[idx + 1 :]:
            if G.has_edge(src, dst) or G.has_edge(dst, src):
                continue
            if (
                not include_latent_latent
                and src not in observed
                and dst not in observed
            ):
                continue
            pairs.append((src, dst))
    return pairs


def _acyclic_add_edge(G: nx.DiGraph, u: str, v: str) -> bool:
    if u == v or G.has_edge(u, v):
        logger.debug("Skip add edge %s->%s: self-loop or already exists", u, v)
        return False
    G.add_edge(u, v)
    if not nx.is_directed_acyclic_graph(G):
        G.remove_edge(u, v)
        logger.debug("Reject edge %s->%s: would introduce a cycle", u, v)
        return False
    logger.debug("Added edge %s->%s", u, v)
    return True


def _augment_graph_to_n(
    G: nx.DiGraph,
    n_total: int,
    p_extra_edge: float = 0.25,
    ensure_connected: bool = False,
    connect_isolates: bool = True,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    rng = random.Random(seed)
    H = G.copy()
    logger.info(
        "Augmenting graph to %d nodes (p_extra_edge=%.3f, ensure_connected=%s, connect_isolates=%s)",
        n_total,
        p_extra_edge,
        ensure_connected,
        connect_isolates,
    )
    # Add new nodes V1, V2, ... that don't conflict with existing names
    i = 1
    while H.number_of_nodes() < n_total:
        name = f"V{i}"
        i += 1
        if name not in H:
            H.add_node(name)
    # Backbone on a topo-consistent order
    order = list(H.nodes())
    rng.shuffle(order)
    for i in range(len(order) - 1):
        _acyclic_add_edge(H, order[i], order[i + 1])
    for i, u in enumerate(order):
        for v in order[i + 1 :]:
            if rng.random() < p_extra_edge:
                _acyclic_add_edge(H, u, v)

    while nx.number_weakly_connected_components(H) > 1:
        topo = toposort(H)
        comps = [
            sorted(c, key=lambda n: topo.index(n))
            for c in nx.weakly_connected_components(H)
        ]
        for i in range(len(comps) - 1):
            _acyclic_add_edge(H, comps[i][-1], comps[i + 1][0])

    if connect_isolates:
        _attach_isolates(H, seed=rng.randrange(10**9))
    if ensure_connected:
        _backbone_and_bridge(H, seed=rng.randrange(10**9))

    logger.debug(
        "Augmented graph has %d nodes, %d edges",
        H.number_of_nodes(),
        H.number_of_edges(),
    )
    return H


def _attach_isolates(G: nx.DiGraph, seed: Optional[int] = None) -> None:
    """Give each isolate at least one in- or out-edge while preserving acyclicity."""

    rng = random.Random(seed)
    topo = toposort(G)
    index = {n: i for i, n in enumerate(topo)}
    isolates = [v for v in topo if G.in_degree(v) == 0 and G.out_degree(v) == 0]
    logger.debug("Attaching %d isolate(s)", len(isolates))
    for v in isolates:
        i = index[v]
        before = topo[:i]
        after = topo[i + 1 :]
        # choose in-edge or out-edge depending on availability (random if both)
        if before and after:
            if rng.random() < 0.5:
                u = rng.choice(before)
                _acyclic_add_edge(G, u, v)
            else:
                w = rng.choice(after)
                _acyclic_add_edge(G, v, w)
        elif before:
            u = rng.choice(before)
            _acyclic_add_edge(G, u, v)
        elif after:
            w = rng.choice(after)
            _acyclic_add_edge(G, v, w)
        # else: the graph has a single node; nothing to do


def _backbone_and_bridge(G, seed=None):
    rng = random.Random(seed)
    order = list(G.nodes())
    rng.shuffle(order)
    for i in range(len(order) - 1):
        _acyclic_add_edge(G, order[i], order[i + 1])
    if nx.number_weakly_connected_components(G) > 1:
        logger.debug(
            "Bridging %d weak components",
            nx.number_weakly_connected_components(G),
        )
        topo = toposort(G)
        comps = [
            sorted(c, key=lambda n: topo.index(n))
            for c in nx.weakly_connected_components(G)
        ]
        for i in range(len(comps) - 1):
            _acyclic_add_edge(G, comps[i][-1], comps[i + 1][0])


def _choose_treatment_outcome(
    nodes: Iterable[str], seed: Optional[int] = None
) -> Tuple[str, str]:
    rng = random.Random(seed)
    nodes = list(nodes)
    t, y = rng.sample(nodes, 2)
    logger.debug("Chosen treatment=%s, outcome=%s", t, y)
    return t, y


# NOTE: to be expanded for ADMGs
# Might need it for non-identifiability!
def add_latent_confounder(
    G: nx.DiGraph, a: str, b: str, u_name: str = "U"
) -> Tuple[nx.DiGraph, str]:
    """Inject a hidden parent U -> a, U -> b. Returns (graph, U_name)."""
    H = G.copy()
    base = u_name
    counter = 1
    while u_name in H:
        u_name = f"{base}_{counter}"
        counter += 1
    H.add_node(u_name)
    H.add_edge(u_name, a)
    H.add_edge(u_name, b)
    logger.info("Injected latent confounder %s -> {%s,%s}", u_name, a, b)
    return H, u_name


# -----------------------------
# NOTE: smart augmentation
# -----------------------------


def _subdivide_edge(H: nx.DiGraph, u: str, v: str, new_name: str) -> bool:
    """Replace u->v with u->new_name->v (preserves acyclicity & motif semantics).

    Returns True if subdivision succeeded, False if edge didn't exist.
    """
    if not H.has_edge(u, v):
        logger.error("Edge %s->%s not found for subdivision", u, v)
        return False

    H.remove_edge(u, v)
    H.add_node(new_name)
    H.add_edge(u, new_name)
    H.add_edge(new_name, v)
    return True


def _augment_by_edge_subdivision(
    G: nx.DiGraph,
    n_total: int,
    allowed_edges: Optional[List[Tuple[str, str]]] = None,
    seed: Optional[int] = None,
    prefix: str = "E",
) -> nx.DiGraph:
    """
    Insert new nodes only by subdividing edges in 'allowed_edges'.
    If allowed_edges=None, defaults to all current edges.
    """
    rng = random.Random(seed)
    H = G.copy()
    if H.number_of_nodes() >= n_total:
        return H

    # Start with allowed set (intersect with current edges)
    if allowed_edges is None:
        current = list(H.edges())
    else:
        current = [(u, v) for (u, v) in allowed_edges if H.has_edge(u, v)]

    i = 1

    def fresh():
        nonlocal i
        name = f"{prefix}{i}"
        while name in H:
            i += 1
            name = f"{prefix}{i}"
        i += 1
        return name

    # Subdivide edges until we reach target n_total
    # Each time we split (u,v), we allow further splits on the two new edges.
    while H.number_of_nodes() < n_total:
        if not current:
            logger.warning(
                "No more edges available for subdivision; stopping augmentation early at %d nodes",
                H.number_of_nodes(),
            )
            break

        u, v = rng.choice(current)
        new_name = fresh()

        if not _subdivide_edge(H, u, v, new_name):
            # Edge was already removed or doesn't exist; remove from pool and retry
            if (u, v) in current:
                current.remove((u, v))
            continue

        # Update the edge pool: replace (u,v) with (u,new) and (new,v)
        if (u, v) in current:
            current.remove((u, v))
        current.extend([(u, new_name), (new_name, v)])

    return H


# -----------------------------
# Motif builders
# -----------------------------


def _motif_chain(include_direct: bool = False) -> Tuple[nx.DiGraph, str, str]:
    # X -> M -> Y (optionally X->Y)
    G = nx.DiGraph()
    G.add_edges_from([("X", "M"), ("M", "Y")])
    if include_direct:
        G.add_edge("X", "Y")
    return G, "X", "Y"


def _motif_confounding(
    include_direct: bool = True,
) -> Tuple[nx.DiGraph, str, str]:
    # Z -> X, Z -> Y, optionally X -> Y
    G = nx.DiGraph()
    G.add_edges_from([("Z", "X"), ("Z", "Y")])
    if include_direct:
        G.add_edge("X", "Y")
    return G, "X", "Y"


def _motif_collider() -> Tuple[nx.DiGraph, str, str]:
    # X -> Z <- Y
    G = nx.DiGraph()
    G.add_edges_from([("X", "Z"), ("Y", "Z")])
    return G, "X", "Y"


def _motif_mediation(
    include_direct: bool = True,
) -> Tuple[nx.DiGraph, str, str]:
    return _motif_chain(include_direct=include_direct)


def _motif_iv() -> Tuple[nx.DiGraph, str, str, Dict[str, Any]]:
    # Valid IV with unobserved confounding U: Z -> X, X -> Y, U -> {X, Y}
    G = nx.DiGraph()
    G.add_edges_from([("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")])
    meta = {"latent_nodes": ["U"]}
    return G, "X", "Y", meta


def _motif_frontdoor() -> Tuple[nx.DiGraph, str, str, Dict[str, Any]]:
    """Front-door: X -> Z -> Y with unobserved U confounding X and Y."""
    G = nx.DiGraph()
    G.add_edges_from([("X", "Z"), ("Z", "Y"), ("U", "X"), ("U", "Y")])
    meta = {"latent_nodes": ["U"]}
    return G, "X", "Y", meta


# NOTE:  wrong in fig 6
def _motif_fork() -> Tuple[nx.DiGraph, str, str]:
    # Common-cause without X->Y
    G, X, Y = _motif_confounding(include_direct=False)
    return G, X, Y


def _motif_arrowhead() -> Tuple[nx.DiGraph, str, str]:
    G = nx.DiGraph()
    G.add_edges_from([("X", "V2"), ("V1", "V2"), ("X", "Y"), ("V1", "Y"), ("V2", "Y")])
    return G, "X", "Y"


def _motif_diamond(cross_edge: bool = False) -> Tuple[nx.DiGraph, str, str]:
    G = nx.DiGraph()
    G.add_edges_from([("X", "M1"), ("M1", "Y"), ("X", "M2"), ("M2", "Y")])
    if cross_edge:
        _acyclic_add_edge(G, "M1", "M2")
    return G, "X", "Y"


def _motif_diamondcut() -> Tuple[nx.DiGraph, str, str]:
    G = nx.DiGraph()
    G.add_edges_from([("V1", "V2"), ("V1", "X"), ("X", "Y"), ("V2", "Y")])
    return G, "X", "Y"


def _motif_double_nc(include_Z_to_A: bool = False, include_A_to_Y: bool = True):
    """
    Double Negative Control motif with latent U.
    Nodes: U (latent), A (treatment), Y (outcome), Z (NCE), W (NCO)
    Edges: U->{A,Y,Z,W}, optional Z->A (typical), optional A->Y (typical).
    Forbidden: A->W (NCO not affected by A), Z->Y (NCE not affecting Y).
    """
    G = nx.DiGraph()
    G.add_nodes_from(["U", "A", "Y", "Z", "W"])
    edges = [("U", "A"), ("U", "Y"), ("U", "Z"), ("U", "W")]
    if include_Z_to_A:
        edges.append(("Z", "A"))
    if include_A_to_Y:
        edges.append(("A", "Y"))
    G.add_edges_from(edges)
    # Mark U as latent, and annotate which nodes act as NCE/NCO
    meta = {"latent_nodes": ["U"], "nce": "Z", "nco": "W"}
    return G, "A", "Y", meta


def _motif_triangle() -> Tuple[nx.DiGraph, str, str]:
    """Triangle motif: X→Y, X→V3, Y→V3.  V3 is a collider on X and Y."""
    G = nx.DiGraph()
    G.add_edges_from([("X", "Y"), ("X", "V3"), ("Y", "V3")])
    return G, "X", "Y"


_MOTIF_BUILDERS = {
    Motif.CHAIN: _motif_chain,
    Motif.MEDIATION: _motif_mediation,
    Motif.CONFOUNDING: _motif_confounding,
    Motif.FORK: _motif_fork,
    Motif.COLLIDER: _motif_collider,
    Motif.ARROWHEAD: _motif_arrowhead,
    Motif.DIAMOND: _motif_diamond,
    Motif.DIAMONDCUT: _motif_diamondcut,
    Motif.FRONTDOOR: _motif_frontdoor,
    Motif.IV: _motif_iv,
    Motif.DOUBLE_NC: _motif_double_nc,
    Motif.TRIANGLE: _motif_triangle,
}


# NOTE: WIP - smart edge augmentation
# Which base edges are ALLOWED to be subdivided for each motif:
_RESTRICTED_AUG_EDGES = {
    Motif.CHAIN: [("X", "M"), ("M", "Y")],
    Motif.MEDIATION: [
        ("X", "M"),
        ("M", "Y"),
        ("X", "Y"),
    ],  # ("X", "Y") included for explicitness; subdividing direct effect is allowed
    Motif.COLLIDER: [("X", "Z"), ("Y", "Z")],
    Motif.FORK: [("Z", "X"), ("Z", "Y")],
    Motif.CONFOUNDING: [("Z", "X"), ("Z", "Y")],
    Motif.ARROWHEAD: [("X", "Y"), ("V1", "Y")],
    Motif.IV: [("Z", "X"), ("X", "Y")],  # preserve IV exclusion
    Motif.DIAMOND: [("X", "M1"), ("M1", "Y"), ("X", "M2"), ("M2", "Y")],
    Motif.DIAMONDCUT: [
        ("X", "V1"),
        ("V1", "Y"),
        ("X", "V2"),
        ("V2", "Y"),
    ],
    Motif.FRONTDOOR: [("X", "M"), ("M", "Y")],
    Motif.DOUBLE_NC: [("U", "A"), ("U", "Y"), ("U", "Z"), ("U", "W"), ("A", "Y")],
    Motif.TRIANGLE: [("X", "Y"), ("X", "V3"), ("Y", "V3")],
}


_BASIC_AUXILIARY_MOTIFS = (
    Motif.CHAIN,
    Motif.MEDIATION,
    Motif.CONFOUNDING,
    Motif.FORK,
    Motif.COLLIDER,
    Motif.TRIANGLE,
)

_DEFAULT_GRAFT_SAFE_MAIN_MOTIFS = (
    Motif.CHAIN,
    Motif.MEDIATION,
    Motif.CONFOUNDING,
    Motif.FORK,
    Motif.COLLIDER,
    Motif.ARROWHEAD,
    Motif.DIAMOND,
    Motif.DIAMONDCUT,
    Motif.IV,
    Motif.TRIANGLE,
)


def _coerce_motif_list(
    raw_motifs: Optional[List[Union[str, Motif]]],
    *,
    field_name: str,
) -> List[Motif]:
    """Parse a motif allowlist from config/user input."""
    pool: List[Motif] = []
    for raw in raw_motifs or []:
        if isinstance(raw, Motif):
            pool.append(raw)
            continue
        txt = str(raw).strip().lower()
        if not txt:
            continue
        try:
            pool.append(Motif(txt))
        except ValueError as exc:
            raise ValueError(
                f"Unknown motif '{raw}' in {field_name}. Available: {[m.value for m in Motif]}"
            ) from exc
    if raw_motifs is not None and not pool:
        raise ValueError(f"{field_name} was provided but no valid motifs were found.")
    return pool


def _resolve_main_graph_motif_pool(
    *,
    main_graph_motifs: Optional[List[Union[str, Motif]]],
) -> List[Motif]:
    """Resolve the graft-safe motif pool for sampled main graphs."""
    if main_graph_motifs is None:
        return list(_DEFAULT_GRAFT_SAFE_MAIN_MOTIFS)
    return _coerce_motif_list(main_graph_motifs, field_name="main_graph_motifs")


def _normalize_requested_motif(
    motif: Optional[Union[str, Motif]],
) -> Optional[Union[str, Motif]]:
    """Normalize motif spec while preserving the random-DAG / random-motif sentinels."""
    if motif is None:
        return None
    if isinstance(motif, Motif):
        return motif
    txt = str(motif).strip().lower()
    if not txt or txt in {"none", "random_dag"}:
        return None
    if txt == "random":
        return "random"
    try:
        return Motif(txt)
    except ValueError as exc:
        raise ValueError(
            f"Unknown motif '{motif}'. Available: {[m.value for m in Motif]} plus 'random'/'none'."
        ) from exc


def _resolve_requested_graft_count(
    *,
    augmentation_mode: str,
    aux_graft_count: int,
    rng: random.Random,
) -> int:
    """Resolve how many auxiliary graphs to attempt for this sampled graph."""
    max_grafts = max(0, int(aux_graft_count or 0))
    mode = str(augmentation_mode or "optional").strip().lower()
    if max_grafts <= 0 or mode in {"none", "off", "disabled"}:
        return 0
    if mode in {"optional", "sample", "sampled"}:
        return rng.randint(0, max_grafts)
    if mode in {"fixed", "always", "anchor_auxiliary_graft", "anchor_graft"}:
        return max_grafts
    raise ValueError(
        "augmentation_mode must be one of: none, optional, fixed "
        "(legacy aliases: anchor_auxiliary_graft, anchor_graft)."
    )


def _resolve_main_graph_motif_for_sampling(
    *,
    motif: Optional[Union[str, Motif]],
    requested_grafts: int,
    main_graph_restrict_when_grafting: bool,
    main_graph_motifs: Optional[List[Union[str, Motif]]],
    rng: random.Random,
) -> Optional[Union[str, Motif]]:
    """Resolve the effective main-graph motif for this scene."""
    normalized = _normalize_requested_motif(motif)

    if normalized is None:
        return None

    if normalized == "random":
        pool = list(Motif)
        if requested_grafts > 0 and main_graph_restrict_when_grafting:
            pool = _resolve_main_graph_motif_pool(main_graph_motifs=main_graph_motifs)
        return rng.choice(pool)

    return normalized


def _resolve_auxiliary_motif_pool(
    *,
    restrict_to_basic: bool,
    custom_motifs: Optional[List[Union[str, Motif]]],
) -> List[Motif]:
    """Resolve auxiliary motif candidates for anchor-graft augmentation."""
    if custom_motifs:
        return _coerce_motif_list(custom_motifs, field_name="auxiliary_custom_motifs")

    if restrict_to_basic:
        return list(_BASIC_AUXILIARY_MOTIFS)

    return list(Motif)


def _resolve_auxiliary_motif_weights(
    *,
    motif_pool: List[Motif],
    custom_motif_weights: Optional[Dict[str, float]],
) -> List[float]:
    """Resolve optional auxiliary-motif sampling weights aligned to motif_pool."""
    if not custom_motif_weights:
        return [1.0] * len(motif_pool)

    raw_weights = {
        str(key).strip().lower(): float(value)
        for key, value in dict(custom_motif_weights).items()
    }
    unknown = sorted(set(raw_weights) - {motif.value for motif in motif_pool})
    if unknown:
        raise ValueError(
            "auxiliary_motif_weights references motifs outside the auxiliary pool: "
            f"{unknown}"
        )

    weights: List[float] = []
    for motif in motif_pool:
        weight = float(raw_weights.get(motif.value, 1.0))
        if weight < 0.0:
            raise ValueError(
                f"auxiliary_motif_weights[{motif.value!r}] must be >= 0, got {weight}"
            )
        weights.append(weight)

    if not any(weight > 0.0 for weight in weights):
        raise ValueError(
            "auxiliary_motif_weights must contain at least one positive weight."
        )
    return weights


def _fresh_aux_node_name(H: nx.DiGraph, graft_index: int, role_name: str) -> str:
    """Return a collision-free node name for auxiliary graft nodes."""
    base = f"AUX{graft_index}_{role_name}"
    cand = base
    idx = 2
    while cand in H:
        cand = f"{base}_{idx}"
        idx += 1
    return cand


def _graft_auxiliary_motifs(
    *,
    base: SampledGraph,
    n_grafts: int,
    grafting_policy: str,
    main_graph_restrict_when_grafting: bool,
    main_graph_motifs: Optional[List[Union[str, Motif]]],
    restrict_to_basic: bool,
    custom_motifs: Optional[List[Union[str, Motif]]],
    custom_motif_weights: Optional[Dict[str, float]],
    allow_treatment_outcome_anchor: bool,
    preserve_treatment_outcome: bool,
    max_retries_per_graft: int,
    require_all_grafts: bool,
    rng: random.Random,
) -> SampledGraph:
    """Attach auxiliary motifs through a single shared anchor node."""
    if n_grafts <= 0:
        return base

    motif_pool = _resolve_auxiliary_motif_pool(
        restrict_to_basic=restrict_to_basic,
        custom_motifs=custom_motifs,
    )
    motif_weights = _resolve_auxiliary_motif_weights(
        motif_pool=motif_pool,
        custom_motif_weights=custom_motif_weights,
    )
    if not motif_pool:
        logger.warning(
            "Auxiliary graft mode requested but motif pool is empty; skipping."
        )
        return base

    logger.info(
        "Starting auxiliary graft augmentation: requested_grafts=%d, restrict_basic=%s, preserve_treatment_outcome=%s",
        n_grafts,
        restrict_to_basic,
        preserve_treatment_outcome,
    )

    H = base.graph.copy()
    base_meta = copy.deepcopy(base.meta) if isinstance(base.meta, dict) else {}
    latent_nodes = set(base.latent_nodes or base_meta.get("latent_nodes", []))
    main_graph_nodes = toposort_node_subset(H, list(H.nodes()))

    auxiliary_graph_grafts: List[Dict[str, Any]] = []
    mapping_sequence: List[Dict[str, Any]] = [
        {
            "stage_id": "main_graph",
            "kind": "main_graph",
            "order": 0,
            "node_ids": main_graph_nodes,
            "fixed_nodes": [],
            "needs_names": main_graph_nodes,
            "anchor_node": None,
            "new_node_ids": [],
            "motif": base.motif,
        }
    ]

    applied_grafts = 0
    for graft_index in range(1, n_grafts + 1):
        graft_success = False
        for attempt in range(1, max(1, max_retries_per_graft) + 1):
            motif_choice = rng.choices(motif_pool, weights=motif_weights, k=1)[0]
            aux = build_motif(motif_choice)
            aux_graph = aux.graph.copy()
            aux_meta = aux.meta or {}
            aux_latent = set(aux.latent_nodes or aux_meta.get("latent_nodes", []))

            anchor_role_candidates = [
                n for n in aux_graph.nodes() if n not in aux_latent
            ]
            if not anchor_role_candidates:
                anchor_role_candidates = list(aux_graph.nodes())
            if not anchor_role_candidates:
                continue

            anchor_candidates = [n for n in H.nodes() if n not in latent_nodes]
            if not allow_treatment_outcome_anchor:
                anchor_candidates = [
                    n
                    for n in anchor_candidates
                    if n not in (base.treatment, base.outcome)
                ]
            if not anchor_candidates:
                anchor_candidates = [n for n in H.nodes() if n not in latent_nodes]
            if not anchor_candidates:
                continue

            anchor_role = rng.choice(anchor_role_candidates)
            anchor_node = rng.choice(anchor_candidates)

            relabel_map: Dict[str, str] = {}
            for node in aux_graph.nodes():
                if node == anchor_role:
                    relabel_map[node] = anchor_node
                else:
                    relabel_map[node] = _fresh_aux_node_name(H, graft_index, str(node))

            aux_relabeled = nx.relabel_nodes(aux_graph, relabel_map, copy=True)
            H_candidate = nx.compose(H, aux_relabeled)
            if not nx.is_directed_acyclic_graph(H_candidate):
                logger.debug(
                    "Rejected graft %d attempt %d (%s): cycle introduced",
                    graft_index,
                    attempt,
                    motif_choice.value,
                )
                continue

            prev_edges = set(H.edges())
            H = H_candidate
            applied_grafts += 1
            graft_success = True

            new_nodes = [relabel_map[n] for n in aux_graph.nodes() if n != anchor_role]
            auxiliary_graph_nodes = toposort_node_subset(H, [anchor_node] + new_nodes)
            new_nodes_topo = toposort_node_subset(H, new_nodes)
            mapped_latent = [
                relabel_map[n]
                for n in aux_latent
                if n in relabel_map and relabel_map[n] != anchor_node
            ]
            latent_nodes.update(mapped_latent)
            edges_added = [
                (u, v) for (u, v) in aux_relabeled.edges() if (u, v) not in prev_edges
            ]

            auxiliary_graph_grafts.append(
                {
                    "stage_id": f"graft_{graft_index}",
                    "order": graft_index,
                    "motif": motif_choice.value,
                    "anchor_node": anchor_node,
                    "anchor_role": anchor_role,
                    "auxiliary_relabel_map": relabel_map,
                    "auxiliary_graph_nodes": auxiliary_graph_nodes,
                    "new_nodes": new_nodes_topo,
                    "edges_added": edges_added,
                    "latent_nodes_added": sorted(mapped_latent),
                }
            )
            mapping_sequence.append(
                {
                    "stage_id": f"graft_{graft_index}",
                    "kind": "auxiliary_graph",
                    "order": graft_index,
                    "motif": motif_choice.value,
                    "node_ids": auxiliary_graph_nodes,
                    "new_node_ids": new_nodes_topo,
                    "anchor_node": anchor_node,
                    "fixed_nodes": [anchor_node],
                    "needs_names": new_nodes_topo,
                }
            )

            logger.info(
                "Applied auxiliary graft %d/%d: motif=%s, anchor=%s, new_nodes=%s",
                graft_index,
                n_grafts,
                motif_choice.value,
                anchor_node,
                new_nodes_topo,
            )
            break

        if not graft_success:
            msg = (
                f"Failed to graft auxiliary motif {graft_index}/{n_grafts} "
                f"within max_retries_per_graft={max_retries_per_graft}."
            )
            if require_all_grafts:
                raise RuntimeError(msg)
            logger.warning("%s Stopping further grafts.", msg)
            break

    treatment = base.treatment
    outcome = base.outcome
    if not preserve_treatment_outcome:
        selectable = [n for n in H.nodes() if n not in latent_nodes]
        if len(selectable) >= 2:
            treatment, outcome = _choose_treatment_outcome(
                selectable, seed=rng.randrange(10**9)
            )

    meta = base_meta or {}
    meta["latent_nodes"] = sorted(latent_nodes)
    meta["augmentation"] = {
        "mode": "anchor_auxiliary_graft",
        "policy": str(grafting_policy or "fixed").strip().lower(),
        "requested_grafts": int(n_grafts),
        "applied_grafts": int(applied_grafts),
        "main_graph_restrict_when_grafting": bool(main_graph_restrict_when_grafting),
        "main_graph_motifs": (
            [
                m.value
                for m in _resolve_main_graph_motif_pool(
                    main_graph_motifs=main_graph_motifs
                )
            ]
            if main_graph_restrict_when_grafting
            else None
        ),
        "restrict_to_basic_aux_motifs": bool(restrict_to_basic),
        "aux_motif_pool": [m.value for m in motif_pool],
        "aux_motif_weights": {
            motif.value: float(weight)
            for motif, weight in zip(motif_pool, motif_weights)
        },
        "preserve_treatment_outcome": bool(preserve_treatment_outcome),
        "allow_treatment_outcome_anchor": bool(allow_treatment_outcome_anchor),
    }
    meta["main_graph"] = {
        "stage_id": "main_graph",
        "node_ids": main_graph_nodes,
        "motif": base.motif,
    }
    meta["auxiliary_graph_grafts"] = auxiliary_graph_grafts
    meta["mapping_sequence"] = mapping_sequence

    observed_nodes = [n for n in H.nodes() if n not in latent_nodes]
    logger.info(
        "Completed auxiliary graft augmentation: applied_grafts=%d, nodes=%d, edges=%d",
        applied_grafts,
        H.number_of_nodes(),
        H.number_of_edges(),
    )
    return SampledGraph(
        graph=H,
        treatment=treatment,
        outcome=outcome,
        motif=base.motif,
        observed_nodes=observed_nodes,
        latent_nodes=sorted(latent_nodes),
        meta=meta,
    )


def build_motif(motif: Union[str, Motif], **kwargs) -> SampledGraph:

    # Convert string to Motif enum if needed
    if isinstance(motif, str):
        try:
            motif_enum = Motif(motif.lower())
        except ValueError:
            raise ValueError(
                f"Unknown motif '{motif}'. Available: {[m.value for m in Motif]}"
            )
    else:
        motif_enum = motif

    if motif_enum not in _MOTIF_BUILDERS:
        raise ValueError(
            f"Unknown motif '{motif_enum}'. Available: {[m.value for m in Motif]}"
        )

    ret = _MOTIF_BUILDERS[motif_enum](**kwargs)
    try:
        G, X, Y, meta = ret
    except ValueError:
        G, X, Y = ret
        meta = {}

    logger.info(
        "Built motif '%s' with treatment=%s, outcome=%s, meta=%s",
        motif_enum.value,
        X,
        Y,
        meta,
    )
    return SampledGraph(
        graph=G,
        treatment=X,
        outcome=Y,
        motif=motif_enum.value,
        observed_nodes=list(G.nodes()),
        latent_nodes=meta.get("latent_nodes", []),
        meta=meta | {"kwargs": kwargs},
    )


# -----------------------------
# Random DAGs + identifiability gating
# -----------------------------


# Modification for more connected graphs
def sample_random_dag(
    n_nodes: int,
    edge_prob: float = 0.25,
    ensure_connected: bool = False,
    connect_isolates: bool = True,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    assert n_nodes >= 2
    rng = random.Random(seed)
    logger.info(
        "Sampling random DAG (n_nodes=%d, edge_prob=%.3f, ensure_connected=%s, connect_isolates=%s)",
        n_nodes,
        edge_prob,
        ensure_connected,
        connect_isolates,
    )
    # NOTE: to be consistent with augmentation, which starts at 1
    order = [f"V{i + 1}" for i in range(n_nodes)]
    rng.shuffle(order)
    G = nx.DiGraph()
    G.add_nodes_from(order)

    # 1) Backbone chain for guaranteed weak connectivity
    for i in range(n_nodes - 1):
        _acyclic_add_edge(G, order[i], order[i + 1])

    # 2) Sprinkle extra forward edges
    for i, u in enumerate(order):
        for v in order[i + 1 :]:
            if rng.random() < edge_prob:
                _acyclic_add_edge(G, u, v)

    # 3) Bridge any remaining weakly-connected components
    #    (usually unnecessary with the chain, but harmless)
    while nx.number_weakly_connected_components(G) > 1:
        topo = toposort(G)
        comps = [
            sorted(c, key=lambda n: topo.index(n))
            for c in nx.weakly_connected_components(G)
        ]
        for i in range(len(comps) - 1):
            u, v = comps[i][-1], comps[i + 1][0]
            _acyclic_add_edge(G, u, v)

    if connect_isolates:
        _attach_isolates(G, seed=rng.randrange(10**9))
    if ensure_connected:
        _backbone_and_bridge(G, seed=rng.randrange(10**9))

    logger.debug(
        "Sampled random DAG: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def is_identifiable(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    observed_nodes: Optional[Iterable[str]] = None,
    latent_nodes: Optional[Iterable[str]] = None,
) -> Optional[bool]:
    """Return whether the population ATE is identifiable under the graph."""
    # Validate that treatment and outcome exist in graph
    if treatment not in G.nodes():
        logger.warning(
            "is_identifiable: treatment '%s' not in graph nodes %s",
            treatment,
            list(G.nodes()),
        )
        return False
    if outcome not in G.nodes():
        logger.warning(
            "is_identifiable: outcome '%s' not in graph nodes %s",
            outcome,
            list(G.nodes()),
        )
        return False

    result = identify_population_ate(
        G,
        treatment,
        outcome,
        observed_nodes=observed_nodes,
        latent_nodes=latent_nodes,
    )
    logger.debug(
        "Population-ATE identifiability via %s: identifiable=%s, error=%s",
        result.engine,
        result.identifiable,
        result.error_type,
    )
    return result.identifiable


def _build_sample_base_graph(
    *,
    motif: Optional[Union[str, Motif]],
    n_nodes: Optional[int],
    p_extra_edge: float,
    ensure_connected: bool,
    connect_isolates: bool,
    augmentation_mode: str,
    aux_graft_count: int,
    main_graph_restrict_when_grafting: bool,
    main_graph_motifs: Optional[List[Union[str, Motif]]],
    aux_restrict_basic_motifs: bool,
    aux_custom_motifs: Optional[List[Union[str, Motif]]],
    aux_custom_motif_weights: Optional[Dict[str, float]],
    aux_allow_treatment_outcome_anchor: bool,
    aux_preserve_treatment_outcome: bool,
    aux_max_retries_per_graft: int,
    aux_require_all_grafts: bool,
    rng: random.Random,
) -> SampledGraph:
    """Build a motif-based or random base graph before latent injection."""
    requested_motif = motif
    requested_grafts = _resolve_requested_graft_count(
        augmentation_mode=augmentation_mode,
        aux_graft_count=aux_graft_count,
        rng=rng,
    )
    motif = _resolve_main_graph_motif_for_sampling(
        motif=motif,
        requested_grafts=requested_grafts,
        main_graph_restrict_when_grafting=main_graph_restrict_when_grafting,
        main_graph_motifs=main_graph_motifs,
        rng=rng,
    )
    logger.info(
        "Resolved main graph sampling: requested_motif=%s, effective_motif=%s, augmentation_mode=%s, drawn_grafts=%d",
        requested_motif,
        motif,
        augmentation_mode,
        requested_grafts,
    )

    if motif is None:
        G = sample_random_dag(
            n_nodes=n_nodes,
            edge_prob=p_extra_edge,
            ensure_connected=ensure_connected,
            connect_isolates=connect_isolates,
            seed=rng.randrange(10**9),
        )
        X, Y = _choose_treatment_outcome(G.nodes(), seed=rng.randrange(10**9))
        base_sample = SampledGraph(
            graph=G,
            treatment=X,
            outcome=Y,
            motif="random",
            observed_nodes=list(G.nodes()),
            latent_nodes=[],
            meta={},
        )
    else:
        sg = build_motif(motif)
        G = sg.graph
        base_n = G.number_of_nodes()
        target_n = n_nodes if n_nodes is not None else base_n
        if target_n < base_n:
            logger.info(
                "Requested n_nodes=%d is less than motif base nodes=%d; ignoring n_nodes",
                target_n,
                base_n,
            )
            target_n = base_n
        if target_n > base_n:
            motif_key = motif if isinstance(motif, Motif) else Motif(str(motif).lower())
            if motif_key in _RESTRICTED_AUG_EDGES:
                G = _augment_by_edge_subdivision(
                    G,
                    n_total=target_n,
                    allowed_edges=_RESTRICTED_AUG_EDGES[motif_key],
                    seed=rng.randrange(10**9),
                    prefix="E",
                )
            else:
                G = _augment_graph_to_n(
                    G,
                    n_total=target_n,
                    p_extra_edge=p_extra_edge,
                    ensure_connected=ensure_connected,
                    connect_isolates=connect_isolates,
                    seed=rng.randrange(10**9),
                )

        motif_latent = sg.meta.get("latent_nodes", []) if sg.meta else []
        obs_nodes = [v for v in G.nodes() if v not in motif_latent]
        base_sample = SampledGraph(
            graph=G,
            treatment=sg.treatment,
            outcome=sg.outcome,
            motif=sg.motif,
            observed_nodes=obs_nodes,
            latent_nodes=motif_latent,
            meta=sg.meta,
        )
    if requested_grafts > 0:
        base_sample = _graft_auxiliary_motifs(
            base=base_sample,
            n_grafts=requested_grafts,
            grafting_policy=augmentation_mode,
            main_graph_restrict_when_grafting=main_graph_restrict_when_grafting,
            main_graph_motifs=main_graph_motifs,
            restrict_to_basic=aux_restrict_basic_motifs,
            custom_motifs=aux_custom_motifs,
            custom_motif_weights=aux_custom_motif_weights,
            allow_treatment_outcome_anchor=aux_allow_treatment_outcome_anchor,
            preserve_treatment_outcome=aux_preserve_treatment_outcome,
            max_retries_per_graft=aux_max_retries_per_graft,
            require_all_grafts=aux_require_all_grafts,
            rng=rng,
        )
    return base_sample


def _inject_latent_confounders(
    *,
    base: SampledGraph,
    p_latent_xy: float,
    latent_pairs: Optional[List[Tuple[str, str]]],
    observed_nodes: Optional[List[str]],
    rng: random.Random,
) -> SampledGraph:
    """Inject optional latent confounders and build observed/latent node lists."""
    G = base.graph
    latent_nodes: List[str] = []
    if base.meta and "latent_nodes" in base.meta:
        latent_nodes.extend([n for n in base.meta["latent_nodes"] if n in G])

    if rng.random() < p_latent_xy:
        G, U = add_latent_confounder(G, base.treatment, base.outcome)
        latent_nodes.append(U)

    if latent_pairs:
        for a, b in latent_pairs:
            if a in G and b in G:
                G, U = add_latent_confounder(G, a, b)
                latent_nodes.append(U)

    if observed_nodes is None:
        obs = [v for v in G.nodes() if v not in latent_nodes]
    else:
        obs = [v for v in observed_nodes if v in G and v not in latent_nodes]

    return SampledGraph(
        graph=G,
        treatment=base.treatment,
        outcome=base.outcome,
        motif=base.motif,
        observed_nodes=obs,
        latent_nodes=latent_nodes,
        meta=base.meta,
    )


def sample_graph(
    motif: Optional[Union[str, Motif]] = None,
    n_nodes: Optional[int] = None,
    p_extra_edge: float = 0.2,
    p_latent_xy: float = 0.0,
    require_identifiable: bool = True,
    ensure_connected: bool = False,
    connect_isolates: bool = True,
    observed_nodes: Optional[List[str]] = None,
    latent_pairs: Optional[List[Tuple[str, str]]] = None,
    max_tries: int = 50,
    seed: Optional[int] = None,
    augmentation_mode: str = "optional",
    aux_graft_count: int = 1,
    main_graph_restrict_when_grafting: bool = True,
    main_graph_motifs: Optional[List[Union[str, Motif]]] = None,
    aux_restrict_basic_motifs: bool = True,
    aux_custom_motifs: Optional[List[Union[str, Motif]]] = None,
    aux_custom_motif_weights: Optional[Dict[str, float]] = None,
    aux_allow_treatment_outcome_anchor: bool = True,
    aux_preserve_treatment_outcome: bool = True,
    aux_max_retries_per_graft: int = 25,
    aux_require_all_grafts: bool = False,
) -> SampledGraph:
    """Sample a motif-based or random DAG with optional augmentation and ID gating.

    Parameters
    ----------
    motif : specific motif name, "random" for a random motif, or None/"none" for a pure random DAG
    n_nodes : total nodes after augmentation
    p_extra_edge : probability of adding a forward edge during augmentation/random sampling
    require_identifiable : if True, reject graphs whose population ATE is non-identifiable
    observed_nodes : the set of observed nodes for ID; if None, assume all current nodes are observed
    latent_pairs : list of (a,b) pairs to confound by injecting a hidden parent U -> a, U -> b
    max_tries : attempts before giving up when require_identifiable=True
    augmentation_mode : "none", "fixed", or "optional" for anchor-based motif grafting
    aux_graft_count : exact graft count in fixed mode, or the max count for Uniform{0..K} sampling in optional mode
    main_graph_restrict_when_grafting : if True, random main-graph motif sampling uses main_graph_motifs whenever grafting is drawn
    main_graph_motifs : optional graft-safe main motif allowlist (defaults to all motifs except frontdoor/double_nc)
    aux_restrict_basic_motifs : if True, restrict aux motif candidates to simple motifs
    aux_custom_motifs : optional explicit aux motif candidate list (overrides restriction flag)
    aux_custom_motif_weights : optional weight map for auxiliary motif sampling
    aux_allow_treatment_outcome_anchor : allow treatment/outcome to be selected as graft anchors
    aux_preserve_treatment_outcome : keep original treatment/outcome after grafting
    aux_max_retries_per_graft : retries per graft attempt when cycles/invalid merges occur
    aux_require_all_grafts : if True, fail when fewer than the configured/drawn graft count are applied
    """
    rng = random.Random(seed)
    logger.info(
        "Sampling graph (motif=%s, n_nodes=%s, p_extra_edge=%.3f, p_latent_xy=%.3f, require_identifiable=%s, augmentation_mode=%s, aux_graft_count=%d)",
        motif,
        n_nodes,
        p_extra_edge,
        p_latent_xy,
        require_identifiable,
        augmentation_mode,
        aux_graft_count,
    )
    has_identifier = has_population_ate_identifier()
    if require_identifiable and not has_identifier:
        raise RuntimeError(
            "y0 or DoWhy is required for require_identifiable=True. "
            "Install y0 or DoWhy to enable population-ATE identifiability gating."
        )

    tries = 0
    while True:
        tries += 1
        base = _build_sample_base_graph(
            motif=motif,
            n_nodes=n_nodes,
            p_extra_edge=p_extra_edge,
            ensure_connected=ensure_connected,
            connect_isolates=connect_isolates,
            augmentation_mode=augmentation_mode,
            aux_graft_count=aux_graft_count,
            main_graph_restrict_when_grafting=main_graph_restrict_when_grafting,
            main_graph_motifs=main_graph_motifs,
            aux_restrict_basic_motifs=aux_restrict_basic_motifs,
            aux_custom_motifs=aux_custom_motifs,
            aux_custom_motif_weights=aux_custom_motif_weights,
            aux_allow_treatment_outcome_anchor=aux_allow_treatment_outcome_anchor,
            aux_preserve_treatment_outcome=aux_preserve_treatment_outcome,
            aux_max_retries_per_graft=aux_max_retries_per_graft,
            aux_require_all_grafts=aux_require_all_grafts,
            rng=rng,
        )
        sg = _inject_latent_confounders(
            base=base,
            p_latent_xy=p_latent_xy,
            latent_pairs=latent_pairs,
            observed_nodes=observed_nodes,
            rng=rng,
        )

        id_ok = is_identifiable(
            sg.graph,
            sg.treatment,
            sg.outcome,
            observed_nodes=sg.observed_nodes,
            latent_nodes=sg.latent_nodes,
        )
        sg._identifiable = id_ok

        if has_identifier and require_identifiable:
            logger.debug("Try %d/%d: identifiability=%s", tries, max_tries, id_ok)
            if id_ok:
                logger.info(
                    "Sampled identifiable graph on try %d (nodes=%d, edges=%d, motif=%s)",
                    tries,
                    sg.graph.number_of_nodes(),
                    sg.graph.number_of_edges(),
                    sg.motif,
                )
                return sg
        else:
            logger.info(
                "Returning sampled graph without identifiability gating (backend=%s, require=%s)",
                has_identifier,
                require_identifiable,
            )
            return sg

        if tries >= max_tries:
            logger.error(
                "Failed to sample an identifiable graph within max_tries=%d",
                max_tries,
            )
            raise RuntimeError(
                "Failed to sample an identifiable graph within max_tries"
            )


# -----------------------------
# Backdoor / Adjustment Set Utilities
# -----------------------------


def get_node_roles(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> Dict[str, List[str]]:
    """Return node role classifications relative to treatment and outcome.

    Args:
        G: The causal DAG
        treatment: Treatment node ID
        outcome: Outcome node ID

    Returns:
        Dict with keys:
            - ancestors_of_treatment: nodes that causally precede treatment
            - ancestors_of_outcome: nodes that causally precede outcome
            - descendants_of_treatment: nodes causally downstream of treatment
            - descendants_of_outcome: nodes causally downstream of outcome
            - on_causal_path: nodes on directed paths from treatment to outcome
            - potential_confounders: ancestors of both treatment and outcome
    """
    # Ancestors (including the node itself in NetworkX convention)
    ancestors_t = nx.ancestors(G, treatment)
    ancestors_y = nx.ancestors(G, outcome)

    # Descendants
    descendants_t = nx.descendants(G, treatment)
    descendants_y = nx.descendants(G, outcome)

    # Nodes on causal paths from treatment to outcome
    # These are nodes that are both descendants of treatment AND ancestors of outcome
    on_causal_path = set()
    if outcome in descendants_t:
        # Find all simple paths and collect intermediate nodes
        try:
            for path in nx.all_simple_paths(G, treatment, outcome):
                on_causal_path.update(path[1:-1])  # Exclude treatment and outcome
        except nx.NetworkXNoPath:
            pass

    # Potential confounders: common ancestors of treatment and outcome
    potential_confounders = ancestors_t & ancestors_y

    return {
        "ancestors_of_treatment": sorted(ancestors_t - {treatment}),
        "ancestors_of_outcome": sorted(ancestors_y - {outcome}),
        "descendants_of_treatment": sorted(descendants_t - {outcome}),
        "descendants_of_outcome": sorted(descendants_y),
        "on_causal_path": sorted(on_causal_path),
        "potential_confounders": sorted(potential_confounders),
    }


def is_valid_adjustment_set(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    adjustment_set: Iterable[str],
    observed_nodes: Optional[Iterable[str]] = None,
) -> bool:
    """Check if an adjustment set is valid for backdoor adjustment.

    A valid adjustment set Z satisfies:
    1. Z blocks all backdoor paths from treatment to outcome
    2. Z does not contain any descendants of treatment that are not on the causal path

    Args:
        G: The causal DAG
        treatment: Treatment node ID
        outcome: Outcome node ID
        adjustment_set: Set of nodes to adjust for
        observed_nodes: Set of observed nodes (if None, all nodes are observed)

    Returns:
        True if the adjustment set is valid
    """
    Z = set(adjustment_set)
    obs = set(observed_nodes) if observed_nodes is not None else set(G.nodes())

    # Check that Z contains only observed nodes
    if not Z.issubset(obs):
        logger.debug("Adjustment set contains unobserved nodes")
        return False

    # Check that Z doesn't contain treatment or outcome
    if treatment in Z or outcome in Z:
        logger.debug("Adjustment set contains treatment or outcome")
        return False

    # Check that Z doesn't contain descendants of treatment (except those on causal path)
    descendants_t = nx.descendants(G, treatment)
    # Find nodes on causal paths
    on_causal_path = set()
    if outcome in descendants_t:
        try:
            for path in nx.all_simple_paths(G, treatment, outcome):
                on_causal_path.update(path[1:-1])
        except nx.NetworkXNoPath:
            pass

    forbidden = descendants_t - on_causal_path - {outcome}
    if Z & forbidden:
        logger.debug("Adjustment set contains forbidden descendants of treatment")
        return False

    # Check if Z d-separates treatment and outcome when we remove all directed paths
    # This is done by checking if conditioning on Z blocks all backdoor paths
    # We use the moral graph approach: create a graph without T->Y edges
    # and check d-separation

    # Create graph without direct causal paths from treatment to outcome
    H = G.copy()

    # Remove all edges on causal paths from treatment
    edges_to_remove = []
    for path in nx.all_simple_paths(G, treatment, outcome):
        for i in range(len(path) - 1):
            edges_to_remove.append((path[i], path[i + 1]))

    for u, v in edges_to_remove:
        if H.has_edge(u, v):
            H.remove_edge(u, v)

    # Now check if Z d-separates treatment from outcome in H
    # If outcome is still reachable from treatment, there are unblocked backdoor paths
    try:
        return is_d_separator(H, {treatment}, {outcome}, Z)
    except Exception:
        # Fallback: if d-separation check fails, assume invalid
        return False


def list_valid_adjustment_sets(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    observed_nodes: Optional[List[str]] = None,
    max_size: int = 5,
    max_sets: int = 100,
) -> List[List[str]]:
    """Enumerate valid backdoor adjustment sets.

    Args:
        G: The causal DAG
        treatment: Treatment node ID
        outcome: Outcome node ID
        observed_nodes: List of observed nodes (if None, all nodes are observed)
        max_size: Maximum size of adjustment sets to consider
        max_sets: Maximum number of sets to return

    Returns:
        List of valid adjustment sets (each as a sorted list of node IDs)
    """
    obs = list(observed_nodes) if observed_nodes is not None else list(G.nodes())

    # Exclude treatment and outcome from candidates
    candidates = [n for n in obs if n not in (treatment, outcome)]

    # Get descendants of treatment (forbidden to adjust for)
    descendants_t = nx.descendants(G, treatment)

    # Get nodes on causal paths (allowed even if descendants)
    on_causal_path = set()
    if outcome in descendants_t:
        try:
            for path in nx.all_simple_paths(G, treatment, outcome):
                on_causal_path.update(path[1:-1])
        except nx.NetworkXNoPath:
            pass

    # Filter candidates: remove forbidden descendants
    forbidden = descendants_t - on_causal_path - {outcome}
    candidates = [n for n in candidates if n not in forbidden]

    valid_sets = []

    # Check empty set first
    if is_valid_adjustment_set(G, treatment, outcome, set(), obs):
        valid_sets.append([])

    # Enumerate subsets of increasing size
    for size in range(1, min(max_size + 1, len(candidates) + 1)):
        if len(valid_sets) >= max_sets:
            break

        for subset in it.combinations(candidates, size):
            if len(valid_sets) >= max_sets:
                break

            if is_valid_adjustment_set(G, treatment, outcome, set(subset), obs):
                valid_sets.append(sorted(subset))

    logger.debug(
        "Found %d valid adjustment sets for %s -> %s",
        len(valid_sets),
        treatment,
        outcome,
    )
    return valid_sets


def list_minimal_adjustment_sets(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    observed_nodes: Optional[List[str]] = None,
    max_sets: int = 50,
) -> List[List[str]]:
    """Find minimal valid backdoor adjustment sets.

    A minimal set is one where no proper subset is also a valid adjustment set.

    Args:
        G: The causal DAG
        treatment: Treatment node ID
        outcome: Outcome node ID
        observed_nodes: List of observed nodes (if None, all nodes are observed)
        max_sets: Maximum number of sets to return

    Returns:
        List of minimal valid adjustment sets
    """
    # Get all valid sets (up to a reasonable size)
    all_valid = list_valid_adjustment_sets(
        G, treatment, outcome, observed_nodes, max_size=5, max_sets=500
    )

    if not all_valid:
        return []

    # Sort by size
    all_valid.sort(key=len)

    # Filter to minimal sets
    minimal = []
    for candidate in all_valid:
        if len(minimal) >= max_sets:
            break

        candidate_set = frozenset(candidate)
        # Check if any existing minimal set is a subset
        is_superset_of_minimal = any(
            frozenset(m).issubset(candidate_set) and frozenset(m) != candidate_set
            for m in minimal
        )

        if not is_superset_of_minimal:
            minimal.append(candidate)

    return minimal


def compute_v_structures(G: nx.DiGraph) -> List[List[str]]:
    """Compute all v-structures (colliders) in a DAG.

    A v-structure is a triple (u, w, z) where u → z ← w and there is no
    direct edge between u and w in either direction.

    Args:
        G: The causal DAG

    Returns:
        List of [parent1, parent2, child] triples representing v-structures
    """
    vstructs: List[List[str]] = []
    if not isinstance(G, nx.DiGraph):
        return vstructs
    for z in G.nodes():
        parents = list(G.predecessors(z))
        if len(parents) < 2:
            continue
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                u, w = parents[i], parents[j]
                if not G.has_edge(u, w) and not G.has_edge(w, u):
                    vstructs.append([u, w, z])
    return vstructs


def get_collider_nodes(
    G: nx.DiGraph,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
) -> List[str]:
    """Get collider (v-structure child) nodes, optionally filtered by query relevance.

    Without treatment/outcome args, returns ALL v-structure child nodes.
    With treatment/outcome, filters to colliders whose parents connect to
    both sides of the treatment-outcome query.

    Args:
        G: The causal DAG
        treatment: Optional treatment node ID for query-specific filtering
        outcome: Optional outcome node ID for query-specific filtering

    Returns:
        Sorted list of collider node IDs (excluding treatment and outcome)
    """
    vstructs = compute_v_structures(G)
    # All child nodes in v-structures
    all_colliders = sorted({vs[2] for vs in vstructs} - {treatment, outcome})

    if treatment is None or outcome is None:
        return all_colliders

    # Filter to query-relevant colliders: parents from both T-side and Y-side
    ancestors_t = nx.ancestors(G, treatment) | {treatment}
    ancestors_y = nx.ancestors(G, outcome) | {outcome}
    descendants_t = nx.descendants(G, treatment) | {treatment}
    descendants_y = nx.descendants(G, outcome) | {outcome}
    t_side = ancestors_t | descendants_t
    y_side = ancestors_y | descendants_y

    relevant = []
    for vs in vstructs:
        u, w, z = vs
        if z in (treatment, outcome):
            continue
        parents = {u, w}
        if (parents & t_side) and (parents & y_side):
            relevant.append(z)

    return sorted(set(relevant))


def get_forbidden_conditioning_vars(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> Dict[str, List[str]]:
    """Identify variables that should NOT be conditioned on.

    Args:
        G: The causal DAG
        treatment: Treatment node ID
        outcome: Outcome node ID

    Returns:
        Dict with keys:
            - descendants: descendants of treatment that would block causal effect
            - colliders: collider nodes (would open backdoor paths if conditioned on)
    """
    # Descendants of treatment (excluding those on causal path to outcome)
    descendants_t = nx.descendants(G, treatment)

    # Find nodes on causal paths
    on_causal_path = set()
    if outcome in descendants_t:
        try:
            for path in nx.all_simple_paths(G, treatment, outcome):
                on_causal_path.update(path[1:-1])
        except nx.NetworkXNoPath:
            pass

    # Forbidden descendants (would bias the estimate)
    forbidden_desc = sorted(descendants_t - on_causal_path - {outcome})

    # Find colliders via v-structures (proper structural definition)
    colliders = get_collider_nodes(G, treatment, outcome)

    return {
        "descendants": forbidden_desc,
        "colliders": colliders,
    }


def resolve_structural_label(
    metadata: Optional[Dict[str, Any]] = None,
    graph_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve the exam-side structural label for a scene.

    Structural label is the conditioning key used at exam construction time:
    - non-grafted scenes use the scene motif
    - scenes with one or more applied auxiliary grafts use ``grafted``

    This helper accepts the scene ``metadata`` and/or the private ``graph``
    payload from ``ground_truth.json`` so callers can use the same logic during
    generation and exam sampling.
    """
    metadata = metadata or {}
    graph_info = graph_info or {}

    structural_label = metadata.get("structural_label")
    if structural_label is not None:
        return str(structural_label)

    augmentation = graph_info.get("augmentation")
    if isinstance(augmentation, dict):
        applied_grafts = augmentation.get("applied_grafts")
        try:
            if applied_grafts is not None and int(applied_grafts) > 0:
                return "grafted"
        except (TypeError, ValueError):
            pass

    auxiliary_graph_grafts = graph_info.get("auxiliary_graph_grafts")
    if isinstance(auxiliary_graph_grafts, list) and auxiliary_graph_grafts:
        return "grafted"

    motif = metadata.get("motif")
    if motif is not None:
        return str(motif)

    main_graph = graph_info.get("main_graph")
    if isinstance(main_graph, dict):
        main_motif = main_graph.get("motif")
        if main_motif is not None:
            return str(main_motif)

    return None


# -----------------------------
# Convenience: a small motif registry (exportable)
# -----------------------------
MOTIFS = tuple(m.value for m in Motif)

__all__ = [
    "Motif",
    "SampledGraph",
    "build_motif",
    "is_identifiable",
    "sample_random_dag",
    "sample_graph",
    "add_latent_confounder",
    "MOTIFS",
    # New exports for backdoor utilities
    "get_node_roles",
    "is_valid_adjustment_set",
    "list_valid_adjustment_sets",
    "list_minimal_adjustment_sets",
    "get_forbidden_conditioning_vars",
    "resolve_structural_label",
    "compute_v_structures",
    "get_collider_nodes",
]
