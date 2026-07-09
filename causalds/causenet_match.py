import copy
import logging
import random
from typing import Dict, List, Optional, Set, Tuple

from causalds.causenet_extract import (
    choose_best_nl_source,
    extract_nl_provenance_fields,
)
from causalds.graph import SampledGraph

logger = logging.getLogger(__name__)


def _simple_causenet_pair_quality_key(
    support: int,
    has_wiki: bool,
) -> Tuple[int, int]:
    """Return the existing simple-matcher quality ordering.

    Historically, the simple matcher treated candidate quality lexicographically:
    Wikipedia provenance first, then CauseNet support. The default sampling path
    still keeps the current filtered-uniform behavior, while the optional
    weighted mode reuses these same quality signals explicitly.
    """
    return (1 if has_wiki else 0, max(int(support or 0), 0))


def _simple_causenet_pair_weight(
    support: int,
    has_wiki: bool,
    *,
    max_support: int,
    mixed_wiki_candidates: bool,
) -> float:
    """Convert the existing quality ordering into a positive sampling weight.

    When both wiki-backed and non-wiki candidates are present, every wiki-backed
    candidate gets a bonus of ``max_support + 1``. This preserves the original
    lexicographic preference: wiki-backed candidates always outrank non-wiki
    ones, and support orders candidates within each group. If all candidates
    share the same wiki status, weighting reduces to support-only sampling.
    """
    base_support = max(int(support or 0), 1)
    if mixed_wiki_candidates and has_wiki:
        return float(base_support + max_support + 1)
    return float(base_support)


def _weighted_choice_index(rng: random.Random, weights: List[float]) -> int:
    """Select one index proportionally to a list of non-negative weights."""
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))

    threshold = rng.random() * total
    cumulative = 0.0
    for idx, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= threshold:
            return idx
    return len(weights) - 1


def _order_simple_causenet_candidates(
    candidates: List[Tuple[str, str, int, bool]],
    *,
    rng: random.Random,
    sampling_strategy: str,
    max_items: int,
) -> List[Tuple[str, str, int, bool]]:
    """Produce the candidate visitation order for simple CauseNet matching.

    ``uniform`` preserves the current behavior: after filtering, candidates are
    shuffled uniformly and tried in that order.

    ``quality_weighted`` samples without replacement using the existing
    lexicographic quality signals from :func:`_simple_causenet_pair_quality_key`.
    """
    ordered = list(candidates)
    if not ordered:
        return ordered

    if sampling_strategy == "uniform":
        rng.shuffle(ordered)
        return ordered[:max_items]

    if sampling_strategy != "quality_weighted":
        logger.warning(
            "Unknown simple CauseNet sampling strategy '%s'; falling back to uniform",
            sampling_strategy,
        )
        rng.shuffle(ordered)
        return ordered[:max_items]

    mixed_wiki_candidates = any(has_wiki for _, _, _, has_wiki in ordered) and any(
        not has_wiki for _, _, _, has_wiki in ordered
    )
    max_support = max((support for _, _, support, _ in ordered), default=0)
    remaining = list(ordered)
    sampled: List[Tuple[str, str, int, bool]] = []

    while remaining and len(sampled) < max_items:
        weights = [
            _simple_causenet_pair_weight(
                support,
                has_wiki,
                max_support=max_support,
                mixed_wiki_candidates=mixed_wiki_candidates,
            )
            for _, _, support, has_wiki in remaining
        ]
        idx = _weighted_choice_index(rng, weights)
        sampled.append(remaining.pop(idx))

    return sampled


def _choose(rng, S):
    S = list(S)
    result = rng.choice(S) if S else None
    if result is None:
        logger.debug("_choose: empty set provided")
    return result


def _choose2(rng, S):
    """Randomly select two distinct elements from a set."""
    S = list(S)
    result = tuple(rng.sample(S, 2)) if len(S) >= 2 else None
    if result is None:
        logger.debug("_choose2: insufficient elements (need 2, got %d)", len(S))
    return result


def _distinct(*xs):
    is_distinct = len(set(xs)) == len(xs)
    if not is_distinct:
        logger.debug("_distinct check failed for: %s", xs)
    return is_distinct


########################################################################################
# METADATA HELPER
########################################################################################
def _make_prov_record(cause: str, effect: str, rec) -> Dict:
    """Standardize provenance records and attach NL-friendly fields."""
    logger.debug("Creating provenance record: %s -> %s", cause, effect)
    sources = rec.sources if rec else None
    nl_sources = extract_nl_provenance_fields(sources)
    return {
        "cause": cause,
        "effect": effect,
        "support": (rec.support if rec else None),
        "sources": sources,
        "nl_sources": nl_sources,
        "best_nl_source": choose_best_nl_source(nl_sources),
    }


########################################################################################
#  MOTIFS
########################################################################################
# ------------------------------------------------------------

MATCHER = {
    "chain": lambda ch, pa, rng: pick_chain(ch, pa, rng),
    "fork": lambda ch, pa, rng: pick_fork(ch, rng),
    "collider": lambda ch, pa, rng: pick_collider(pa, rng),
    "confounding": lambda ch, pa, rng: pick_confounding(ch, pa, rng),
    "mediation": lambda ch, pa, rng: pick_mediation(ch, rng, require_direct=True),
    "iv": lambda ch, pa, rng: pick_iv(ch, pa, rng),
    "arrowhead": lambda ch, pa, rng: pick_arrowhead(ch, rng),
    "diamond": lambda ch, pa, rng: pick_diamond(ch, rng),
    "diamondcut": lambda ch, pa, rng: pick_diamondcut(ch, rng),
    "frontdoor": lambda ch, pa, rng: pick_frontdoor(ch, rng),
    "double_nc": lambda ch, pa, rng: None,  # no CauseNet picker
    "triangle": lambda ch, pa, rng: None,  # no CauseNet picker
}


# ---- motif pickers (return (role->concept, motif_edges)) ----


def pick_chain(children, parents, rng):
    logger.debug("Attempting to pick chain motif")
    Bs = [b for b in parents if parents[b] and children.get(b)]
    if not Bs:
        logger.debug("No valid B nodes for chain motif")
        return None
    b = rng.choice(Bs)
    a = _choose(rng, parents[b])
    c = _choose(rng, children[b])
    if not (a and c) or not _distinct(a, b, c):
        return None
    return {"X": a, "M": b, "Y": c}, [("X", "M"), ("M", "Y")]


def pick_fork(children, rng):
    logger.debug("Attempting to pick fork motif")
    As = [a for a, outs in children.items() if len(outs) >= 2]
    if not As:
        logger.debug("No valid fork nodes (need at least 2 children)")
        return None
    a = rng.choice(As)
    b, c = _choose2(rng, children[a]) or (None, None)
    if not (b and c) or not _distinct(a, b, c):
        return None
    return {"Z": a, "X": b, "Y": c}, [("Z", "X"), ("Z", "Y")]


def pick_collider(parents, rng):
    logger.debug("Attempting to pick collider motif")
    As = [a for a, ins in parents.items() if len(ins) >= 2]
    if not As:
        logger.debug("No valid collider nodes (need at least 2 parents)")
        return None
    a = rng.choice(As)
    b, c = _choose2(rng, parents[a]) or (None, None)
    if not (b and c) or not _distinct(a, b, c):
        return None
    return {"X": b, "Y": c, "Z": a}, [("X", "Z"), ("Y", "Z")]


def pick_confounding(children, parents, rng):
    logger.debug("Attempting to pick confounding motif")
    Xs = [x for x in children if children[x]]
    rng.shuffle(Xs)
    for x in Xs:
        Ys = list(children[x])
        rng.shuffle(Ys)
        for y in Ys:
            Zs = parents.get(x, set()) & parents.get(y, set())
            if not Zs:
                continue
            z = _choose(rng, Zs)
            if z and _distinct(x, y, z):
                return {"X": x, "Y": y, "Z": z}, [("Z", "X"), ("Z", "Y"), ("X", "Y")]
    logger.debug("No valid confounding structure found")
    return None


def pick_mediation(children, rng, require_direct=True):
    # X->M->Y and (optionally) X->Y (common in “mediation”)
    Xs = [x for x in children if children[x]]
    rng.shuffle(Xs)
    for x in Xs:
        for m in list(children.get(x, ())):
            Ys = list(children.get(m, ()))
            if not Ys:
                continue
            y = _choose(rng, Ys)
            if not y or not _distinct(x, m, y):
                continue
            if require_direct and y not in children.get(x, set()):
                continue
            return {"X": x, "M": m, "Y": y}, [("X", "M"), ("M", "Y")] + (
                [("X", "Y")] if require_direct else []
            )
    return None


def pick_iv(children, parents, rng):
    # Z->X->Y with NO direct Z->Y (weak IV proxy)
    Xs = [x for x in children if children[x]]
    rng.shuffle(Xs)
    for x in Xs:
        Zs = list(parents.get(x, ()))
        Ys = list(children.get(x, ()))
        if not Zs or not Ys:
            continue
        z = _choose(rng, Zs)
        y = _choose(rng, Ys)
        if not (z and y) or not _distinct(z, x, y):
            continue
        if y in children.get(z, set()):
            continue
        return {"Z": z, "X": x, "Y": y}, [("Z", "X"), ("X", "Y")]
    return None


def pick_arrowhead(children, rng):
    # X->Y and W->Y ; avoid X<->W to keep the minimal shape
    # scan all Y with at least 2 distinct parents
    parents_of = {
        y: [p for p, outs in children.items() if y in outs]
        for y in set(children) | {v for S in children.values() for v in S}
    }
    Ys = [y for y, ps in parents_of.items() if len(ps) >= 2]
    rng.shuffle(Ys)
    for y in Ys:
        x, w = _choose2(rng, parents_of[y]) or (None, None)
        if not (x and w) or not _distinct(x, w, y):
            continue
        if w in children.get(x, set()) or x in children.get(w, set()):
            continue
        return {"X": x, "W": w, "Y": y}, [("X", "Y"), ("W", "Y")]
    return None


def pick_diamond(children, rng):
    # X->M1, X->M2, M1->Y, M2->Y
    Xs = [x for x, outs in children.items() if len(outs) >= 2]
    rng.shuffle(Xs)
    for x in Xs:
        m1, m2 = _choose2(rng, children[x]) or (None, None)
        if not (m1 and m2) or m1 == m2:
            continue
        Ys = children.get(m1, set()) & children.get(m2, set())
        if not Ys:
            continue
        y = _choose(rng, Ys)
        if y and _distinct(x, m1, m2, y):
            return {"X": x, "M1": m1, "M2": m2, "Y": y}, [
                ("X", "M1"),
                ("M1", "Y"),
                ("X", "M2"),
                ("M2", "Y"),
            ]
    return None


def pick_diamondcut(children, rng):
    # “diamond” where Y is child of exactly one of {M1,M2}
    Xs = [x for x, outs in children.items() if len(outs) >= 2]
    rng.shuffle(Xs)
    for x in Xs:
        m1, m2 = _choose2(rng, children[x]) or (None, None)
        if not (m1 and m2) or m1 == m2:
            continue
        only_m1 = children.get(m1, set()) - children.get(m2, set())
        only_m2 = children.get(m2, set()) - children.get(m1, set())
        cand = list(only_m1 | only_m2)
        if not cand:
            continue
        y = _choose(rng, cand)
        if y and _distinct(x, m1, m2, y):
            edges = [("X", "M1"), ("X", "M2")] + (
                [("M1", "Y")] if y in children[m1] else [("M2", "Y")]
            )
            return {"X": x, "M1": m1, "M2": m2, "Y": y}, edges
    return None


def pick_frontdoor(children, rng):
    # X->M->Y with NO direct X->Y; latent U is conceptual (naming only)
    Xs = [x for x in children if children[x]]
    rng.shuffle(Xs)
    for x in Xs:
        for m in list(children.get(x, ())):
            Ys = list(children.get(m, ()))
            if not Ys:
                continue
            y = _choose(rng, Ys)
            if not y or not _distinct(x, m, y):
                continue
            if y in children.get(x, set()):  # exclude direct X->Y
                continue
            return {"X": x, "M": m, "Y": y}, [("X", "M"), ("M", "Y")]
    return None


########################################################################################
# CAUSENET PROVENANCE FILTERING
########################################################################################


def _prov_ok(info_entry, prefer_wikipedia: bool, support_floor: int) -> bool:
    """Check if a CauseNet edge meets provenance requirements.

    Args:
        info_entry: CNEdge metadata object or None
        prefer_wikipedia: If True, require at least one Wikipedia source
        support_floor: Minimum support threshold

    Returns:
        True if edge meets requirements, False otherwise
    """
    if info_entry is None:
        # If no metadata, accept only if prefer_wikipedia is False and support_floor <= 0
        return (not prefer_wikipedia) and (support_floor <= 0)
    sup = getattr(info_entry, "support", None)
    if sup is not None and int(sup) < int(support_floor):
        logger.debug("Edge rejected: support %d < floor %d", sup, support_floor)
        return False
    if prefer_wikipedia:
        srcs = getattr(info_entry, "sources", None) or []
        # Accept if any source is a Wikipedia-derived type
        for s in srcs:
            t = (s or {}).get("type", "")
            if isinstance(t, str) and t.startswith("wikipedia_"):
                return True
        logger.debug("Edge rejected: no Wikipedia sources found")
        return False
    return True


########################################################################################
# SIMPLE CAUSENET MATCHING (X, Y only)
########################################################################################

# Motif-specific mappings for source (cause) and sink (effect) nodes
# These are used as fallback when graph structure doesn't yield unique source/sink
_MOTIF_SOURCE_SINK_MAPPING = {
    "fork": {"source": "Z", "sink": None},  # Z is source, X and Y are both sinks
    "collider": {"source": None, "sink": "Z"},  # X and Y are sources, Z is sink
    "mediation": {"source": "X", "sink": "Y"},
    "confounding": {"source": "X", "sink": "Y"},  # X->Y with confounder Z
    "arrowhead": {"source": "X", "sink": "Y"},
    "double_nc": {"source": "A", "sink": "Y"},
    # Problematic motifs: use manual mapping as default (direct unmediated edge to Y)
    "chain": {"source": "M", "sink": "Y"},
    "diamond": {"source": "M1", "sink": "Y"},
    "diamondcut": {"source": "X", "sink": "Y"},
    "iv": {"source": "X", "sink": "Y"},
    "frontdoor": {"source": "Z", "sink": "Y"},
}

# Motifs where we should use the manual mapping as the PRIMARY approach
# (not fallback) because automatic source/sink detection would pick wrong nodes.
# These motifs need to assign CauseNet concepts to direct unmediated edges to outcome.
_MANUAL_MAPPING_MOTIFS = {"chain", "diamond", "diamondcut", "iv", "frontdoor"}


def _score_source_sink_pair(
    G, source: str, sink: str, outcome: Optional[str], latent_nodes: Set[str]
) -> Tuple[int, int, int, int]:
    """Score a (source, sink) pair for CauseNet assignment.

    Returns a tuple for sorting (higher is better):
        (is_direct_edge, sink_is_outcome, source_not_latent, sink_not_latent)

    Preferences (in order):
        1. Direct edge from source to sink (distance 1)
        2. Sink is the outcome node Y
        3. Both nodes are observed (not latent)
    """

    # Check if direct edge exists
    is_direct = 1 if G.has_edge(source, sink) else 0

    # Prefer sink being the outcome
    sink_is_outcome = 1 if sink == outcome else 0

    # Prefer observed nodes
    source_not_latent = 0 if source in latent_nodes else 1
    sink_not_latent = 0 if sink in latent_nodes else 1

    return (is_direct, sink_is_outcome, source_not_latent, sink_not_latent)


def _find_best_pair_from_candidates(
    G,
    sources: List[str],
    sinks: List[str],
    outcome: Optional[str],
    latent_nodes: Set[str],
    rng,
) -> Tuple[Optional[str], Optional[str]]:
    """Find the best (source, sink) pair from candidate lists.

    Prefers pairs that:
        1. Have a direct edge (source -> sink)
        2. Have sink == outcome node
        3. Are both observed (not latent)

    Falls back to random selection if no clear winner.
    """
    if not sources or not sinks:
        return None, None

    # Generate all valid pairs (source != sink)
    pairs = [(s, t) for s in sources for t in sinks if s != t]
    if not pairs:
        return None, None

    # Score each pair
    scored = [
        (pair, _score_source_sink_pair(G, pair[0], pair[1], outcome, latent_nodes))
        for pair in pairs
    ]

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Get best score
    best_score = scored[0][1]

    # Find all pairs with the best score (for tie-breaking)
    best_pairs = [p for p, s in scored if s == best_score]

    if len(best_pairs) == 1:
        winner = best_pairs[0]
        logger.debug(
            "Best pair by scoring: %s -> %s (score=%s)",
            winner[0],
            winner[1],
            best_score,
        )
        return winner

    # Tie-break randomly among best pairs
    winner = rng.choice(best_pairs)
    logger.debug(
        "Randomly selected from %d tied best pairs: %s -> %s (score=%s)",
        len(best_pairs),
        winner[0],
        winner[1],
        best_score,
    )
    return winner


def _find_source_sink_from_structure(
    G, motif: Optional[str], rng, outcome: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find source and sink nodes from graph structure.

    For motifs in _MANUAL_MAPPING_MOTIFS, use the manual mapping directly
    (these need direct unmediated edges to the outcome for CauseNet assignment).

    For other motifs: use in-degree/out-degree to find sources and sinks.
    When multiple candidates exist, prefer pairs that:
        1. Have a direct edge (source -> sink)
        2. Have sink == outcome node
        3. Are both observed (not latent)

    Args:
        G: The graph
        motif: Motif name (optional)
        rng: Random number generator
        outcome: The outcome node name (optional, used for preferring sink=outcome)

    Returns (source_node, sink_node) or (None, None) if not found.
    """
    motif_lower = motif.lower() if motif else None

    # For problematic motifs, use manual mapping as the PRIMARY approach
    # These motifs need CauseNet concepts assigned to direct unmediated edges to Y
    if motif_lower and motif_lower in _MANUAL_MAPPING_MOTIFS:
        mapping = _MOTIF_SOURCE_SINK_MAPPING.get(motif_lower, {})
        source_node = mapping.get("source")
        sink_node = mapping.get("sink")

        # Verify nodes exist in graph
        if source_node and source_node not in G.nodes():
            logger.warning(
                "Manual mapping source '%s' not found in graph for motif '%s'",
                source_node,
                motif,
            )
            source_node = None
        if sink_node and sink_node not in G.nodes():
            logger.warning(
                "Manual mapping sink '%s' not found in graph for motif '%s'",
                sink_node,
                motif,
            )
            sink_node = None

        if source_node and sink_node:
            logger.debug(
                "Using manual mapping for motif '%s': source=%s, sink=%s",
                motif,
                source_node,
                sink_node,
            )
            return source_node, sink_node
        # If manual mapping failed, fall through to automatic detection

    # Identify latent nodes (commonly named "U" or starting with "U_")
    latent_names = {"U", "U_"}
    latent_nodes = {n for n in G.nodes() if n in latent_names or n.startswith("U_")}

    # Find all sources (in-degree 0) and sinks (out-degree 0), excluding latents
    sources = [n for n in G.nodes() if G.in_degree(n) == 0 and n not in latent_nodes]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0 and n not in latent_nodes]

    logger.debug(
        "Graph structure: sources=%s, sinks=%s (after filtering latents)",
        sources,
        sinks,
    )

    # If unique source and sink, use them directly
    if len(sources) == 1 and len(sinks) == 1 and sources[0] != sinks[0]:
        logger.debug(
            "Found unique source=%s and sink=%s from structure",
            sources[0],
            sinks[0],
        )
        return sources[0], sinks[0]

    # Multiple candidates: use scoring to find best pair
    if len(sources) >= 1 and len(sinks) >= 1:
        best_source, best_sink = _find_best_pair_from_candidates(
            G, sources, sinks, outcome, latent_nodes, rng
        )
        if best_source and best_sink:
            return best_source, best_sink

    # Fallback: try to find at least one of each
    source_node = None
    sink_node = None

    if len(sources) == 1:
        source_node = sources[0]
    elif len(sources) > 1:
        # Check motif mapping
        if motif_lower and motif_lower in _MOTIF_SOURCE_SINK_MAPPING:
            mapped_source = _MOTIF_SOURCE_SINK_MAPPING[motif_lower].get("source")
            if mapped_source and mapped_source in sources:
                source_node = mapped_source
                logger.debug("Using motif mapping for source: %s", source_node)
        if source_node is None:
            source_node = rng.choice(sources)
            logger.debug("Randomly selected source: %s", source_node)

    available_sinks = [s for s in sinks if s != source_node]
    if len(available_sinks) == 1:
        sink_node = available_sinks[0]
    elif len(available_sinks) > 1:
        # Check motif mapping
        if motif_lower and motif_lower in _MOTIF_SOURCE_SINK_MAPPING:
            mapped_sink = _MOTIF_SOURCE_SINK_MAPPING[motif_lower].get("sink")
            if mapped_sink and mapped_sink in available_sinks:
                sink_node = mapped_sink
                logger.debug("Using motif mapping for sink: %s", sink_node)
        if sink_node is None:
            sink_node = rng.choice(available_sinks)
            logger.debug("Randomly selected sink: %s", sink_node)
    elif not available_sinks and sinks:
        sink_node = rng.choice(sinks) if sinks else None
        logger.debug("Fallback sink selection: %s", sink_node)

    # Fallback to motif-specific hard mapping if structure didn't work
    if source_node is None and motif and motif.lower() in _MOTIF_SOURCE_SINK_MAPPING:
        mapped_source = _MOTIF_SOURCE_SINK_MAPPING[motif.lower()].get("source")
        if mapped_source and mapped_source in G.nodes():
            source_node = mapped_source
            logger.debug(
                "Fallback: using motif hard mapping for source: %s", source_node
            )

    if sink_node is None and motif and motif.lower() in _MOTIF_SOURCE_SINK_MAPPING:
        mapped_sink = _MOTIF_SOURCE_SINK_MAPPING[motif.lower()].get("sink")
        if mapped_sink and mapped_sink in G.nodes() and mapped_sink != source_node:
            sink_node = mapped_sink
            logger.debug("Fallback: using motif hard mapping for sink: %s", sink_node)

    # Last resort: check for literal "X" and "Y" nodes
    if source_node is None and "X" in G.nodes():
        source_node = "X"
        logger.debug("Last resort: using 'X' as source")
    if sink_node is None and "Y" in G.nodes() and "Y" != source_node:
        sink_node = "Y"
        logger.debug("Last resort: using 'Y' as sink")

    return source_node, sink_node


def simple_causenet_matching(
    sg,
    children,
    info,
    *,
    prefer_wikipedia: bool = True,
    support_floor: int = 2,
    max_tries: int = 100,
    sampling_strategy: str = "uniform",
    seed: Optional[int] = None,
) -> Tuple[SampledGraph, Dict[str, str]]:
    """Simple CauseNet matching: map one CauseNet relation to source and sink nodes.

    This function:
    1. Finds the source node (no parents) and sink node (no children) from graph structure
    2. Falls back to motif-specific mappings if structure is ambiguous
    3. Samples a CauseNet relation (cause, effect) and maps source→cause, sink→effect

    Args:
        sg: SampledGraph object containing the graph and motif metadata
        children: Dict mapping CauseNet concepts to their children (effects)
        info: Dict mapping (cause, effect) tuples to CNEdge metadata
        prefer_wikipedia: If True, prefer edges with Wikipedia sources
        support_floor: Minimum support threshold
        max_tries: Maximum attempts to find a valid CauseNet pair
        sampling_strategy: Candidate sampling strategy after filtering.
            ``uniform`` preserves the current behavior. ``quality_weighted``
            biases sampling toward higher-quality pairs using Wikipedia
            provenance and support.
        seed: Random seed for reproducibility

    Returns:
        Tuple of (modified_sg, applied_mapping) where:
            - modified_sg: Copy of sg with source and sink renamed
            - applied_mapping: Dict of {old_id -> new_concept} applied
        Or (sg, {}) if no valid match found

    Side effects:
        Updates sg.meta with strategy, applied_renames, fixed_nodes, needs_names
    """
    logger.info(
        "Starting simple CauseNet matching (prefer_wikipedia=%s, support_floor=%d, sampling_strategy=%s, seed=%s)",
        prefer_wikipedia,
        support_floor,
        sampling_strategy,
        seed if seed is not None else "None",
    )

    rng = random.Random(seed if seed is not None else random.randint(0, 1_000_000))
    G = sg.graph
    motif = getattr(sg, "motif", None)
    outcome = getattr(sg, "outcome", None)

    # 1) Find source and sink using graph structure (primary) with motif fallback
    # Pass outcome to prefer sink=outcome when multiple candidates exist
    X, Y = _find_source_sink_from_structure(G, motif, rng, outcome=outcome)

    if X is None:
        logger.warning(
            "Could not identify a source node for simple matching. Cannot proceed."
        )
        sg.meta = sg.meta or {}
        sg.meta["strategy"] = "simple_causenet"
        sg.meta.setdefault("applied_renames", {})
        sg.meta.setdefault("fixed_nodes", [])
        needs_names = sorted(set(G.nodes) - set(sg.meta.get("fixed_nodes", [])))
        sg.meta["needs_names"] = needs_names
        sg.needs_names = needs_names
        return sg, {}

    if Y is None:
        logger.warning(
            "Could not identify a sink node for simple matching. Cannot proceed."
        )
        sg.meta = sg.meta or {}
        sg.meta["strategy"] = "simple_causenet"
        sg.meta.setdefault("applied_renames", {})
        sg.meta.setdefault("fixed_nodes", [])
        needs_names = sorted(set(G.nodes) - set(sg.meta.get("fixed_nodes", [])))
        sg.meta["needs_names"] = needs_names
        sg.needs_names = needs_names
        return sg, {}

    if X == Y:
        logger.warning("Source and sink are the same node (%s). Cannot proceed.", X)
        sg.meta = sg.meta or {}
        sg.meta["strategy"] = "simple_causenet"
        sg.meta.setdefault("applied_renames", {})
        sg.meta.setdefault("fixed_nodes", [])
        needs_names = sorted(set(G.nodes) - set(sg.meta.get("fixed_nodes", [])))
        sg.meta["needs_names"] = needs_names
        sg.needs_names = needs_names
        return sg, {}

    logger.info("Identified source=%s, sink=%s for motif=%s", X, Y, motif)

    # 2) Prepare CauseNet candidate pairs
    cn_pairs: List[Tuple[str, str, int, bool]] = (
        []
    )  # (cause, effect, support, has_wiki)

    for (c, d), meta in info.items():
        if not _prov_ok(meta, prefer_wikipedia, support_floor):
            continue

        sup = getattr(meta, "support", 0) or 0
        has_wiki = False
        srcs = getattr(meta, "sources", None) or []
        for s in srcs:
            t = (s or {}).get("type", "")
            if isinstance(t, str) and t.startswith("wikipedia_"):
                has_wiki = True
                break

        cn_pairs.append((c, d, int(sup), bool(has_wiki)))

    if not cn_pairs:
        logger.warning("No valid CauseNet pairs found matching provenance criteria")
        sg.meta = sg.meta or {}
        sg.meta["strategy"] = "simple_causenet"
        sg.meta.setdefault("applied_renames", {})
        sg.meta.setdefault("fixed_nodes", [])
        needs_names = sorted(set(G.nodes) - set(sg.meta.get("fixed_nodes", [])))
        sg.meta["needs_names"] = needs_names
        sg.needs_names = needs_names
        return sg, {}

    candidate_order = _order_simple_causenet_candidates(
        cn_pairs,
        rng=rng,
        sampling_strategy=sampling_strategy,
        max_items=max_tries,
    )
    if candidate_order:
        best_quality = max(
            (
                _simple_causenet_pair_quality_key(support, has_wiki)
                for _, _, support, has_wiki in candidate_order
            ),
            default=(0, 0),
        )
        logger.debug(
            "Prepared %d CauseNet candidate pairs for strategy=%s (best_quality=%s)",
            len(candidate_order),
            sampling_strategy,
            best_quality,
        )

    # 3) Sample a CauseNet pair and apply mapping
    tried = 0
    for cause, effect, sup, has_wiki in candidate_order:
        tried += 1
        if tried > max_tries:
            logger.debug("Reached max_tries=%d without finding valid pair", max_tries)
            break

        # Verify the edge exists in children
        if effect not in children.get(cause, set()):
            continue

        # Found a valid pair
        rename_map = {X: cause, Y: effect}
        logger.info(
            "Matched CauseNet relation: %s→%s to motif %s→%s (support=%d, wiki=%s)",
            cause,
            effect,
            X,
            Y,
            sup,
            has_wiki,
        )

        # Apply rename
        sg_copy = copy.deepcopy(sg)
        applied, sg_copy = sg_copy.rename_nodes(rename_map)

        # Bookkeeping
        sg_copy.meta = sg_copy.meta or {}
        sg_copy.meta["strategy"] = "simple_causenet"
        sg_copy.meta["applied_renames"] = applied
        sg_copy.meta["fixed_nodes"] = sorted(applied.values())

        # Track provenance
        prov_rec = info.get((cause, effect))
        sg_copy.meta["concept_provenance"] = [
            _make_prov_record(cause, effect, prov_rec)
        ]

        # Determine which nodes still need names
        fixed_nodes = set(applied.values())
        needs_names = sorted([n for n in sg_copy.graph.nodes if n not in fixed_nodes])
        sg_copy.meta["needs_names"] = needs_names
        sg_copy.needs_names = needs_names

        logger.info(
            "Simple matching complete: 2 nodes renamed (%s→%s, %s→%s), %d nodes need names",
            X,
            cause,
            Y,
            effect,
            len(needs_names),
        )

        return sg_copy, applied

    # If we exhausted all tries
    logger.warning("Failed to find valid CauseNet pair after %d tries", tried)
    sg.meta = sg.meta or {}
    sg.meta["strategy"] = "simple_causenet"
    sg.meta.setdefault("applied_renames", {})
    sg.meta.setdefault("fixed_nodes", [])
    needs_names = sorted(set(G.nodes) - set(sg.meta.get("fixed_nodes", [])))
    sg.meta["needs_names"] = needs_names
    sg.needs_names = needs_names

    return sg, {}
