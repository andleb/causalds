"""
CYAML (Causal‑YAML) and fallback encodings for causal graphs.

This module serializes `SampledGraph`, `SCM`, or `DataGenerator` into:
  1) CYAML (primary, LLM‑friendly, lossless)
  2) Parents‑JSON (compact adjacency form)
  3) Natural edge list (readable prose with per‑node disclaimers)
  4) List of conditional independencies

Key features implemented here (see function docstrings for details):
- Deterministic ordering (stable topological order across sections)
- Explicit v‑structures/colliders
- Latents as bidirected pairs (ADMG) via `latents.connects: [[A,B], ...]`
- Middle‑ground non‑edges: `constraints.non_edges_parents_topo` lists, for each
  node V, the set of earlier (in topological order) *non‑parents* of V among
  observed nodes. This avoids full O(n^2) enumeration while keeping absence
  information explicit in the most semantically relevant direction.
- Safer sign & monotonicity inference from SCM mechanisms
- Optional lightweight data summary (means/sds; edge‑pair association signs)
- Graceful fallbacks when optional deps (yaml, networkx) are missing

The CYAML header includes `cyaml_version` so the format can evolve safely.

Public API:
  - serialize_cyaml(obj, include_data=True, n_samples=500, seed=None,
                      include_non_edges="parents_topo") -> str
  - serialize_parents_json(obj) -> str
  - serialize_edge_list(obj) -> str
"""

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


import networkx as nx
import yaml

from causalds.causenet_extract import (
    choose_best_nl_source,
    extract_nl_provenance_fields,
)
from causalds.graph import (
    all_pairwise_cond_independencies,
    compute_v_structures,
    minimal_pairwise_cond_independencies,
    toposort,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _is_nx_digraph(G: Any) -> bool:
    return nx is not None and isinstance(G, nx.DiGraph)


def _as_graph(obj) -> Tuple[Any, Dict[str, str], List[str], List[str]]:
    """
    Return (G, node_types, observed_nodes, latents).
    """
    logger.debug("Extracting graph from object of type: %s", type(obj).__name__)
    # SampledGraph
    if hasattr(obj, "graph"):
        G = obj.graph
        observed = list(getattr(obj, "observed_nodes", None) or list(G.nodes()))
        # DataGenerator carries node_types, SampledGraph may not
        node_types = dict(getattr(obj, "node_types", {}))
        # If DataGenerator
        if hasattr(obj, "scm"):
            # obj is likely a DataGenerator
            scm = getattr(obj, "scm")
            node_types.update(getattr(scm, "node_types", {}))
        # SCM object
    elif hasattr(obj, "G"):
        G = obj.G
        observed = list(getattr(obj, "observed_nodes", []) or list(G.nodes()))
        node_types = dict(getattr(obj, "node_types", {}))
    else:
        raise TypeError(
            "Unsupported object type. Expected SampledGraph, SCM, or DataGenerator."
        )
    latents = [v for v in G.nodes() if v not in observed]
    return G, node_types, observed, latents


def _serialization_context(obj: Any) -> Dict[str, Any]:
    """Build common serialization context shared by all output formats."""
    G, node_types, observed_nodes, latents = _as_graph(obj)
    topo = toposort(G, strict=False)
    idx = {n: i for i, n in enumerate(topo)}
    scm = getattr(obj, "scm", None)
    if scm is None and hasattr(obj, "mechanisms"):
        scm = obj
    return {
        "graph": G,
        "node_types": node_types,
        "observed_nodes": observed_nodes,
        "latents": latents,
        "topo": topo,
        "idx": idx,
        "scm": scm,
        "meta": _get_meta(obj),
    }


def _infer_edge_sign(scm: Any, child: str, parent: str) -> Tuple[str, bool, str]:
    """
    Infer qualitative sign and monotonicity of parent->child using SCM parameters.
    """
    if scm is None:
        logger.debug("No SCM available for sign inference: %s->%s", parent, child)
        return "unknown", False, "medium"
    mechanisms = getattr(scm, "mechanisms", None)
    if not mechanisms or child not in mechanisms:
        return "unknown", False, "medium"
    mech = mechanisms[child]
    params = getattr(mech, "params", {}) or {}
    w = params.get("w", {}) or {}
    w_nl = params.get("w_nl", {}) or {}
    w_pair = params.get("w_pair", {}) or {}
    linear = float(w.get(parent, 0.0))
    nonlin = float(w_nl.get(parent, 0.0))
    # Any interaction touching this parent?
    has_interaction = False
    if isinstance(w_pair, dict):
        for (p1, p2), coef in w_pair.items():
            if parent == p1 or parent == p2:
                has_interaction = True
                break
    if linear == 0.0 and nonlin == 0.0:
        return "unknown", False, "medium"
    total = linear + nonlin
    sign = "+" if total > 0 else "-"
    # Monotone only if no interactions and linear/nonlinear do not conflict
    same_dir = (linear == 0) or (nonlin == 0) or ((linear > 0) == (nonlin > 0))
    monotonic = (not has_interaction) and same_dir
    mag = abs(total)
    strength = "small" if mag < 0.3 else "medium" if mag < 0.8 else "large"
    logger.debug(
        "Inferred edge sign %s->%s: sign=%s, monotonic=%s, strength=%s",
        parent,
        child,
        sign,
        monotonic,
        strength,
    )
    return sign, monotonic, strength


def _compute_v_structures(G) -> List[List[str]]:
    """Delegate to graph.compute_v_structures (canonical implementation)."""
    if not _is_nx_digraph(G):
        return []
    return compute_v_structures(G)


def _compute_non_edges_parents_topo(
    G, topo: List[str], observed: List[str]
) -> Dict[str, List[str]]:
    """
    For each node v, list earlier (in topo) observed nodes that are *not* parents of v.
    """
    logger.debug(
        "Computing non-edges (parents_topo) for %d observed nodes", len(observed)
    )
    if not _is_nx_digraph(G):
        return {}
    observed_set = set(observed)
    idx = {n: i for i, n in enumerate(topo)}
    res: Dict[str, List[str]] = {}
    for v in topo:
        if v not in observed_set:
            continue
        earlier = [u for u in topo[: idx[v]] if u in observed_set]
        non_parents = [u for u in earlier if not G.has_edge(u, v)]
        if non_parents:
            res[v] = non_parents
    logger.debug("Computed non_edges_parents_topo: %d nodes with non-parents", len(res))
    return res


def _node_metadata(n: str, node_types: Dict[str, str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"id": n}
    vtype = node_types.get(n)
    if vtype:
        meta["type"] = vtype
    return meta


# Simple YAML fallback dumper (when PyYAML isn't available)
def _dump_yaml_like(obj: Any, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            key_line = f"{sp}{k}:"
            if isinstance(v, (dict, list)):
                lines.append(key_line)
                lines.append(_dump_yaml_like(v, indent + 1))
            else:
                if v is None:
                    lines.append(f"{key_line} null")
                elif v == {}:
                    lines.append(f"{key_line} {{}}")
                elif v == []:
                    lines.append(f"{key_line} []")
                elif isinstance(v, bool):
                    lines.append(f"{key_line} {'true' if v else 'false'}")
                else:
                    lines.append(
                        f"{key_line} {v!r}" if isinstance(v, str) else f"{key_line} {v}"
                    )
        return "\n".join(lines)
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(_dump_yaml_like(item, indent + 1))
            else:
                if item is None:
                    lines.append(f"{sp}- null")
                elif isinstance(item, bool):
                    lines.append(f"{sp}- {'true' if item else 'false'}")
                else:
                    lines.append(
                        f"{sp}- {item!r}" if isinstance(item, str) else f"{sp}- {item}"
                    )
        return "\n".join(lines)
    else:
        # scalars
        if obj is None:
            return "null"
        if isinstance(obj, bool):
            return "true" if obj else "false"
        return repr(obj) if isinstance(obj, str) else str(obj)


def _yaml_dump(data: Dict[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(data, sort_keys=False)
    return _dump_yaml_like(data)


# ------------------------------------------------------------------
# Optional: CauseNet provenance (NL) + statistical properties
# ------------------------------------------------------------------
def _get_meta(obj: Any) -> Optional[Dict[str, Any]]:
    """Return `meta` dict from SampledGraph/SCM or from obj.sg (DataGenerator), if present."""
    if hasattr(obj, "sg"):  # DataGenerator
        return getattr(obj.sg, "meta", None)
    if hasattr(obj, "meta"):
        return getattr(obj, "meta", None)
    return None


def _choose_best_nl_source(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Best-effort selection of one NL-ish provenance source dict."""

    # NOTE:  Prefer the one already computed in causenet_match
    best = entry.get("best_nl_source")

    if isinstance(best, dict):
        return best

    # Otherwise: choose from nl_sources
    nl_sources = entry.get("nl_sources")
    if isinstance(nl_sources, list) and nl_sources:
        return choose_best_nl_source(nl_sources, shuffle=True)

    # Fallback: extract from raw sources (keep existing logic for backward compat)
    sources = entry.get("sources")
    if not isinstance(sources, list):
        return None
    extracted = extract_nl_provenance_fields(sources)
    return choose_best_nl_source(extracted, shuffle=True) if extracted else None


def _format_source_context(src: Dict[str, Any], include_url: bool) -> str:
    ctx: List[str] = []
    title = src.get("wikipedia_page_title") or src.get("infobox_title")
    if isinstance(title, str) and title.strip():
        ctx.append(f"Wikipedia: {title.strip()}")

    # Headings that are often very NL-useful
    heading = (
        src.get("sentence_section_heading")
        or src.get("list_toc_section_heading")
        or src.get("list_toc_parent_title")
    )
    if isinstance(heading, str) and heading.strip():
        ctx.append(f"section '{heading.strip()}'")

    ib = src.get("infobox_template")
    if isinstance(ib, str) and ib.strip():
        ctx.append(f"infobox '{ib.strip()}'")

    arg = src.get("infobox_argument")
    if isinstance(arg, str) and arg.strip():
        ctx.append(f"field '{arg.strip()}'")

    if include_url:
        url = src.get("url")
        if isinstance(url, str) and url.strip():
            ctx.append(url.strip())

    return "; ".join(ctx)


def render_concept_provenance_nl(
    concept_provenance: Any,
    *,
    target_format: str = "text",
    include_url: bool = False,
    max_sentence_chars: int = 240,
) -> Union[str, List[str]]:
    """
    Render concept provenance into a compact, natural-language-like representation.
    """
    logger.debug(
        "Rendering concept provenance: format=%s, include_url=%s",
        target_format,
        include_url,
    )
    if not isinstance(concept_provenance, list) or not concept_provenance:
        return [] if target_format in ("cyaml", "json", "markdown") else ""

    lines: List[str] = []
    for entry in concept_provenance:
        if not isinstance(entry, dict):
            continue
        cause = entry.get("cause")
        effect = entry.get("effect")
        support = entry.get("support")

        edge = f"{cause} -> {effect}" if cause and effect else "unknown_edge"
        if support is not None:
            edge += f" (support={support})"

        best = _choose_best_nl_source(entry)
        sentence = None
        ctx = ""
        if best:
            sentence = best.get("sentence") or best.get("surface")
            if isinstance(sentence, str):
                sentence = sentence.strip()
                if len(sentence) > max_sentence_chars:
                    sentence = sentence[: max_sentence_chars - 1] + "…"
            ctx = _format_source_context(best, include_url=include_url)

        if sentence:
            sbit = f'Evidence: "{sentence}"'
            if ctx:
                sbit += f" [{ctx}]"
            line = f"CauseNet evidence for {edge}. {sbit}"
        else:
            line = f"CauseNet evidence for {edge}."

        lines.append(line)

    if target_format == "markdown":
        return [f"- {ln}" for ln in lines]
    if target_format in ("cyaml", "json"):
        return lines
    # default: plain text block
    return "\n".join(lines)


def _ci_preface() -> str:
    return "Statistical implications of the graph (conditional independencies):"


def format_conditional_independencies_section(
    obj: Any,
    *,
    mode: str = "minimal",
    target_format: str = "text",  # "text" | "cyaml" | "json" | "markdown"
) -> Union[str, Dict[str, Any]]:
    """
    Produce a CI 'section' suitable for embedding into other serializers.

    - For structured formats ("cyaml","json"): returns dict with description + list-form CIs.
    - For text formats: returns a string block with a preface and markdown bullets.
    """
    if target_format in ("cyaml", "json"):
        ci = serialize_conditional_independencies(obj, mode=mode, format="list")
        if not ci:
            return {}
        return {
            "description": _ci_preface(),
            "conditional_independencies": ci,
        }

    # text/markdown style - don't include metadata since the parent serializer already includes it
    lines = serialize_conditional_independencies(
        obj, mode=mode, format="markdown", include_metadata=False
    )
    if not lines:
        return ""
    header = _ci_preface()
    return header + "\n" + lines


# ------------------------------------------------------------------
# Public serializers
# ------------------------------------------------------------------
def serialize_cyaml(
    obj: Any,
    include_non_edges: str = "parents_topo",  # "none" | "parents_topo" | "all"
    include_concept_provenance_nl: bool = False,
    include_conditional_independencies: bool = False,
    ci_mode: str = "minimal",
    include_v_structures: bool = True,
) -> str:
    """
    Serialize SampledGraph, SCM, or DataGenerator to simplified CYAML.

    Matches the simplified CYAML_SYSTEM schema with:
    - nodes: just id and optional type
    - edges: just from/to (no sign/strength/monotonic)
    - graph: observed_nodes, unobserved_nodes, topological_order, v_structures, optional non_edges
    - meta: optional annotations
    - statistical_properties: optional CI statements

    Parameters
    ----------
    include_non_edges :
        - "none": do not emit any non-edge listing
        - "parents_topo": (default) for each node, lists earlier *observed* nodes
            that are not direct parents
        - "all": list every ordered non-edge pair [u, v]

    Returns
    -------
    YAML string.
    """
    ctx = _serialization_context(obj)
    G = ctx["graph"]
    node_types = ctx["node_types"]
    observed_nodes = ctx["observed_nodes"]
    topo = ctx["topo"]
    idx = ctx["idx"]

    # Nodes section - simplified (just id and optional type)
    nodes_section = [_node_metadata(n, node_types=node_types) for n in G.nodes()]
    nodes_section.sort(key=lambda d: idx[d["id"]])

    # Edges section - simplified (just from/to)
    edges_section: List[Dict[str, Any]] = []
    for u, v in G.edges():
        edges_section.append({"from": u, "to": v})
    edges_section.sort(key=lambda e: (idx[e["from"]], idx[e["to"]]))

    # Graph section - structure information
    unobserved_nodes = [v for v in G.nodes() if v not in observed_nodes]
    graph_section: Dict[str, Any] = {
        "observed_nodes": observed_nodes,
        "unobserved_nodes": unobserved_nodes,
        "topological_order": topo,
    }
    if include_v_structures:
        graph_section["v_structures"] = sorted(
            _compute_v_structures(G), key=lambda t: (idx[t[0]], idx[t[1]], idx[t[2]])
        )

    # Add non_edges based on mode
    if include_non_edges == "parents_topo":
        non_edges_pt = _compute_non_edges_parents_topo(
            G, topo=topo, observed=observed_nodes
        )
        if non_edges_pt:
            graph_section["non_edges_parents_topo"] = non_edges_pt
    elif include_non_edges == "all":
        all_non_edges: List[List[str]] = []
        if _is_nx_digraph(G):
            for u in G.nodes():
                for v in G.nodes():
                    if u == v:
                        continue
                    if not G.has_edge(u, v):
                        all_non_edges.append([u, v])
        if all_non_edges:
            graph_section["non_edges"] = all_non_edges

    # Meta section
    meta_section: Dict[str, Any] = {}
    for key in ("treatment", "outcome", "motif"):
        if hasattr(obj, "sg"):  # DataGenerator
            val = getattr(obj.sg, key, None)
            if val is not None:
                meta_section[key] = val
        elif hasattr(obj, key):
            val = getattr(obj, key, None)
            if val is not None:
                meta_section[key] = val

    sg_meta = ctx["meta"]
    if isinstance(sg_meta, dict):
        if include_concept_provenance_nl and "concept_provenance" in sg_meta:
            cp = sg_meta.get("concept_provenance")
            nl = render_concept_provenance_nl(cp, target_format="cyaml")
            if nl:
                meta_section["concept_provenance_nl"] = nl

        if "fixed_nodes" in sg_meta:
            meta_section["fixed_nodes"] = sg_meta["fixed_nodes"]
        if "needs_names" in sg_meta:
            meta_section["needs_names"] = sg_meta["needs_names"]

    # Build payload
    payload = {
        "nodes": nodes_section,
        "edges": edges_section,
        "graph": graph_section,
    }

    if meta_section:
        payload["meta"] = meta_section

    if include_conditional_independencies:
        stat = format_conditional_independencies_section(
            obj, mode=ci_mode, target_format="cyaml"
        )
        if stat:
            payload["statistical_properties"] = stat

    logger.info(
        "serialize_cyaml: nodes=%d, edges=%d, unobserved=%d, non_edges_mode=%s",
        len(nodes_section),
        len(edges_section),
        len(unobserved_nodes),
        include_non_edges,
    )
    return _yaml_dump(payload)


def _build_parents_data(
    obj: Any,
    include_concept_provenance_nl: bool = False,
    include_conditional_independencies: bool = False,
    ci_mode: str = "minimal",
    include_non_edges: str = "none",
    include_v_structures: bool = False,
) -> Dict[str, Any]:
    """Build the dict underlying serialize_parents_json and serialize_parents_xml.

    Returns
    -------
    Dict ready for JSON serialization (or XML conversion).
    """
    ctx = _serialization_context(obj)
    G = ctx["graph"]
    observed_nodes = ctx["observed_nodes"]
    topo = ctx["topo"]
    idx = ctx["idx"]
    scm = ctx["scm"]

    mapping: Dict[str, Dict[str, Any]] = {}
    for v in topo:
        parents = (
            sorted(list(G.predecessors(v)), key=lambda u: idx[u])
            if _is_nx_digraph(G)
            else []
        )
        signs = {}
        for u in parents:
            sign, _, _ = _infer_edge_sign(scm, child=v, parent=u)
            if sign != "unknown":
                signs[u] = sign
        entry: Dict[str, Any] = {"parents": parents}
        if signs:  # Only include signs if non-empty
            entry["signs"] = signs
        mapping[v] = entry

    out: Dict[str, Any] = {
        "nodes": mapping,
        "observed_nodes": sorted(observed_nodes),
    }

    # Add structural constraints (non-edges, v-structures)
    constraints: Dict[str, Any] = {}
    if include_non_edges == "parents_topo":
        non_edges_pt = _compute_non_edges_parents_topo(
            G, topo=topo, observed=observed_nodes
        )
        if non_edges_pt:
            constraints["non_edges"] = non_edges_pt
    elif include_non_edges == "all":
        all_non_edges: List[List[str]] = []
        if _is_nx_digraph(G):
            for u in G.nodes():
                for v in G.nodes():
                    if u == v:
                        continue
                    if not G.has_edge(u, v):
                        all_non_edges.append([u, v])
        if all_non_edges:
            constraints["non_edges"] = all_non_edges

    if include_v_structures:
        vstructs = _compute_v_structures(G)
        if vstructs:
            constraints["v_structures"] = sorted(
                vstructs,
                key=lambda t: (idx.get(t[0], 0), idx.get(t[1], 0), idx.get(t[2], 0)),
            )
    if constraints:
        out["constraints"] = constraints

    # Optional metadata (CauseNet provenance) and statistical properties
    meta_out: Dict[str, Any] = {}
    sg_meta = ctx["meta"]
    if isinstance(sg_meta, dict):
        if include_concept_provenance_nl and "concept_provenance" in sg_meta:
            cp = sg_meta.get("concept_provenance")
            nl = render_concept_provenance_nl(cp, target_format="json")
            if nl:
                meta_out["concept_provenance_nl"] = nl

        if "fixed_nodes" in sg_meta:
            meta_out["fixed_nodes"] = sg_meta["fixed_nodes"]
        if "needs_names" in sg_meta:
            meta_out["needs_names"] = sg_meta["needs_names"]

    if meta_out:
        out["meta"] = meta_out

    if include_conditional_independencies:
        stat = format_conditional_independencies_section(
            obj, mode=ci_mode, target_format="json"
        )
        if stat:
            out["statistical_properties"] = stat

    return out


def serialize_parents_json(
    obj: Any,
    include_concept_provenance_nl: bool = False,
    include_conditional_independencies: bool = False,
    ci_mode: str = "minimal",
    include_non_edges: str = "none",  # "none" | "parents_topo" | "all"
    include_v_structures: bool = False,
) -> str:
    """
    Parents‑JSON: for each node, the list of parents and optional qualitative signs.

    Output JSON keys are sorted for reproducibility.

    Parameters
    ----------
    include_non_edges : str
        - "none": do not emit any non-edge listing
        - "parents_topo": for each node, lists earlier *observed* nodes that are not direct parents
        - "all": list every ordered non-edge pair [u, v]
    include_v_structures : bool
        If True, include v-structures/colliders (u→z←w with no u↔w).
        Useful for diamond/collider motifs.
    """
    out = _build_parents_data(
        obj,
        include_concept_provenance_nl=include_concept_provenance_nl,
        include_conditional_independencies=include_conditional_independencies,
        ci_mode=ci_mode,
        include_non_edges=include_non_edges,
        include_v_structures=include_v_structures,
    )
    logger.debug("serialize_parents_json: nodes=%d", len(out.get("nodes", {})))
    return json.dumps(out, sort_keys=True, indent=2)


def _dict_to_parents_xml(data: Dict[str, Any]) -> str:
    """Convert the parents-data dict to pretty-printed XML.

    Produces a ``<causal_graph>`` document mirroring the parents-JSON structure
    using XML elements and attributes.
    """
    root = ET.Element("causal_graph")

    # --- nodes ---
    nodes_el = ET.SubElement(root, "nodes")
    nodes_dict = data.get("nodes", {})
    for node_id, node_info in nodes_dict.items():
        node_el = ET.SubElement(nodes_el, "node", id=node_id)
        parents_el = ET.SubElement(node_el, "parents")
        signs = node_info.get("signs", {})
        for p in node_info.get("parents", []):
            attrs: Dict[str, str] = {"id": p}
            if p in signs:
                attrs["sign"] = signs[p]
            ET.SubElement(parents_el, "parent", **attrs)

    # --- observed_nodes ---
    obs_el = ET.SubElement(root, "observed_nodes")
    for n in data.get("observed_nodes", []):
        ET.SubElement(obs_el, "node", id=n)

    # --- constraints ---
    constraints = data.get("constraints")
    if constraints:
        con_el = ET.SubElement(root, "constraints")

        non_edges = constraints.get("non_edges")
        if non_edges:
            ne_el = ET.SubElement(con_el, "non_edges")
            if isinstance(non_edges, dict):
                # parents_topo style: {child: [non_parents]}
                for child, non_parents in sorted(non_edges.items()):
                    entry = ET.SubElement(ne_el, "entry", child=child)
                    for np_id in non_parents:
                        ET.SubElement(entry, "non_parent", id=np_id)
            elif isinstance(non_edges, list):
                # all style: [[u, v], ...]
                for pair in non_edges:
                    if isinstance(pair, list) and len(pair) == 2:
                        ET.SubElement(ne_el, "pair", **{"from": pair[0], "to": pair[1]})

        vstructs = constraints.get("v_structures")
        if vstructs:
            vs_el = ET.SubElement(con_el, "v_structures")
            for triple in vstructs:
                if isinstance(triple, list) and len(triple) == 3:
                    ET.SubElement(
                        vs_el, "triple", u=triple[0], w=triple[1], z=triple[2]
                    )

    # --- meta ---
    meta = data.get("meta")
    if meta:
        meta_el = ET.SubElement(root, "meta")
        if "fixed_nodes" in meta:
            fn_el = ET.SubElement(meta_el, "fixed_nodes")
            for n in meta["fixed_nodes"]:
                ET.SubElement(fn_el, "node", id=n)
        if "needs_names" in meta:
            nn_el = ET.SubElement(meta_el, "needs_names")
            for n in meta["needs_names"]:
                ET.SubElement(nn_el, "node", id=n)
        if "concept_provenance_nl" in meta:
            cp_el = ET.SubElement(meta_el, "concept_provenance_nl")
            for evidence_str in meta["concept_provenance_nl"]:
                ev = ET.SubElement(cp_el, "evidence")
                ev.text = str(evidence_str)

    # --- statistical_properties ---
    stat = data.get("statistical_properties")
    if stat:
        sp_el = ET.SubElement(root, "statistical_properties")
        desc = stat.get("description")
        if desc:
            d_el = ET.SubElement(sp_el, "description")
            d_el.text = desc
        cis = stat.get("conditional_independencies")
        if cis:
            ci_el = ET.SubElement(sp_el, "conditional_independencies")
            for ci in cis:
                attrs = {"x": ci["x"], "y": ci["y"]}
                ci_item = ET.SubElement(ci_el, "ci", **attrs)
                for g in ci.get("given", []):
                    ET.SubElement(ci_item, "given").text = g

    # Pretty-print via minidom
    from xml.dom.minidom import parseString

    rough = ET.tostring(root, encoding="unicode")
    pretty = parseString(rough).toprettyxml(indent="  ")
    # Remove the XML declaration line that minidom adds
    lines = pretty.split("\n")
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]
    return "\n".join(line for line in lines if line.strip())


def serialize_parents_xml(
    obj: Any,
    include_concept_provenance_nl: bool = False,
    include_conditional_independencies: bool = False,
    ci_mode: str = "minimal",
    include_non_edges: str = "none",
    include_v_structures: bool = False,
) -> str:
    """
    Parents‑XML: XML equivalent of parents-JSON format.

    For each node, lists its direct parents with optional edge signs.
    Same information as ``serialize_parents_json`` but encoded as XML elements
    and attributes for A/B testing of serialization format effects on LLMs.

    Parameters
    ----------
    include_non_edges : str
        - "none": do not emit any non-edge listing
        - "parents_topo": for each node, lists earlier *observed* nodes that are not direct parents
        - "all": list every ordered non-edge pair [u, v]
    include_v_structures : bool
        If True, include v-structures/colliders.
    """
    out = _build_parents_data(
        obj,
        include_concept_provenance_nl=include_concept_provenance_nl,
        include_conditional_independencies=include_conditional_independencies,
        ci_mode=ci_mode,
        include_non_edges=include_non_edges,
        include_v_structures=include_v_structures,
    )
    logger.debug("serialize_parents_xml: nodes=%d", len(out.get("nodes", {})))
    return _dict_to_parents_xml(out)


def serialize_simple_json(
    obj: Any,
    include_concept_provenance_nl: bool = False,
    include_conditional_independencies: bool = False,
    ci_mode: str = "minimal",
    include_non_edges: str = "none",  # "none" | "parents_topo" | "all"
    include_v_structures: bool = False,
) -> str:
    """
    Serialize to simple JSON format with natural language annotations.

    The graph structure (nodes with parents) is output as JSON, while all other
    information (observed nodes, constraints, metadata, statistical properties)
    is provided in plain natural language text.

    This format matches the SIMPLE_JSON_SYSTEM schema in schemas.py.

    Parameters
    ----------
    obj : SampledGraph, SCM, or DataGenerator
    include_concept_provenance_nl : bool
        If True, include CauseNet provenance in natural language.
    include_conditional_independencies : bool
        If True, include conditional independence statements.
    ci_mode : str
        "minimal" or "all" for CI computation.
    include_non_edges : str
        - "none": do not emit any non-edge listing
        - "parents_topo": for each node, lists earlier *observed* nodes
          that are not direct parents
        - "all": list every ordered non-edge pair [u, v]
    include_v_structures : bool
        If True, include v-structures/colliders.

    Returns
    -------
    str : Combined JSON (for graph structure) and natural language text.
    """
    G, _, observed_nodes, latents = _as_graph(obj)
    topo = toposort(G, strict=False)
    idx = {n: i for i, n in enumerate(topo)}

    scm = getattr(obj, "scm", None)
    if scm is None and hasattr(obj, "mechanisms"):
        scm = obj

    # Build nodes JSON (parent-child adjacency)
    mapping: Dict[str, Dict[str, Any]] = {}
    for v in topo:
        parents = (
            sorted(list(G.predecessors(v)), key=lambda u: idx[u])
            if _is_nx_digraph(G)
            else []
        )
        signs = {}
        for u in parents:
            sign, _, _ = _infer_edge_sign(scm, child=v, parent=u)
            if sign != "unknown":
                signs[u] = sign
        entry: Dict[str, Any] = {"parents": parents}
        if signs:
            entry["signs"] = signs
        mapping[v] = entry

    nodes_json = json.dumps({"nodes": mapping}, sort_keys=True, indent=2)

    # Build natural language sections
    sections: List[str] = []

    # Section: Observed and unobserved nodes
    sections.append(f"Observed nodes: {', '.join(sorted(observed_nodes))}")
    if latents:
        sections.append(f"Unobserved/latent nodes: {', '.join(sorted(latents))}")

    # Section: Structural constraints
    constraint_lines: List[str] = []

    # Non-edges
    if include_non_edges != "none":
        if include_non_edges == "parents_topo":
            non_edges_map = _compute_non_edges_parents_topo(
                G, topo=topo, observed=observed_nodes
            )
            if non_edges_map:
                constraint_lines.append("Non-edges (pairs with NO direct causal link):")
                for child, non_parents in sorted(
                    non_edges_map.items(), key=lambda x: idx.get(x[0], 0)
                ):
                    if non_parents:
                        for np in non_parents:
                            constraint_lines.append(
                                f"  {np} does not directly cause {child}."
                            )
        elif include_non_edges == "all":
            non_edge_pairs: List[Tuple[str, str]] = []
            if _is_nx_digraph(G):
                for u in G.nodes():
                    for v in G.nodes():
                        if u == v:
                            continue
                        if not G.has_edge(u, v):
                            non_edge_pairs.append((u, v))
            if non_edge_pairs:
                constraint_lines.append("Non-edges (pairs with NO direct causal link):")
                for u, v in sorted(
                    non_edge_pairs, key=lambda p: (idx.get(p[0], 0), idx.get(p[1], 0))
                ):
                    constraint_lines.append(f"  {u} does not directly cause {v}.")

    # V-structures
    if include_v_structures:
        vstructs = _compute_v_structures(G)
        if vstructs:
            constraint_lines.append("V-structures (colliders):")
            for u, w, z in sorted(
                vstructs,
                key=lambda t: (idx.get(t[0], 0), idx.get(t[1], 0), idx.get(t[2], 0)),
            ):
                constraint_lines.append(
                    f"  {u} → {z} ← {w} "
                    f"(both {u} and {w} cause {z}, but {u} and {w} are not directly connected)"
                )

    if constraint_lines:
        sections.append("\n".join(constraint_lines))

    # Section: Metadata
    sg_meta = _get_meta(obj)
    meta_lines: List[str] = []
    if isinstance(sg_meta, dict):
        if "fixed_nodes" in sg_meta:
            fixed = sg_meta["fixed_nodes"]
            meta_lines.append(f"Fixed nodes (do NOT rename): {', '.join(fixed)}")
        if "needs_names" in sg_meta:
            needs = sg_meta["needs_names"]
            meta_lines.append(
                f"Nodes needing names (you MUST name these): {', '.join(needs)}"
            )
        if include_concept_provenance_nl and "concept_provenance" in sg_meta:
            cp = sg_meta.get("concept_provenance")
            prov = render_concept_provenance_nl(cp, target_format="text")
            if prov:
                meta_lines.append(f"CauseNet provenance:\n{prov}")

    if meta_lines:
        sections.append("\n".join(meta_lines))

    # Section: Statistical properties (conditional independencies)
    if include_conditional_independencies:
        ci_block = format_conditional_independencies_section(
            obj, mode=ci_mode, target_format="text"
        )
        if ci_block:
            sections.append(ci_block)

    # Combine JSON and text
    result = nodes_json + "\n\n" + "\n\n".join(sections)

    logger.debug("serialize_simple_json: nodes=%d", len(mapping))
    return result


def serialize_edge_list(
    obj: Any,
    include_concept_provenance_nl: bool = False,
    include_conditional_independencies: bool = False,
    ci_mode: str = "minimal",
    include_non_edges: str = "none",  # "none" | "parents_topo" | "all"
    include_v_structures: bool = False,
) -> str:
    """
    Natural‑edge list with per‑node disclaimers about absent direct causes/children.

    Parameters
    ----------
    include_non_edges : str
        - "none": do not emit any non-edge listing
        - "parents_topo": for each node, lists earlier *observed* nodes that are not direct parents
        - "all": list every ordered non-edge pair [u, v]
    include_v_structures : bool
        If True, include v-structures/colliders.
    """
    G, _, observed_nodes, latents = _as_graph(obj)
    topo = toposort(G, strict=False)
    idx = {n: i for i, n in enumerate(topo)}

    scm = getattr(obj, "scm", None)
    if scm is None and hasattr(obj, "mechanisms"):
        scm = obj

    lines: List[str] = []

    # Observed / unobserved node lists
    lines.append(f"Observed nodes: {sorted(observed_nodes)}")
    if latents:
        lines.append(f"Unobserved/latent nodes: {sorted(latents)}")
    lines.append("")

    lines.append("Edges (all and only):")
    if _is_nx_digraph(G):
        edges_sorted = sorted(G.edges(), key=lambda e: (idx[e[0]], idx[e[1]]))
    else:
        edges_sorted = list(getattr(G, "edges", lambda: [])())
    for u, v in edges_sorted:
        sign, _, _ = _infer_edge_sign(scm, child=v, parent=u)
        if sign == "+":
            verb = "increases"
        elif sign == "-":
            verb = "reduces"
        else:
            verb = "causes"
        lines.append(
            f"{u} {verb} {v} ({'+' if sign == '+' else '-' if sign == '-' else 'unknown'})."
        )

    # Per‑node adjacency (observed nodes only)
    lines.append("")
    lines.append("Per-node adjacency (observed only):")
    if _is_nx_digraph(G):
        for v in topo:
            if v not in observed_nodes:
                continue
            parents = sorted(list(G.predecessors(v)), key=lambda u: idx[u])
            children = sorted(list(G.successors(v)), key=lambda w: idx[w])
            lines.append(
                f"  {v}: parents={parents if parents else '[]'}, children={children if children else '[]'}"
            )

    # Add non-edges if requested
    if include_non_edges != "none":
        lines.append("")
        lines.append("Non-edges (pairs with NO direct causal link):")
        if include_non_edges == "parents_topo":
            non_edges_map = _compute_non_edges_parents_topo(
                G, topo=topo, observed=observed_nodes
            )
            for child, non_parents in sorted(
                non_edges_map.items(), key=lambda x: idx.get(x[0], 0)
            ):
                if non_parents:
                    lines.append(f"  {child}: {non_parents}")
        elif include_non_edges == "all":
            non_edge_pairs: List[Tuple[str, str]] = []
            if _is_nx_digraph(G):
                for u in G.nodes():
                    for v in G.nodes():
                        if u == v:
                            continue
                        if not G.has_edge(u, v):
                            non_edge_pairs.append((u, v))
            for u, v in sorted(
                non_edge_pairs, key=lambda p: (idx.get(p[0], 0), idx.get(p[1], 0))
            ):
                lines.append(f"  [{u}, {v}]")

    # Add v-structures if requested
    if include_v_structures:
        vstructs = _compute_v_structures(G)
        if vstructs:
            lines.append("")
            lines.append("V-structures (colliders):")
            for u, w, z in sorted(
                vstructs,
                key=lambda t: (idx.get(t[0], 0), idx.get(t[1], 0), idx.get(t[2], 0)),
            ):
                lines.append(f"  {u} → {z} ← {w}")

    # Add metadata section (fixed_nodes, needs_names, provenance)
    sg_meta = _get_meta(obj)
    meta_lines: List[str] = []
    if isinstance(sg_meta, dict):
        if "fixed_nodes" in sg_meta:
            meta_lines.append(f"Fixed nodes (already named): {sg_meta['fixed_nodes']}")
        if "needs_names" in sg_meta:
            meta_lines.append(f"Nodes needing names: {sg_meta['needs_names']}")
        if include_concept_provenance_nl and "concept_provenance" in sg_meta:
            cp = sg_meta.get("concept_provenance")
            nl_lines = render_concept_provenance_nl(cp, target_format="markdown")
            if nl_lines:
                meta_lines.append("")
                meta_lines.append("CauseNet provenance (natural language):")
                meta_lines.extend(nl_lines)

    if meta_lines:
        lines.append("")
        lines.extend(meta_lines)

    if include_conditional_independencies:
        block = format_conditional_independencies_section(
            obj, mode=ci_mode, target_format="text"
        )
        if block:
            lines.append("")
            lines.extend(block.splitlines())

    logger.debug("serialize_edge_list: lines=%d", len(lines))
    return "\n".join(lines)


def serialize_text_simple(
    obj: Any,
    include_concept_provenance_nl: bool = False,
    include_conditional_independencies: bool = False,
    ci_mode: str = "minimal",
    include_non_edges: str = "none",  # "none" | "parents_topo" | "all"
    include_v_structures: bool = False,
) -> str:
    """
    Serialize to simple text format: "A->B, B->C" with latent node annotations.

    Uses actual node names from the graph (including any CauseNet renamings).

    Optional extras (off by default):
      - CauseNet provenance in a natural-language-like form
      - Conditional independencies implied by the graph (minimal list)
      - Non-edges (pairs with NO direct causal link)
      - V-structures (colliders)

    Parameters
    ----------
    obj : SampledGraph, SCM, or DataGenerator
    include_non_edges : str
        - "none": do not emit any non-edge listing
        - "parents_topo": for each node, lists earlier *observed* nodes that are not direct parents
        - "all": list every ordered non-edge pair [u, v]
    include_v_structures : bool
        If True, include v-structures/colliders.

    Returns
    -------
    str : Simple text representation like "A->B, B->C. Node U is an unmeasured confounder of {X, Y}."
    """
    G, _, observed_nodes, latents = _as_graph(obj)

    # Get edges in a consistent order
    if _is_nx_digraph(G):
        topo = toposort(G, strict=False)
        idx = {n: i for i, n in enumerate(topo)}
        edges = sorted(G.edges(), key=lambda e: (idx[e[0]], idx[e[1]]))
    else:
        edges = list(getattr(G, "edges", lambda: [])())

    # Format edges
    parts: List[str] = []
    if not edges:
        parts.append("No edges.")
    else:
        edge_strs = [f"{u}->{v}" for u, v in edges]
        parts.append(", ".join(edge_strs) + ".")

    # Add latent node information
    if latents:
        for u in sorted(latents):
            if _is_nx_digraph(G):
                children = sorted(G.successors(u))
                if len(children) >= 2:
                    parts.append(
                        f"Node {u} is an unmeasured confounder of {{{', '.join(children)}}}."
                    )
                elif len(children) == 1:
                    parts.append(
                        f"Node {u} is an unmeasured latent variable with child {children[0]}."
                    )
                else:
                    parts.append(f"Node {u} is an unmeasured latent variable.")
            else:
                parts.append(f"Node {u} is an unmeasured latent variable.")

    base = " ".join(parts)

    extra_blocks: List[str] = []

    # Add non-edges if requested
    if include_non_edges != "none":
        topo = toposort(G, strict=False)
        idx = {n: i for i, n in enumerate(topo)}

        non_edge_lines: List[str] = ["Non-edges (pairs with NO direct causal link):"]
        if include_non_edges == "parents_topo":
            non_edges_map = _compute_non_edges_parents_topo(
                G, topo=topo, observed=observed_nodes
            )
            for child, non_parents in sorted(
                non_edges_map.items(), key=lambda x: idx.get(x[0], 0)
            ):
                if non_parents:
                    non_edge_lines.append(f"  {child}: {non_parents}")
        elif include_non_edges == "all":
            non_edge_pairs: List[Tuple[str, str]] = []
            if _is_nx_digraph(G):
                for u in G.nodes():
                    for v in G.nodes():
                        if u == v:
                            continue
                        if not G.has_edge(u, v):
                            non_edge_pairs.append((u, v))
            for u, v in sorted(
                non_edge_pairs, key=lambda p: (idx.get(p[0], 0), idx.get(p[1], 0))
            ):
                non_edge_lines.append(f"  [{u}, {v}]")
        if len(non_edge_lines) > 1:
            extra_blocks.append("\n".join(non_edge_lines))

    # Add v-structures if requested
    if include_v_structures:
        vstructs = _compute_v_structures(G)
        if vstructs:
            topo = toposort(G, strict=False)
            idx = {n: i for i, n in enumerate(topo)}
            vstruct_lines = ["V-structures (colliders):"]
            for u, w, z in sorted(
                vstructs,
                key=lambda t: (idx.get(t[0], 0), idx.get(t[1], 0), idx.get(t[2], 0)),
            ):
                vstruct_lines.append(f"  {u} → {z} ← {w}")
            extra_blocks.append("\n".join(vstruct_lines))

    # Add metadata (fixed_nodes, needs_names, provenance)
    sg_meta = _get_meta(obj)
    if isinstance(sg_meta, dict):
        meta_parts: List[str] = []
        if "fixed_nodes" in sg_meta:
            meta_parts.append(f"Fixed nodes (already named): {sg_meta['fixed_nodes']}")
        if "needs_names" in sg_meta:
            meta_parts.append(f"Nodes needing names: {sg_meta['needs_names']}")
        if meta_parts:
            extra_blocks.append("\n".join(meta_parts))

        if include_concept_provenance_nl and "concept_provenance" in sg_meta:
            cp = sg_meta.get("concept_provenance")
            prov = render_concept_provenance_nl(cp, target_format="text")
            if prov:
                extra_blocks.append(
                    "CauseNet provenance (natural language):\n" + str(prov)
                )

    if include_conditional_independencies:
        block = format_conditional_independencies_section(
            obj, mode=ci_mode, target_format="text"
        )
        if block:
            extra_blocks.append(block)

    if extra_blocks:
        return base + "\n\n" + "\n\n".join(extra_blocks)
    return base


def serialize_conditional_independencies(
    obj: Any,
    mode: str = "all",
    format: str = "list",
    include_concept_provenance_nl: bool = False,
    include_non_edges: str = "none",  # "none" | "parents_topo" | "all"
    include_v_structures: bool = False,
    include_metadata: bool = True,  # Whether to include fixed_nodes/needs_names/provenance
) -> Union[List, str]:
    """
    Serialize conditional independencies from a graph.

    Parameters
    ----------
    include_non_edges : str
        - "none": do not emit any non-edge listing
        - "parents_topo": for each node, lists earlier *observed* nodes that are not direct parents
        - "all": list every ordered non-edge pair [u, v]
    include_v_structures : bool
        If True, include v-structures/colliders.
    """
    logger.debug(
        "Serializing conditional independencies: mode=%s, format=%s", mode, format
    )
    # Extract graph
    G, _, _, _ = _as_graph(obj)

    if not _is_nx_digraph(G):
        logger.warning(
            "serialize_conditional_independencies: input graph is not a DiGraph, returning empty list"
        )
        return []

    # Compute independencies
    try:
        if mode == "all":
            independencies = all_pairwise_cond_independencies(G)
        elif mode == "minimal":
            independencies = minimal_pairwise_cond_independencies(G)
        else:
            logger.warning("Unknown mode '%s', returning empty list", mode)
            return []
    except Exception as e:
        logger.warning("Failed to compute conditional independencies: %s", e)
        # Fallback to minimal if all fails
        if mode == "all":
            try:
                independencies = minimal_pairwise_cond_independencies(G)
            except Exception:
                return [] if format == "list" else ""
        else:
            return [] if format == "list" else ""

    # Format output
    if format == "list":
        # Convert frozensets to sorted lists for serialization
        result = []
        for x, y, z_set in independencies:
            z_list = sorted(list(z_set))
            result.append({"x": x, "y": y, "given": z_list})
        logger.debug(
            "serialize_conditional_independencies: mode=%s, format=list, count=%d",
            mode,
            len(result),
        )
        return result

    elif format == "markdown":
        # Format as readable markdown lines
        _, _, observed_nodes, latents = _as_graph(obj)
        lines = []
        lines.append(f"Observed nodes: {sorted(observed_nodes)}")
        if latents:
            lines.append(f"Unobserved/latent nodes: {sorted(latents)}")
        lines.append("")
        for x, y, z_set in independencies:
            z_list = sorted(list(z_set))
            if len(z_list) == 0:
                lines.append(f"- {x} is (unconditionally) independent of {y}")
            else:
                z_str = ", ".join(z_list)
                lines.append(f"- {x} is conditionally independent of {y} given {z_str}")

        # Add non-edges if requested
        if include_non_edges != "none":
            topo = toposort(G, strict=False)
            idx = {n: i for i, n in enumerate(topo)}

            lines.append("")
            lines.append("Non-edges (pairs with NO direct causal link):")
            if include_non_edges == "parents_topo":
                non_edges_map = _compute_non_edges_parents_topo(
                    G, topo=topo, observed=observed_nodes
                )
                for child, non_parents in sorted(
                    non_edges_map.items(), key=lambda x: idx.get(x[0], 0)
                ):
                    if non_parents:
                        lines.append(f"  {child}: {non_parents}")
            elif include_non_edges == "all":
                non_edge_pairs: List[Tuple[str, str]] = []
                for u in G.nodes():
                    for v in G.nodes():
                        if u == v:
                            continue
                        if not G.has_edge(u, v):
                            non_edge_pairs.append((u, v))
                for u, v in sorted(
                    non_edge_pairs, key=lambda p: (idx.get(p[0], 0), idx.get(p[1], 0))
                ):
                    lines.append(f"  [{u}, {v}]")

        # Add v-structures if requested
        if include_v_structures:
            vstructs = _compute_v_structures(G)
            if vstructs:
                topo = toposort(G, strict=False)
                idx = {n: i for i, n in enumerate(topo)}
                lines.append("")
                lines.append("V-structures (colliders):")
                for u, w, z in sorted(
                    vstructs,
                    key=lambda t: (
                        idx.get(t[0], 0),
                        idx.get(t[1], 0),
                        idx.get(t[2], 0),
                    ),
                ):
                    lines.append(f"  {u} → {z} ← {w}")

        # Append metadata (fixed_nodes, needs_names, provenance) - only if requested
        if include_metadata:
            sg_meta = _get_meta(obj)
            if isinstance(sg_meta, dict):
                meta_lines: List[str] = []
                if "fixed_nodes" in sg_meta:
                    meta_lines.append(
                        f"Fixed nodes (already named): {sg_meta['fixed_nodes']}"
                    )
                if "needs_names" in sg_meta:
                    meta_lines.append(f"Nodes needing names: {sg_meta['needs_names']}")
                if meta_lines:
                    lines.append("")
                    lines.extend(meta_lines)

                if include_concept_provenance_nl and "concept_provenance" in sg_meta:
                    cp = sg_meta.get("concept_provenance")
                    prov_lines = render_concept_provenance_nl(
                        cp, target_format="markdown"
                    )
                    if prov_lines:
                        lines.append("")
                        lines.append("CauseNet provenance (natural language):")
                        lines.extend(prov_lines)

        logger.debug(
            "serialize_conditional_independencies: mode=%s, format=markdown, count=%d",
            mode,
            len(lines),
        )
        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown format '{format}'. Expected 'list' or 'markdown'")
