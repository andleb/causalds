# causalds/mapping_audit.py
"""
Audit utilities for validating variable mappings against graph constraints.

This module provides:
- AUDIT_SYSTEM_PROMPT: System prompt for the LLM auditor
- build_audit_user_prompt(): Build user prompt for audit
- run_audit_check(): Run a full audit check on a mapping
- is_unfixable_fixed_nodes_failure(): Check if violations are unfixable
"""
import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


from .graph import audit_edges, audit_non_edge_pairs_undirected
from .schemas import (
    AUDITOR_WEB_SEARCH_TOOL_INSTRUCTION,
    DERIVED_NAME_EDGE_RED_FLAG,
    MAPPING_AUDIT_ARTIFACT_EDGE_EXCEPTION,
    NON_EDGE_BIDIRECTIONAL_CHECK_RULE,
    RESTRICTIVE_QUALIFIER_RULE,
)
from .utils import collect_node_name_yellow_flags, parse_llm_output

# ----------------------
# Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

_UNKNOWN_TYPE_TOKENS = {
    "",
    "null",
    "none",
    "unknown",
    "n/a",
    "na",
    "unspecified",
    "missing",
}

_AUDIT_SYSTEM_PROMPT_TEMPLATE = """You are a strict causal consistency auditor.

{stage_specific_instruction_block}You will be given:
1) A list of DIRECT edges (u -> v) that must be plausible as direct causal effects.
2) A list of NON-EDGE pairs (u, v) meaning there must be NO plausible direct causal link in EITHER direction.
3) Optionally, a list of conditional independence (CI) relations implied by the graph.
4) A proposed variable mapping from node ids to real-world variable meanings.
5) Optionally, expected node type hints from generation (e.g., binary/continuous).

Your job:
1. For each NON-EDGE pair (U, V), decide if ANY DIRECT CAUSAL LINK exists between the mapped variables.
  We adopt the following working definition of DIRECT CAUSAL LINK:
      'There exists an intervention on U that changes V **while holding fixed the other variables in the graph** (especially the other nodes that could mediate the effect).'
  {non_edge_bidirectional_rule}

  You MUST include an entry in "non_edge_attestations" for EVERY non-edge pair provided in the input.
  - If NO direct link exists, set "no_direct_link": true and give a brief justification.
  - If a direct link DOES exist, set "no_direct_link": false, give a justification, AND record it as a VIOLATION in the "violations" list.

2. For each DIRECT edge (U -> V), decide if a plausible direct causal link exists from U to V.
  Be GENEROUS here: any reasonable mechanism (even weak, partial, or context-dependent) is sufficient.
  Only flag a VIOLATION if the edge is clearly implausible — i.e., there is no sensible real-world mechanism by which U could directly influence V.
  {artifact_edge_exception}

3. For each CI statement, judge if it is plausible given the variable meanings.
  If clearly implausible, that is a VIOLATION. In the explanation, be specific:
  - State which variables are involved and what the CI relation claims.
  - Explain WHY the chosen variable meanings make this independence implausible
    (e.g., "Rainfall and Crop Yield cannot be independent given Soil Moisture, because rainfall
    also affects yield through humidity, pest pressure, etc. — pathways not captured by Soil Moisture alone").
  - In the HINT, suggest how the mapper could reframe variable meanings to restore the independence
    (e.g., narrower scope, different operationalization, or a variable that better blocks the path).

4. Type consistency : if expected node type hints are provided, compare them against mapping "type" fields.
  Be lenient and pragmatic:
  - Treat close families as compatible when reasonable (e.g., count/integer/discrete vs continuous scale).
  - Skip nodes where either expected or mapped type is missing/unknown.
  - Only flag a VIOLATION for clear contradictions that would mislead downstream reasoning
    (e.g., expected binary but mapped as clearly non-binary ratio/continuous physiology variable).

INTERPRETATION NOTES:
We mostly care about the NON-EDGE violations. Use the definition of the DIRECT CAUSAL LINK above very strictly.
For example, a verbal aggregation of a detailed effect X -> M -> Y is NOT a separate direct effect X -> Y! For that, we require a SEPARATE direct pathway!
{derived_name_edge_red_flag}
{restrictive_qualifier_rule}
For the second criterion - DIRECT EDGE PLAUSIBILITY - be lenient. We do not care if the effect could be bi-directional, as long as the stated direction is plausible. Even indirect, weak, or speculative mechanisms count. Only reject edges that are clearly nonsensical.
The third criterion - the CI STATEMENTS - is not as high priority as the first two for pass/fail decisions. Treat them as a soft constraint: think 'common sense independence' rather than 'theoretical absolute conditional independence'. However, CI violations still provide valuable feedback to the mapper. When you DO flag a CI violation, make the explanation detailed and actionable — explain what alternative pathways or confounders make the independence implausible, and hint at how variable meanings could be narrowed or reframed to fix it.
Then fourth criterion - TYPE CONSISTENCY - is also a soft constraint. Only flag clear contradictions that would cause confusion or mislead downstream users.
 
FEEDBACK QUALITY REQUIREMENT (critical for iterative repair):
- For every entry in "violations", the "explanation" must be actionable. End each explanation with a short 'HINT:' clause suggesting how the violation might be resolved. It is not your job to fix the violation, so 100% fixes are not required. In the HINT:
    - Do not suggest changing the graph structure as that cannot be done, e.g. removing edges or adding new ones.
    - Do NOT suggest changing fixed nodes. (see FIXED NODE IDS in the user prompt).
    - Keep suggestions within the PROPOSED DOMAIN if it can be done.
- If a violation is caused by two fixed nodes (unrepairable without changing fixed nodes),
  say so explicitly in the 'HINT:' clause (e.g., 'HINT: unrecoverable with fixed nodes; reject this sample.').

{output_format_instruction}
"""

_AUDIT_OUTPUT_INSTRUCTION_JSON = """Return ONLY valid JSON in the required schema below. No prose.
REQUIRED OUTPUT JSON SCHEMA:
{{
  "pass": boolean,
  "violations": [
    {{
      "kind": string,
      "pair": [string, string] | null,
      "story_pair": [string, string] | null,
      "explanation": string
    }}
  ],
  "non_edge_attestations": [
    {{
      "pair": [string, string],
      "story_pair": [string, string],
      "no_direct_link": boolean,
      "justification": string
    }}
  ],
  "summary": string
}}"""

_AUDIT_OUTPUT_INSTRUCTION_XML = """Return ONLY valid XML in the required schema below. No prose.
REQUIRED OUTPUT XML SCHEMA:
<audit>
  <pass>true or false</pass>
  <violations>
    <violation kind="..." explanation="...">
      <pair><node>...</node><node>...</node></pair>
      <story_pair><node>...</node><node>...</node></story_pair>
    </violation>
  </violations>
  <non_edge_attestations>
    <attestation no_direct_link="true or false" justification="...">
      <pair><node>...</node><node>...</node></pair>
      <story_pair><node>...</node><node>...</node></story_pair>
    </attestation>
  </non_edge_attestations>
  <summary>...</summary>
</audit>"""

# Default (JSON) for backward compatibility
AUDIT_SYSTEM_PROMPT = _AUDIT_SYSTEM_PROMPT_TEMPLATE.format(
    stage_specific_instruction_block="",
    non_edge_bidirectional_rule=NON_EDGE_BIDIRECTIONAL_CHECK_RULE,
    derived_name_edge_red_flag=DERIVED_NAME_EDGE_RED_FLAG,
    artifact_edge_exception=MAPPING_AUDIT_ARTIFACT_EDGE_EXCEPTION,
    restrictive_qualifier_rule=RESTRICTIVE_QUALIFIER_RULE,
    output_format_instruction=_AUDIT_OUTPUT_INSTRUCTION_JSON,
)


def _normalize_audit_stage(audit_stage: Optional[str]) -> str:
    stage = str(audit_stage or "main_graph").strip().lower()
    if stage in {"auxiliary_graph", "auxiliary_graph_local"}:
        return "auxiliary_graph_local"
    if stage in {"final_augmented_graph", "global_augmented_graph"}:
        return "final_augmented_graph"
    return "main_graph"


def _audit_stage_instruction(audit_stage: Optional[str]) -> str:
    stage = _normalize_audit_stage(audit_stage)
    if stage == "auxiliary_graph_local":
        return (
            "ADDITIONAL REQUIREMENTS:\n"
            "- Treat the shared anchor as immutable. Do not reinterpret, paraphrase, or rename it.\n"
            "- Pay extra attention to name collisions or meanings that would imply unintended direct links to already-mapped variables.\n\n"
        )
    if stage == "final_augmented_graph":
        return (
            "ADDITIONAL REQUIREMENTS:\n"
            "- Treat the provided mapping as an already-proposed full-graph candidate mapping and audit it directly.\n"
            "- Pay extra attention to cross-graph direct-link implications, duplicate semantics across different grafts, and anchor stability.\n\n"
        )
    return ""


def build_audit_system_prompt(
    output_format: str = "json",
    audit_stage: str = "main_graph",
    enable_web: bool = False,
) -> str:
    """Build audit system prompt with the appropriate output format instruction."""
    if output_format == "xml":
        instruction = _AUDIT_OUTPUT_INSTRUCTION_XML
    else:
        instruction = _AUDIT_OUTPUT_INSTRUCTION_JSON
    prompt = _AUDIT_SYSTEM_PROMPT_TEMPLATE.format(
        stage_specific_instruction_block=_audit_stage_instruction(audit_stage),
        non_edge_bidirectional_rule=NON_EDGE_BIDIRECTIONAL_CHECK_RULE,
        derived_name_edge_red_flag=DERIVED_NAME_EDGE_RED_FLAG,
        artifact_edge_exception=MAPPING_AUDIT_ARTIFACT_EDGE_EXCEPTION,
        restrictive_qualifier_rule=RESTRICTIVE_QUALIFIER_RULE,
        output_format_instruction=instruction,
    )
    if enable_web:
        prompt = f"{prompt}\n\n{AUDITOR_WEB_SEARCH_TOOL_INSTRUCTION}"
    return prompt


def build_audit_user_prompt(
    mapping_rows: List[Dict[str, Any]],
    edges: List[Tuple[str, str]],
    non_edges: List[Tuple[str, str]],
    ci_lines: Optional[List[str]] = None,
    proposed_domain: Optional[str] = None,
    fixed_nodes: Optional[List[str]] = None,
    expected_node_types: Optional[Dict[str, Any]] = None,
    output_format: str = "json",
    audit_stage: str = "main_graph",
    existing_graph_mapping_rows: Optional[List[Dict[str, Any]]] = None,
    shared_anchor_context: Optional[Dict[str, Any]] = None,
) -> str:
    logger.debug(
        "Building audit prompt: %d edges, %d non-edges, %d CI lines",
        len(edges),
        len(non_edges),
        len(ci_lines) if ci_lines else 0,
    )

    # Normalize CI lines (handle both list and string inputs)
    if isinstance(ci_lines, str):
        ci_block = ci_lines
    elif isinstance(ci_lines, list):
        ci_block = (
            "\n".join(str(line).lstrip("- ") for line in ci_lines) if ci_lines else None
        )
    else:
        ci_block = None

    domain_block = proposed_domain.strip() if isinstance(proposed_domain, str) else None
    expected_types_block: Dict[str, str] = {}
    if isinstance(expected_node_types, dict):
        expected_types_block = {
            str(k): str(v)
            for k, v in expected_node_types.items()
            if k is not None and v is not None and str(v).strip()
        }

    # NOTE: let's build the prompt programmatically for clarity
    parts: List[str] = []
    parts.append("AUDIT INPUTS")

    if domain_block:
        parts.append(
            "PROPOSED DOMAIN (context only; mapping below is authoritative):\n"
            + domain_block
        )

    existing_context_rows = []
    for row in existing_graph_mapping_rows or []:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        story_name = str(row.get("story_name", "")).strip()
        if not node_id or not story_name:
            continue
        existing_context_rows.append(
            {
                "id": node_id,
                "story_name": story_name,
                "observed": row.get("observed"),
                "type": row.get("type"),
                "unit": row.get("unit"),
            }
        )
    if existing_context_rows:
        parts.append(
            "ALREADY-MAPPED MAIN-GRAPH / PRIOR-GRAFT VARIABLES "
            "(context for cross-graph collisions and unintended direct links):\n"
            + json.dumps(existing_context_rows, indent=2)
        )

    anchor_context_block: Dict[str, str] = {}
    if isinstance(shared_anchor_context, dict):
        anchor_context_block = {
            str(k): str(v)
            for k, v in shared_anchor_context.items()
            if k is not None and v is not None and str(v).strip()
        }
    if anchor_context_block:
        parts.append(
            "SHARED ANCHOR CONTEXT:\n" + json.dumps(anchor_context_block, indent=2)
        )

    parts.append(
        "DIRECT EDGES (use the given definition of DIRECT CAUSAL LINK):\n"
        + json.dumps(edges, indent=2)
    )
    parts.append(
        "NON-EDGE PAIRS (use the given definition of DIRECT CAUSAL LINK):\n"
        + json.dumps(non_edges, indent=2)
    )

    if ci_block:
        parts.append(
            "CONDITIONAL INDEPENDENCE RELATIONS (secondary plausibility check):\n"
            + str(ci_block)
        )
    parts.append(
        "FIXED NODE IDS (immutable; you cannot propose changing these meanings):\n"
        + json.dumps(fixed_nodes, indent=2)
    )
    parts.append(
        "PROPOSED VARIABLE MAPPING (id -> meaning):\n"
        + json.dumps(mapping_rows, indent=2)
    )
    yellow_flags = collect_node_name_yellow_flags(mapping_rows)
    if yellow_flags:
        parts.append(
            "NODE-NAME YELLOW FLAGS (heuristic only; scrutinize these carefully, but do not fail solely for this reason):\n"
            + json.dumps(yellow_flags, indent=2)
        )
    parts.append(
        "EXPECTED NODE TYPE HINTS (soft constraint; use only when provided):\n"
        + (
            json.dumps(expected_types_block, indent=2)
            if expected_types_block
            else "null"
        )
    )

    output_format_name = "XML" if output_format == "xml" else "JSON"

    if output_format == "xml":
        schema_block = (
            "REQUIRED OUTPUT XML SCHEMA:\n"
            "<audit>\n"
            "  <pass>true or false</pass>\n"
            "  <violations>\n"
            '    <violation kind="..." explanation="...">\n'
            "      <pair><node>...</node><node>...</node></pair>\n"
            "      <story_pair><node>...</node><node>...</node></story_pair>\n"
            "    </violation>\n"
            "  </violations>\n"
            "  <non_edge_attestations>\n"
            '    <attestation no_direct_link="true or false" justification="...">\n'
            "      <pair><node>...</node><node>...</node></pair>\n"
            "      <story_pair><node>...</node><node>...</node></story_pair>\n"
            "    </attestation>\n"
            "  </non_edge_attestations>\n"
            "  <summary>...</summary>\n"
            "</audit>\n"
        )
    else:
        schema_block = (
            "REQUIRED OUTPUT JSON SCHEMA:\n"
            "{\n"
            '  "pass": boolean,\n'
            '  "violations": [\n'
            "    {\n"
            '      "kind": string,\n'
            '      "pair": [string, string] | null,\n'
            '      "story_pair": [string, string] | null,\n'
            '      "explanation": string\n'
            "    }\n"
            "  ],\n"
            '  "non_edge_attestations": [\n'
            "    {\n"
            '      "pair": [string, string],\n'
            '      "story_pair": [string, string],\n'
            '      "no_direct_link": boolean,\n'
            '      "justification": string\n'
            "    }\n"
            "  ],\n"
            '  "summary": string\n'
            "}\n"
        )

    parts.append(
        "GROUNDING RULES:\n"
        "- Use ONLY the meanings given in PROPOSED VARIABLE MAPPING.\n"
        '- When you fill any "story_pair", copy the corresponding story_name strings verbatim from the mapping.\n'
        "- Do NOT invent alternative domains, studies, or variables not present in the mapping or given context.\n\n"
        "SOFT TYPE RULE:\n"
        "- If EXPECTED NODE TYPE HINTS are provided, prefer mapped types compatible with those hints.\n"
        "- Be lenient (e.g., count/integer/discrete vs continuous can be acceptable in context).\n"
        "- Only flag clear type contradictions as violations.\n\n"
        "FEEDBACK REQUIREMENT:\n"
        '- For each violation, end "explanation" and a HINT of how the violation might be resolved.\n'
        "- For the HINT, note that we CANNOT modify the graph or change the fixed nodes (besides slight renaming).\n"
        "- If violation involves only fixed nodes, say so explicitly in 'HINT:' clause.\n"
        'IMPORTANT: "non_edge_attestations" must contain exactly one entry for EVERY pair in NON-EDGE PAIRS.\n'
        'The "pair" field must exactly match the Node IDs given in the input (NOT story names).\n'
        f"Return ONLY the {output_format_name} object, no extra keys, no prose.\n\n"
        + schema_block
    )

    return "\n".join(parts)


###############################################################################
# Audit Check Runner
###############################################################################

# Violation kinds that are NOT considered "unfixable fixed nodes" failures
UNFIXABLE_IGNORE_KINDS = frozenset(
    {
        "audit_parse_error",
        "missing_attestation",
    }
)


def _normalize_type_label(raw_type: Any) -> Optional[str]:
    """Normalize free-form type strings to coarse categories."""
    if raw_type is None:
        return None
    txt = str(raw_type).strip().lower()
    if txt in _UNKNOWN_TYPE_TOKENS:
        return None
    txt_norm = re.sub(r"[_\-]+", " ", txt)

    if any(
        token in txt_norm
        for token in ("binary", "boolean", "bool", "bernoulli", "indicator", "flag")
    ):
        return "binary"
    if any(
        token in txt_norm
        for token in ("count", "integer", "int", "discrete", "poisson", "ordinal")
    ):
        return "count"
    if any(
        token in txt_norm
        for token in (
            "continuous",
            "real",
            "float",
            "numeric",
            "number",
            "ratio",
            "interval",
            "gaussian",
        )
    ):
        return "continuous"
    if any(token in txt_norm for token in ("categorical", "category", "nominal")):
        return "categorical"
    return txt_norm


def _types_are_compatible(expected: Optional[str], mapped: Optional[str]) -> bool:
    """Lenient compatibility relation for type hints."""
    if expected is None or mapped is None:
        return True
    if expected == mapped:
        return True

    # Lenient bucket: count/integer/discrete often represented numerically.
    if expected == "continuous" and mapped == "count":
        return True
    if expected == "count" and mapped == "continuous":
        return True

    # Categorical is ambiguous without cardinality; keep this soft.
    if expected == "categorical" or mapped == "categorical":
        return True

    return False


def _compute_soft_type_mismatches(
    mapping_rows: List[Dict[str, Any]],
    expected_node_types: Dict[str, str],
) -> List[Dict[str, str]]:
    """Detect clear expected-vs-mapped type contradictions (non-blocking)."""
    if not expected_node_types:
        return []

    row_by_id: Dict[str, Dict[str, Any]] = {}
    for row in mapping_rows or []:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        if node_id:
            row_by_id[node_id] = row

    mismatches: List[Dict[str, str]] = []
    for node_id, expected_raw in expected_node_types.items():
        row = row_by_id.get(str(node_id))
        if row is None:
            continue
        mapped_raw = row.get("type")
        expected_norm = _normalize_type_label(expected_raw)
        mapped_norm = _normalize_type_label(mapped_raw)

        # Ignore unknown/missing values in the soft checker.
        if expected_norm is None or mapped_norm is None:
            continue

        if not _types_are_compatible(expected_norm, mapped_norm):
            mismatches.append(
                {
                    "id": str(node_id),
                    "expected_type": str(expected_raw),
                    "mapped_type": str(mapped_raw),
                    "message": (
                        f"Expected '{expected_raw}' but mapping provided '{mapped_raw}'. "
                        "This is a soft type inconsistency."
                    ),
                },
            )

    return mismatches


def is_unfixable_fixed_nodes_failure(
    violations: List[Dict[str, Any]],
    fixed_nodes: List[str],
) -> bool:
    """Check if any violation is caused by two fixed nodes (unfixable).

    Args:
        violations: List of violation dicts from audit result
        fixed_nodes: List of fixed node IDs that cannot be changed

    Returns:
        True if there's an unfixable violation between two fixed nodes
    """
    fixed = set(fixed_nodes or [])
    for v in violations or []:
        kind = str(v.get("kind", "")).strip().lower()

        # These are auditor-format failures, never "unfixable fixed nodes"
        if kind in UNFIXABLE_IGNORE_KINDS:
            continue

        # CI is soft in our setup; do not rejection-sample on CI alone
        if kind.startswith("ci") or "independ" in kind:
            continue

        # Edge implausibility between fixed nodes is NOT unfixable —
        # the mapper can try a different domain interpretation.
        # Only non-edge violations are truly unfixable.
        if "implausib" in kind or "edge_plausib" in kind:
            continue

        # Only treat NON-EDGE violations as potentially unfixable.
        if not ("non" in kind and "edge" in kind):
            continue

        pair = v.get("pair")
        if not (isinstance(pair, list) and len(pair) == 2):
            continue

        u, w = str(pair[0]).strip(), str(pair[1]).strip()
        if u in fixed and w in fixed:
            return True

    return False


def run_audit_check(
    *,
    client: Any = None,
    api_key: str = None,
    session: Any = None,  # ChatSession - reuse for multi-iteration audits
    model: str,
    reasoning: Any,
    sg: Any,  # SampledGraph
    mapping_rows: List[Dict[str, Any]],
    serialization_format: str,
    proposed_domain: Optional[str] = None,
    temperature: float = 0.0,
    output_format: str = "json",
    enable_web: bool = False,
    max_tool_loops: int = 3,
    max_api_retries: int = 3,
) -> Tuple[Dict[str, Any], str, str, Any]:
    """Run a single audit check on a variable mapping.

    Args:
        client: Optional LLMClient instance (preferred over api_key)
        api_key: API key for LLM provider (used if client not provided)
        session: Optional ChatSession to reuse (maintains conversation context across iterations)
        model: Model identifier
        reasoning: Reasoning configuration (True, dict, or None)
        sg: SampledGraph object with graph structure
        mapping_rows: List of mapping dicts with id, story_name, etc.
        serialization_format: Serialization format used (for context)
        proposed_domain: Optional domain context
        temperature: LLM temperature (default 0.0 for consistency)

    Returns:
        Tuple of (audit_result_dict, audit_user_prompt, audit_response_text, session)
        audit_result_dict contains:
            - 'pass': bool - whether audit passed
            - 'violations': list - specific issues found
            - 'non_edge_attestations': list - explicit attestations for non-edges
            - 'soft_type_mismatches': list - deterministic expected-vs-mapped type warnings
            - 'summary': str - brief summary
            - '_audited_proposed_domain': str - domain that was audited
            - '_audited_mapping_rows': list - mapping that was audited
        session: The ChatSession used (for reuse in subsequent calls)
    """
    from . import llm_client as cl
    from . import serialization as cs
    from .utils import retry_with_backoff

    G = sg.graph
    observed = [str(node) for node in (sg.observed_nodes or list(G.nodes()))]
    observed_set = set(observed)
    latent = [str(node) for node in G.nodes() if str(node) not in observed_set]
    edges = audit_edges(G, observed)
    non_edges = audit_non_edge_pairs_undirected(G, observed)
    fixed_nodes = (sg.meta or {}).get("fixed_nodes", [])
    audit_meta = sg.meta or {}
    audit_stage = _normalize_audit_stage(
        audit_meta.get("audit_stage") or audit_meta.get("stage_kind")
    )
    existing_graph_mapping_rows = (
        audit_meta.get("existing_graph_mapping_rows", []) or []
    )
    shared_anchor_context = audit_meta.get("shared_anchor_context", {}) or {}
    expected_node_types: Dict[str, str] = {}
    if hasattr(sg, "node_types") and isinstance(getattr(sg, "node_types"), dict):
        expected_node_types = {
            str(k): str(v)
            for k, v in getattr(sg, "node_types").items()
            if k is not None and v is not None and str(v).strip()
        }
    elif isinstance((sg.meta or {}).get("node_types"), dict):
        expected_node_types = {
            str(k): str(v)
            for k, v in (sg.meta or {}).get("node_types", {}).items()
            if k is not None and v is not None and str(v).strip()
        }

    # Get CI list based on format
    ci_mode = "minimal"
    ci_lines = cs.serialize_conditional_independencies(
        sg, mode=ci_mode, format="markdown"
    )
    if isinstance(ci_lines, str):
        ci_lines = [ln.strip() for ln in ci_lines.splitlines() if ln.strip()]

    logger.info(
        (
            "Running audit check: %d edges, %d non-edges, %d CI lines, %d fixed nodes, "
            "%d type hints, observed=%d, latent=%d"
        ),
        len(edges),
        len(non_edges),
        len(ci_lines) if ci_lines else 0,
        len(fixed_nodes),
        len(expected_node_types),
        len(observed),
        len(latent),
    )

    # Use provided session, or create one from client
    audit_system_prompt = build_audit_system_prompt(
        output_format,
        audit_stage=audit_stage,
        enable_web=enable_web,
    )
    if session is None:
        if client is None:
            if api_key is None:
                raise ValueError("Either client, session, or api_key must be provided")
            client = cl.LLMClient(api_key=api_key, default_model=model)
        session = client.create_session(model=model, system_prompt=audit_system_prompt)

    audit_prompt = build_audit_user_prompt(
        mapping_rows=mapping_rows,
        edges=edges,
        non_edges=non_edges,
        ci_lines=ci_lines,
        proposed_domain=proposed_domain,
        fixed_nodes=fixed_nodes,
        expected_node_types=expected_node_types,
        output_format=output_format,
        audit_stage=audit_stage,
        existing_graph_mapping_rows=existing_graph_mapping_rows,
        shared_anchor_context=shared_anchor_context,
    )

    try:
        audit_text, audit_raw = retry_with_backoff(
            lambda: session.chat(
                audit_prompt,
                temperature=temperature,
                tools=enable_web,
                max_tool_loops=max_tool_loops if enable_web else 0,
                reasoning=reasoning,
            ),
            max_retries=max_api_retries,
        )
    except Exception as e:
        logger.error("Audit request failed after retries: %s", e)
        audit_json = {
            "pass": False,
            "violations": [
                {
                    "kind": "audit_request_failed",
                    "pair": None,
                    "story_pair": None,
                    "explanation": (
                        "The audit request failed after exhausting API retries: "
                        f"{e}. Treating this as a normal failed audit iteration."
                    ),
                }
            ],
            "non_edge_attestations": [],
            "soft_type_mismatches": [],
            "_audited_proposed_domain": proposed_domain or "",
            "_audited_mapping_rows": mapping_rows or [],
            "_audit_stage": audit_stage,
            "_audit_system_prompt": audit_system_prompt,
            "_audit_user_prompt": audit_prompt,
            "_error": str(e),
            "summary": "Audit request failed after exhausting API retries.",
        }
        logger.info(
            "Audit complete: pass=%s, violations=%d",
            audit_json.get("pass"),
            len(audit_json.get("violations", [])),
        )
        return audit_json, audit_prompt, f"ERROR: {e}", session

    try:
        audit_json = parse_llm_output(audit_text, output_format, silent=True)

        # Ensure result is a dict (LLM may have returned unexpected structure)
        if not isinstance(audit_json, dict):
            logger.warning(
                "Audit parse returned non-dict (type=%s), creating error result",
                type(audit_json).__name__,
            )
            raise ValueError(
                f"Audit parse returned {type(audit_json).__name__}, expected dict"
            )

        # After parse, attach "what was audited" for reporting:
        audit_json["_audited_proposed_domain"] = proposed_domain or ""
        audit_json["_audited_mapping_rows"] = mapping_rows or []
        audit_json["_audit_stage"] = audit_stage
        audit_json["_audit_system_prompt"] = audit_system_prompt
        audit_json["_audit_user_prompt"] = audit_prompt

        # Deterministic soft type consistency checks (non-blocking by default)
        soft_type_mismatches = _compute_soft_type_mismatches(
            mapping_rows=mapping_rows,
            expected_node_types=expected_node_types,
        )
        audit_json["soft_type_mismatches"] = soft_type_mismatches
        if soft_type_mismatches:
            logger.warning(
                "Soft type checker found %d mismatch(es); not hard-failing audit",
                len(soft_type_mismatches),
            )
            old_summary = str(audit_json.get("summary", "")).strip()
            note = (
                f"[SYSTEM: {len(soft_type_mismatches)} soft type mismatch(es) detected]"
            )
            audit_json["summary"] = (old_summary + " " + note).strip()

        # Enforce non-edge attestations check
        # The auditor must explicitly attest to every non-edge pair.
        # If any are missing, we treat it as a failure (violation).

        # Build lookup for story_name -> id to handle cases where LLM uses story names
        story_to_id = {}
        for row in mapping_rows:
            sid = row.get("id")
            sname = row.get("story_name")
            if sid and sname:
                story_to_id[sname] = sid
                story_to_id[sname.strip()] = sid

        attestations = audit_json.get("non_edge_attestations", [])
        attested_pairs = set()
        for att in attestations:
            p = att.get("pair")
            if p and isinstance(p, list) and len(p) == 2:
                u_raw, v_raw = str(p[0]).strip(), str(p[1]).strip()
                # Try to resolve to IDs if they are not already in G
                u = u_raw if u_raw in G else story_to_id.get(u_raw, u_raw)
                v = v_raw if v_raw in G else story_to_id.get(v_raw, v_raw)
                attested_pairs.add((u, v))

        # Check if all required non_edges are present
        missing_attestations = []
        for u, v in non_edges:
            if (u, v) not in attested_pairs and (v, u) not in attested_pairs:
                missing_attestations.append((u, v))

        if missing_attestations:
            # Force fail
            audit_json["pass"] = False
            if "violations" not in audit_json:
                audit_json["violations"] = []

            for u, v in missing_attestations:
                audit_json["violations"].append(
                    {
                        "kind": "missing_attestation",
                        "pair": [u, v],
                        "story_pair": None,
                        "explanation": (
                            f"Auditor failed to explicitly attest safety for non-edge pair ({u}, {v}). "
                            "This does not necessarily imply that the mapping is incorrect, "
                            "so take it with a grain of salt."
                        ),
                    }
                )

            # Update summary
            old_summary = audit_json.get("summary", "")
            audit_json["summary"] = (
                old_summary + " [SYSTEM: Failed due to missing non-edge attestations]"
            ).strip()

    except Exception as e:
        logger.error("Audit parse failed: %s", e)
        audit_json = {
            "pass": False,
            "violations": [
                {
                    "kind": "audit_parse_error",
                    "pair": None,
                    "story_pair": None,
                    "explanation": str(e),
                }
            ],
            "non_edge_attestations": [],
            "soft_type_mismatches": [],
            "_audited_proposed_domain": proposed_domain or "",
            "_audited_mapping_rows": mapping_rows or [],
            "_audit_stage": audit_stage,
            "_audit_system_prompt": audit_system_prompt,
            "_audit_user_prompt": audit_prompt,
            "summary": "Audit failed to produce parseable JSON.",
        }

    logger.info(
        "Audit complete: pass=%s, violations=%d",
        audit_json.get("pass"),
        len(audit_json.get("violations", [])),
    )

    return audit_json, audit_prompt, audit_text, session
