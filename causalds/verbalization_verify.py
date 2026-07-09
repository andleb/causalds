"""
Story-to-DAG verification helpers.

This module mirrors the mapping-audit style:
- Build a strict auditor prompt
- Require per-node and per-edge attestations
- Apply deterministic post-checks for exact node mentions and abstract-id leaks
- Optionally drive an iterative rewrite loop from the story pipeline
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .graph import SampledGraph, induced_edges, non_edge_pairs_undirected
from .schemas import (
    AUDITOR_WEB_SEARCH_TOOL_INSTRUCTION,
    DERIVED_NAME_EDGE_RED_FLAG,
    NON_EDGE_BIDIRECTIONAL_CHECK_RULE,
    RESTRICTIVE_QUALIFIER_RULE,
    STORY_AUDIT_ARTIFACT_PLAUSIBILITY_RED_FLAG,
)
from .utils import (
    coerce_observed_flag,
    collect_node_name_yellow_flags,
    parse_llm_output,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)


GRAPH_JARGON = re.compile(
    r"\b(DAG|graph|edge|node|collider|v-structure|parent|child|adjacency|topological)\b",
    re.IGNORECASE,
)


_STORY_AUDIT_SYSTEM_PROMPT_TEMPLATE = """You are a strict story-to-DAG auditor.

You will be given:
1) The REQUIRED VARIABLE MAPPING (node ids -> story names, including observed flags).
2) The list of DIRECT edges (u -> v) that the STORY must explicitly support.
3) A list of NON-EDGE pairs (u, v) for soft checking only.
4) The proposed domain.
5) The STORY text.

Your job has two tiers:

HARD REQUIREMENTS:
1. Every required variable must be explicitly mentioned in the STORY using its provided story name.
   - If the STORY only gestures at the concept vaguely, that does NOT count.
   - However, capitalization and punctuation can be ignored.
   - If the variable is unobserved/latent, it may be narrated as a hidden, background, or unmeasured factor, but it still must be mentioned explicitly.
   - You MUST include one entry in "node_attestations" for EVERY required variable.

2. Every DIRECT edge (U -> V) must be clearly supported by the STORY itself.
   - The STORY must state or strongly imply that U affects V.
   - Be lenient about wording, but not about omission.
   - A path-level claim does NOT automatically cover every edge on the path.
     Example: saying "A eventually raises C" does not by itself prove that A -> B is mentioned.
   - You MUST include one entry in "edge_attestations" for EVERY direct edge.

3. If the STORY contradicts the stated direction of an edge, mark that as a hard violation.

SOFT CHECKS:
4. Warn if the STORY appears to introduce extra direct causal claims between NON-EDGE pairs.
   - {non_edge_bidirectional_rule}
5. Warn if the STORY drifts away from the proposed domain/setting.
6. Warn if the STORY uses graph jargon.
7. Warn if the STORY is implausible, globally incoherent, or semantically mixed in a way that makes the scenario hard to believe as a real setting.

INTERPRETATION NOTES:
- We primarily care about exact story-to-DAG faithfulness: variable mentions, required direct edges, and direction consistency.
- The plausibility/coherence check is NOT as high priority as those hard requirements for pass/fail decisions. Treat it as a soft constraint: think common-sense realism and domain coherence, not nitpicky perfection or exhaustive fact-checking.
- Only raise a plausibility/coherence warning when the problem is clear and actionable, such as incompatible domains mixed together, impossible mechanisms, or variables that do not belong in one believable study/system.
- {derived_name_edge_red_flag}
- {restrictive_qualifier_rule}
- {artifact_plausibility_red_flag}

DECISION RULE:
- Set "pass" to true only if all hard requirements are satisfied.
- Put ONLY hard failures in "violations".
- Put lower-priority issues in "warnings".

FEEDBACK QUALITY REQUIREMENT:
- Each issue must be actionable.
- End each explanation with a short "HINT:" clause suggesting how to rewrite the story to remove the violations.

{output_format_instruction}
"""

_STORY_AUDIT_OUTPUT_INSTRUCTION_JSON = """Return ONLY valid JSON in the required schema below. No prose.
REQUIRED OUTPUT JSON SCHEMA:
{
  "pass": boolean,
  "violations": [
    {
      "kind": string,
      "severity": "hard" | "soft",
      "node": string | null,
      "story_name": string | null,
      "pair": [string, string] | null,
      "story_pair": [string, string] | null,
      "explanation": string
    }
  ],
  "warnings": [
    {
      "kind": string,
      "severity": "hard" | "soft",
      "node": string | null,
      "story_name": string | null,
      "pair": [string, string] | null,
      "story_pair": [string, string] | null,
      "explanation": string
    }
  ],
  "node_attestations": [
    {
      "id": string,
      "story_name": string,
      "mentioned": boolean,
      "justification": string
    }
  ],
  "edge_attestations": [
    {
      "pair": [string, string],
      "story_pair": [string, string],
      "supported": boolean,
      "contradicted": boolean,
      "justification": string
    }
  ],
  "summary": string
}"""

_STORY_AUDIT_OUTPUT_INSTRUCTION_XML = """Return ONLY valid XML in the required schema below. No prose.
REQUIRED OUTPUT XML SCHEMA:
<story_audit>
  <pass>true or false</pass>
  <violations>
    <violation kind="..." severity="hard or soft" explanation="...">
      <node>...</node>
      <story_name>...</story_name>
      <pair><node>...</node><node>...</node></pair>
      <story_pair><node>...</node><node>...</node></story_pair>
    </violation>
  </violations>
  <warnings>
    <warning kind="..." severity="hard or soft" explanation="...">
      <node>...</node>
      <story_name>...</story_name>
      <pair><node>...</node><node>...</node></pair>
      <story_pair><node>...</node><node>...</node></story_pair>
    </warning>
  </warnings>
  <node_attestations>
    <attestation mentioned="true or false" justification="...">
      <id>...</id>
      <story_name>...</story_name>
    </attestation>
  </node_attestations>
  <edge_attestations>
    <attestation supported="true or false" contradicted="true or false" justification="...">
      <pair><node>...</node><node>...</node></pair>
      <story_pair><node>...</node><node>...</node></story_pair>
    </attestation>
  </edge_attestations>
  <summary>...</summary>
</story_audit>"""


STORY_AUDIT_SYSTEM_PROMPT = _STORY_AUDIT_SYSTEM_PROMPT_TEMPLATE.format(
    non_edge_bidirectional_rule=NON_EDGE_BIDIRECTIONAL_CHECK_RULE,
    derived_name_edge_red_flag=DERIVED_NAME_EDGE_RED_FLAG,
    restrictive_qualifier_rule=RESTRICTIVE_QUALIFIER_RULE,
    artifact_plausibility_red_flag=STORY_AUDIT_ARTIFACT_PLAUSIBILITY_RED_FLAG,
    output_format_instruction=_STORY_AUDIT_OUTPUT_INSTRUCTION_JSON,
)


def build_story_audit_system_prompt(
    output_format: str = "json", enable_web: bool = False
) -> str:
    """Build story-audit system prompt with output-format instructions."""
    if output_format == "xml":
        instruction = _STORY_AUDIT_OUTPUT_INSTRUCTION_XML
    else:
        instruction = _STORY_AUDIT_OUTPUT_INSTRUCTION_JSON
    prompt = _STORY_AUDIT_SYSTEM_PROMPT_TEMPLATE.format(
        non_edge_bidirectional_rule=NON_EDGE_BIDIRECTIONAL_CHECK_RULE,
        derived_name_edge_red_flag=DERIVED_NAME_EDGE_RED_FLAG,
        restrictive_qualifier_rule=RESTRICTIVE_QUALIFIER_RULE,
        artifact_plausibility_red_flag=STORY_AUDIT_ARTIFACT_PLAUSIBILITY_RED_FLAG,
        output_format_instruction=instruction,
    )
    if enable_web:
        prompt = f"{prompt}\n\n{AUDITOR_WEB_SEARCH_TOOL_INSTRUCTION}"
    return prompt


def build_story_audit_inputs(
    sg: SampledGraph,
    mapping_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Prepare canonical node/edge lists for story verification."""
    G = sg.graph
    observed = list(sg.observed_nodes or G.nodes())

    row_by_id: Dict[str, Dict[str, Any]] = {}
    row_by_story_name: Dict[str, Dict[str, Any]] = {}
    required_nodes: List[Dict[str, Any]] = []
    for row in mapping_rows or []:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        story_name = str(row.get("story_name", "")).strip()
        if not node_id or not story_name:
            continue
        normalized = {
            "id": node_id,
            "story_name": story_name,
            "observed": bool(coerce_observed_flag(row.get("observed", True))),
            "type": row.get("type"),
            "unit": row.get("unit"),
        }
        if node_id not in row_by_id:
            required_nodes.append(normalized)
        row_by_id[node_id] = normalized
        row_by_story_name[story_name] = normalized

    graph_node_to_canonical_id: Dict[str, str] = {}
    for node_id in G.nodes():
        node_id_str = str(node_id).strip()
        matched = row_by_id.get(node_id_str) or row_by_story_name.get(node_id_str)
        if matched is None:
            matched = {
                "id": node_id_str,
                "story_name": node_id_str,
                "observed": node_id in observed,
                "type": None,
                "unit": None,
            }
            row_by_id[node_id_str] = matched
            row_by_story_name[node_id_str] = matched
            required_nodes.append(matched)
        graph_node_to_canonical_id[node_id_str] = str(matched["id"]).strip()

    graph_node_order = [str(node_id).strip() for node_id in G.nodes()]
    edges_story = induced_edges(G, graph_node_order)
    non_edges_story = non_edge_pairs_undirected(G, graph_node_order)

    edges: List[Tuple[str, str]] = []
    seen_edges: set = set()
    for src, dst in edges_story:
        canonical_pair = (
            graph_node_to_canonical_id.get(str(src).strip(), str(src).strip()),
            graph_node_to_canonical_id.get(str(dst).strip(), str(dst).strip()),
        )
        if canonical_pair not in seen_edges:
            edges.append(canonical_pair)
            seen_edges.add(canonical_pair)

    non_edges: List[Tuple[str, str]] = []
    seen_non_edges: set = set()
    for src, dst in non_edges_story:
        canonical_pair = (
            graph_node_to_canonical_id.get(str(src).strip(), str(src).strip()),
            graph_node_to_canonical_id.get(str(dst).strip(), str(dst).strip()),
        )
        if canonical_pair not in seen_non_edges:
            non_edges.append(canonical_pair)
            seen_non_edges.add(canonical_pair)

    return {
        "required_nodes": required_nodes,
        "edges": edges,
        "non_edges": non_edges,
        "row_by_id": row_by_id,
    }


def build_story_audit_user_prompt(
    mapping_rows: List[Dict[str, Any]],
    edges: List[Tuple[str, str]],
    non_edges: List[Tuple[str, str]],
    story: str,
    proposed_domain: Optional[str] = None,
    output_format: str = "json",
) -> str:
    """Build the user prompt for story verification."""
    parts: List[str] = ["AUDIT INPUTS"]

    if proposed_domain:
        parts.append("PROPOSED DOMAIN:\n" + str(proposed_domain).strip())

    parts.append(
        "REQUIRED VARIABLE MAPPING:\n" + json.dumps(mapping_rows or [], indent=2)
    )
    yellow_flags = collect_node_name_yellow_flags(mapping_rows or [])
    if yellow_flags:
        parts.append(
            "NODE-NAME YELLOW FLAGS (heuristic only; scrutinize these carefully, but do not fail solely for this reason):\n"
            + json.dumps(yellow_flags, indent=2)
        )
    parts.append("DIRECT EDGES:\n" + json.dumps(edges, indent=2))
    parts.append("SOFT-CHECK NON-EDGE PAIRS:\n" + json.dumps(non_edges, indent=2))
    parts.append("STORY TO AUDIT:\n```text\n" + (story or "") + "\n```")

    output_format_name = "XML" if output_format == "xml" else "JSON"

    if output_format == "xml":
        schema_block = _STORY_AUDIT_OUTPUT_INSTRUCTION_XML
    else:
        schema_block = _STORY_AUDIT_OUTPUT_INSTRUCTION_JSON

    parts.append(
        "GROUNDING RULES:\n"
        "- Use ONLY the provided mapping and the STORY text.\n"
        "- For node mentions, require the exact provided story_name to appear in the STORY.\n"
        "- For edge support, be lenient about paraphrase but do not give credit for omitted links.\n"
        "- For plausibility/coherence, use common-sense judgment and warn only for clear problems.\n"
        "- When filling any story_pair, copy the provided story_name strings verbatim.\n"
        "- Do not let the causal_justifications appendix substitute for missing content in the STORY.\n"
        f"Return ONLY the {output_format_name} object, no extra keys, no prose.\n\n"
        + schema_block
    )

    return "\n".join(parts)


def _story_mentions_name(story: str, story_name: str) -> bool:
    """Check if the story contains the exact story name, case-insensitively."""
    normalized_story = re.sub(r"\s+", " ", story or "").strip()
    normalized_name = re.sub(r"\s+", " ", story_name or "").strip()
    if not normalized_name:
        return False
    pattern = r"(?<!\w)" + re.escape(normalized_name) + r"(?!\w)"
    return re.search(pattern, normalized_story, flags=re.IGNORECASE) is not None


def _looks_like_abstract_node_id(node_id: str) -> bool:
    """Heuristic for placeholder graph ids that should never appear in stories."""
    return bool(re.fullmatch(r"(?:[VUMZ]\d*|[XYW])", node_id.strip(), flags=re.I))


def _make_issue(
    *,
    kind: str,
    explanation: str,
    severity: str = "hard",
    node: Optional[str] = None,
    story_name: Optional[str] = None,
    pair: Optional[List[str]] = None,
    story_pair: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "severity": severity,
        "node": node,
        "story_name": story_name,
        "pair": pair,
        "story_pair": story_pair,
        "explanation": explanation,
    }


def _issue_key(issue: Dict[str, Any]) -> Tuple[Any, ...]:
    pair = tuple(issue.get("pair") or [])
    story_pair = tuple(issue.get("story_pair") or [])
    return (
        issue.get("kind"),
        issue.get("severity"),
        issue.get("node"),
        issue.get("story_name"),
        pair,
        story_pair,
    )


def _append_issue_once(issues: List[Dict[str, Any]], issue: Dict[str, Any]) -> None:
    """Append issue unless an identical issue is already present."""
    existing_keys = {_issue_key(existing) for existing in issues}
    if _issue_key(issue) not in existing_keys:
        issues.append(issue)


def finalize_story_audit_result(
    audit_result: Dict[str, Any],
    *,
    mapping_rows: List[Dict[str, Any]],
    edges: List[Tuple[str, str]],
    story: str,
) -> Dict[str, Any]:
    """Normalize and enforce the story-audit contract deterministically."""
    result = dict(audit_result or {})
    result["violations"] = list(result.get("violations") or [])
    result["warnings"] = list(result.get("warnings") or [])
    result["node_attestations"] = list(result.get("node_attestations") or [])
    result["edge_attestations"] = list(result.get("edge_attestations") or [])
    result["_audited_story"] = story or ""
    result["_audited_mapping_rows"] = mapping_rows or []

    required_nodes: List[Dict[str, Any]] = []
    story_name_to_id: Dict[str, str] = {}
    for row in mapping_rows or []:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        story_name = str(row.get("story_name", "")).strip()
        if not node_id or not story_name:
            continue
        required_nodes.append({"id": node_id, "story_name": story_name})
        story_name_to_id[story_name] = node_id
        story_name_to_id[story_name.strip()] = node_id

    node_att_by_id: Dict[str, Dict[str, Any]] = {}
    for att in result["node_attestations"]:
        if not isinstance(att, dict):
            continue
        node_id = str(att.get("id", "")).strip()
        story_name = str(att.get("story_name", "")).strip()
        if not node_id and story_name:
            node_id = story_name_to_id.get(story_name, "")
        if node_id:
            node_att_by_id[node_id] = att

    edge_att_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for att in result["edge_attestations"]:
        if not isinstance(att, dict):
            continue
        pair = att.get("pair")
        story_pair = att.get("story_pair")
        if isinstance(pair, list) and len(pair) == 2:
            key = (str(pair[0]).strip(), str(pair[1]).strip())
            edge_att_by_pair[key] = att
            continue
        if isinstance(story_pair, list) and len(story_pair) == 2:
            src = story_name_to_id.get(str(story_pair[0]).strip())
            dst = story_name_to_id.get(str(story_pair[1]).strip())
            if src and dst:
                edge_att_by_pair[(src, dst)] = att

    for node_info in required_nodes:
        node_id = node_info["id"]
        story_name = node_info["story_name"]
        attestation = node_att_by_id.get(node_id)

        if attestation is None:
            result["pass"] = False
            _append_issue_once(
                result["violations"],
                _make_issue(
                    kind="missing_node_attestation",
                    node=node_id,
                    story_name=story_name,
                    explanation=(
                        f"The auditor did not provide a node attestation for {node_id} / {story_name}. "
                        "HINT: explicitly confirm whether each variable name appears in the story."
                    ),
                ),
            )
        else:
            if not bool(attestation.get("mentioned")):
                result["pass"] = False
                _append_issue_once(
                    result["violations"],
                    _make_issue(
                        kind="missing_node_mention",
                        node=node_id,
                        story_name=story_name,
                        explanation=(
                            f"The story does not explicitly mention {story_name}. "
                            "HINT: name this variable directly in the narrative."
                        ),
                    ),
                )

        if not _story_mentions_name(story, story_name):
            result["pass"] = False
            _append_issue_once(
                result["violations"],
                _make_issue(
                    kind="deterministic_missing_node_mention",
                    node=node_id,
                    story_name=story_name,
                    explanation=(
                        f"The exact story name '{story_name}' does not appear in the story text. "
                        "HINT: use the provided story name verbatim in the narrative."
                    ),
                ),
            )

        if _looks_like_abstract_node_id(node_id) and re.search(
            r"(?<!\w)" + re.escape(node_id) + r"(?!\w)",
            story or "",
            flags=re.IGNORECASE,
        ):
            result["pass"] = False
            _append_issue_once(
                result["violations"],
                _make_issue(
                    kind="raw_node_id_mention",
                    node=node_id,
                    story_name=story_name,
                    explanation=(
                        f"The raw node id '{node_id}' appears in the story. "
                        "HINT: replace raw graph ids with the mapped story names."
                    ),
                ),
            )

    for edge in edges:
        src, dst = str(edge[0]).strip(), str(edge[1]).strip()
        attestation = edge_att_by_pair.get((src, dst))
        src_name = next(
            (row["story_name"] for row in required_nodes if row["id"] == src),
            src,
        )
        dst_name = next(
            (row["story_name"] for row in required_nodes if row["id"] == dst),
            dst,
        )

        if attestation is None:
            result["pass"] = False
            _append_issue_once(
                result["violations"],
                _make_issue(
                    kind="missing_edge_attestation",
                    pair=[src, dst],
                    story_pair=[src_name, dst_name],
                    explanation=(
                        f"The auditor did not attest whether the story supports the edge {src} -> {dst}. "
                        "HINT: explicitly review every required edge one by one."
                    ),
                ),
            )
            continue

        if not bool(attestation.get("supported")):
            result["pass"] = False
            _append_issue_once(
                result["violations"],
                _make_issue(
                    kind="missing_edge_mention",
                    pair=[src, dst],
                    story_pair=[src_name, dst_name],
                    explanation=(
                        f"The story does not clearly support the direct edge {src_name} -> {dst_name}. "
                        "HINT: add a sentence making this influence explicit."
                    ),
                ),
            )

        if bool(attestation.get("contradicted")):
            result["pass"] = False
            _append_issue_once(
                result["violations"],
                _make_issue(
                    kind="edge_direction_contradiction",
                    pair=[src, dst],
                    story_pair=[src_name, dst_name],
                    explanation=(
                        f"The story appears to contradict or reverse the required edge {src_name} -> {dst_name}. "
                        "HINT: rewrite the causal direction so the source affects the target."
                    ),
                ),
            )

    if GRAPH_JARGON.search(story or ""):
        _append_issue_once(
            result["warnings"],
            _make_issue(
                kind="graph_jargon",
                severity="soft",
                explanation=(
                    "The story contains graph jargon. "
                    "HINT: rewrite with domain language instead of graph terminology."
                ),
            ),
        )

    if not isinstance(result.get("summary"), str):
        result["summary"] = ""

    hard_violations = [
        issue
        for issue in result["violations"]
        if str(issue.get("severity", "hard")).strip().lower() != "soft"
    ]
    result["pass"] = not hard_violations

    return result


def run_story_audit_check(
    *,
    client: Any = None,
    api_key: Optional[str] = None,
    session: Any = None,
    model: str,
    reasoning: Any,
    sg: SampledGraph,
    mapping_rows: List[Dict[str, Any]],
    story: str,
    proposed_domain: Optional[str] = None,
    temperature: float = 0.0,
    output_format: str = "json",
    enable_web: bool = False,
    max_tool_loops: int = 3,
    max_api_retries: int = 3,
) -> Tuple[Dict[str, Any], str, str, Any]:
    """Run a single auditor pass on a generated story."""
    from . import llm_client as cl

    inputs = build_story_audit_inputs(sg=sg, mapping_rows=mapping_rows)
    required_nodes = inputs["required_nodes"]
    edges = inputs["edges"]
    non_edges = inputs["non_edges"]

    logger.info(
        "Running story audit: %d nodes, %d edges, %d non-edges",
        len(required_nodes),
        len(edges),
        len(non_edges),
    )

    if session is None:
        if client is None:
            if api_key is None:
                raise ValueError("Either client, session, or api_key must be provided")
            client = cl.LLMClient(api_key=api_key, default_model=model)
        system_prompt = build_story_audit_system_prompt(
            output_format,
            enable_web=enable_web,
        )
        session = client.create_session(model=model, system_prompt=system_prompt)

    audit_prompt = build_story_audit_user_prompt(
        mapping_rows=required_nodes,
        edges=edges,
        non_edges=non_edges,
        story=story,
        proposed_domain=proposed_domain,
        output_format=output_format,
    )

    try:
        audit_text, _ = retry_with_backoff(
            lambda: session.chat(
                audit_prompt,
                temperature=temperature,
                tools=enable_web,
                max_tool_loops=max_tool_loops if enable_web else 0,
                reasoning=reasoning,
            ),
            max_retries=max_api_retries,
        )
    except Exception as exc:
        logger.error("Story audit request failed after retries: %s", exc)
        audit_result = finalize_story_audit_result(
            {
                "pass": False,
                "violations": [
                    _make_issue(
                        kind="story_audit_request_failed",
                        explanation=(
                            "The story audit request failed after exhausting API "
                            f"retries: {exc}. Treating this as a normal failed "
                            "audit iteration."
                        ),
                    )
                ],
                "warnings": [],
                "node_attestations": [],
                "edge_attestations": [],
                "_error": str(exc),
                "summary": "Story audit request failed after exhausting API retries.",
            },
            mapping_rows=required_nodes,
            edges=edges,
            story=story,
        )
        logger.info(
            "Story audit complete: pass=%s, violations=%d, warnings=%d",
            audit_result.get("pass"),
            len(audit_result.get("violations", [])),
            len(audit_result.get("warnings", [])),
        )
        return audit_result, audit_prompt, f"ERROR: {exc}", session

    try:
        parsed = parse_llm_output(audit_text, output_format, silent=True)
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Story audit parse returned {type(parsed).__name__}, expected dict"
            )
        audit_result = finalize_story_audit_result(
            parsed,
            mapping_rows=required_nodes,
            edges=edges,
            story=story,
        )
    except Exception as exc:
        logger.error("Story audit parse failed: %s", exc)
        audit_result = finalize_story_audit_result(
            {
                "pass": False,
                "violations": [
                    _make_issue(
                        kind="story_audit_parse_error",
                        explanation=(
                            f"Story audit failed to produce parseable output: {exc}. "
                            "HINT: return the requested schema exactly."
                        ),
                    )
                ],
                "warnings": [],
                "node_attestations": [],
                "edge_attestations": [],
                "summary": "Story audit failed to produce parseable output.",
            },
            mapping_rows=required_nodes,
            edges=edges,
            story=story,
        )

    logger.info(
        "Story audit complete: pass=%s, violations=%d, warnings=%d",
        audit_result.get("pass"),
        len(audit_result.get("violations", [])),
        len(audit_result.get("warnings", [])),
    )
    return audit_result, audit_prompt, audit_text, session


__all__ = [
    "GRAPH_JARGON",
    "STORY_AUDIT_SYSTEM_PROMPT",
    "build_story_audit_inputs",
    "build_story_audit_system_prompt",
    "build_story_audit_user_prompt",
    "finalize_story_audit_result",
    "run_story_audit_check",
]
