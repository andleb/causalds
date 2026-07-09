# causalds/pre_auditor.py
"""
Pre-auditor for CauseNet mapping validation.

Checks if the CauseNet-fixed node concepts can plausibly satisfy the graph's
structural constraints BEFORE the mapper runs. This prevents wasting LLM calls
on hopeless fixed node combinations.
"""
import logging
from typing import Any, Dict, Tuple, Union

from .graph import audit_edges, audit_non_edge_pairs_undirected
from .schemas import AUDITOR_WEB_SEARCH_TOOL_INSTRUCTION, FORMAT_SYSTEM_PROMPTS
from .utils import parse_llm_output, retry_with_backoff

logger = logging.getLogger(__name__)


# ----------------------
# System Prompt
# ----------------------
PRE_AUDIT_SYSTEM_PROMPT = """You are a causal structure feasibility checker. 

You will be given:
1. A causal graph structure with FIXED NODE CONCEPTS (already named from a knowledge base)
2. The structural constraints imposed by the graph:
   - DIRECT EDGES: Causal relationships that MUST be plausible
   - NON-EDGES: Pairs where NO direct causal link must exist

YOUR TASK: Decide whether there is at least one MAINSTREAM, non-contrived interpretation under which the FIXED NODE concepts satisfy the structural constraints.
Do not require that MOST interpretations work, but do not pass based on a single exotic edge case.
Ask: "Can a skilled mapper typically find a coherent interpretation that makes the constraints work?"

IMPORTANT CONTEXT:
- This is a PRE-FILTER. A separate, thorough audit runs AFTER mapping to catch actual violations.
- The mapper is skilled at finding interpretations (e.g., "exercise" as "gym membership enrollment" rather than physical activity).
- Placeholder nodes (not yet named) give the mapper flexibility to construct mediating variables that make the structure work.
- Your job is to catch only OBVIOUS failures, not borderline cases. When in doubt, let it through.

WORKING DEFINITION of DIRECT CAUSAL LINK:
'There exists an intervention on U that changes V **while holding fixed the other variables in the graph** (especially the other nodes that could mediate the effect).'

NON-EDGE CHECK:
For each NON-EDGE pair of fixed nodes, classify the inevitability of a direct causal link in mainstream usage:
- HIGH: A direct effect is the default interpretation and would likely persist even after introducing a mediator (i.e., residual direct pathway remains plausible).
- MEDIUM: A direct effect is plausible, but a mainstream operationalization can reasonably remove it (e.g., measurement/administrative proxy, eligibility rule, time-indexed exposure).
- LOW: A direct effect is not a typical interpretation.
Only mark the sample HOPELESS (feasible=false) if ANY fixed–fixed non-edge is HIGH.
If any is MEDIUM, set feasible=true but confidence="low" and explain the needed operationalization.
If all are LOW, confidence can be "medium" or "high".

Examples where non-edge IS salvageable (FEASIBLE):
- "exercise" and "weight_loss" with mediation required: mapper can interpret "exercise" as "gym membership" (administrative) and mediate through "physical activity"
- "education" and "income" with mediation required: "education" as "years of schooling" mediated through "skills acquired"
- "stress" and "heart_disease" with mediation required: "stress" as "self-reported stress score" mediated through physiological responses
- "mining" and "habitat_destruction" with mediation required: "mining" as "mining permits issued" mediated through "actual extraction activity"

Examples where non-edge is TRULY hopeless (reject):
- "gunshot_wound" and "death" with full mediation required (gunshot directly causes death — no reframing helps)
- "decapitation" and "death" with full mediation required (immediately fatal)
- "drinking_poison" and "poisoning" with full mediation required (tautological direct link)

EDGE PLAUSIBILITY CHECK (apply very generously):
For each DIRECT EDGE between fixed nodes, ask: "Is there ANY mechanism — even indirect, weak, or context-dependent — by which U could influence V?"
Only flag as HOPELESS if the edge is physically impossible or logically contradictory.

DECISION RULE: Default to FEASIBLE (feasible=true) with confidence="low" when uncertain. when uncertain. Let the full audit catch actual problems.

=== GRAPH FORMAT ===
{format_description}

{web_tool_instruction}

{output_format_instruction}
"""

_PRE_AUDIT_OUTPUT_INSTRUCTION_JSON = """Output ONLY valid JSON:
{{
  "feasible": boolean,
  "confidence": "high" | "medium" | "low",
  "reason": string,
  "problematic_pairs": [
    {{
      "pair": [node1, node2],
      "constraint": "non_edge" | "edge_implausible" | "direction_wrong",
      "explanation": string
    }}
  ],
  "suggested_domain": string | null
}}"""

_PRE_AUDIT_OUTPUT_INSTRUCTION_XML = """Output ONLY valid XML:
<pre_audit>
  <feasible>true or false</feasible>
  <confidence>high | medium | low</confidence>
  <reason>...</reason>
  <problematic_pairs>
    <pair constraint="non_edge | edge_implausible | direction_wrong" explanation="...">
      <node>node1</node>
      <node>node2</node>
    </pair>
  </problematic_pairs>
  <suggested_domain>... or empty</suggested_domain>
</pre_audit>"""


def build_pre_audit_system_prompt(
    serialization_format: str,
    output_format: str = "json",
    enable_web: bool = False,
) -> str:
    """Build the pre-audit system prompt with the appropriate format description.

    Args:
        serialization_format: One of 'cyaml', 'parents_json', 'parents_xml', 'edge_list', 'text_simple', 'ci_only'
        output_format: "json" or "xml" — LLM output format

    Returns:
        Complete system prompt string
    """
    format_description = FORMAT_SYSTEM_PROMPTS.get(
        serialization_format, FORMAT_SYSTEM_PROMPTS["simple_json"]
    )
    if output_format == "xml":
        output_instruction = _PRE_AUDIT_OUTPUT_INSTRUCTION_XML
    else:
        output_instruction = _PRE_AUDIT_OUTPUT_INSTRUCTION_JSON
    web_tool_instruction = AUDITOR_WEB_SEARCH_TOOL_INSTRUCTION if enable_web else ""
    return PRE_AUDIT_SYSTEM_PROMPT.format(
        format_description=format_description,
        web_tool_instruction=web_tool_instruction,
        output_format_instruction=output_instruction,
    )


# ----------------------
# Prompt Builder
# ----------------------
def build_pre_audit_user_prompt(sg) -> str:
    """Build pre-audit prompt to check CauseNet mapping feasibility.

    Args:
        sg: SampledGraph object with meta containing fixed_nodes, needs_names

    Returns:
        User prompt string for the pre-auditor
    """
    meta = getattr(sg, "meta", None) or {}
    fixed_nodes = meta.get("fixed_nodes", [])
    needs_names = meta.get("needs_names", [])
    motif = getattr(sg, "motif", "unknown")

    G = sg.graph
    observed = getattr(sg, "observed_nodes", None) or list(G.nodes())

    # Audit coverage should include latent-related structure too; only pure
    # latent-latent non-edges are skipped as low-value/brittle.
    edges = audit_edges(G, observed)
    all_non_edges = audit_non_edge_pairs_undirected(G, observed)

    # Filter non-edges to those involving fixed nodes
    fixed_set = set(fixed_nodes)
    non_edges_fixed = [
        (u, v) for u, v in all_non_edges if u in fixed_set and v in fixed_set
    ]

    # Build fixed nodes section (no provenance - pre-auditor should imagine any domain)
    fixed_nodes_lines = [f"  - {fn}" for fn in fixed_nodes]
    fixed_nodes_str = "\n".join(fixed_nodes_lines) if fixed_nodes_lines else "  (none)"

    # Build edges section
    edges_str = "\n".join([f"  - {u} -> {v}" for u, v in edges]) or "  (none)"

    # Build non-edges sections
    non_edges_fixed_str = (
        "\n".join(
            [f"  - {u} <-> {v} (NO DIRECT LINK ALLOWED)" for u, v in non_edges_fixed]
        )
        or "  (none - fixed nodes are directly connected)"
    )
    all_non_edges_str = (
        "\n".join([f"  - {u} <-> {v}" for u, v in all_non_edges]) or "  (none)"
    )

    # Build needs_names section
    needs_names_str = ", ".join(needs_names) if needs_names else "(none)"

    prompt = f"""PRE-AUDIT CHECK

MOTIF TYPE: {motif}

FIXED NODE CONCEPTS (from CauseNet - CANNOT BE CHANGED):
{fixed_nodes_str}

PLACEHOLDER NODES (to be named by mapper):
  {needs_names_str}

STRUCTURAL CONSTRAINTS:

DIRECT EDGES (must be plausible causal relationships):
{edges_str}

NON-EDGES between fixed nodes (NO direct causal link allowed):
{non_edges_fixed_str}

ALL NON-EDGES (for context):
{all_non_edges_str}

Question: Can a skilled mapper typically find A reasonable interpretation of these fixed concepts that satisfies the constraints?
- Remember: the mapper can use narrow/specialized interpretations (e.g., "exercise" as "gym enrollment records")
- Remember: placeholder nodes give flexibility to construct appropriate mediators
- For NON-EDGES: Would a direct link be the DEFAULT mainstream interpretation?
  - If yes and hard to operationalize away → feasible: false
  - If yes but can be operationalized away in a mainstream way → feasible: true, confidence: low
  - If no → feasible: true
- For DIRECT EDGES: is there any plausible mechanism at all? If not physically impossible → feasible: true
- Do not rely on exotic edge cases.
- If uncertain, feasible: true with confidence: low (explain the uncertainty).. The full audit will catch actual problems.

Return ONLY the JSON object."""

    return prompt


# ----------------------
# Pre-Audit Runner
# ----------------------
def run_pre_audit_check(
    *,
    client: Any = None,
    api_key: str = None,
    session: Any = None,  # ChatSession - reuse for multi-attempt pre-audits
    model: str,
    reasoning: Union[bool, Dict[str, Any], None],
    sg,
    serialization_format: str = "simple_json",
    output_format: str = "json",
    enable_web: bool = False,
    max_tool_loops: int = 3,
    max_api_retries: int = 3,
) -> Tuple[Dict[str, Any], str, str, Any]:
    """Run pre-audit check on CauseNet mapping.

    Args:
        client: Optional LLMClient instance (preferred over api_key)
        api_key: API key for LLM provider (used if client not provided)
        session: Optional ChatSession to reuse (maintains conversation context across retries)
        model: Model identifier
        reasoning: Reasoning configuration (True, dict, or None)
        sg: SampledGraph object with CauseNet-fixed nodes
        serialization_format: Graph format (used to include format description in system prompt)

    Returns:
        Tuple of (pre_audit_result_dict, user_prompt, response_text, session)
        pre_audit_result_dict contains:
            - 'feasible': bool - whether CauseNet fixed nodes are workable
            - 'confidence': str - confidence level
            - 'reason': str - explanation
            - 'problematic_pairs': list - specific issues
            - 'suggested_domain': str | None - alternative domain suggestion
        session: The ChatSession used (for reuse in subsequent calls)
    """
    from . import llm_client as cl

    logger.info(
        "Starting pre-audit check (model=%s, fixed_nodes=%s)",
        model,
        (sg.meta or {}).get("fixed_nodes", []),
    )

    system_prompt = build_pre_audit_system_prompt(
        serialization_format,
        output_format,
        enable_web=enable_web,
    )

    # Use provided session, or create one from client
    if session is None:
        if client is None:
            if api_key is None:
                raise ValueError("Either client, session, or api_key must be provided")
            client = cl.LLMClient(api_key=api_key, default_model=model)
        session = client.create_session(model=model, system_prompt=system_prompt)

    user_prompt = build_pre_audit_user_prompt(sg)

    try:
        response_text, raw = retry_with_backoff(
            lambda: session.chat(
                user_prompt,
                temperature=0.0,
                tools=enable_web,
                max_tool_loops=max_tool_loops if enable_web else 0,
                reasoning=reasoning,
            ),
            max_retries=max_api_retries,
        )

        result = parse_llm_output(response_text, output_format, silent=True)

        # Ensure result is a dict (LLM may have returned unexpected structure)
        if not isinstance(result, dict):
            logger.warning(
                "Pre-audit parse returned non-dict (type=%s); treating result as infeasible",
                type(result).__name__,
            )
            result = {
                "feasible": False,
                "confidence": "low",
                "reason": f"Pre-audit parse failure: unexpected result type {type(result).__name__}",
                "problematic_pairs": [],
                "suggested_domain": None,
                "_parse_error": True,
                "_parse_result": result,
            }
        else:
            if "feasible" not in result:
                logger.warning(
                    "Pre-audit response missing required 'feasible' field; treating result as infeasible"
                )
                result["_parse_error"] = True
                result["feasible"] = False
                result.setdefault(
                    "reason",
                    "Pre-audit parse failure: missing required 'feasible' field",
                )

            result.setdefault("confidence", "low")
            result.setdefault("reason", "Incomplete pre-audit response")
            result.setdefault("problematic_pairs", [])
            result.setdefault("suggested_domain", None)

        logger.info(
            "Pre-audit complete: feasible=%s, confidence=%s, reason=%s",
            result.get("feasible"),
            result.get("confidence"),
            result.get("reason", "")[:150],
        )

    except Exception as e:
        logger.error("Pre-audit check failed: %s", e, exc_info=True)
        result = {
            "feasible": False,
            "confidence": "low",
            "reason": f"Pre-audit error: {str(e)}",
            "problematic_pairs": [],
            "suggested_domain": None,
            "_error": str(e),
            "_parse_error": True,
        }
        response_text = f"ERROR: {str(e)}"

    return result, user_prompt, response_text, session
