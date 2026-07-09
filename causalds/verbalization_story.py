# causalds/verbalization_story.py
"""
Story verbalization pipeline for causal graphs.

This module provides story generation from variable mappings:
- run_story_generation(): Run the story generation pipeline
- StoryResult: Dataclass containing all story outputs
- Prompt templates and builder functions

The pipeline takes a MappingResult (from run_variable_mapping) and generates
a natural language story with causal justifications.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from .graph import SampledGraph
from .llm_client import LLMClient
from .reporting import story_result_core_outputs_md
from .schemas import (
    CYAML_SYSTEM,
    VERBALIZATION_JSON_SCHEMA,
    build_user_prompt_passB_story,
)
from .utils import parse_json_from_llm, retry_with_backoff
from .var_mapping import MappingResult
from .verbalization_verify import build_story_audit_system_prompt, run_story_audit_check

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .var_mapping import MappingResult

from . import llm_client as cl
from . import serialization as cs
from .schemas import FORMAT_SYSTEM_PROMPTS, WEB_SEARCH_TOOL_INSTRUCTION

###############################################################################
# Prompt Templates (following existing schemas.py patterns)
###############################################################################


# System prompt for story generation - based on CYAML_SYSTEM + story instructions
STORY_SYSTEM_PROMPT = """
You are an expert in narrating a cohesive story based on a causal graph with a domain and domain-relevant variables.
The graph will be provided in the following structured format:
{format_system}

Your task is to write a SHORT, CONCRETE STORY (2-4 paragraphs) that:
- is situated in the provided PROPOSED DOMAIN
- uses ONLY the provided STORY NAMES from the VARIABLE MAPPING (NEVER use raw node IDs like V0, V1, etc.)
- mentions EVERY mapped variable explicitly in the story, including latent/background factors when present
- makes EVERY direct causal relationship in the graph explicit in the story itself
- does NOT introduce unsupported extra direct causal relationships
- avoids graph jargon (no "edges", "nodes", "colliders", "DAG", etc.)
- is scientifically plausible and engaging
- may be slightly denser than usual if needed to cover the full graph exactly
"""

# User prompt template - follows build_user_prompt_passB_story pattern
STORY_USER_PROMPT_TEMPLATE = """You will be given a PROPOSED domain, a VARIABLE MAPPING (ids -> story names) in JSON, and a CAUSAL GRAPH in the format prescribed by the system prompt.

Your task is to write a short, concrete STORY (2-4 paragraphs) using ONLY the provided story names from the VARIABLE MAPPING (no raw ids).
The story needs to reflect the causal relationships in the graph and mention every mapped variable explicitly within the narrative, BUT WITHOUT using any graph jargon revealing its structure.
Every direct edge in the graph must be stated or clearly implied in the STORY itself. Do not rely on the appendix/justifications to cover missing edges.
Do not introduce unsupported extra direct causes that are not in the graph.
Retain the units and variable types (e.g., categorical, continuous) given by the mapping.
Additionally return CAUSAL JUSTIFICATIONS explaining how the story reflects the causal structure of the graph.

Return the story, and causal justifications ONLY in a JSON object with this structure:
{{
  "story": "Your story here...",
  "causal_justifications": "Explain how the story reflects the causal structure of the graph, including specific causal relationships and observed variables HERE ...",
}}


The details of the inputs are as follows:

Proposed domain:
{proposed_domain}

Variable mapping (JSON):
```json
{variable_mapping_json}
```

Graph structure:
{graph_structure}
{extra_instruction}"""

# Feedback prompt for regeneration
STORY_FEEDBACK_PROMPT_TEMPLATE = """
Your previous story needs revision.

Current story:
```text
{previous_story}
```

Current causal justifications:
```text
{previous_justifications}
```

Issues to fix:
{feedback}

Rewrite the STORY from scratch if needed. Preserve the domain and use the exact provided story names.
Return ONLY the JSON object in the same format as before.
"""


###############################################################################
# Prompt Builder Functions
###############################################################################


def build_story_system_prompt(format_type: str = "parents_json") -> str:
    """Build system prompt for story generation.

    Args:
        format_type: The serialization format used for the graph

    Returns:
        System prompt string
    """

    format_system = FORMAT_SYSTEM_PROMPTS.get(format_type, "")
    return STORY_SYSTEM_PROMPT.format(format_system=format_system)


def build_story_user_prompt(
    variable_mapping: List[Dict[str, Any]],
    sg: SampledGraph,
    proposed_domain: str,
    serialization_format: str = "parents_json",
    extra_instruction: str = "",
) -> str:
    """Build user prompt for story generation.

    Args:
        variable_mapping: List of mapping rows [{id, story_name, observed, ...}, ...]
        sg: SampledGraph with the graph structure
        serialization_format: Format to use for serializing the graph
        extra_instruction: Additional instruction (e.g., web search note)

    Returns:
        User prompt string
    """

    # Serialize graph structure
    common_kwargs = {
        "include_concept_provenance_nl": False,
        "include_conditional_independencies": False,
        "include_non_edges": "None",
        "include_v_structures": False,
    }

    if serialization_format == "cyaml":
        graph_structure = cs.serialize_cyaml(sg, **common_kwargs)
    elif serialization_format == "parents_json":
        graph_structure = cs.serialize_parents_json(sg, **common_kwargs)
    elif serialization_format == "simple_json":
        graph_structure = cs.serialize_simple_json(sg, **common_kwargs)
    elif serialization_format == "edge_list":
        graph_structure = cs.serialize_edge_list(sg, **common_kwargs)
    else:
        graph_structure = cs.serialize_parents_json(sg, **common_kwargs)

    return STORY_USER_PROMPT_TEMPLATE.format(
        variable_mapping_json=json.dumps(variable_mapping, indent=2),
        graph_structure=graph_structure,
        proposed_domain=proposed_domain,
        extra_instruction=("\n" + extra_instruction) if extra_instruction else "",
    )


def build_story_feedback_prompt(
    feedback: str,
    previous_story: str = "",
    previous_justifications: Any = None,
) -> str:
    """Build feedback prompt for story regeneration."""
    if isinstance(previous_justifications, (dict, list)):
        justifications_text = json.dumps(
            previous_justifications,
            indent=2,
            ensure_ascii=False,
        )
    else:
        justifications_text = str(previous_justifications or "")

    return STORY_FEEDBACK_PROMPT_TEMPLATE.format(
        feedback=feedback,
        previous_story=previous_story or "",
        previous_justifications=justifications_text,
    )


def _run_single_story_generation(
    *,
    session: Any,
    prompt: str,
    temperature: Optional[float],
    reasoning: Optional[Union[bool, Dict[str, Any]]],
    enable_web: bool,
    max_tool_loops: int,
    max_parse_retries: int,
    max_api_retries: int,
) -> Dict[str, Any]:
    """Run one story-generation attempt and return the parsed payload plus trace info."""
    text = None
    raw = None
    parsed = None
    last_error = None

    for attempt in range(max_parse_retries):
        try:
            text, raw = retry_with_backoff(
                lambda: session.chat(
                    prompt,
                    temperature=temperature,
                    tools=enable_web,
                    max_tool_loops=max_tool_loops,
                    reasoning=reasoning,
                ),
                max_retries=max_api_retries,
            )
            parsed = parse_json_from_llm(text, silent=True)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Story generation returned {type(parsed).__name__}, expected dict"
                )
            break
        except ValueError as exc:
            last_error = str(exc)
            logger.warning(
                "Story parse attempt %d/%d failed: %s",
                attempt + 1,
                max_parse_retries,
                exc,
            )
            if attempt < max_parse_retries - 1:
                time.sleep(1.0)

    if parsed is None:
        raise ValueError(last_error or "Could not parse story JSON.")

    used_web = bool(
        isinstance(raw, dict) and (raw.get("used_web") or raw.get("used_tools"))
    )
    tool_trace = raw.get("tool_trace", []) if isinstance(raw, dict) else []

    return {
        "story": str(parsed.get("story", "") or ""),
        "causal_justifications": parsed.get("causal_justifications", {}),
        "text": text,
        "raw": raw,
        "used_web": used_web,
        "tool_trace": tool_trace,
        "session": session,
    }


def _build_story_feedback_from_audit(
    audit_result: Dict[str, Any],
    *,
    previous_story: str,
    previous_justifications: Any,
) -> str:
    """Build regeneration feedback from a story-audit result."""
    lines: List[str] = []

    hard_issues = list(audit_result.get("violations", []) or [])
    soft_issues = list(audit_result.get("warnings", []) or [])

    if hard_issues:
        lines.append("HARD ISSUES (must fix):")
        for issue in hard_issues:
            lines.append(
                f"- [{issue.get('kind', 'issue')}] {issue.get('explanation', '')}"
            )

    if soft_issues:
        lines.append("")
        lines.append("SOFT ISSUES (fix if possible without breaking the graph):")
        for issue in soft_issues:
            lines.append(
                f"- [{issue.get('kind', 'issue')}] {issue.get('explanation', '')}"
            )

    lines.append("")
    lines.append("Rewrite requirements:")
    lines.append("- Mention every mapped variable using its exact story name.")
    lines.append("- Make every direct edge explicit in the story itself.")
    lines.append("- Avoid unsupported extra direct effects.")
    lines.append(
        "- Improve scientific plausibility and domain coherence when possible without breaking the required graph."
    )
    lines.append("- Keep the story concrete and free of graph jargon.")

    return build_story_feedback_prompt(
        "\n".join(lines).strip(),
        previous_story=previous_story,
        previous_justifications=previous_justifications,
    )


def _make_story_attempt_trace(
    *,
    attempt: int,
    phase: str,
    prompt: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize one story-generation attempt for reporting."""
    payload = result or {}
    return {
        "attempt": int(attempt),
        "phase": str(phase),
        "prompt": prompt or "",
        "response_text": payload.get("text", "") or "",
        "story": payload.get("story", "") or "",
        "causal_justifications": payload.get("causal_justifications", {}),
        "used_web": bool(payload.get("used_web")),
        "tool_trace": list(payload.get("tool_trace", []) or []),
        "error": error,
    }


def _run_story_audit_loop(
    *,
    client: "LLMClient",
    sg: SampledGraph,
    variable_mapping: List[Dict[str, Any]],
    proposed_domain: str,
    initial_story_result: Dict[str, Any],
    writer_session: Any,
    model: Optional[str],
    reasoning: Optional[Union[bool, Dict[str, Any]]],
    temperature: Optional[float],
    enable_web: bool,
    audit_enable_web: bool,
    max_tool_loops: int,
    max_parse_retries: int,
    max_api_retries: int,
    max_audit_iterations: int,
    audit_output_format: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Iteratively verify and repair the story until it passes or hits the cap."""
    audit_info = {
        "enabled": True,
        "iterations": [],
        "final_pass": False,
    }
    current_result = dict(initial_story_result)
    generation_trace = [
        _make_story_attempt_trace(
            attempt=1,
            phase="initial",
            prompt=current_result.get("prompt", ""),
            result=current_result,
        )
    ]
    accumulated_tool_trace = list(current_result.get("tool_trace", []) or [])
    used_web = bool(current_result.get("used_web"))
    audit_session = None

    for iteration in range(max_audit_iterations):
        logger.info("Story audit iteration %d/%d", iteration + 1, max_audit_iterations)

        audit_result, audit_prompt, audit_text, audit_session = run_story_audit_check(
            client=client,
            session=audit_session,
            model=model,
            reasoning=reasoning,
            sg=sg,
            mapping_rows=variable_mapping,
            story=current_result.get("story", ""),
            proposed_domain=proposed_domain,
            output_format=audit_output_format,
            enable_web=audit_enable_web,
            max_tool_loops=max_tool_loops,
            max_api_retries=max_api_retries,
        )

        audit_info["iterations"].append(
            {
                "iteration": iteration + 1,
                "result": audit_result,
                "prompt": audit_prompt,
                "response": audit_text,
            }
        )
        if audit_result.get("_error"):
            logger.warning(
                "Resetting story-audit session after transport failure on iteration %d/%d",
                iteration + 1,
                max_audit_iterations,
            )
            audit_session = None

        if audit_result.get("pass", False):
            audit_info["final_pass"] = True
            break

        if iteration >= max_audit_iterations - 1:
            break

        feedback_prompt = _build_story_feedback_from_audit(
            audit_result,
            previous_story=current_result.get("story", ""),
            previous_justifications=current_result.get("causal_justifications", {}),
        )

        logger.info(
            "Regenerating story after audit failure (%d hard issue(s), %d warning(s))",
            len(audit_result.get("violations", []) or []),
            len(audit_result.get("warnings", []) or []),
        )

        try:
            regenerated = _run_single_story_generation(
                session=writer_session,
                prompt=feedback_prompt,
                temperature=temperature,
                reasoning=reasoning,
                enable_web=enable_web,
                max_tool_loops=max_tool_loops,
                max_parse_retries=max_parse_retries,
                max_api_retries=max_api_retries,
            )
        except ValueError as exc:
            generation_trace.append(
                _make_story_attempt_trace(
                    attempt=len(generation_trace) + 1,
                    phase="rewrite",
                    prompt=feedback_prompt,
                    error=str(exc),
                )
            )
            audit_info["regeneration_error"] = str(exc)
            logger.warning("Story regeneration after audit failed: %s", exc)
            break

        regenerated["prompt"] = feedback_prompt
        generation_trace.append(
            _make_story_attempt_trace(
                attempt=len(generation_trace) + 1,
                phase="rewrite",
                prompt=feedback_prompt,
                result=regenerated,
            )
        )
        writer_session = regenerated["session"]
        accumulated_tool_trace.extend(regenerated.get("tool_trace", []) or [])
        used_web = used_web or bool(regenerated.get("used_web"))
        current_result = regenerated

    current_result["tool_trace"] = accumulated_tool_trace
    current_result["used_web"] = used_web
    current_result["session"] = writer_session
    return current_result, audit_info, generation_trace


###############################################################################
# Result Dataclass
###############################################################################


@dataclass
class StoryResult:
    """Result from the story generation pipeline."""

    # Core outputs
    story: str
    causal_justifications: Union[str, Dict[str, Any], List[Any]]

    # Input context
    variable_mapping: List[Dict[str, Any]]
    sg: SampledGraph
    proposed_domain: str = ""

    # Status
    success: bool = True
    parse_error: Optional[str] = None

    # Trace info
    audit_info: Optional[Dict[str, Any]] = None
    generation_trace: List[Dict[str, Any]] = field(default_factory=list)
    prompts: Dict[str, str] = field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None
    response_text: Optional[str] = None
    used_web: bool = False
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    config: Optional[Union[DictConfig, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "story": self.story,
            "causal_justifications": self.causal_justifications,
            "variable_mapping": self.variable_mapping,
            "proposed_domain": self.proposed_domain,
            "success": self.success,
            "parse_error": self.parse_error,
            "prompts": self.prompts,
            "raw_response": self.raw_response,
            "response_text": self.response_text,
            "used_web": self.used_web,
            "tool_trace": self.tool_trace,
            "fallback_used": self.fallback_used,
            "audit_info": self.audit_info,
            "generation_trace": self.generation_trace,
            "config": (
                None
                if self.config is None
                else (
                    OmegaConf.to_container(self.config, resolve=True)
                    if OmegaConf.is_config(self.config)
                    else dict(self.config)
                )
            ),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], *, sg: SampledGraph) -> "StoryResult":
        """Reconstruct a story result from :meth:`to_dict` output."""
        payload = dict(payload or {})
        config_raw = payload.get("config")
        return cls(
            story=str(payload.get("story", "")),
            causal_justifications=payload.get("causal_justifications", {}),
            variable_mapping=list(payload.get("variable_mapping") or []),
            sg=sg,
            proposed_domain=str(payload.get("proposed_domain", "")),
            success=bool(payload.get("success", False)),
            parse_error=payload.get("parse_error"),
            audit_info=payload.get("audit_info"),
            generation_trace=list(payload.get("generation_trace") or []),
            prompts=dict(payload.get("prompts") or {}),
            raw_response=payload.get("raw_response"),
            response_text=payload.get("response_text"),
            used_web=bool(payload.get("used_web", False)),
            tool_trace=list(payload.get("tool_trace") or []),
            fallback_used=bool(payload.get("fallback_used", False)),
            config=(None if config_raw is None else OmegaConf.create(config_raw)),
        )

    def __str__(self) -> str:
        return story_result_core_outputs_md(self)


###############################################################################
# Main Entry Point
###############################################################################


def run_story_generation(
    mapping_result: MappingResult,
    *,
    client: Optional["LLMClient"] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    request_timeout_sec: Optional[float] = None,
    serialization_format: Optional[str] = None,
    enable_web: bool = True,
    temperature: Optional[float] = None,
    reasoning: Optional[Union[bool, Dict[str, Any]]] = True,
    enable_audit: bool = True,
    audit_enable_web: Optional[bool] = None,
    require_audit_pass: bool = True,
    max_audit_iterations: int = 3,
    audit_output_format: str = "json",
    max_tool_loops: int = 3,
    max_parse_retries: int = 3,
    max_api_retries: int = 3,
) -> StoryResult:
    """Run the story generation pipeline.

    Args:
        mapping_result: MappingResult from run_variable_mapping
        client: Optional LLMClient instance
        api_key: API key (defaults to env var)
        model: LLM model identifier
        request_timeout_sec: Per-request LLM HTTP timeout in seconds
        serialization_format: Graph serialization format (default from mapping config)
        enable_web: Enable web search tools
        temperature: LLM temperature
        reasoning: Reasoning configuration
        enable_audit: Whether to run the story verification loop
        audit_enable_web: Enable web search tools for the story auditor
        require_audit_pass: Mark the story-generation run as failed if the
            story audit does not pass after the allotted iterations
        max_audit_iterations: Max verifier-driven rewrite attempts
        audit_output_format: Auditor output format ("json" or "xml")
        max_tool_loops: Maximum tool call iterations
        max_parse_retries: Max JSON parse retry attempts
        max_api_retries: Max API call retry attempts

    Returns:
        StoryResult with story, justifications, and trace info
    """

    # Extract from MappingResult
    variable_mapping = mapping_result.mapping_rows
    sg = mapping_result.sg_renamed
    proposed_domain = mapping_result.proposed_domain

    # Infer format from mapping config if not specified
    if serialization_format is None:
        if mapping_result.config:
            serialization_format = mapping_result.config.get(
                "serialization_format", "parents_json"
            )
        else:
            serialization_format = "parents_json"

    # Resolve client
    if client is None:
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )
        if not api_key:
            raise ValueError("No API key provided")
        client = cl.LLMClient(
            api_key=api_key,
            default_model=model,
            request_timeout_sec=request_timeout_sec,
        )

    logger.info(
        "Starting story generation: %d variables, domain=%s",
        len(variable_mapping),
        proposed_domain[:50] if proposed_domain else "unknown",
    )

    # Build prompts
    system_prompt = build_story_system_prompt(format_type=serialization_format)

    extra_instruction = ""
    if enable_web:
        extra_instruction = WEB_SEARCH_TOOL_INSTRUCTION

    user_prompt = build_story_user_prompt(
        variable_mapping=variable_mapping,
        sg=sg,
        proposed_domain=proposed_domain,
        serialization_format=serialization_format,
        extra_instruction=extra_instruction,
    )

    session = client.create_session(model=model, system_prompt=system_prompt)
    audit_enable_web = (
        bool(enable_web) if audit_enable_web is None else bool(audit_enable_web)
    )
    auditor_system_prompt = (
        build_story_audit_system_prompt(
            audit_output_format,
            enable_web=audit_enable_web,
        )
        if enable_audit
        else ""
    )
    resolved_config = {
        "model": model or getattr(client, "default_model", None),
        "serialization_format": serialization_format,
        "enable_web": bool(enable_web),
        "temperature": temperature,
        "reasoning": reasoning,
        "enable_audit": bool(enable_audit),
        "audit_enable_web": bool(audit_enable_web),
        "require_audit_pass": bool(require_audit_pass),
        "max_audit_iterations": max(1, int(max_audit_iterations)),
        "audit_output_format": audit_output_format,
        "max_tool_loops": max_tool_loops,
        "max_parse_retries": max_parse_retries,
        "max_api_retries": max_api_retries,
    }

    try:
        story_payload = _run_single_story_generation(
            session=session,
            prompt=user_prompt,
            temperature=temperature,
            reasoning=reasoning,
            enable_web=enable_web,
            max_tool_loops=max_tool_loops,
            max_parse_retries=max_parse_retries,
            max_api_retries=max_api_retries,
        )
        story_payload["prompt"] = user_prompt
    except ValueError as exc:
        logger.error("All story parse attempts failed: %s", exc)
        return StoryResult(
            story="",
            causal_justifications={},
            variable_mapping=variable_mapping,
            sg=sg,
            proposed_domain=proposed_domain,
            success=False,
            parse_error=str(exc),
            audit_info=None,
            generation_trace=[
                _make_story_attempt_trace(
                    attempt=1,
                    phase="initial",
                    prompt=user_prompt,
                    error=str(exc),
                )
            ],
            prompts={
                "system": system_prompt,
                "user": user_prompt,
                "auditor_system": auditor_system_prompt,
            },
            raw_response=None,
            response_text=None,
            used_web=False,
            tool_trace=[],
            fallback_used=False,
            config=resolved_config,
        )

    audit_info = None
    generation_trace = [
        _make_story_attempt_trace(
            attempt=1,
            phase="initial",
            prompt=user_prompt,
            result=story_payload,
        )
    ]
    if enable_audit:
        story_payload, audit_info, generation_trace = _run_story_audit_loop(
            client=client,
            sg=sg,
            variable_mapping=variable_mapping,
            proposed_domain=proposed_domain,
            initial_story_result=story_payload,
            writer_session=story_payload["session"],
            model=model,
            reasoning=reasoning,
            temperature=temperature,
            enable_web=enable_web,
            audit_enable_web=audit_enable_web,
            max_tool_loops=max_tool_loops,
            max_parse_retries=max_parse_retries,
            max_api_retries=max_api_retries,
            max_audit_iterations=max(1, int(max_audit_iterations)),
            audit_output_format=audit_output_format,
        )

    logger.info(
        "Story generation complete: %d chars used_web=%s",
        len(story_payload.get("story", "")),
        story_payload.get("used_web"),
    )

    success = bool(story_payload.get("story"))
    parse_error: Optional[str] = None
    if enable_audit and require_audit_pass:
        final_pass = bool(audit_info and audit_info.get("final_pass", False))
        success = success and final_pass
        if not final_pass:
            parse_error = (
                audit_info.get("regeneration_error") if audit_info else None
            ) or "Story audit did not pass after max iterations."
            logger.warning("Story generation marked failed: %s", parse_error)

    return StoryResult(
        story=story_payload.get("story", ""),
        causal_justifications=story_payload.get("causal_justifications", {}),
        variable_mapping=variable_mapping,
        sg=sg,
        proposed_domain=proposed_domain,
        success=success,
        parse_error=parse_error,
        audit_info=audit_info,
        generation_trace=generation_trace,
        prompts={
            "system": system_prompt,
            "user": user_prompt,
            "auditor_system": auditor_system_prompt,
        },
        raw_response=story_payload.get("raw"),
        response_text=story_payload.get("text"),
        used_web=bool(story_payload.get("used_web")),
        tool_trace=story_payload.get("tool_trace", []),
        fallback_used=False,
        config=resolved_config,
    )


###############################################################################
# Deprecated: structured-outputs 2-step story generation
###############################################################################
# The following functions are ported from llm_client.py for backwards
# compatibility. They use OpenRouter structured outputs (response_format).
# For new code, use run_story_generation() instead.
###############################################################################


def verbalize_cyaml_one_shot(
    client: LLMClient,
    cyaml: str,
    model: Optional[str] = None,
    use_json_schema: bool = True,
    enable_web: bool = True,
    temperature: Optional[float] = None,
    max_tool_loops: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    DEPRECATED: Use run_variable_mapping() + run_story_generation() instead.

    One-shot structured verbalization (naming + story + justifications).
    Returns (parsed_json, raw_response_with_tool_trace).

    Requires OpenRouter structured outputs.
    """

    sess = client.create_session(model=model, system_prompt=CYAML_SYSTEM)
    return sess.chat_structured_verbalization(
        cyaml=cyaml,
        model=model,
        temperature=temperature,
        enable_web=enable_web,
        use_json_schema=use_json_schema,
        max_tool_loops=max_tool_loops,
    )


def verbalize_cyaml_multi_shot(
    client: LLMClient,
    cyaml: str,
    model: Optional[str] = None,
    use_json_schema: bool = True,
    enable_web: bool = True,
    temperature: Optional[float] = None,
    max_tool_loops: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    DEPRECATED: Use run_variable_mapping() + run_story_generation() instead.

    Multi-shot structured verbalization (Pass A: mapping, Pass B: story).
    Returns (combined_json, raw_response).

    Requires OpenRouter structured outputs.
    """

    sess = client.create_session(model=model, system_prompt=CYAML_SYSTEM)
    passA_json, passB_json, passB_raw = sess.chat_structured_multipass(
        cyaml=cyaml,
        model=model,
        temperature=temperature,
        enable_web=enable_web,
        use_json_schema=use_json_schema,
        max_tool_loops=max_tool_loops,
    )

    return (
        {
            "story": passB_json["story"],
            "variable_mapping": passA_json["variable_mapping"],
            "causal_justifications": passB_json["causal_justifications"],
        },
        passB_raw,
    )


def chat_story_given_mapping(
    session: Any,  # ChatSession
    cyaml: Optional[str] = None,
    mapping_json: Optional[Dict[str, Any]] = None,
    user_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    enable_web: bool = True,
    use_json_schema: bool = True,
    max_tool_loops: int = 3,
    submit_name: str = "submit",
    reasoning: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    DEPRECATED: Use run_story_generation() instead.

    Pass B: Generate story + justifications given a variable mapping.
    Returns (verbalization_json, raw_response_with_trace).

    Requires OpenRouter structured outputs.
    """

    if user_prompt is None:
        if cyaml is None or mapping_json is None:
            raise ValueError(
                "Either 'cyaml' and 'mapping_json', or 'user_prompt' must be provided"
            )
        mapping_str = json.dumps(mapping_json, indent=2)
        user_prompt = build_user_prompt_passB_story(cyaml, mapping_str)
    elif mapping_json is not None:
        mapping_str = json.dumps(mapping_json, indent=2)
        user_prompt = user_prompt.replace("{passA_mapping}", mapping_str)

    return session._complete_with_schema_and_tools(
        user_prompt=user_prompt,
        schema=VERBALIZATION_JSON_SCHEMA,
        model=model,
        temperature=temperature,
        enable_web=enable_web,
        use_json_schema=use_json_schema,
        max_tool_loops=max_tool_loops,
        submit_name=submit_name,
        reasoning=reasoning,
    )
