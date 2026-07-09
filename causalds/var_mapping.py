# causalds/var_mapping.py
"""
Variable mapping pipeline for causal graphs.

This module provides the main entry point for variable mapping:
- run_variable_mapping(): Run the full variable mapping pipeline
- MappingResult: Dataclass containing all mapping outputs
- DEFAULT_CONFIG: Default OmegaConf configuration
"""
import copy
import json
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from .graph import SampledGraph

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .llm_client import LLMClient

###############################################################################
# Default Configuration
###############################################################################

# NOTE: Default config is now nested (llm/serialization/causenet/etc.).
# We flatten relevant sections for the variable-mapping pipeline.

_VAR_MAPPING_SECTIONS = (
    "llm",
    "serialization",
    "var_mapping",
    "causenet",
    "pre_audit",
    "audit",
    "retry",
)


def _extract_var_mapping_config(config: Optional[DictConfig]) -> DictConfig:
    """Extract and merge variable-mapping-relevant sections from a config.

    Supports both the new nested config and the legacy flat config.
    """
    if config is None:
        return OmegaConf.create({})

    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)

    if any(section in config for section in _VAR_MAPPING_SECTIONS):
        merged = OmegaConf.create({})
        for section in _VAR_MAPPING_SECTIONS:
            if section in config and config[section] is not None:
                section_cfg = config[section]
                if section == "pre_audit" and "enable_web" in section_cfg:
                    section_cfg = OmegaConf.create(
                        {
                            **OmegaConf.to_container(section_cfg, resolve=True),
                            "pre_audit_enable_web": section_cfg.get("enable_web"),
                        }
                    )
                    del section_cfg["enable_web"]
                elif section == "audit" and "enable_web" in section_cfg:
                    section_cfg = OmegaConf.create(
                        {
                            **OmegaConf.to_container(section_cfg, resolve=True),
                            "audit_enable_web": section_cfg.get("enable_web"),
                        }
                    )
                    del section_cfg["enable_web"]
                merged = OmegaConf.merge(merged, section_cfg)

        # Allow legacy top-level keys to override section defaults
        extra = {k: v for k, v in config.items() if k not in _VAR_MAPPING_SECTIONS}
        if extra:
            merged = OmegaConf.merge(merged, OmegaConf.create(extra))
        return merged

    # Legacy flat config
    return config


# NOTE: Prefer the on-disk default config instead of this hardcoded one.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_config_path = os.path.join(
    _project_root, "exp", "configs", "generation_default.yaml"
)

if os.path.exists(_default_config_path):
    logger.info("Loading default config from %s", _default_config_path)
    DEFAULT_CONFIG = _extract_var_mapping_config(OmegaConf.load(_default_config_path))
else:
    logger.info("Default config file not found. Using hardcoded defaults.")
    DEFAULT_CONFIG = _extract_var_mapping_config(
        OmegaConf.create(
            {
                # LLM settings
                "model": "openai/gpt-oss-120b",
                # "temperature": 0.1,
                "max_tool_loops": 3,
                "reasoning": True,  # bool or dict
                # Serialization settings
                "serialization_format": "parents_json",  # cyaml, parents_json, simple_json, edge_list, text_simple, ci_only
                "json_mode": "prompt",  # "schema" or "prompt"
                "include_ci": True,
                "include_provenance": True,
                "enable_web": True,
                "web_conservative": False,  # Use conservative web search instruction
                # CauseNet settings
                "enable_causenet": True,
                "causenet_path": "data/causenet/causenet-precision.jsonl.bz2",  # Path to causenet-precision.jsonl.bz2
                "min_support": 2,
                "prefer_wikipedia": True,
                "domain_regex": None,
                "simple_pair_sampling": "uniform",
                "causenet_max_tries": 100,  # Max attempts to find valid CauseNet pair (within single matching call)
                # Pre-audit settings
                "enable_pre_audit": True,  # Run pre-audit by default for full pipeline
                "pre_audit_max_retries": 10,  # Max CauseNet+pre-audit retry attempts
                "pre_audit_skip_causenet": False,  # Skip CauseNet on pre-audit failure vs reject
                "pre_audit_enable_web": True,
                # Audit settings
                "enable_audit": True,
                "max_audit_iterations": 10,
                "audit_enable_web": True,
                # Grafting-augmentation mapping settings
                "enable_grafting_augmentation_mapping": True,
                "per_graph_max_mapping_attempts": 3,
                "per_graph_enable_local_audit": True,
                "final_augmented_graph_enable_audit": True,
                "final_augmented_graph_require_pass": True,
                # Retry settings
                "max_parse_retries": 3,
                "max_api_retries": 3,
                "api_retry_base_sleep": 2.0,
            }
        )
    )


###############################################################################
# Result Dataclass
###############################################################################


@dataclass
class MappingResult:
    """Result from the variable mapping pipeline.

    Contains all information needed for downstream report generation.
    """

    # Core outputs
    proposed_domain: str
    mapping_rows: List[Dict[str, Any]]  # [{id, story_name, observed, type, unit}, ...]
    applied_mapping: Dict[str, str]  # old_id -> new_name
    sg_renamed: SampledGraph  # Graph with renamed nodes
    sg_original: SampledGraph  # Original graph before renaming (for comparison)

    # Status
    success: bool
    parse_error: Optional[str] = None

    # CauseNet info
    causenet_applied: Dict[str, str] = field(default_factory=dict)  # old_id -> concept
    causenet_provenance: List[Dict[str, Any]] = field(
        default_factory=list
    )  # provenance records

    # Pre-audit info
    pre_audit_result: Optional[Dict[str, Any]] = (
        None  # {feasible, confidence, reason, ...}
    )

    # Audit info (if enabled) - full trace for report generation
    audit_info: Optional[Dict[str, Any]] = (
        None  # {enabled, iterations, final_pass, unfixable}
    )

    # Trace info for debugging/reports
    prompts: Dict[str, str] = field(default_factory=dict)  # {system, user}
    raw_response: Optional[Dict[str, Any]] = None  # LLM raw response
    response_text: Optional[str] = None  # LLM response text (for prompt mode)
    used_web: bool = False
    tool_trace: List[Dict[str, Any]] = field(
        default_factory=list
    )  # web search tool calls
    mapping_json: Dict[str, Any] = field(
        default_factory=dict
    )  # full parsed JSON from LLM

    # Config used (for reproducibility)
    config: Optional[DictConfig] = None

    def __str__(self) -> str:
        from .reporting import mapping_result_core_outputs_md

        return mapping_result_core_outputs_md(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable payload for checkpoints/manifests."""
        return {
            "proposed_domain": self.proposed_domain,
            "mapping_rows": copy.deepcopy(self.mapping_rows),
            "applied_mapping": copy.deepcopy(self.applied_mapping),
            "sg_renamed": self.sg_renamed.to_dict(),
            "sg_original": self.sg_original.to_dict(),
            "success": self.success,
            "parse_error": self.parse_error,
            "causenet_applied": copy.deepcopy(self.causenet_applied),
            "causenet_provenance": copy.deepcopy(self.causenet_provenance),
            "pre_audit_result": copy.deepcopy(self.pre_audit_result),
            "audit_info": copy.deepcopy(self.audit_info),
            "prompts": copy.deepcopy(self.prompts),
            "raw_response": copy.deepcopy(self.raw_response),
            "response_text": self.response_text,
            "used_web": self.used_web,
            "tool_trace": copy.deepcopy(self.tool_trace),
            "mapping_json": copy.deepcopy(self.mapping_json),
            "config": (
                None
                if self.config is None
                else OmegaConf.to_container(self.config, resolve=True)
            ),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MappingResult":
        """Reconstruct a mapping result from :meth:`to_dict` output."""
        payload = dict(payload or {})
        config_raw = payload.get("config")
        return cls(
            proposed_domain=str(payload.get("proposed_domain", "")),
            mapping_rows=list(payload.get("mapping_rows") or []),
            applied_mapping=dict(payload.get("applied_mapping") or {}),
            sg_renamed=SampledGraph.from_dict(dict(payload.get("sg_renamed") or {})),
            sg_original=SampledGraph.from_dict(dict(payload.get("sg_original") or {})),
            success=bool(payload.get("success", False)),
            parse_error=payload.get("parse_error"),
            causenet_applied=dict(payload.get("causenet_applied") or {}),
            causenet_provenance=list(payload.get("causenet_provenance") or []),
            pre_audit_result=payload.get("pre_audit_result"),
            audit_info=payload.get("audit_info"),
            prompts=dict(payload.get("prompts") or {}),
            raw_response=payload.get("raw_response"),
            response_text=payload.get("response_text"),
            used_web=bool(payload.get("used_web", False)),
            tool_trace=list(payload.get("tool_trace") or []),
            mapping_json=dict(payload.get("mapping_json") or {}),
            config=(None if config_raw is None else OmegaConf.create(config_raw)),
        )


###############################################################################
# Internal Helpers
###############################################################################


def _merge_config(
    config: Optional[DictConfig] = None,
    **overrides,
) -> DictConfig:
    """Merge configuration with defaults and overrides.

    Args:
        config: Optional base config to merge with defaults
        **overrides: Individual parameter overrides (None values are ignored)

    Returns:
        Merged OmegaConf DictConfig
    """
    # Start with defaults (flattened from nested config if needed)
    cfg = _extract_var_mapping_config(config)
    merged = OmegaConf.merge(DEFAULT_CONFIG, cfg or OmegaConf.create({}))

    # Apply non-None overrides
    override_dict = {k: v for k, v in overrides.items() if v is not None}
    if override_dict:
        merged = OmegaConf.merge(merged, OmegaConf.create(override_dict))

    return merged


def _extract_sampled_graph(
    input_obj: Any,
) -> SampledGraph:
    """Extract SampledGraph from various input types.

    Args:
        input_obj: SampledGraph, DataGenerator, or SCM object

    Returns:
        SampledGraph object

    Raises:
        TypeError: If input type is not supported
    """

    def _attach_node_types(
        sg_obj: SampledGraph, node_types: Optional[Dict[str, Any]]
    ) -> SampledGraph:
        """Attach node type hints on SampledGraph for downstream serialization/audit."""
        if not node_types:
            return sg_obj
        normalized = {
            str(k): str(v) for k, v in dict(node_types).items() if k is not None and v
        }
        if not normalized:
            return sg_obj
        setattr(sg_obj, "node_types", normalized)
        sg_obj.meta = sg_obj.meta or {}
        sg_obj.meta["node_types"] = normalized
        return sg_obj

    # Direct SampledGraph
    if isinstance(input_obj, SampledGraph):
        if hasattr(input_obj, "node_types"):
            return _attach_node_types(input_obj, getattr(input_obj, "node_types"))
        if getattr(input_obj, "meta", None) and "node_types" in input_obj.meta:
            return _attach_node_types(input_obj, input_obj.meta.get("node_types"))
        return input_obj

    # DataGenerator (has .sg attribute)
    if hasattr(input_obj, "sg") and isinstance(input_obj.sg, SampledGraph):
        sg_copy = copy.deepcopy(input_obj.sg)
        node_types = {}
        if hasattr(input_obj, "node_types"):
            node_types.update(getattr(input_obj, "node_types") or {})
        if hasattr(input_obj, "scm") and hasattr(input_obj.scm, "node_types"):
            node_types.update(getattr(input_obj.scm, "node_types") or {})
        return _attach_node_types(sg_copy, node_types)

    # SCM (has .G attribute but need to wrap in SampledGraph)
    if hasattr(input_obj, "G") and hasattr(input_obj, "order"):
        # This is an SCM object - wrap it
        sg = SampledGraph(
            graph=input_obj.G,
            treatment=getattr(input_obj, "treatment", None) or "X",
            outcome=getattr(input_obj, "outcome", None) or "Y",
            motif="from_scm",
            observed_nodes=(
                list(input_obj.observed_nodes)
                if hasattr(input_obj, "observed_nodes")
                else list(input_obj.G.nodes())
            ),
            latent_nodes=[],
            meta={},
        )
        return _attach_node_types(sg, getattr(input_obj, "node_types", None))

    raise TypeError(
        f"Unsupported input type: {type(input_obj)}. "
        "Expected SampledGraph, DataGenerator, or SCM."
    )


def _run_causenet_matching(
    sg: SampledGraph,
    config: DictConfig,
    causenet_index: Optional[Tuple] = None,
    seed: Optional[int] = None,
) -> Tuple[SampledGraph, Dict[str, str], List[Dict[str, Any]]]:
    """Apply CauseNet matching to the graph.

    Args:
        sg: Input SampledGraph
        config: Configuration with causenet settings
        causenet_index: Optional pre-built (children, parents, info) tuple

    Returns:
        Tuple of (modified_sg, applied_mapping, provenance_list)
    """
    from .causenet_extract import build_index
    from .causenet_match import simple_causenet_matching

    # Build or use provided index
    if causenet_index is not None:
        children, parents, info = causenet_index
    else:
        causenet_path = config.get("causenet_path")
        if not causenet_path:
            # Try default path relative to project root
            default_path = os.path.join(
                _project_root, "data/causenet/causenet-precision.jsonl.bz2"
            )
            if os.path.exists(default_path):
                causenet_path = default_path
            else:
                logger.warning(
                    "No CauseNet path provided and default not found at %s. Skipping CauseNet matching.",
                    default_path,
                )
                return sg, {}, []
        elif not os.path.isabs(causenet_path) and not os.path.exists(causenet_path):
            # If relative path doesn't exist, try relative to project root
            project_relative = os.path.join(_project_root, causenet_path)
            if os.path.exists(project_relative):
                causenet_path = project_relative

        logger.info("Building CauseNet index from %s", causenet_path)
        children, parents, info = build_index(
            causenet_path,
            min_support=config.get("min_support", 2),
            domain_regex=config.get("domain_regex"),
        )

    # Run simple matching (source/sink only)
    sg_matched, applied = simple_causenet_matching(
        sg,
        children,
        info,
        prefer_wikipedia=config.get("prefer_wikipedia", True),
        support_floor=config.get("min_support", 2),
        max_tries=config.get("causenet_max_tries", 100),
        sampling_strategy=config.get("simple_pair_sampling", "uniform"),
        seed=seed,
    )

    # Extract provenance
    provenance = []
    if sg_matched.meta and "concept_provenance" in sg_matched.meta:
        provenance = sg_matched.meta["concept_provenance"]

    logger.info(
        "CauseNet matching: %d nodes renamed, %d provenance records",
        len(applied),
        len(provenance),
    )

    return sg_matched, applied, provenance


def _run_pre_audit(
    sg: SampledGraph,
    config: DictConfig,
    client: Any,  # LLMClient
    session: Optional[Any] = None,  # ChatSession - reuse for multi-attempt pre-audits
) -> Tuple[bool, Optional[Dict[str, Any]], Any]:
    """Run pre-audit check on CauseNet mapping.

    Args:
        sg: SampledGraph with CauseNet-fixed nodes
        config: Configuration
        client: LLMClient instance
        session: Optional ChatSession to reuse (maintains conversation context across retries)

    Returns:
        Tuple of (should_proceed, pre_audit_result_dict, session)
    """
    from .pre_auditor import run_pre_audit_check

    result, prompt, text, session = run_pre_audit_check(
        client=client,
        session=session,
        model=config.get("model"),
        reasoning=config.get("reasoning"),
        sg=sg,
        serialization_format=config.get("serialization_format", "simple_json"),
        output_format=config.get("output_format", "json"),
        enable_web=bool(config.get("pre_audit_enable_web", True)),
        max_tool_loops=int(config.get("max_tool_loops", 3)),
        max_api_retries=int(config.get("max_api_retries", 3)),
    )

    feasible = result.get("feasible", False)
    confidence = result.get("confidence", "low")

    logger.info(
        "Pre-audit result: feasible=%s, confidence=%s, parse_error=%s",
        feasible,
        confidence,
        bool(result.get("_parse_error") or result.get("_error")),
    )

    return feasible, result, session


def _serialize_graph(
    sg: SampledGraph,
    config: DictConfig,
) -> str:
    """Serialize graph based on format configuration.

    Args:
        sg: SampledGraph to serialize
        config: Configuration with serialization settings

    Returns:
        Serialized graph string
    """
    from . import serialization as cs

    fmt = config.get("serialization_format", "cyaml")
    include_ci = config.get("include_ci", False)
    include_provenance = config.get("include_provenance", False)

    common_kwargs = {
        "include_concept_provenance_nl": include_provenance,
        "include_conditional_independencies": include_ci,
        "ci_mode": "minimal",
        "include_non_edges": "all",
        "include_v_structures": True,
    }

    if fmt == "ci_only":
        serialized = cs.serialize_conditional_independencies(
            sg,
            mode="all",
            format="markdown",
            include_concept_provenance_nl=include_provenance,
            include_non_edges="all",
            include_v_structures=True,
        )
        if isinstance(serialized, list):
            serialized = "\n".join(serialized)
    elif fmt == "cyaml":
        serialized = cs.serialize_cyaml(sg, **common_kwargs)
    elif fmt == "parents_json":
        serialized = cs.serialize_parents_json(sg, **common_kwargs)
    elif fmt == "parents_xml":
        serialized = cs.serialize_parents_xml(sg, **common_kwargs)
    elif fmt == "simple_json":
        serialized = cs.serialize_simple_json(sg, **common_kwargs)
    elif fmt == "edge_list":
        serialized = cs.serialize_edge_list(sg, **common_kwargs)
    elif fmt == "text_simple":
        serialized = cs.serialize_text_simple(sg, **common_kwargs)
    else:
        raise ValueError(f"Unknown serialization_format: {fmt}")

    return serialized


def _build_independence_section(
    sg: SampledGraph,
    include_ci: Union[bool, str],
) -> str:
    """Build the conditional independence section for user prompt.

    Args:
        sg: SampledGraph
        include_ci: Whether to include CI (bool or "only")

    Returns:
        Formatted independence section string
    """
    from . import serialization as cs

    if not include_ci:
        return ""

    ci_mode = "all" if include_ci == "only" else "minimal"
    lines = cs.serialize_conditional_independencies(sg, mode=ci_mode, format="markdown")
    if isinstance(lines, str):
        lines = [ln.strip() for ln in lines.splitlines() if ln.strip()]

    if lines:
        bullets = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            bullets.append(ln if ln.startswith("-") else f"- {ln}")
        prompt_lines = [f"  {b}" for b in bullets]
        return (
            "Additionally, your naming must obey the following statistical properties:\n"
            "Conditional Independence Relations:\n" + "\n".join(prompt_lines) + "\n"
        )
    else:
        return (
            "Additionally, your naming must obey the following statistical properties:\n"
            "Conditional Independence Relations:\n  None (All variables may be dependent).\n"
        )


def _prepare_mapping_prompts(
    sg: SampledGraph,
    config: DictConfig,
) -> Tuple[str, str]:
    """Build system and user prompts for variable mapping."""
    from .schemas import (
        WEB_SEARCH_TOOL_INSTRUCTION,
        WEB_SEARCH_TOOL_INSTRUCTION_CONSERVATIVE,
        build_grafting_auxiliary_mapping_requirements,
        build_grafting_final_mapping_requirements,
        build_var_mapping_system_prompt,
        build_var_mapping_user_prompt,
    )

    serialized = _serialize_graph(sg, config)
    fmt = config.get("serialization_format", "cyaml")
    json_mode = config.get("json_mode", "prompt")
    enable_web = config.get("enable_web", False)
    include_ci = config.get("include_ci", False)
    output_format = config.get("output_format", "json")

    include_independencies: Union[bool, str]
    if fmt == "ci_only":
        include_independencies = "only"
    else:
        include_independencies = include_ci

    meta = sg.meta or {}
    fixed_nodes = meta.get("fixed_nodes", [])
    needs_names = getattr(sg, "needs_names", None) or meta.get("needs_names", [])
    strict_fixed_nodes = bool(meta.get("strict_fixed_nodes", False))
    fixed_name_assignments = meta.get("fixed_name_assignments", {}) or {}
    forbidden_story_names = meta.get("forbidden_story_names", []) or []
    domain_hint = meta.get("domain_hint", "")
    stage_kind = str(meta.get("stage_kind", "") or "").strip().lower()
    if stage_kind == "auxiliary_graph":
        existing_graph_mapping_rows = meta.get("existing_graph_mapping_rows", []) or []
        shared_anchor_context = meta.get("shared_anchor_context", {}) or {}
    else:
        existing_graph_mapping_rows = []
        shared_anchor_context = {}
    additional_requirements = ""
    fixed_nodes_instruction_override = None
    if stage_kind == "auxiliary_graph":
        additional_requirements = build_grafting_auxiliary_mapping_requirements()
    elif stage_kind == "final_augmented_graph":
        additional_requirements = build_grafting_final_mapping_requirements()
        fixed_nodes_instruction_override = (
            "These are anchor variables carried over from earlier stages. "
            "Minor wording refinements are allowed only if the same variable meaning, "
            "scope, type, unit, and causal role are preserved."
        )

    system_prompt = build_var_mapping_system_prompt(
        format_type=fmt,
        include_independencies=bool(include_independencies),
        json_mode=json_mode,
        output_format=output_format,
    )

    independence_section = _build_independence_section(sg, include_independencies)

    extra_instruction = ""
    if enable_web:
        web_conservative = config.get("web_conservative", False)
        web_instruction = (
            WEB_SEARCH_TOOL_INSTRUCTION_CONSERVATIVE
            if web_conservative
            else WEB_SEARCH_TOOL_INSTRUCTION
        )
        extra_instruction = "\n" + web_instruction

    user_prompt = build_var_mapping_user_prompt(
        serialized_graph=serialized,
        format_type=fmt,
        fixed_nodes=fixed_nodes,
        needs_names=needs_names,
        independence_section=independence_section,
        enable_web=enable_web,
        extra_instruction=extra_instruction,
        output_format=output_format,
        strict_fixed_nodes=strict_fixed_nodes,
        fixed_name_assignments=fixed_name_assignments,
        forbidden_story_names=forbidden_story_names,
        domain_hint=domain_hint,
        additional_requirements=additional_requirements,
        existing_graph_mapping_rows=existing_graph_mapping_rows,
        shared_anchor_context=shared_anchor_context,
        fixed_nodes_instruction_override=fixed_nodes_instruction_override,
    )
    return system_prompt, user_prompt


def _apply_mapping_to_graph(
    sg: SampledGraph,
    mapping_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], SampledGraph]:
    """Apply mapping rows to a deep-copied graph and return applied map + renamed graph."""
    name_map: Dict[str, str] = {}
    for row in mapping_rows:
        vid = row.get("id")
        sname = row.get("story_name")
        if vid and sname:
            name_map[str(vid)] = str(sname)

    sg_renamed = copy.deepcopy(sg)
    applied_mapping, sg_renamed = sg_renamed.rename_nodes(name_map)
    return applied_mapping, sg_renamed


def _tool_trace_indicates_web_usage(tool_trace: Any) -> bool:
    """Return True when the trace contains successful web tool calls."""
    if not isinstance(tool_trace, list):
        return False

    for entry in tool_trace:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if name in {"web_search", "web_open"} and not entry.get("error"):
            return True
    return False


def _raw_indicates_web_usage(raw: Optional[Dict[str, Any]]) -> bool:
    """Return True when a raw model response indicates any web/tool usage."""
    if not isinstance(raw, dict):
        return False
    if raw.get("used_web") or raw.get("used_tools"):
        return True
    return _tool_trace_indicates_web_usage(raw.get("tool_trace", []))


def _extend_tool_trace(target: List[Dict[str, Any]], source: Any) -> None:
    """Append normalized trace entries from source into target."""
    if not isinstance(source, list):
        return
    for entry in source:
        if isinstance(entry, dict):
            target.append(copy.deepcopy(entry))


def _payload_indicates_web_usage(payload: Optional[Dict[str, Any]]) -> bool:
    """Return True when a mapping payload indicates any web usage."""
    if not isinstance(payload, dict):
        return False
    if payload.get("used_web"):
        return True
    if _tool_trace_indicates_web_usage(payload.get("tool_trace", [])):
        return True
    return _raw_indicates_web_usage(payload.get("raw"))


def _apply_usage_aggregate_to_payload(
    *,
    payload: Dict[str, Any],
    any_used_web: bool,
    tool_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply run-level usage aggregates to a mapping payload."""
    payload["used_web"] = bool(payload.get("used_web", False) or any_used_web)
    if tool_trace:
        payload["tool_trace"] = copy.deepcopy(tool_trace)

    raw = payload.get("raw")
    if isinstance(raw, dict):
        raw["used_web_any_attempt"] = bool(
            raw.get("used_web_any_attempt", False) or payload["used_web"]
        )

    return payload


def _build_mapping_payload(
    *,
    proposed_domain: str,
    used_web: bool,
    mapping_rows: List[Dict[str, Any]],
    applied_mapping: Dict[str, str],
    prompts: Dict[str, str],
    raw: Optional[Dict[str, Any]],
    text: Optional[str],
    mapping_json: Dict[str, Any],
    sg_renamed: SampledGraph,
    parse_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Create the standard payload used across mapping and audit stages."""
    return {
        "proposed_domain": proposed_domain,
        "used_web": used_web,
        "mapping_rows": mapping_rows,
        "applied_mapping": applied_mapping,
        "prompts": prompts,
        "raw": raw,
        "text": text,
        "mapping_json": mapping_json,
        "sg_renamed": sg_renamed,
        "parse_error": parse_error,
        "tool_trace": raw.get("tool_trace", []) if isinstance(raw, dict) else [],
    }


def _mapping_result_object(
    *,
    proposed_domain: str,
    mapping_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the canonical mapping object from resolved rows."""
    return {
        "proposed_domain": proposed_domain,
        "variable_mapping": copy.deepcopy(mapping_rows),
    }


def _mapping_result_text(
    *,
    proposed_domain: str,
    mapping_rows: List[Dict[str, Any]],
    json_mode: str,
    output_format: str,
) -> str:
    """Serialize a mapping object into the format expected by the active pipeline."""
    payload = _mapping_result_object(
        proposed_domain=proposed_domain,
        mapping_rows=mapping_rows,
    )

    if json_mode == "schema" or output_format != "xml":
        return json.dumps(payload, indent=2, ensure_ascii=False)

    root = ET.Element("mapping")
    ET.SubElement(root, "proposed_domain").text = str(proposed_domain or "")
    variable_mapping_el = ET.SubElement(root, "variable_mapping")
    for row in mapping_rows or []:
        if not isinstance(row, dict):
            continue
        attrs: Dict[str, str] = {}
        for key in ("id", "story_name", "observed", "type", "unit"):
            value = row.get(key)
            if isinstance(value, bool):
                attrs[key] = "true" if value else "false"
            elif value is None:
                attrs[key] = ""
            else:
                attrs[key] = str(value)
        ET.SubElement(variable_mapping_el, "variable", attrib=attrs)
    return ET.tostring(root, encoding="unicode")


def _payload_from_existing_mapping(
    *,
    sg: SampledGraph,
    config: DictConfig,
    proposed_domain: str,
    mapping_rows: List[Dict[str, Any]],
    used_web: bool,
    tool_trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a mapping payload for an already-proposed mapping on a given graph."""
    system_prompt, user_prompt = _prepare_mapping_prompts(sg, config)
    applied_mapping, sg_renamed = _apply_mapping_to_graph(sg, mapping_rows)
    mapping_json = _mapping_result_object(
        proposed_domain=proposed_domain,
        mapping_rows=mapping_rows,
    )
    text = _mapping_result_text(
        proposed_domain=proposed_domain,
        mapping_rows=mapping_rows,
        json_mode=config.get("json_mode", "prompt"),
        output_format=config.get("output_format", "json"),
    )
    tool_trace = tool_trace or []
    raw = {
        "seeded_from_existing_mapping": True,
        "tool_trace": tool_trace,
        "used_tools": bool(tool_trace),
        "used_web": bool(used_web),
    }
    return _build_mapping_payload(
        proposed_domain=proposed_domain,
        used_web=used_web,
        mapping_rows=mapping_rows,
        applied_mapping=applied_mapping,
        prompts={"system": system_prompt, "user": user_prompt},
        raw=raw,
        text=text,
        mapping_json=mapping_json,
        sg_renamed=sg_renamed,
        parse_error=None,
    )


def _seed_mapper_session_from_payload(
    *,
    client: Any,
    config: DictConfig,
    payload: Dict[str, Any],
) -> Any:
    """Create a mapper session primed with an existing mapping exchange."""
    prompts = payload.get("prompts", {}) or {}
    system_prompt = str(prompts.get("system", "") or "")
    user_prompt = str(prompts.get("user", "") or "")
    assistant_text = str(
        payload.get("text")
        or json.dumps(payload.get("mapping_json", {}), indent=2, ensure_ascii=False)
    )

    session = client.create_session(
        model=config.get("model"),
        system_prompt=system_prompt,
    )
    session.messages = []
    if system_prompt:
        session.messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        session.messages.append({"role": "user", "content": user_prompt})
    session.messages.append({"role": "assistant", "content": assistant_text})
    return session


def _rows_by_id(mapping_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in mapping_rows or []:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        if node_id:
            out[node_id] = row
    return out


def _validate_mapping_constraints(
    sg: SampledGraph,
    mapping_rows: List[Dict[str, Any]],
) -> List[str]:
    """Validate optional strict constraints carried in sg.meta."""
    meta = sg.meta or {}
    violations: List[str] = []
    row_by_id = _rows_by_id(mapping_rows)

    strict_fixed_nodes = bool(meta.get("strict_fixed_nodes", False))
    fixed_nodes = [
        str(x) for x in (meta.get("fixed_nodes", []) or []) if str(x).strip()
    ]
    fixed_name_assignments_raw = meta.get("fixed_name_assignments", {}) or {}
    fixed_name_assignments = {
        str(k): str(v).strip()
        for k, v in fixed_name_assignments_raw.items()
        if str(k).strip() and str(v).strip()
    }

    if strict_fixed_nodes or fixed_name_assignments:
        for fixed_id in fixed_nodes:
            row = row_by_id.get(fixed_id)
            if row is None:
                violations.append(
                    f"Missing required fixed node '{fixed_id}' in variable_mapping."
                )
                continue
            expected = fixed_name_assignments.get(fixed_id, fixed_id)
            got = str(row.get("story_name", "")).strip()
            if got != expected:
                violations.append(
                    f"Fixed node '{fixed_id}' must keep story_name '{expected}' exactly, got '{got}'."
                )

    require_complete_mapping = bool(meta.get("require_complete_mapping", False))
    if require_complete_mapping:
        missing_ids = [str(n) for n in sg.graph.nodes() if str(n) not in row_by_id]
        if missing_ids:
            violations.append(
                f"Mapping is incomplete. Missing node ids: {sorted(missing_ids)}."
            )

    enforce_unique_story_names = bool(meta.get("enforce_unique_story_names", False))
    if enforce_unique_story_names:
        seen_name_to_id: Dict[str, str] = {}
        for row in mapping_rows or []:
            node_id = str(row.get("id", "")).strip()
            story_name = str(row.get("story_name", "")).strip()
            if not node_id or not story_name:
                continue
            prev = seen_name_to_id.get(story_name)
            if prev is not None and prev != node_id:
                violations.append(
                    f"Duplicate story_name '{story_name}' used by ids '{prev}' and '{node_id}'."
                )
            else:
                seen_name_to_id[story_name] = node_id

    forbidden_story_names = {
        str(x).strip()
        for x in (meta.get("forbidden_story_names", []) or [])
        if str(x).strip()
    }
    if forbidden_story_names:
        for row in mapping_rows or []:
            node_id = str(row.get("id", "")).strip()
            story_name = str(row.get("story_name", "")).strip()
            if not node_id or not story_name:
                continue
            expected = fixed_name_assignments.get(node_id)
            if expected and story_name == expected:
                continue
            if story_name in forbidden_story_names:
                violations.append(
                    f"story_name '{story_name}' for id '{node_id}' collides with an existing graph variable."
                )

    return violations


def _mapping_feedback_from_error(
    error: str,
    output_format: str,
) -> str:
    from .schemas import build_var_mapping_feedback_prompt

    violation_block = f"- [mapping_constraint] {error}"
    return build_var_mapping_feedback_prompt(
        violation_block,
        output_format=output_format,
    )


def _build_mapping_stage_graph(
    sg_full: SampledGraph,
    stage_spec: Dict[str, Any],
) -> SampledGraph:
    """Build a SampledGraph view for one mapping stage using an induced subgraph."""
    stage_nodes = [str(n) for n in (stage_spec.get("node_ids", []) or [])]
    if not stage_nodes:
        raise ValueError(
            f"Mapping stage '{stage_spec.get('stage_id', 'unknown')}' has no node_ids."
        )

    missing = [n for n in stage_nodes if n not in sg_full.graph]
    if missing:
        raise ValueError(
            f"Mapping stage '{stage_spec.get('stage_id', 'unknown')}' references unknown nodes: {missing}"
        )

    G_stage = sg_full.graph.subgraph(stage_nodes).copy()
    observed_full = list(sg_full.observed_nodes or sg_full.graph.nodes())
    latent_full = set(sg_full.latent_nodes or [])
    observed_stage = [n for n in observed_full if n in G_stage]
    latent_stage = [n for n in stage_nodes if n in latent_full]

    treatment = (
        sg_full.treatment
        if sg_full.treatment in G_stage
        else (stage_nodes[0] if stage_nodes else sg_full.treatment)
    )
    outcome = (
        sg_full.outcome
        if sg_full.outcome in G_stage and sg_full.outcome != treatment
        else (
            stage_nodes[1]
            if len(stage_nodes) > 1 and stage_nodes[1] != treatment
            else treatment
        )
    )

    meta = copy.deepcopy(sg_full.meta) if isinstance(sg_full.meta, dict) else {}
    fixed_nodes = [str(x) for x in (stage_spec.get("fixed_nodes", []) or [])]
    needs_names = [str(x) for x in (stage_spec.get("needs_names", []) or [])]
    meta["fixed_nodes"] = fixed_nodes
    meta["needs_names"] = needs_names
    meta["stage_id"] = stage_spec.get("stage_id")
    meta["stage_kind"] = stage_spec.get("kind", "mapping_stage")

    return SampledGraph(
        graph=G_stage,
        treatment=str(treatment),
        outcome=str(outcome),
        motif=str(stage_spec.get("motif", sg_full.motif)),
        observed_nodes=observed_stage,
        latent_nodes=latent_stage,
        meta=meta,
    )


def _resolve_rows_to_original_ids(
    *,
    original_ids: List[str],
    mapping_rows: List[Dict[str, Any]],
    old_to_current_id: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Resolve mapping rows keyed by possibly-renamed IDs back to original IDs."""
    row_by_id = _rows_by_id(mapping_rows)
    resolved: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for original_id in original_ids:
        current_id = str(old_to_current_id.get(original_id, original_id))
        row = row_by_id.get(current_id) or row_by_id.get(str(original_id))
        if row is None:
            missing.append(str(original_id))
            continue
        resolved_row = copy.deepcopy(row)
        resolved_row["id"] = str(original_id)
        resolved[str(original_id)] = resolved_row
    return resolved, missing


def _canonicalize_stage_ids_from_meta(
    *,
    stage_node_ids: List[str],
    sg: SampledGraph,
    old_to_current_id: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Recover canonical original ids and a complete old->current map for a stage.

    This uses `sg.meta["original_ids"]` (new -> old) when available, which is populated
    by `SampledGraph.rename_nodes(...)`. That keeps stage bookkeeping stable even if the
    input graph was already renamed before entering the mapping pipeline.
    """
    reverse_original_ids_raw = (sg.meta or {}).get("original_ids", {}) or {}
    current_to_original = {
        str(current_id): str(original_id)
        for current_id, original_id in reverse_original_ids_raw.items()
        if str(current_id).strip() and str(original_id).strip()
    }

    canonical_original_ids: List[str] = []
    seen_original_ids: set[str] = set()
    for node_id in stage_node_ids:
        canonical_id = current_to_original.get(str(node_id), str(node_id))
        if canonical_id in seen_original_ids:
            continue
        seen_original_ids.add(canonical_id)
        canonical_original_ids.append(canonical_id)

    recovered_old_to_current = {
        original_id: current_id
        for current_id, original_id in current_to_original.items()
    }
    explicit_old_to_current = {
        str(old_id): str(current_id)
        for old_id, current_id in (old_to_current_id or {}).items()
        if str(old_id).strip() and str(current_id).strip()
    }
    effective_old_to_current = {
        **recovered_old_to_current,
        **explicit_old_to_current,
    }
    if current_to_original:
        logger.info(
            "Recovered canonical original ids for mapping stage via original_ids metadata: stage_nodes=%d, renamed_nodes=%d",
            len(stage_node_ids),
            len(current_to_original),
        )
    return canonical_original_ids, effective_old_to_current


def _run_mapping_with_retries(
    *,
    client: Any,
    sg: SampledGraph,
    config: DictConfig,
    max_attempts: int,
    run_audit: bool,
    require_audit_pass: bool,
    session: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Any, Optional[str]]:
    """Run mapping (and optional audit) with feedback retries on failures."""
    feedback_prompt: Optional[str] = None
    current_session = session
    last_result: Optional[Dict[str, Any]] = None
    last_audit: Optional[Dict[str, Any]] = None
    attempts = max(1, int(max_attempts))
    used_web_any_attempt = False
    tool_trace_any_attempt: List[Dict[str, Any]] = []

    def _capture_usage(payload: Dict[str, Any]) -> None:
        nonlocal used_web_any_attempt
        used_web_any_attempt = used_web_any_attempt or _payload_indicates_web_usage(
            payload
        )
        _extend_tool_trace(tool_trace_any_attempt, payload.get("tool_trace", []))

    for attempt in range(1, attempts + 1):
        result, current_session = _run_single_mapping(
            client=client,
            sg=sg,
            config=config,
            feedback_prompt=feedback_prompt,
            session=current_session,
        )
        last_result = result
        last_audit = None

        parse_error = result.get("parse_error")
        if parse_error:
            _capture_usage(result)
            if attempt < attempts:
                feedback_prompt = _mapping_feedback_from_error(
                    str(parse_error),
                    output_format=config.get("output_format", "json"),
                )
                continue
            result = _apply_usage_aggregate_to_payload(
                payload=result,
                any_used_web=used_web_any_attempt,
                tool_trace=tool_trace_any_attempt,
            )
            return result, None, current_session, str(parse_error)

        if run_audit and config.get("enable_audit", False):
            audited_result, audit_info = _run_audit_loop(
                client=client,
                sg=sg,
                initial_result=result,
                config=config,
                mapper_session=current_session,
            )
            result = audited_result
            last_result = audited_result
            last_audit = audit_info
            _capture_usage(result)
            if require_audit_pass and not audit_info.get("final_pass", False):
                err = "Audit failed to pass after max iterations."
                if attempt < attempts:
                    feedback_prompt = _mapping_feedback_from_error(
                        err,
                        output_format=config.get("output_format", "json"),
                    )
                    continue
                result = _apply_usage_aggregate_to_payload(
                    payload=result,
                    any_used_web=used_web_any_attempt,
                    tool_trace=tool_trace_any_attempt,
                )
                return result, audit_info, current_session, err

        if not (run_audit and config.get("enable_audit", False)):
            _capture_usage(result)

        result = _apply_usage_aggregate_to_payload(
            payload=result,
            any_used_web=used_web_any_attempt,
            tool_trace=tool_trace_any_attempt,
        )
        return result, last_audit, current_session, None

    if last_result is None:
        last_result = _build_mapping_payload(
            proposed_domain="",
            used_web=used_web_any_attempt,
            mapping_rows=[],
            applied_mapping={},
            prompts={},
            raw=None,
            text=None,
            mapping_json={},
            sg_renamed=sg,
            parse_error="Mapping attempts did not execute.",
        )
    last_result = _apply_usage_aggregate_to_payload(
        payload=last_result,
        any_used_web=used_web_any_attempt,
        tool_trace=tool_trace_any_attempt,
    )
    return (
        last_result,
        last_audit,
        current_session,
        str(last_result.get("parse_error") or "Mapping attempts exhausted."),
    )


def _run_single_mapping(
    *,
    client: Any,  # LLMClient
    sg: SampledGraph,
    config: DictConfig,
    feedback_prompt: Optional[str] = None,
    session: Optional[Any] = None,  # ChatSession - reuse for context continuity
) -> Tuple[Dict[str, Any], Any]:  # Returns (result_dict, session)
    """Run a single variable mapping attempt.

    Args:
        client: LLMClient instance
        sg: SampledGraph to map
        config: Configuration
        feedback_prompt: Optional feedback from failed audit (for regeneration)
        session: Optional existing ChatSession to reuse (maintains conversation context)

    Returns:
        Tuple of (result_dict, session) - session can be reused for follow-up calls
    """
    from .utils import normalize_mapping_rows, parse_llm_output, retry_with_backoff

    system_prompt, user_prompt = _prepare_mapping_prompts(sg, config)
    json_mode = config.get("json_mode", "prompt")
    enable_web = config.get("enable_web", False)
    temperature = config.get("temperature", 0.1)
    max_tool_loops = config.get("max_tool_loops", 3)
    reasoning = config.get("reasoning")
    max_parse_retries = config.get("max_parse_retries", 3)
    output_format = config.get("output_format", "json")
    used_web_any_attempt = False
    tool_trace_any_attempt: List[Dict[str, Any]] = []

    def _capture_raw_usage(raw_payload: Optional[Dict[str, Any]]) -> None:
        nonlocal used_web_any_attempt
        used_web_any_attempt = used_web_any_attempt or _raw_indicates_web_usage(
            raw_payload
        )
        if isinstance(raw_payload, dict):
            _extend_tool_trace(
                tool_trace_any_attempt, raw_payload.get("tool_trace", [])
            )

    # Session handling: reuse existing session for context continuity
    is_followup = session is not None and feedback_prompt is not None
    if session is None:
        session = client.create_session(
            model=config.get("model"), system_prompt=system_prompt
        )

    # For follow-up calls, just send the feedback (session has full context)
    # For initial calls, send the full user_prompt
    if is_followup:
        effective_prompt = feedback_prompt
    else:
        effective_prompt = user_prompt

    if json_mode == "schema":
        # Use structured outputs

        def _do():
            return session.chat_variable_mapping(
                user_prompt=effective_prompt,
                use_json_schema=True,
                temperature=temperature,
                enable_web=enable_web,
                max_tool_loops=max_tool_loops,
                reasoning=reasoning,
            )

        mapping_json, raw = retry_with_backoff(
            _do, max_retries=config.get("max_api_retries", 3)
        )
        _capture_raw_usage(raw)
        proposed_domain = mapping_json.get("proposed_domain", "")
        mapping = mapping_json.get("variable_mapping", []) or []
        text = None

    elif json_mode == "prompt":
        # Use unstructured prompting with retry logic

        def _do():
            return session.chat(
                effective_prompt,
                temperature=temperature,
                tools=enable_web,
                max_tool_loops=max_tool_loops,
                reasoning=reasoning,
            )

        text = None
        raw = None
        parsed = None
        last_error = None

        for attempt in range(max_parse_retries):
            try:
                text, raw = retry_with_backoff(
                    _do, max_retries=config.get("max_api_retries", 3)
                )
                _capture_raw_usage(raw)
                parsed = parse_llm_output(text, output_format, silent=True)
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"Parse returned {type(parsed).__name__}, expected dict"
                    )
                break
            except ValueError as e:
                last_error = str(e)
                logger.warning(
                    "Parse attempt %d/%d failed: %s", attempt + 1, max_parse_retries, e
                )
                if attempt < max_parse_retries - 1:
                    import time

                    time.sleep(1.0)
                else:
                    logger.error("All %d parse attempts failed.", max_parse_retries)
                    failed_payload = _build_mapping_payload(
                        proposed_domain="",
                        used_web=used_web_any_attempt,
                        mapping_rows=[],
                        applied_mapping={},
                        prompts={"system": system_prompt, "user": user_prompt},
                        raw=raw,
                        text=text,
                        mapping_json={},
                        sg_renamed=sg,
                        parse_error=last_error,
                    )
                    return (
                        _apply_usage_aggregate_to_payload(
                            payload=failed_payload,
                            any_used_web=used_web_any_attempt,
                            tool_trace=tool_trace_any_attempt,
                        ),
                        session,
                    )

        proposed_domain = parsed.get("proposed_domain", "")
        mapping = parsed.get("variable_mapping", []) or []
        mapping_json = parsed
    else:
        raise ValueError(f"Unknown json_mode: {json_mode}")

    # Normalize mapping
    normalized_rows = normalize_mapping_rows(mapping)
    constraint_violations = _validate_mapping_constraints(sg, normalized_rows)
    if constraint_violations:
        error_text = " ".join(constraint_violations)
        logger.warning("Mapping constraint violations: %s", error_text)
        invalid_payload = _build_mapping_payload(
            proposed_domain=proposed_domain,
            used_web=used_web_any_attempt,
            mapping_rows=normalized_rows,
            applied_mapping={},
            prompts={"system": system_prompt, "user": user_prompt},
            raw=raw,
            text=text,
            mapping_json=mapping_json if json_mode == "prompt" else {},
            sg_renamed=sg,
            parse_error=error_text,
        )
        return (
            _apply_usage_aggregate_to_payload(
                payload=invalid_payload,
                any_used_web=used_web_any_attempt,
                tool_trace=tool_trace_any_attempt,
            ),
            session,
        )

    applied_mapping, sg_renamed = _apply_mapping_to_graph(sg, normalized_rows)
    used_web = used_web_any_attempt

    success_payload = _build_mapping_payload(
        proposed_domain=proposed_domain,
        used_web=used_web,
        mapping_rows=normalized_rows,
        applied_mapping=applied_mapping,
        prompts={"system": system_prompt, "user": user_prompt},
        raw=raw,
        text=text,
        mapping_json=mapping_json if json_mode == "prompt" else {},
        sg_renamed=sg_renamed,
        parse_error=None,
    )
    return (
        _apply_usage_aggregate_to_payload(
            payload=success_payload,
            any_used_web=used_web_any_attempt,
            tool_trace=tool_trace_any_attempt,
        ),
        session,
    )


def _run_audit_loop(
    *,
    client: Any,  # LLMClient
    sg: SampledGraph,
    initial_result: Dict[str, Any],
    config: DictConfig,
    mapper_session: Optional[Any] = None,  # ChatSession from initial mapping
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run audit loop with regeneration on failures.

    Args:
        client: LLMClient instance
        sg: SampledGraph being mapped
        initial_result: Initial mapping result from _run_single_mapping
        config: Configuration
        mapper_session: Optional ChatSession from initial mapping (for context continuity)

    Returns:
        Tuple of (final_mapping_result, audit_info)
    """
    from .mapping_audit import is_unfixable_fixed_nodes_failure, run_audit_check
    from .schemas import build_var_mapping_feedback_prompt

    audit_info = {
        "enabled": True,
        "iterations": [],
        "mapping_attempts": [],
        "final_pass": False,
        "unfixable_fixed_nodes": False,
    }

    current_result = initial_result
    current_session = mapper_session  # Maintain mapper session across iterations
    audit_session = None  # Will be created on first audit call and reused
    max_iterations = config.get("max_audit_iterations", 3)
    used_web_any_attempt = False
    tool_trace_any_attempt: List[Dict[str, Any]] = []

    for iteration in range(max_iterations):
        logger.info("Audit iteration %d/%d", iteration + 1, max_iterations)
        used_web_any_attempt = used_web_any_attempt or _payload_indicates_web_usage(
            current_result
        )
        _extend_tool_trace(tool_trace_any_attempt, current_result.get("tool_trace", []))

        # Record the mapping candidate that is about to be audited.
        # This captures regeneration attempts for downstream trace rendering.
        audit_info["mapping_attempts"].append(
            {
                "iteration": iteration + 1,
                "mapping_result": {
                    "proposed_domain": str(
                        current_result.get("proposed_domain", "")
                    ).strip(),
                    "used_web": bool(current_result.get("used_web", False)),
                    "mapping_rows": copy.deepcopy(
                        current_result.get("mapping_rows", [])
                    ),
                    "parse_error": current_result.get("parse_error"),
                    "prompts": copy.deepcopy(current_result.get("prompts", {}) or {}),
                    "text": current_result.get("text", ""),
                    "mapping_json": copy.deepcopy(
                        current_result.get("mapping_json", {}) or {}
                    ),
                    "tool_trace": copy.deepcopy(
                        current_result.get("tool_trace", []) or []
                    ),
                },
            }
        )

        current_parse_error = current_result.get("parse_error")
        if current_parse_error:
            logger.warning(
                "Skipping audit on iteration %d due to invalid mapping candidate: %s",
                iteration + 1,
                current_parse_error,
            )
            audit_info["iterations"].append(
                {
                    "iteration": iteration + 1,
                    "stage": str((sg.meta or {}).get("audit_stage", "") or ""),
                    "system_prompt": "",
                    "result": {
                        "pass": False,
                        "violations": [
                            {
                                "kind": "mapping_constraint_violation",
                                "pair": None,
                                "story_pair": None,
                                "explanation": str(current_parse_error),
                            }
                        ],
                        "non_edge_attestations": [],
                        "summary": str(current_parse_error),
                        "_audit_stage": str(
                            (sg.meta or {}).get("audit_stage", "") or ""
                        ),
                        "_audit_system_prompt": "",
                    },
                    "prompt": "",
                    "response": str(current_parse_error),
                    "skipped_due_to_mapping_error": True,
                }
            )
            if iteration < max_iterations - 1:
                feedback = _mapping_feedback_from_error(
                    str(current_parse_error),
                    output_format=config.get("output_format", "json"),
                )
                current_result, current_session = _run_single_mapping(
                    client=client,
                    sg=sg,
                    config=config,
                    feedback_prompt=feedback,
                    session=current_session,
                )
                continue
            break

        # Run audit check (reuse audit_session across iterations)
        audit_result, audit_prompt, audit_text, audit_session = run_audit_check(
            client=client,
            session=audit_session,  # Reuse session for conversation continuity
            model=config.get("model"),
            reasoning=config.get("reasoning"),
            sg=sg,
            mapping_rows=current_result["mapping_rows"],
            serialization_format=config.get("serialization_format"),
            proposed_domain=current_result["proposed_domain"],
            output_format=config.get("output_format", "json"),
            enable_web=bool(config.get("audit_enable_web", True)),
            max_tool_loops=int(config.get("max_tool_loops", 3)),
            max_api_retries=int(config.get("max_api_retries", 3)),
        )

        audit_info["iterations"].append(
            {
                "iteration": iteration + 1,
                "stage": audit_result.get("_audit_stage", ""),
                "system_prompt": audit_result.get("_audit_system_prompt", ""),
                "result": audit_result,
                "prompt": audit_prompt,
                "response": audit_text,
            }
        )
        if audit_result.get("_error"):
            logger.warning(
                "Resetting audit session after transport failure on iteration %d/%d",
                iteration + 1,
                max_iterations,
            )
            audit_session = None

        if audit_result.get("pass", False):
            logger.info("Audit passed on iteration %d", iteration + 1)
            audit_info["final_pass"] = True
            break

        # Check for unfixable violations
        fixed_nodes = (sg.meta or {}).get("fixed_nodes", [])
        if is_unfixable_fixed_nodes_failure(
            audit_result.get("violations", []), fixed_nodes
        ):
            logger.warning("Unfixable fixed nodes failure detected")
            audit_info["unfixable_fixed_nodes"] = True
            break

        # If not last iteration, regenerate with feedback
        if iteration < max_iterations - 1:
            violations = audit_result.get("violations", [])
            violation_lines = []
            for v in violations:
                kind = v.get("kind", "unknown")
                pair = v.get("pair")
                story_pair = v.get("story_pair")
                explanation = v.get("explanation", "")
                line = f"- [{kind}]"
                if pair:
                    line += f" {pair}"
                if story_pair:
                    line += f" ({story_pair})"
                line += f": {explanation}"
                violation_lines.append(line)

            violation_block = "\n".join(violation_lines)
            feedback = build_var_mapping_feedback_prompt(
                violation_block,
                output_format=config.get("output_format", "json"),
            )

            logger.info(
                "Regenerating with feedback from %d violations",
                len(violations),
            )
            current_result, current_session = _run_single_mapping(
                client=client,
                sg=sg,
                config=config,
                feedback_prompt=feedback,
                session=current_session,
            )

    current_result = _apply_usage_aggregate_to_payload(
        payload=current_result,
        any_used_web=used_web_any_attempt,
        tool_trace=tool_trace_any_attempt,
    )
    return current_result, audit_info


def _resolve_mapping_client(
    client: Optional["LLMClient"],
    api_key: Optional[str],
    config: DictConfig,
) -> Tuple["LLMClient", Optional[str]]:
    """Resolve or construct an LLMClient for mapping/audit stages."""
    from . import llm_client as cl

    if client is not None and not isinstance(client, cl.LLMClient):
        raise TypeError("client must be an instance of LLMClient")

    resolved_api_key = api_key
    resolved_client = client

    if resolved_client is None:
        if resolved_api_key is None:
            resolved_api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )
        if not resolved_api_key:
            raise ValueError(
                "No API key provided. Set OPENROUTER_API_KEY or OPENAI_API_KEY env var, "
                "or pass api_key parameter."
            )
        resolved_client = cl.LLMClient(
            api_key=resolved_api_key,
            default_model=config.get("model"),
            request_timeout_sec=config.get("request_timeout_sec"),
        )
    else:
        if resolved_api_key is None:
            resolved_api_key = getattr(resolved_client, "api_key", None)
        if resolved_api_key is None and (
            config.get("enable_pre_audit") or config.get("enable_audit")
        ):
            raise ValueError(
                "No API key available for audit/pre-audit. Provide api_key or "
                "ensure client.api_key is set."
            )

    return resolved_client, resolved_api_key


def _run_causenet_pre_audit_stage(
    *,
    sg_original: SampledGraph,
    client: "LLMClient",
    config: DictConfig,
    causenet_index: Optional[Tuple],
    seed: Optional[int],
) -> Tuple[
    SampledGraph,
    Dict[str, str],
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
    Optional[str],
]:
    """Run CauseNet + pre-audit retries and return final graph state.

    Returns:
        (sg, causenet_applied, causenet_provenance, pre_audit_result, pre_audit_attempts, fatal_error)
    """
    sg = copy.deepcopy(sg_original)
    causenet_applied: Dict[str, str] = {}
    causenet_provenance: List[Dict[str, Any]] = []
    pre_audit_result: Optional[Dict[str, Any]] = None
    pre_audit_attempts: List[Dict[str, Any]] = []
    pre_audit_session = None

    enable_causenet = config.get("enable_causenet", True)
    enable_pre_audit = config.get("enable_pre_audit", False)
    max_retries = config.get("pre_audit_max_retries", 10) if enable_pre_audit else 1

    if not enable_causenet:
        return (
            sg,
            causenet_applied,
            causenet_provenance,
            pre_audit_result,
            pre_audit_attempts,
            None,
        )

    causenet_success = False
    for attempt in range(max_retries):
        attempt_seed = (seed + attempt) if seed is not None else None
        sg = copy.deepcopy(sg_original)
        sg, causenet_applied, causenet_provenance = _run_causenet_matching(
            sg, config, causenet_index, seed=attempt_seed
        )

        if not causenet_applied:
            logger.warning(
                "CauseNet matching attempt %d/%d: no match found",
                attempt + 1,
                max_retries,
            )
            continue

        if enable_pre_audit:
            feasible, pre_audit_result, pre_audit_session = _run_pre_audit(
                sg, config, client, session=pre_audit_session
            )
            pre_audit_attempts.append(
                {
                    "attempt": attempt + 1,
                    "seed": attempt_seed,
                    "causenet_applied": causenet_applied.copy(),
                    "feasible": feasible,
                    "result": pre_audit_result,
                }
            )
            if pre_audit_result and (
                pre_audit_result.get("_parse_error") or pre_audit_result.get("_error")
            ):
                logger.warning(
                    "Resetting pre-audit session after malformed/error response on attempt %d/%d",
                    attempt + 1,
                    max_retries,
                )
                pre_audit_session = None
            if feasible:
                logger.info(
                    "CauseNet+pre-audit succeeded on attempt %d/%d",
                    attempt + 1,
                    max_retries,
                )
                causenet_success = True
                break

            logger.warning(
                "Pre-audit failed on attempt %d/%d (reason: %s), retrying with different CauseNet match",
                attempt + 1,
                max_retries,
                pre_audit_result.get("reason", "Unknown")[:50],
            )
        else:
            causenet_success = True
            break

    if causenet_success:
        return (
            sg,
            causenet_applied,
            causenet_provenance,
            pre_audit_result,
            pre_audit_attempts,
            None,
        )

    if config.get("pre_audit_skip_causenet", False):
        logger.warning(
            "All %d CauseNet+pre-audit attempts failed, skipping CauseNet and using original graph",
            max_retries,
        )
        sg = copy.deepcopy(sg_original)
        sg.meta = sg.meta or {}
        sg.meta["fixed_nodes"] = []
        sg.meta["needs_names"] = list(sg.graph.nodes())
        sg.needs_names = list(sg.graph.nodes())
        return (
            sg,
            {},
            [],
            pre_audit_result,
            pre_audit_attempts,
            None,
        )

    error = f"Pre-audit failed after {max_retries} attempts: " + (
        pre_audit_result.get("reason", "Unknown")
        if pre_audit_result
        else "No valid CauseNet match"
    )
    return (
        sg,
        causenet_applied,
        causenet_provenance,
        pre_audit_result,
        pre_audit_attempts,
        error,
    )


def _get_mapping_sequence(sg: SampledGraph) -> List[Dict[str, Any]]:
    """Return ordered grafting-augmentation mapping stages from sg.meta."""
    meta = sg.meta or {}
    raw_plan = meta.get("mapping_sequence")
    if not isinstance(raw_plan, list):
        logger.debug("No mapping_sequence metadata found on graph.")
        return []

    plan: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_plan):
        if not isinstance(item, dict):
            continue
        node_ids = [str(n) for n in (item.get("node_ids", []) or []) if str(n).strip()]
        if not node_ids:
            continue
        record = copy.deepcopy(item)
        record["node_ids"] = node_ids
        record["order"] = int(record.get("order", idx))
        record["stage_id"] = str(record.get("stage_id", f"stage_{idx}"))
        plan.append(record)

    plan.sort(key=lambda x: x.get("order", 0))
    logger.info(
        "Resolved grafting augmentation mapping sequence with %d stage(s).", len(plan)
    )
    return plan


def _run_grafting_augmentation_mapping(
    *,
    sg_original: SampledGraph,
    client: "LLMClient",
    config: DictConfig,
    causenet_index: Optional[Tuple],
    seed: Optional[int],
) -> MappingResult:
    """Run main-graph-first, then graft-wise mapping with immutable anchor constraints."""
    plan = _get_mapping_sequence(sg_original)
    if not plan:
        logger.error(
            "Grafting augmentation mapping requested, but graph has no mapping_sequence metadata."
        )
        raise ValueError(
            "Grafting augmentation mapping requested, but no mapping_sequence was found."
        )

    logger.info(
        "Starting grafting augmentation mapping workflow: total_stages=%d, per_graph_attempts=%s, local_audit=%s, final_audit=%s",
        len(plan),
        config.get("per_graph_max_mapping_attempts", 3),
        config.get("per_graph_enable_local_audit", True),
        config.get("final_augmented_graph_enable_audit", True),
    )

    main_graph_spec = plan[0]
    main_graph_sg = _build_mapping_stage_graph(sg_original, main_graph_spec)
    main_graph_sg.meta = main_graph_sg.meta or {}
    main_graph_sg.meta["require_complete_mapping"] = True
    main_graph_sg.meta["enforce_unique_story_names"] = True
    main_graph_sg.meta["audit_stage"] = "main_graph"
    logger.info(
        "Running isolated main-graph mapping stage (%s) with %d node(s) before any auxiliary graft mapping.",
        main_graph_spec.get("stage_id", "main_graph"),
        main_graph_sg.graph.number_of_nodes(),
    )

    (
        main_graph_sg,
        causenet_applied,
        causenet_provenance,
        pre_audit_result,
        pre_audit_attempts,
        causenet_error,
    ) = _run_causenet_pre_audit_stage(
        sg_original=main_graph_sg,
        client=client,
        config=config,
        causenet_index=causenet_index,
        seed=seed,
    )

    if causenet_error is not None:
        logger.error("Main-graph CauseNet/pre-audit stage failed: %s", causenet_error)
        return MappingResult(
            proposed_domain="",
            mapping_rows=[],
            applied_mapping={},
            sg_renamed=sg_original,
            sg_original=sg_original,
            success=False,
            parse_error=causenet_error,
            causenet_applied=causenet_applied,
            causenet_provenance=causenet_provenance,
            pre_audit_result={
                "attempts": pre_audit_attempts,
                "final_result": pre_audit_result,
            },
            audit_info=None,
            config=config,
        )

    local_audit_enabled = bool(config.get("per_graph_enable_local_audit", True))
    mapping_attempts = int(config.get("per_graph_max_mapping_attempts", 3) or 3)
    per_graph_config = OmegaConf.merge(
        config,
        OmegaConf.create(
            {
                "enable_causenet": False,
                "enable_pre_audit": False,
            }
        ),
    )

    base_result, base_mapper_session = _run_single_mapping(
        client=client,
        sg=main_graph_sg,
        config=config,
    )
    base_audit_info = None
    base_error = base_result.get("parse_error")
    if base_error is None and config.get("enable_audit", False):
        base_result, base_audit_info = _run_audit_loop(
            client=client,
            sg=main_graph_sg,
            initial_result=base_result,
            config=config,
            mapper_session=base_mapper_session,
        )
        if not base_audit_info.get("final_pass", False):
            if base_audit_info.get("unfixable_fixed_nodes"):
                base_error = "Main-graph audit found unfixable fixed-node violations."
            else:
                base_error = "Main-graph audit did not pass after max iterations."

    if base_error:
        logger.warning("Main-graph mapping stage failed: %s", base_error)
        return MappingResult(
            proposed_domain="",
            mapping_rows=base_result.get("mapping_rows", []),
            applied_mapping=base_result.get("applied_mapping", {}),
            sg_renamed=base_result.get("sg_renamed", sg_original),
            sg_original=sg_original,
            success=False,
            parse_error=f"Main-graph mapping stage failed: {base_error}",
            causenet_applied=causenet_applied,
            causenet_provenance=causenet_provenance,
            pre_audit_result={
                "attempts": pre_audit_attempts,
                "final_result": pre_audit_result,
            },
            audit_info={"main_graph_audit": base_audit_info},
            prompts=base_result.get("prompts", {}),
            raw_response=base_result.get("raw"),
            response_text=base_result.get("text"),
            used_web=base_result.get("used_web", False),
            tool_trace=base_result.get("tool_trace", []),
            mapping_json=base_result.get("mapping_json", {}),
            config=config,
        )

    base_original_ids, effective_old_to_current = _canonicalize_stage_ids_from_meta(
        stage_node_ids=[str(n) for n in main_graph_spec.get("node_ids", [])],
        sg=main_graph_sg,
        old_to_current_id=causenet_applied,
    )
    resolved_base_rows, missing_base = _resolve_rows_to_original_ids(
        original_ids=base_original_ids,
        mapping_rows=base_result.get("mapping_rows", []),
        old_to_current_id=effective_old_to_current,
    )
    logger.info(
        "Main-graph mapping stage complete: mapped_nodes=%d, proposed_domain=%s",
        len(resolved_base_rows),
        str(base_result.get("proposed_domain", "")).strip() or "<empty>",
    )
    if missing_base:
        return MappingResult(
            proposed_domain="",
            mapping_rows=[],
            applied_mapping={},
            sg_renamed=sg_original,
            sg_original=sg_original,
            success=False,
            parse_error=(
                "Main-graph mapping stage missing original ids after CauseNet remap: "
                f"{missing_base}"
            ),
            causenet_applied=causenet_applied,
            causenet_provenance=causenet_provenance,
            pre_audit_result={
                "attempts": pre_audit_attempts,
                "final_result": pre_audit_result,
            },
            audit_info={"main_graph_audit": base_audit_info},
            config=config,
        )

    row_by_original_id: Dict[str, Dict[str, Any]] = resolved_base_rows
    story_by_original_id: Dict[str, str] = {
        node_id: str(row.get("story_name", "")).strip()
        for node_id, row in resolved_base_rows.items()
        if str(row.get("story_name", "")).strip()
    }
    if len(story_by_original_id) != len(resolved_base_rows):
        missing_story = sorted(
            [
                node_id
                for node_id, row in resolved_base_rows.items()
                if not str(row.get("story_name", "")).strip()
            ]
        )
        return MappingResult(
            proposed_domain="",
            mapping_rows=[],
            applied_mapping={},
            sg_renamed=sg_original,
            sg_original=sg_original,
            success=False,
            parse_error=f"Main-graph mapping stage returned empty story names: {missing_story}",
            causenet_applied=causenet_applied,
            causenet_provenance=causenet_provenance,
            pre_audit_result={
                "attempts": pre_audit_attempts,
                "final_result": pre_audit_result,
            },
            audit_info={"main_graph_audit": base_audit_info},
            config=config,
        )

    stage_traces: List[Dict[str, Any]] = [
        {
            "stage_id": main_graph_spec.get("stage_id", "main_graph"),
            "kind": "main_graph",
            "mapping_result": base_result,
            "audit_info": base_audit_info,
            "causenet_applied": causenet_applied,
            "causenet_provenance_count": len(causenet_provenance or []),
        }
    ]

    for spec in plan[1:]:
        stage_id = str(spec.get("stage_id", "auxiliary_graph"))
        anchor_node = str(spec.get("anchor_node", "")).strip()
        if not anchor_node:
            logger.error(
                "Auxiliary graph stage '%s' is missing anchor_node in metadata.",
                stage_id,
            )
            return MappingResult(
                proposed_domain="",
                mapping_rows=[],
                applied_mapping={},
                sg_renamed=sg_original,
                sg_original=sg_original,
                success=False,
                parse_error=f"Auxiliary graph stage '{stage_id}' is missing anchor_node.",
                causenet_applied=causenet_applied,
                causenet_provenance=causenet_provenance,
                pre_audit_result={
                    "attempts": pre_audit_attempts,
                    "final_result": pre_audit_result,
                },
                audit_info={"grafting_stage_traces": stage_traces},
                config=config,
            )
        if anchor_node not in story_by_original_id:
            logger.error(
                "Auxiliary graph stage '%s' anchor '%s' has no prior mapped story name.",
                stage_id,
                anchor_node,
            )
            return MappingResult(
                proposed_domain="",
                mapping_rows=[],
                applied_mapping={},
                sg_renamed=sg_original,
                sg_original=sg_original,
                success=False,
                parse_error=(
                    f"Auxiliary graph stage '{stage_id}' anchor '{anchor_node}' has no prior mapped story name."
                ),
                causenet_applied=causenet_applied,
                causenet_provenance=causenet_provenance,
                pre_audit_result={
                    "attempts": pre_audit_attempts,
                    "final_result": pre_audit_result,
                },
                audit_info={"grafting_stage_traces": stage_traces},
                config=config,
            )

        auxiliary_graph_sg = _build_mapping_stage_graph(sg_original, spec)
        auxiliary_graph_sg.meta = auxiliary_graph_sg.meta or {}
        anchor_story_name = story_by_original_id[anchor_node]
        existing_story_names = {
            str(v).strip() for v in story_by_original_id.values() if str(v).strip()
        }
        forbidden_names = sorted(existing_story_names - {anchor_story_name})

        auxiliary_graph_sg.meta["fixed_nodes"] = [anchor_node]
        auxiliary_graph_sg.meta["needs_names"] = [
            str(n)
            for n in (
                spec.get("needs_names")
                or spec.get("new_node_ids")
                or spec.get("node_ids")
                or []
            )
            if str(n) != anchor_node
        ]
        auxiliary_graph_sg.meta["strict_fixed_nodes"] = True
        auxiliary_graph_sg.meta["fixed_name_assignments"] = {
            anchor_node: anchor_story_name
        }
        auxiliary_graph_sg.meta["forbidden_story_names"] = forbidden_names
        auxiliary_graph_sg.meta["enforce_unique_story_names"] = True
        auxiliary_graph_sg.meta["require_complete_mapping"] = True
        auxiliary_graph_sg.meta["domain_hint"] = str(
            base_result.get("proposed_domain", "")
        ).strip()
        auxiliary_graph_sg.meta["audit_stage"] = "auxiliary_graph_local"
        auxiliary_graph_sg.meta["existing_graph_mapping_rows"] = [
            {
                "id": str(node_id),
                "story_name": str(row.get("story_name", "")).strip(),
                "observed": row.get("observed"),
                "type": row.get("type"),
                "unit": row.get("unit"),
            }
            for node_id, row in row_by_original_id.items()
            if isinstance(row, dict) and str(row.get("story_name", "")).strip()
        ]
        auxiliary_graph_sg.meta["shared_anchor_context"] = {
            "anchor_node": anchor_node,
            "anchor_story_name": anchor_story_name,
        }
        logger.info(
            "Mapping auxiliary graph stage '%s': anchor=%s, new_nodes=%d.",
            stage_id,
            anchor_node,
            len(auxiliary_graph_sg.meta.get("needs_names", [])),
        )

        frag_result, frag_audit_info, _, frag_error = _run_mapping_with_retries(
            client=client,
            sg=auxiliary_graph_sg,
            config=per_graph_config,
            max_attempts=mapping_attempts,
            run_audit=local_audit_enabled,
            require_audit_pass=local_audit_enabled,
            session=None,
        )
        stage_traces.append(
            {
                "stage_id": stage_id,
                "kind": "auxiliary_graph",
                "mapping_result": frag_result,
                "audit_info": frag_audit_info,
                "anchor_node": anchor_node,
                "anchor_story_name": anchor_story_name,
            }
        )
        if frag_error:
            logger.warning(
                "Auxiliary graph stage '%s' mapping failed: %s", stage_id, frag_error
            )
            return MappingResult(
                proposed_domain="",
                mapping_rows=[],
                applied_mapping={},
                sg_renamed=sg_original,
                sg_original=sg_original,
                success=False,
                parse_error=f"Auxiliary graph stage '{stage_id}' mapping failed: {frag_error}",
                causenet_applied=causenet_applied,
                causenet_provenance=causenet_provenance,
                pre_audit_result={
                    "attempts": pre_audit_attempts,
                    "final_result": pre_audit_result,
                },
                audit_info={"grafting_stage_traces": stage_traces},
                prompts=frag_result.get("prompts", {}),
                raw_response=frag_result.get("raw"),
                response_text=frag_result.get("text"),
                used_web=frag_result.get("used_web", False),
                tool_trace=frag_result.get("tool_trace", []),
                mapping_json=frag_result.get("mapping_json", {}),
                config=config,
            )
        logger.info("Auxiliary graph stage '%s' mapping complete.", stage_id)

        row_by_id = _rows_by_id(frag_result.get("mapping_rows", []))
        for node_id in [str(n) for n in spec.get("node_ids", [])]:
            row = row_by_id.get(node_id)
            if row is None:
                return MappingResult(
                    proposed_domain="",
                    mapping_rows=[],
                    applied_mapping={},
                    sg_renamed=sg_original,
                    sg_original=sg_original,
                    success=False,
                    parse_error=(
                        f"Auxiliary graph stage '{stage_id}' did not return mapping row for node '{node_id}'."
                    ),
                    causenet_applied=causenet_applied,
                    causenet_provenance=causenet_provenance,
                    pre_audit_result={
                        "attempts": pre_audit_attempts,
                        "final_result": pre_audit_result,
                    },
                    audit_info={"grafting_stage_traces": stage_traces},
                    config=config,
                )
            story_name = str(row.get("story_name", "")).strip()
            if not story_name:
                return MappingResult(
                    proposed_domain="",
                    mapping_rows=[],
                    applied_mapping={},
                    sg_renamed=sg_original,
                    sg_original=sg_original,
                    success=False,
                    parse_error=(
                        f"Auxiliary graph stage '{stage_id}' returned empty story_name for node '{node_id}'."
                    ),
                    causenet_applied=causenet_applied,
                    causenet_provenance=causenet_provenance,
                    pre_audit_result={
                        "attempts": pre_audit_attempts,
                        "final_result": pre_audit_result,
                    },
                    audit_info={"grafting_stage_traces": stage_traces},
                    config=config,
                )

            if node_id == anchor_node and story_name != anchor_story_name:
                return MappingResult(
                    proposed_domain="",
                    mapping_rows=[],
                    applied_mapping={},
                    sg_renamed=sg_original,
                    sg_original=sg_original,
                    success=False,
                    parse_error=(
                        f"Anchor drift in auxiliary graph stage '{stage_id}': node '{anchor_node}' "
                        f"must remain '{anchor_story_name}', got '{story_name}'."
                    ),
                    causenet_applied=causenet_applied,
                    causenet_provenance=causenet_provenance,
                    pre_audit_result={
                        "attempts": pre_audit_attempts,
                        "final_result": pre_audit_result,
                    },
                    audit_info={"grafting_stage_traces": stage_traces},
                    config=config,
                )

            if node_id != anchor_node:
                existing_story = story_by_original_id.get(node_id)
                if existing_story and existing_story != story_name:
                    return MappingResult(
                        proposed_domain="",
                        mapping_rows=[],
                        applied_mapping={},
                        sg_renamed=sg_original,
                        sg_original=sg_original,
                        success=False,
                        parse_error=(
                            f"Node '{node_id}' was assigned conflicting story names "
                            f"('{existing_story}' vs '{story_name}')."
                        ),
                        causenet_applied=causenet_applied,
                        causenet_provenance=causenet_provenance,
                        pre_audit_result={
                            "attempts": pre_audit_attempts,
                            "final_result": pre_audit_result,
                        },
                        audit_info={"grafting_stage_traces": stage_traces},
                        config=config,
                    )
                if story_name in existing_story_names:
                    return MappingResult(
                        proposed_domain="",
                        mapping_rows=[],
                        applied_mapping={},
                        sg_renamed=sg_original,
                        sg_original=sg_original,
                        success=False,
                        parse_error=(
                            f"Auxiliary graph stage '{stage_id}' proposed duplicate story_name "
                            f"'{story_name}' for node '{node_id}'."
                        ),
                        causenet_applied=causenet_applied,
                        causenet_provenance=causenet_provenance,
                        pre_audit_result={
                            "attempts": pre_audit_attempts,
                            "final_result": pre_audit_result,
                        },
                        audit_info={"grafting_stage_traces": stage_traces},
                        config=config,
                    )
                copied = copy.deepcopy(row)
                copied["id"] = node_id
                row_by_original_id[node_id] = copied
                story_by_original_id[node_id] = story_name
                existing_story_names.add(story_name)

    full_rows: List[Dict[str, Any]] = []
    observed_set = set(sg_original.observed_nodes or list(sg_original.graph.nodes()))
    for node_id in [str(n) for n in sg_original.graph.nodes()]:
        row = row_by_original_id.get(node_id)
        if row is None:
            return MappingResult(
                proposed_domain="",
                mapping_rows=[],
                applied_mapping={},
                sg_renamed=sg_original,
                sg_original=sg_original,
                success=False,
                parse_error=f"Final merged mapping is missing node '{node_id}'.",
                causenet_applied=causenet_applied,
                causenet_provenance=causenet_provenance,
                pre_audit_result={
                    "attempts": pre_audit_attempts,
                    "final_result": pre_audit_result,
                },
                audit_info={"grafting_stage_traces": stage_traces},
                config=config,
            )
        merged_row = {
            "id": node_id,
            "story_name": str(row.get("story_name", "")).strip(),
            "observed": node_id in observed_set,
            "type": row.get("type"),
            "unit": row.get("unit"),
        }
        full_rows.append(merged_row)

    # Final collision guard before applying rename.
    seen_story: Dict[str, str] = {}
    for row in full_rows:
        name = str(row.get("story_name", "")).strip()
        node_id = str(row.get("id", "")).strip()
        if not name:
            return MappingResult(
                proposed_domain="",
                mapping_rows=[],
                applied_mapping={},
                sg_renamed=sg_original,
                sg_original=sg_original,
                success=False,
                parse_error=f"Merged mapping produced empty story_name for id '{node_id}'.",
                causenet_applied=causenet_applied,
                causenet_provenance=causenet_provenance,
                pre_audit_result={
                    "attempts": pre_audit_attempts,
                    "final_result": pre_audit_result,
                },
                audit_info={"grafting_stage_traces": stage_traces},
                config=config,
            )
        prev = seen_story.get(name)
        if prev is not None and prev != node_id:
            return MappingResult(
                proposed_domain="",
                mapping_rows=[],
                applied_mapping={},
                sg_renamed=sg_original,
                sg_original=sg_original,
                success=False,
                parse_error=(
                    f"Merged mapping has duplicate story_name '{name}' for ids '{prev}' and '{node_id}'."
                ),
                causenet_applied=causenet_applied,
                causenet_provenance=causenet_provenance,
                pre_audit_result={
                    "attempts": pre_audit_attempts,
                    "final_result": pre_audit_result,
                },
                audit_info={"grafting_stage_traces": stage_traces},
                config=config,
            )
        seen_story[name] = node_id

    final_payload = _payload_from_existing_mapping(
        sg=sg_original,
        config=config,
        proposed_domain=str(base_result.get("proposed_domain", "")).strip(),
        used_web=bool(
            base_result.get("used_web")
            or any(
                bool((trace.get("mapping_result") or {}).get("used_web"))
                for trace in stage_traces
            )
        ),
        mapping_rows=full_rows,
    )
    logger.info(
        "Merged main-graph and auxiliary-graph mappings into full graph: nodes=%d, mapped_rows=%d",
        sg_original.graph.number_of_nodes(),
        len(full_rows),
    )

    global_audit_info = None
    global_audit_enabled = bool(config.get("final_augmented_graph_enable_audit", True))
    require_global_audit_pass = bool(
        config.get("final_augmented_graph_require_pass", True)
    )
    if config.get("enable_audit", False) and global_audit_enabled:
        logger.info(
            "Running final full-graph audit on merged main graph plus auxiliary grafts."
        )
        global_sg = copy.deepcopy(sg_original)
        global_sg.meta = global_sg.meta or {}
        global_sg.meta["audit_stage"] = "final_augmented_graph"
        global_sg.meta["stage_kind"] = "final_augmented_graph"
        global_sg.meta["existing_graph_mapping_rows"] = [
            {
                "id": str(row.get("id", "")).strip(),
                "story_name": str(row.get("story_name", "")).strip(),
                "observed": row.get("observed"),
                "type": row.get("type"),
                "unit": row.get("unit"),
            }
            for row in full_rows
            if isinstance(row, dict) and str(row.get("story_name", "")).strip()
        ]
        global_sg.meta["domain_hint"] = str(
            base_result.get("proposed_domain", "")
        ).strip()
        fixed_ids_for_global = sorted(
            {
                str((spec.get("anchor_node") or "")).strip()
                for spec in plan[1:]
                if str((spec.get("anchor_node") or "")).strip()
            }
        )
        fixed_assignments = {
            node_id: story_by_original_id[node_id]
            for node_id in fixed_ids_for_global
            if node_id in story_by_original_id
        }
        if fixed_ids_for_global:
            global_sg.meta["fixed_nodes"] = fixed_ids_for_global
            global_sg.meta["soft_fixed_name_assignments"] = fixed_assignments
        global_sg.meta["enforce_unique_story_names"] = True

        final_payload = _payload_from_existing_mapping(
            sg=global_sg,
            config=config,
            proposed_domain=str(base_result.get("proposed_domain", "")).strip(),
            used_web=bool(final_payload.get("used_web", False)),
            mapping_rows=full_rows,
        )
        final_mapper_session = _seed_mapper_session_from_payload(
            client=client,
            config=config,
            payload=final_payload,
        )

        audited_payload, global_audit_info = _run_audit_loop(
            client=client,
            sg=global_sg,
            initial_result=final_payload,
            config=config,
            mapper_session=final_mapper_session,
        )
        final_payload = audited_payload

    success = final_payload.get("parse_error") is None
    parse_error = final_payload.get("parse_error")
    if global_audit_info and require_global_audit_pass:
        if global_audit_info.get("unfixable_fixed_nodes"):
            success = False
            parse_error = (
                parse_error or "Global audit found unfixable fixed-node violations."
            )
            logger.error("%s", parse_error)
        elif not global_audit_info.get("final_pass", False):
            success = False
            parse_error = (
                parse_error or "Global audit did not pass after max iterations."
            )
            logger.warning("%s", parse_error)

    combined_audit_info = {
        "grafting_augmentation_mode": True,
        "main_graph_audit": base_audit_info,
        "grafting_stage_traces": stage_traces,
        "global_audit": global_audit_info,
    }
    logger.info(
        "Anchor-graft mapping workflow finished: success=%s, mapped_stages=%d",
        success,
        len(plan),
    )

    return MappingResult(
        proposed_domain=final_payload.get("proposed_domain", ""),
        mapping_rows=final_payload.get("mapping_rows", []),
        applied_mapping=final_payload.get("applied_mapping", {}),
        sg_renamed=final_payload.get("sg_renamed", sg_original),
        sg_original=sg_original,
        success=success,
        parse_error=parse_error,
        causenet_applied=causenet_applied,
        causenet_provenance=causenet_provenance,
        pre_audit_result={
            "attempts": pre_audit_attempts,
            "final_result": pre_audit_result,
            "grafting_augmentation_mode": True,
        },
        audit_info=combined_audit_info,
        prompts=final_payload.get("prompts", {}),
        raw_response=final_payload.get("raw"),
        response_text=final_payload.get("text"),
        used_web=final_payload.get("used_web", False),
        tool_trace=final_payload.get("tool_trace", []),
        mapping_json=final_payload.get("mapping_json", {}),
        config=config,
    )


###############################################################################
# Main Entry Point
###############################################################################


def run_variable_mapping(
    input_obj: Any,
    *,
    client: Optional["LLMClient"] = None,
    api_key: Optional[str] = None,
    config: Optional[DictConfig] = None,
    # Override individual config params
    model: Optional[str] = None,
    serialization_format: Optional[str] = None,
    json_mode: Optional[str] = None,
    enable_web: Optional[bool] = None,
    web_conservative: Optional[bool] = None,
    include_ci: Optional[bool] = None,
    include_provenance: Optional[bool] = None,
    temperature: Optional[float] = None,
    reasoning: Optional[Union[bool, Dict[str, Any]]] = None,
    # CauseNet
    enable_causenet: Optional[bool] = None,
    causenet_index: Optional[Tuple] = None,
    causenet_max_tries: Optional[int] = None,
    # Pre-audit
    enable_pre_audit: Optional[bool] = None,
    pre_audit_max_retries: Optional[int] = None,
    pre_audit_skip_causenet: Optional[bool] = None,
    # Audit
    enable_audit: Optional[bool] = None,
    max_audit_iterations: Optional[int] = None,
    # Grafting-augmentation mapping
    enable_grafting_augmentation_mapping: Optional[bool] = None,
    per_graph_max_mapping_attempts: Optional[int] = None,
    per_graph_enable_local_audit: Optional[bool] = None,
    final_augmented_graph_enable_audit: Optional[bool] = None,
    final_augmented_graph_require_pass: Optional[bool] = None,
    # Output format
    output_format: Optional[str] = None,
    # Misc
    seed: Optional[int] = None,
) -> MappingResult:
    """Run the full variable mapping pipeline.

    This function runs the complete pipeline:
    1. CauseNet matching (optional, default=True): Maps source/sink nodes to real-world concepts
    2. Pre-audit check (optional, default=True): Validates CauseNet concepts satisfy graph constraints
    3. LLM variable mapping: Generates names for remaining nodes
    4. Audit loop (optional, default=False): Validates and regenerates mapping on failures

    To run just the LLM mapping without CauseNet/pre-audit, set enable_causenet=False
    and enable_pre_audit=False.

    Args:
        input_obj: SampledGraph, DataGenerator, or SCM object
        client: Optional LLMClient instance (reuses existing client for all LLM calls)
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        config: OmegaConf DictConfig (merged with DEFAULT_CONFIG)
        model: LLM model identifier
        serialization_format: Graph serialization format
        json_mode: "schema" or "prompt"
        enable_web: Enable web search tools
        web_conservative: Use conservative web search instruction (discourages excessive searches)
        include_ci: Include conditional independencies
        include_provenance: Include CauseNet provenance in serialization
        temperature: LLM temperature
        reasoning: Reasoning configuration (bool or dict)
        enable_causenet: Enable CauseNet matching (default=True)
        causenet_index: Pre-built (children, parents, info) tuple
        causenet_max_tries: Max attempts to find valid CauseNet pair (default=100)
        enable_pre_audit: Enable pre-audit check (default=True for full pipeline)
        pre_audit_max_retries: Max CauseNet+pre-audit retry attempts (default=10)
        pre_audit_skip_causenet: Skip CauseNet on pre-audit failure instead of rejecting
        enable_audit: Enable audit loop (default=False)
        max_audit_iterations: Maximum audit regeneration attempts
        enable_grafting_augmentation_mapping: Enable stage-wise mapping when graph metadata carries a grafting sequence
        per_graph_max_mapping_attempts: Max mapping+repair attempts for the main graph and each auxiliary graph
        per_graph_enable_local_audit: Run a local audit loop on the main graph and each auxiliary graph mapping
        final_augmented_graph_enable_audit: Run a final full-graph audit after merging the auxiliary graphs
        final_augmented_graph_require_pass: Mark run failed if the final augmented-graph audit does not pass
        seed: Random seed for CauseNet matching (for reproducibility)

    Returns:
        MappingResult with mapping, renamed graph, and full trace info
    """
    # Merge configuration
    merged_config = _merge_config(
        config,
        model=model,
        serialization_format=serialization_format,
        json_mode=json_mode,
        enable_web=enable_web,
        web_conservative=web_conservative,
        include_ci=include_ci,
        include_provenance=include_provenance,
        temperature=temperature,
        reasoning=reasoning,
        enable_causenet=enable_causenet,
        causenet_max_tries=causenet_max_tries,
        enable_pre_audit=enable_pre_audit,
        pre_audit_max_retries=pre_audit_max_retries,
        pre_audit_skip_causenet=pre_audit_skip_causenet,
        enable_audit=enable_audit,
        max_audit_iterations=max_audit_iterations,
        enable_grafting_augmentation_mapping=enable_grafting_augmentation_mapping,
        per_graph_max_mapping_attempts=per_graph_max_mapping_attempts,
        per_graph_enable_local_audit=per_graph_enable_local_audit,
        final_augmented_graph_enable_audit=final_augmented_graph_enable_audit,
        final_augmented_graph_require_pass=final_augmented_graph_require_pass,
        output_format=output_format,
    )

    client, _ = _resolve_mapping_client(client, api_key, merged_config)

    # Extract SampledGraph from input
    sg_original = _extract_sampled_graph(input_obj)
    sg = copy.deepcopy(sg_original)

    logger.info(
        "Starting variable mapping pipeline: %d nodes, motif=%s",
        sg.graph.number_of_nodes(),
        sg.motif,
    )

    mapping_sequence = _get_mapping_sequence(sg_original)
    if (
        merged_config.get("enable_grafting_augmentation_mapping", True)
        and mapping_sequence
    ):
        logger.info(
            "Using grafting augmentation variable mapping mode (%d stage(s))",
            len(mapping_sequence),
        )
        return _run_grafting_augmentation_mapping(
            sg_original=sg_original,
            client=client,
            config=merged_config,
            causenet_index=causenet_index,
            seed=seed,
        )

    (
        sg,
        causenet_applied,
        causenet_provenance,
        pre_audit_result,
        pre_audit_attempts,
        causenet_error,
    ) = _run_causenet_pre_audit_stage(
        sg_original=sg_original,
        client=client,
        config=merged_config,
        causenet_index=causenet_index,
        seed=seed,
    )

    if causenet_error is not None:
        logger.error("%s", causenet_error)
        return MappingResult(
            proposed_domain="",
            mapping_rows=[],
            applied_mapping={},
            sg_renamed=sg,
            sg_original=sg_original,
            success=False,
            parse_error=causenet_error,
            causenet_applied=causenet_applied,
            causenet_provenance=causenet_provenance,
            pre_audit_result={
                "attempts": pre_audit_attempts,
                "final_result": pre_audit_result,
            },
            audit_info=None,
            config=merged_config,
        )

    # Step 3: Run LLM mapping
    mapping_result, mapper_session = _run_single_mapping(
        client=client,
        sg=sg,
        config=merged_config,
    )

    # Step 4: Audit loop (optional)
    audit_info = None
    if merged_config.get("enable_audit", False):
        mapping_result, audit_info = _run_audit_loop(
            client=client,
            sg=sg,
            initial_result=mapping_result,
            config=merged_config,
            mapper_session=mapper_session,  # Pass session for context continuity
        )

    # Build final result
    parse_error = mapping_result.get("parse_error")
    success = parse_error is None
    if audit_info:
        # Consider audit result in success determination
        if audit_info.get("unfixable_fixed_nodes"):
            success = False
            parse_error = (
                parse_error or "Audit failed due to unfixable fixed-node violations."
            )
        elif merged_config.get("enable_audit") and not audit_info.get("final_pass"):
            success = False
            parse_error = parse_error or "Audit did not pass after max iterations."
            logger.warning("%s", parse_error)

    return MappingResult(
        proposed_domain=mapping_result.get("proposed_domain", ""),
        mapping_rows=mapping_result.get("mapping_rows", []),
        applied_mapping=mapping_result.get("applied_mapping", {}),
        sg_renamed=mapping_result.get("sg_renamed", sg),
        sg_original=sg_original,
        success=success,
        parse_error=parse_error,
        causenet_applied=causenet_applied,
        causenet_provenance=causenet_provenance,
        pre_audit_result=pre_audit_result,
        audit_info=audit_info,
        prompts=mapping_result.get("prompts", {}),
        raw_response=mapping_result.get("raw"),
        response_text=mapping_result.get("text"),
        used_web=mapping_result.get("used_web", False),
        tool_trace=mapping_result.get("tool_trace", []),
        mapping_json=mapping_result.get("mapping_json", {}),
        config=merged_config,
    )


__all__ = [
    "run_variable_mapping",
    "MappingResult",
    "DEFAULT_CONFIG",
]
