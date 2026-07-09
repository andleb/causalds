# causalds/reporting.py
"""
Report generation utilities for benchmark generation results.

This module provides functions to generate markdown reports and conversation traces
from MappingResult objects.
"""
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import networkx as nx

if TYPE_CHECKING:
    from .var_mapping import MappingResult
    from .verbalization_story import StoryResult

logger = logging.getLogger(__name__)


###############################################################################
# Utility Functions
###############################################################################


def safe_slug(s: str) -> str:
    """Create a safe filename slug from a string."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _coerce_markdown_text(text: Any) -> str:
    """Normalize structured values into displayable markdown text."""
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, (dict, list)):
        return json.dumps(text, indent=2, ensure_ascii=False)
    return str(text)


def escape_code_fences(text: Any) -> str:
    """
    Escape nested code fences in text to prevent markdown formatting issues.
    Replace ``` with ~~~ to allow the text to be wrapped in outer ``` blocks.
    """
    return _coerce_markdown_text(text).replace("```", "~~~")


def _write_tool_trace_markdown(f, tool_trace: Optional[List[Dict[str, Any]]]) -> None:
    """Render a tool-trace section into an open markdown file handle."""
    if not tool_trace:
        return

    f.write("### Tool Calls\n\n")
    for tool_call in tool_trace:
        tool_name = tool_call.get("name", "unknown")

        if tool_name == "web_search":
            args = tool_call.get("args", {})
            query = args.get("query", "")
            k = args.get("k", args.get("max_results", 5))

            f.write(f"**web_search**: `{query}`\n")
            f.write(f"  - k: {k}\n")

            results = tool_call.get("results", [])
            if results:
                f.write(f"  - Returned {len(results)} result(s)\n")

            error_msg = tool_call.get("error")
            if error_msg:
                f.write(f"  - **Error:** {error_msg}\n")
            f.write("\n")

        elif tool_name == "web_open":
            args = tool_call.get("args", {})
            url = tool_call.get("url") or args.get("url", "")
            content_len = tool_call.get("raw_content_len", 0)

            f.write(f"**web_open**: `{url}`\n")
            f.write(f"  - Content length: {content_len} chars\n")

            error_msg = tool_call.get("error")
            if error_msg:
                f.write(f"  - **Error:** {error_msg}\n")
            f.write("\n")

        else:
            f.write(f"**{tool_name}**\n")
            error_msg = tool_call.get("error")
            if error_msg:
                f.write(f"  - **Error:** {error_msg}\n")
            f.write("\n")


def _write_wrapped_markdown_text(f, text: Any) -> None:
    """Write plain markdown text with explicit line breaks for readability."""
    s = _coerce_markdown_text(text).strip()
    if not s:
        f.write("*(empty)*\n\n")
        return
    for line in s.splitlines():
        f.write(line.rstrip() + "  \n")
    f.write("\n")


def _write_story_response_markdown(f, response_text: Any) -> bool:
    """Render story/justifications as markdown sections when response is JSON."""
    raw = _coerce_markdown_text(response_text).strip()
    if not raw or not raw.startswith("{"):
        return False
    try:
        payload = json.loads(raw)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if "story" not in payload and "causal_justifications" not in payload:
        return False

    if "story" in payload:
        f.write("#### Story\n\n")
        _write_wrapped_markdown_text(f, payload.get("story", ""))

    f.write("#### Causal Justifications\n\n")
    justifications = payload.get("causal_justifications", "")
    if isinstance(justifications, (dict, list)):
        pretty = json.dumps(justifications, indent=2, ensure_ascii=False)
        _write_wrapped_markdown_text(f, pretty)
    else:
        _write_wrapped_markdown_text(f, justifications)

    f.write("#### Raw Structured Output\n\n")
    f.write("```json\n")
    f.write(escape_code_fences(raw))
    f.write("\n```\n\n")
    return True


def variable_mapping_table_md(mapping: List[Dict[str, Any]]) -> str:
    """Create markdown table from mapping rows."""
    headers = ["id", "story_name", "observed", "type", "unit"]
    rows = []
    for v in mapping or []:
        row = []
        for h in headers:
            val = v.get(h, "")
            if isinstance(val, bool):
                val = "true" if val else "false"
            if val is None:
                val = ""
            row.append(str(val).replace("|", "\\|"))
        rows.append("| " + " | ".join(row) + " |")
    hdr_line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    return "\n".join([hdr_line, sep] + rows) + "\n"


def applied_mapping_table_md(applied_mapping: Dict[str, str]) -> str:
    """Create markdown table from applied mapping."""
    headers = ["id", "renamed"]
    rows = []
    for old_id, new_name in (applied_mapping or {}).items():
        old_id_str = str(old_id).replace("|", "\\|")
        new_name_str = str(new_name).replace("|", "\\|")
        rows.append("| " + " | ".join([old_id_str, new_name_str]) + " |")
    hdr_line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    return "\n".join([hdr_line, sep] + rows) + "\n"


def mapping_result_core_outputs_md(
    result: "MappingResult",
    include_applied_mapping: bool = False,
) -> str:
    """Format core outputs from a MappingResult as markdown."""
    lines = ["# Core outputs\n\n"]

    proposed_domain = result.proposed_domain or "*(not provided)*"
    proposed_domain = proposed_domain.replace("|", "\\|")

    lines.append("**Results**\n\n")
    lines.append(f"- Proposed domain: {proposed_domain}\n")
    lines.append(f"- Used web search: {str(bool(result.used_web)).lower()}\n\n")

    if result.parse_error:
        lines.append(f"**ERROR**: {result.parse_error}\n\n")

    lines.append("### Variable Mapping\n\n")
    lines.append(variable_mapping_table_md(result.mapping_rows))
    lines.append("\n")

    if include_applied_mapping and result.applied_mapping:
        lines.append("### Applied Mapping\n\n")
        lines.append(applied_mapping_table_md(result.applied_mapping))
        lines.append("\n")

    return "".join(lines).rstrip() + "\n"


def _markdown_cell(value: Any) -> str:
    """Render one markdown table cell safely."""
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def _sorted_count_keys(keys: List[Any]) -> List[str]:
    """Sort count-table keys numerically when possible, else lexicographically."""
    seen = {str(key) for key in keys if str(key) != ""}

    def _sort_key(raw: str):
        try:
            return (0, int(raw))
        except Exception:
            return (1, raw)

    return sorted(seen, key=_sort_key)


def composition_count_table_md(
    *,
    requested: Optional[Dict[str, int]] = None,
    realized_manifest: Optional[Dict[str, int]] = None,
    completed: Optional[Dict[str, int]] = None,
) -> str:
    """Render one composition-axis count comparison as a markdown table."""
    requested = dict(requested or {})
    realized_manifest = dict(realized_manifest or {})
    completed = dict(completed or {})
    values = _sorted_count_keys(
        list(requested.keys()) + list(realized_manifest.keys()) + list(completed.keys())
    )
    if not values:
        return "*(no counts)*\n"

    lines = [
        "| value | requested | manifest_realized | completed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for value in values:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(value),
                    _markdown_cell(requested.get(value, "")),
                    _markdown_cell(realized_manifest.get(value, "")),
                    _markdown_cell(completed.get(value, "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def blueprint_benchmark_summary_md(payload: Dict[str, Any]) -> str:
    """Render a blueprint benchmark composition summary payload as markdown."""
    composition = dict(payload.get("composition") or {})
    requested = dict(composition.get("requested") or {})
    realized_manifest = dict(composition.get("realized_manifest") or {})
    completed = dict(composition.get("completed") or {})

    axis_labels = {
        "main_motif": "Main Motif",
        "graft_count": "Graft Count",
        "identifiability_regime": "Identifiability Regime",
        "treatment_type": "Treatment Type",
        "outcome_type": "Outcome Type",
        "continuous_scm_profile": "Continuous SCM Profile",
        "binary_scm_profile": "Binary SCM Profile",
        "scm_profile": "SCM Profile",
        "observation_variant": "Observation Variant",
    }
    axis_order = [
        "main_motif",
        "graft_count",
        "identifiability_regime",
        "treatment_type",
        "outcome_type",
        "continuous_scm_profile",
        "binary_scm_profile",
        "scm_profile",
        "observation_variant",
    ]

    lines: List[str] = []
    lines.append("# Blueprint Benchmark Composition Summary\n\n")
    lines.append("## Overview\n\n")
    lines.append(f"- Generated at: `{payload.get('generated_at', '')}`\n")
    lines.append(f"- Manifest: `{payload.get('manifest_path', '')}`\n")
    lines.append(f"- Benchmark directory: `{payload.get('benchmark_dir', '')}`\n")
    lines.append(
        f"- Runnable manifest rows: {int(composition.get('n_manifest_rows', 0))}\n"
    )
    lines.append(
        f"- Rows with blueprint metadata: {int(composition.get('n_rows_with_blueprint', 0))}\n"
    )
    lines.append(
        f"- Rows with realized metadata: {int(composition.get('n_rows_with_realized', 0))}\n"
    )
    lines.append(
        f"- Completed scenes: {int(composition.get('n_completed_scenes', 0))}\n"
    )
    lines.append("\n")

    for axis_name in axis_order:
        axis_requested = requested.get(axis_name) or {}
        axis_realized = realized_manifest.get(axis_name) or {}
        axis_completed = completed.get(axis_name) or {}
        if not axis_requested and not axis_realized and not axis_completed:
            continue
        lines.append(f"## {axis_labels.get(axis_name, axis_name)}\n\n")
        lines.append(
            composition_count_table_md(
                requested=axis_requested,
                realized_manifest=axis_realized,
                completed=axis_completed,
            )
        )
        lines.append("\n")

    return "".join(lines).rstrip() + "\n"


###############################################################################
# DAG Plotting
###############################################################################


def save_dag_plot(sg, output_path: Path, title: Optional[str] = None) -> None:
    """Save a DAG visualization to a file.

    Args:
        sg: SampledGraph object with .graph attribute
        output_path: Path to save the PNG file
        title: Optional title for the plot
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        sg.graph,
        with_labels=True,
        ax=ax,
        node_color="lightblue",
        node_size=900,
        font_size=10,
        font_weight="bold",
        arrows=True,
        arrowsize=18,
        arrowstyle="->",
    )
    if title:
        ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Deterministic violation kinds added by finalize_story_audit_result / finalize_mapping_audit_result
_DETERMINISTIC_VIOLATION_KINDS = frozenset(
    {
        "deterministic_missing_node_mention",
        "raw_node_id_mention",
        "missing_node_attestation",
        "missing_edge_attestation",
        "graph_jargon",
    }
)


def _write_finalized_audit_section(
    f: Any,
    finalized_result: Dict[str, Any],
) -> None:
    """Write a concise summary of the finalized audit result after the raw LLM response."""
    violations = list(finalized_result.get("violations") or [])
    warnings = list(finalized_result.get("warnings") or [])
    final_pass = finalized_result.get("pass", False)

    deterministic_violations = [
        v for v in violations if v.get("kind") in _DETERMINISTIC_VIOLATION_KINDS
    ]
    deterministic_warnings = [
        w for w in warnings if w.get("kind") in _DETERMINISTIC_VIOLATION_KINDS
    ]

    if not deterministic_violations and not deterministic_warnings:
        f.write(
            f"### Deterministic Post-Checks\n\n"
            f"No additional issues found. "
            f"Final verdict: **{'PASS' if final_pass else 'FAIL'}** "
            f"({len(violations)} violation(s), {len(warnings)} warning(s) total).\n\n"
        )
        return

    f.write("### Deterministic Post-Checks\n\n")
    f.write(
        f"Final verdict: **{'PASS' if final_pass else 'FAIL'}** "
        f"({len(violations)} violation(s), {len(warnings)} warning(s) total).\n\n"
    )

    if deterministic_violations:
        f.write("**Additional violations from deterministic checks:**\n\n")
        for v in deterministic_violations:
            kind = v.get("kind", "unknown")
            node = v.get("node") or ""
            story_name = v.get("story_name") or ""
            pair = v.get("pair")
            explanation = v.get("explanation", "")
            loc_parts = []
            if node:
                loc_parts.append(f"node={node}")
            if story_name:
                loc_parts.append(f'story_name="{story_name}"')
            if pair:
                loc_parts.append(f"pair={pair}")
            loc = f" [{', '.join(loc_parts)}]" if loc_parts else ""
            f.write(f"- `{kind}`{loc}: {explanation}\n")
        f.write("\n")

    if deterministic_warnings:
        f.write("**Additional warnings from deterministic checks:**\n\n")
        for w in deterministic_warnings:
            kind = w.get("kind", "unknown")
            explanation = w.get("explanation", "")
            f.write(f"- `{kind}`: {explanation}\n")
        f.write("\n")


def append_audit_check(
    trace_path: Path,
    iteration: int,
    user_prompt: str,
    response_text: str,
    system_prompt: str = "",
    passed: bool = False,
    error: Optional[str] = None,
    reasoning: Optional[str] = None,
    unfixable: bool = False,
    finalized_result: Optional[Dict[str, Any]] = None,
) -> None:
    """Append an audit check to the trace.

    Args:
        trace_path: Path to the trace file
        iteration: Iteration number (1-indexed)
        user_prompt: Audit prompt sent to LLM
        response_text: Audit response text
        passed: Whether audit passed
        error: Optional error message
        reasoning: Optional model reasoning text
        unfixable: Whether failure is unfixable (fixed nodes)
        finalized_result: Optional finalized audit result dict (with deterministic
            post-checks applied).  When provided, a summary of violations/warnings
            and the final verdict is appended after the raw LLM response.
    """
    with trace_path.open("a", encoding="utf-8") as f:
        if passed:
            status = "✅ PASS"
        elif unfixable:
            status = "⚠️ UNFIXABLE (Fixed Nodes)"
        else:
            status = "❌ FAIL"
        f.write(f"## Audit Check {iteration} — {status}\n\n")

        if system_prompt:
            f.write("### System Prompt\n\n")
            f.write("```\n")
            f.write(escape_code_fences(system_prompt))
            f.write("\n```\n\n")

        f.write("### Audit Prompt\n\n")
        f.write("```\n")
        f.write(escape_code_fences(user_prompt))
        f.write("\n```\n\n")

        if reasoning:
            f.write("### Model Reasoning\n\n")
            f.write("```\n")
            f.write(escape_code_fences(reasoning))
            f.write("\n```\n\n")

        f.write("### Response\n\n")
        if error:
            f.write(f"**Error**: {error}\n\n")

        f.write("```json\n")
        f.write(escape_code_fences(response_text))
        f.write("\n```\n\n")

        if finalized_result is not None:
            _write_finalized_audit_section(f, finalized_result)

        f.write("---\n\n")


###############################################################################
# Story Result Functions
###############################################################################


def init_story_conversation_trace(
    trace_path: Path,
    scene_id: str,
    extra_meta: Dict[str, Any],
    proposed_domain: str,
    variable_mapping: List[Dict[str, Any]],
    renamed_dag_relpath: Optional[str] = None,
    verbalizer_system_prompt: str = "",
    verbalizer_user_prompt: str = "",
    auditor_system_prompt: str = "",
    audit_enabled: bool = False,
) -> None:
    """Initialize a markdown conversation trace file for story generation."""
    with trace_path.open("w", encoding="utf-8") as f:
        f.write(f"# Story Generation Trace: {scene_id}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Scene ID**: `{scene_id}`\n")
        f.write(f"- **Model**: `{extra_meta.get('model', 'unknown')}`\n")
        f.write(f"- **Temperature**: {extra_meta.get('temperature', 'default')}\n")
        f.write(
            f"- **Serialization Format**: {extra_meta.get('serialization_format', 'unknown')}\n"
        )
        f.write(f"- **Web Tools Enabled**: {extra_meta.get('enable_web', False)}\n")
        f.write(f"- **Audit Enabled**: {audit_enabled}\n")
        f.write(f"- **Fallback Used**: {extra_meta.get('fallback_used', False)}\n")
        if audit_enabled:
            f.write(
                f"- **Max Audit Iterations**: {extra_meta.get('max_audit_iterations', 'unknown')}\n"
            )
            f.write(
                f"- **Audit Output Format**: {extra_meta.get('audit_output_format', 'json')}\n"
            )
        f.write("\n---\n\n")

        f.write("## Pre-Story State\n\n")
        f.write("**Proposed Domain**\n\n")
        f.write("```\n")
        f.write(escape_code_fences(proposed_domain or ""))
        f.write("\n```\n\n")
        f.write("**Variable Mapping**\n\n")
        f.write(variable_mapping_table_md(variable_mapping))
        f.write("\n---\n\n")

        if renamed_dag_relpath:
            f.write("## Graph (Renamed)\n\n")
            f.write(f"![Renamed DAG]({renamed_dag_relpath})\n\n")
            f.write("---\n\n")

        if verbalizer_system_prompt:
            f.write("## Story Verbalizer System Prompt\n\n")
            f.write("```\n")
            f.write(escape_code_fences(verbalizer_system_prompt))
            f.write("\n```\n\n")

        if verbalizer_user_prompt:
            f.write("## Story Verbalizer Initial User Prompt\n\n")
            f.write("```\n")
            f.write(escape_code_fences(verbalizer_user_prompt))
            f.write("\n```\n\n")

        if audit_enabled and auditor_system_prompt:
            f.write("## Story Auditor System Prompt\n\n")
            f.write("```\n")
            f.write(escape_code_fences(auditor_system_prompt))
            f.write("\n```\n\n")

        f.write("---\n\n")


def append_story_attempt(
    trace_path: Path,
    attempt: int,
    prompt: str,
    response_text: Any,
    *,
    phase: str = "initial",
    error: Optional[str] = None,
    used_web: bool = False,
    tool_trace: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Append one story-generation attempt to the trace."""
    phase_label = str(phase).strip().lower()
    title_suffix = f" ({phase_label})" if phase_label else ""
    response_text_str = _coerce_markdown_text(response_text)

    with trace_path.open("a", encoding="utf-8") as f:
        f.write(f"## Story Attempt {attempt}{title_suffix}\n\n")

        f.write("### User Prompt\n\n")
        f.write("```\n")
        f.write(escape_code_fences(prompt or ""))
        f.write("\n```\n\n")

        _write_tool_trace_markdown(f, tool_trace)

        f.write("### Response\n\n")
        if error:
            f.write(f"**Error**: {error}\n\n")
        elif used_web:
            f.write("*(Used web tools)*\n\n")

        rendered = _write_story_response_markdown(f, response_text_str)
        if not rendered:
            fence_lang = "json" if response_text_str.lstrip().startswith("{") else ""
            f.write(f"```{fence_lang}\n")
            f.write(escape_code_fences(response_text_str))
            f.write("\n```\n\n")

        f.write("---\n\n")


def write_story_trace_from_result(
    trace_path: Path,
    result: "StoryResult",
    scene_id: str,
) -> None:
    """Write a full story-generation trace from a StoryResult object."""
    config = result.config or {}
    meta = {
        "model": config.get("model", "unknown"),
        "temperature": config.get("temperature", "default"),
        "serialization_format": config.get("serialization_format", "unknown"),
        "enable_web": config.get("enable_web", False),
        "max_audit_iterations": config.get("max_audit_iterations", "unknown"),
        "audit_output_format": config.get("audit_output_format", "json"),
        "fallback_used": bool(getattr(result, "fallback_used", False)),
    }

    audit_enabled = bool(result.audit_info and result.audit_info.get("enabled"))
    auditor_system_prompt = result.prompts.get("auditor_system", "")
    if audit_enabled and not auditor_system_prompt:
        from .verbalization_verify import build_story_audit_system_prompt

        auditor_system_prompt = build_story_audit_system_prompt(
            str(meta.get("audit_output_format", "json"))
        )

    renamed_dag_relpath: Optional[str] = None
    if getattr(result, "sg", None) is not None:
        dag_filename = f"{safe_slug(scene_id)}_renamed_dag.png"
        dag_path = trace_path.parent / dag_filename
        try:
            save_dag_plot(result.sg, dag_path, title=f"Renamed DAG: {scene_id}")
            renamed_dag_relpath = dag_filename
        except Exception:
            logger.exception("Failed to write story-trace DAG plot for %s", scene_id)

    init_story_conversation_trace(
        trace_path=trace_path,
        scene_id=scene_id,
        extra_meta=meta,
        proposed_domain=result.proposed_domain,
        variable_mapping=result.variable_mapping,
        renamed_dag_relpath=renamed_dag_relpath,
        verbalizer_system_prompt=result.prompts.get("system", ""),
        verbalizer_user_prompt=result.prompts.get("user", ""),
        auditor_system_prompt=auditor_system_prompt,
        audit_enabled=audit_enabled,
    )

    attempts = list(result.generation_trace or [])
    if not attempts:
        response_text = result.response_text or ""
        if not response_text:
            response_text = json.dumps(
                {
                    "story": result.story,
                    "causal_justifications": result.causal_justifications,
                },
                indent=2,
                ensure_ascii=False,
            )
        attempts = [
            {
                "attempt": 1,
                "phase": "initial",
                "prompt": result.prompts.get("user", ""),
                "response_text": response_text,
                "used_web": result.used_web,
                "tool_trace": result.tool_trace,
                "error": result.parse_error,
            }
        ]

    for idx, attempt in enumerate(attempts, 1):
        response_text = attempt.get("response_text", "") or ""
        if not response_text and attempt.get("story"):
            response_text = json.dumps(
                {
                    "story": attempt.get("story", ""),
                    "causal_justifications": attempt.get("causal_justifications", {}),
                },
                indent=2,
                ensure_ascii=False,
            )

        append_story_attempt(
            trace_path=trace_path,
            attempt=int(attempt.get("attempt", idx) or idx),
            phase=str(attempt.get("phase", "initial") or "initial"),
            prompt=str(attempt.get("prompt", "") or ""),
            response_text=response_text,
            error=attempt.get("error"),
            used_web=bool(attempt.get("used_web")),
            tool_trace=attempt.get("tool_trace", []),
        )

        if result.audit_info and idx <= len(result.audit_info.get("iterations", [])):
            audit = result.audit_info["iterations"][idx - 1]
            audit_data = audit.get("result", audit)
            append_audit_check(
                trace_path=trace_path,
                iteration=idx,
                system_prompt="",
                user_prompt=audit.get("prompt", ""),
                response_text=audit.get("response", ""),
                passed=audit_data.get("pass", False),
                error=None,
                unfixable=False,
                finalized_result=audit_data if audit_data is not audit else None,
            )


def story_result_core_outputs_md(result: "StoryResult") -> str:
    """Format core outputs from a StoryResult as markdown."""
    lines: List[str] = ["# Core outputs\n\n"]

    if result.parse_error:
        lines.append(f"**ERROR**: {result.parse_error}\n\n")

    lines.append("### Variable Mapping\n\n")
    lines.append(variable_mapping_table_md(result.variable_mapping))
    lines.append("\n")

    lines.append("### Story\n\n")
    lines.append("```\n")
    lines.append(escape_code_fences(result.story or ""))
    lines.append("\n```\n\n")

    lines.append("### Causal Justifications\n\n")
    lines.append("```\n")
    causal_justifications = result.causal_justifications
    if isinstance(causal_justifications, (dict, list)):
        causal_justifications = json.dumps(
            causal_justifications,
            indent=2,
            ensure_ascii=False,
        )
    lines.append(escape_code_fences(str(causal_justifications or "")))
    lines.append("\n```\n\n")

    return "".join(lines).rstrip() + "\n"


###############################################################################
# Observation-model diagnostics (JSON + plot)
###############################################################################


def _slim_observation_metadata(obs_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Strip bulky grid arrays and index lists from observation metadata.

    Keeps summary stats (min/avg information, baseline R²/AUC, gap, sampling
    attempts, per-proxy type/kind) for post-mortem analysis.
    """
    DROP = {
        # top-level
        "calibration_indices",
        # information_diagnostics
        "grid",
        "bundle_information_grid",
        # per-proxy grids
        "probability_grid",
        "mean_grid",
        "variance_grid",
        "derivative_grid",
        "information_grid",
        "logit_derivative_grid",
        # baseline (recoverable from splits)
        "holdout_row_positions",
    }

    def _strip(d):
        if not isinstance(d, dict):
            return d
        return {k: _strip(v) for k, v in d.items() if k not in DROP}

    return _strip(obs_metadata)


def save_observation_diagnostics_json(
    obs_metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    """Persist observation-model diagnostics as JSON for post-mortem analysis.

    Keeps per-node summary stats (min/avg information, sampling attempts,
    baseline R²/AUC, gap, proxy column specs) but strips dense grid arrays
    and calibration indices to keep the file compact.
    """
    import json

    bundles = obs_metadata.get("bundles", {})
    if not bundles:
        return

    slim = _slim_observation_metadata(obs_metadata)
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(slim, f, indent=2, default=str)


def save_observation_diagnostics_plot(
    obs_metadata: Dict[str, Any],
    output_path: Path,
    title: Optional[str] = "Observation model: proxy recoverability diagnostic",
) -> None:
    """Save a bar chart of naive vs calibrated proxy recoverability per node.

    Produces a horizontal grouped bar chart with naive (single-proxy linear)
    and calibrated (multi-proxy GBM) scores, plus the gap annotation.
    The output format follows the ``output_path`` suffix (PNG for pipeline
    diagnostics, PDF for vector/print use). ``title=None`` omits the in-figure
    title, e.g. when a surrounding caption already carries it.
    """
    bundles = obs_metadata.get("bundles", {})
    if not bundles:
        return

    nodes = []
    naive_scores = []
    cal_scores = []
    metric_labels = []

    for node, b in bundles.items():
        bl = b.get("baseline_diagnostics", {})
        if bl.get("status") != "ok":
            continue
        metric_type = bl.get("metric_type", "continuous")
        if metric_type == "binary":
            naive_val = bl.get("naive", {}).get("auc")
            cal_val = bl.get("calibrated", {}).get("auc")
            label = "AUC"
        else:
            naive_val = bl.get("naive", {}).get("r2")
            cal_val = bl.get("calibrated", {}).get("r2")
            label = "R²"

        if naive_val is None or cal_val is None:
            continue

        nodes.append(textwrap.fill(node, width=26))
        naive_scores.append(float(naive_val))
        cal_scores.append(float(cal_val))
        metric_labels.append(label)

    if not nodes:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    # Two shades of one hue for the bound pair; neutral band; ink for text.
    color_naive = "#86b6ef"
    color_cal = "#2a78d6"
    color_band = "#f0efec"
    color_axis = "#c3c2b7"
    color_grid = "#e1e0d9"
    color_ink = "#52514e"
    color_muted = "#898781"

    n = len(nodes)
    y_pos = np.arange(n)
    bar_h = 0.32

    fig, ax = plt.subplots(figsize=(4.6, 0.72 * n + 1.1))

    # Target operating window for the upper bound (calibrated score)
    ax.axvspan(0.3, 0.8, color=color_band, zorder=0)
    ax.axvline(0.3, color=color_axis, lw=0.8, zorder=1)
    ax.axvline(0.8, color=color_axis, lw=0.8, zorder=1)

    # Bars
    ax.barh(
        y_pos - bar_h / 2 - 0.02,
        naive_scores,
        bar_h,
        label="Naive (single proxy, linear, on calibration)",
        color=color_naive,
        zorder=2,
    )
    ax.barh(
        y_pos + bar_h / 2 + 0.02,
        cal_scores,
        bar_h,
        label="Calibrated (all proxies, GBM, on full data)",
        color=color_cal,
        zorder=2,
    )

    # Gap annotation at the end of the longer bar
    for i in range(n):
        hi = max(naive_scores[i], cal_scores[i])
        gap = cal_scores[i] - naive_scores[i]
        sign = "+" if gap >= 0 else ""
        ax.text(
            hi + 0.02,
            y_pos[i],
            f"Δ={sign}{gap:.3f} {metric_labels[i]}",
            va="center",
            fontsize=7,
            color=color_ink,
        )

    # Band labels at bottom
    ax.text(0.15, -0.62, "too hard", ha="center", fontsize=6.5, color=color_muted)
    ax.text(0.55, -0.62, "target range", ha="center", fontsize=6.5, color=color_muted)
    ax.text(0.95, -0.62, "too easy", ha="center", fontsize=6.5, color=color_muted)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(nodes, fontsize=8)
    ax.set_xlabel("R² / AUC  (higher = easier to recover latent)", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(-1.0, n - 0.5 + 0.5)
    ax.tick_params(axis="x", labelsize=7.5, color=color_axis)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=color_grid, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(color_axis)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        fontsize=6.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved observation diagnostics plot: %s", output_path)
