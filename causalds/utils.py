# Collection of random utils for the CausalDS library

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

UINT32_SEED_MODULUS = 2**32


def normalize_random_seed(seed: Optional[int]) -> Optional[int]:
    """Wrap arbitrary integer seeds into NumPy/scikit-learn's uint32 range."""
    if seed is None:
        return None
    return int(int(seed) % UINT32_SEED_MODULUS)


def offset_random_seed(seed: Optional[int], offset: int) -> int:
    """Add a deterministic offset and wrap it into the uint32 seed range."""
    base_seed = 0 if seed is None else int(seed)
    return int((base_seed + int(offset)) % UINT32_SEED_MODULUS)


def deep_merge_dicts(
    base: Dict[str, Any],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recursively merge plain dicts."""
    merged = deepcopy(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_omegaconf_dicts(
    base: Dict[str, Any],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge plain dict configs using OmegaConf semantics."""
    if not override:
        return deepcopy(base)
    merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(override))
    return OmegaConf.to_container(merged, resolve=True)


def merge_omegaconf_sections(*sections: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple OmegaConf/plain sections into one resolved plain dict."""
    merged = OmegaConf.create({})
    for section in sections:
        if section:
            merged = OmegaConf.merge(merged, section)
    return OmegaConf.to_container(merged, resolve=True)


def ensure_plain(value: Any) -> Any:
    """Resolve OmegaConf containers into plain Python values."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def ensure_plain_dict(value: Any) -> Dict[str, Any]:
    """Resolve OmegaConf/plain mappings into a plain dict."""
    if value is None:
        return {}
    value = ensure_plain(value)
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict-like config, got {type(value).__name__}")
    return dict(value)


def coerce_list(value: Any) -> Optional[List[Any]]:
    """Coerce a config field into a plain list when present."""
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    return list(ensure_plain(value))


def json_safe(value: Any) -> Any:
    """Convert nested objects into JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    if OmegaConf.is_config(value):
        return json_safe(OmegaConf.to_container(value, resolve=True))

    if isinstance(value, nx.DiGraph):
        return {
            "kind": "DiGraph",
            "nodes": [str(n) for n in value.nodes()],
            "edges": [[str(u), str(v)] for (u, v) in value.edges()],
        }

    if isinstance(value, nx.Graph):
        return {
            "kind": value.__class__.__name__,
            "nodes": [str(n) for n in value.nodes()],
            "edges": [[str(u), str(v)] for (u, v) in value.edges()],
        }

    # SampledGraph-like object (duck typing) without importing causalds.graph here.
    if all(hasattr(value, attr) for attr in ("graph", "treatment", "outcome", "motif")):
        return {
            "kind": value.__class__.__name__,
            "treatment": str(getattr(value, "treatment", "")),
            "outcome": str(getattr(value, "outcome", "")),
            "motif": str(getattr(value, "motif", "")),
            "observed_nodes": json_safe(getattr(value, "observed_nodes", None)),
            "latent_nodes": json_safe(getattr(value, "latent_nodes", None)),
            "meta": json_safe(getattr(value, "meta", None)),
            "graph": json_safe(getattr(value, "graph", None)),
        }

    if hasattr(value, "__dict__"):
        return json_safe(vars(value))

    return str(value)


def atomic_write_json(path: Union[str, Path], payload: Any) -> None:
    """Write JSON atomically by replacing a temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(target)


def write_text_atomic(path: Union[str, Path], content: str) -> None:
    """Write text atomically by replacing a temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    tmp_path.replace(target)


def coerce_observed_flag(value: Any) -> bool:
    """Normalize observed flags that may come back as bools or strings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "no", "0", ""}


_NODE_NAME_YELLOW_FLAG_PATTERNS: List[Tuple[str, re.Pattern[str], str]] = [
    (
        "derived_name",
        re.compile(
            r"\b(corrected|residual|deterministic function of|derived from|obtained after applying|downstream of)\b",
            flags=re.IGNORECASE,
        ),
        "Derived or processed wording may imply a direct edge or semantic collapse.",
    ),
    (
        "restrictive_qualifier",
        re.compile(
            r"\b(based only on|determined solely by|used only for bookkeeping|unrelated to|not influenced by other factors)\b",
            flags=re.IGNORECASE,
        ),
        "Restrictive qualifier should be taken literally and may rule out extra substantive causes.",
    ),
]


def collect_node_name_yellow_flags(
    mapping_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect heuristic node-name red flags for auditor prompts.

    These are yellow flags only: they should focus the auditor's attention,
    not deterministically fail a mapping or story.
    """
    yellow_flags: List[Dict[str, Any]] = []
    for row in mapping_rows or []:
        if not isinstance(row, dict):
            continue
        story_name = str(row.get("story_name", "")).strip()
        if not story_name:
            continue

        matches: List[Dict[str, Any]] = []
        for kind, pattern, note in _NODE_NAME_YELLOW_FLAG_PATTERNS:
            found = sorted({m.group(0) for m in pattern.finditer(story_name)})
            if not found:
                continue
            matches.append(
                {
                    "kind": kind,
                    "matched_terms": found,
                    "note": note,
                }
            )

        if not matches:
            continue

        yellow_flags.append(
            {
                "id": str(row.get("id", "")).strip() or None,
                "story_name": story_name,
                "flags": matches,
            }
        )

    return yellow_flags


def parse_json_from_llm(text: str, silent: bool = False) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling code blocks and other common issues."""
    text = (text or "").strip()
    if not silent:
        logger.debug("Parsing JSON from LLM output (length: %d)", len(text))

    if not text:
        raise ValueError("Could not parse JSON from LLM output.")

    def clean_json_str(s: str) -> str:
        # Clean up trailing commas
        s = re.sub(r",\s*([\]}])", r"\1", s)
        # Fix common Unicode issues
        s = (
            s.replace("\u2011", "-")  # NON-BREAKING HYPHEN
            .replace("\u2010", "-")  # HYPHEN
            .replace("\u2013", "-")  # EN DASH
            .replace("\u2014", "-")  # EM DASH
            .replace("\u201c", '"')  # LEFT DOUBLE QUOTE
            .replace("\u201d", '"')  # RIGHT DOUBLE QUOTE
            .replace("\u2018", "'")  # LEFT SINGLE QUOTE
            .replace("\u2019", "'")  # RIGHT SINGLE QUOTE
            .replace("\u00a0", " ")  # NO-BREAK SPACE
        )
        # Normalize Python-y literals to JSON
        s = re.sub(r"\bNone\b", "null", s)
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        return s

    def repair_common_json_issues(s: str) -> str:
        s = clean_json_str(s)
        # Fix missing colon after a quoted key before { or [
        s = re.sub(
            r'(?:(?<=\{)|(?<=,)|^)\s*("[^"]+")\s*(\{|\[)',
            r"\1: \2",
            s,
        )
        # Fix missing colon after an unquoted key before { or [
        s = re.sub(
            r"(?:(?<=\{)|(?<=,)|^)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{|\[)",
            r'"\1": \2',
            s,
        )
        # Common typo: "variable_mapping[" -> "variable_mapping": [
        s = re.sub(
            r'"variable_mapping"\s*\[',
            '"variable_mapping": [',
            s,
        )
        s = re.sub(
            r"\bvariable_mapping\s*\[",
            '"variable_mapping": [',
            s,
        )
        return s

    def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(clean_json_str(s), strict=False)
        except Exception:
            return None

    def try_parse_yaml(s: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = yaml.safe_load(s)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
        # If the model returned just a list of mappings, wrap it
        if (
            isinstance(parsed, list)
            and parsed
            and all(isinstance(x, dict) for x in parsed)
        ):
            return {"variable_mapping": parsed}
        return None

    # 1. Try direct parse
    result = try_parse_json(text)
    if result is not None:
        if not silent:
            logger.debug("Successfully parsed JSON via direct parse")
        return result
    if not silent:
        logger.debug("Direct JSON parse failed, trying pattern matching")

    # 2. Try to find JSON in code blocks or raw text with better patterns
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
        r"\{[\s\S]*\}",  # Raw JSON object (greedy)
        r"\[[\s\S]*\]",  # Raw JSON array (greedy)
    ]

    candidates: List[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if match and match not in candidates:
                candidates.append(match)

    for candidate in candidates:
        result = try_parse_json(candidate)
        if result is not None:
            if not silent:
                logger.debug("Successfully parsed JSON via pattern match")
            return result

    # 2b. Try repaired versions of candidates
    for candidate in candidates:
        repaired = repair_common_json_issues(candidate)
        result = try_parse_json(repaired)
        if result is not None:
            if not silent:
                logger.debug("Successfully parsed JSON after repair")
            return result
        yaml_result = try_parse_yaml(repaired)
        if yaml_result is not None:
            if not silent:
                logger.debug("Successfully parsed JSON via YAML fallback after repair")
            return yaml_result

    # 3. Last attempt: try to find the largest valid JSON object by searching for balanced braces
    brace_level = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == "{":
            if brace_level == 0:
                start_idx = i
            brace_level += 1
        elif char == "}":
            brace_level -= 1
            if brace_level == 0 and start_idx >= 0:
                candidate = text[start_idx : i + 1]
                result = try_parse_json(candidate)
                if result is not None:
                    return result
                # Attempt repair + YAML fallback
                repaired = repair_common_json_issues(candidate)
                result = try_parse_json(repaired)
                if result is not None:
                    return result
                yaml_result = try_parse_yaml(repaired)
                if yaml_result is not None:
                    return yaml_result
                # Continue searching
                start_idx = -1

    # 4. YAML fallback on full text (last resort)
    yaml_result = try_parse_yaml(repair_common_json_issues(text))
    if yaml_result is not None:
        if not silent:
            logger.debug("Successfully parsed JSON via YAML fallback (full text)")
        return yaml_result

    # If we failed, log error and raise
    if not silent:
        logger.error(
            "Failed to parse JSON from LLM output. Text excerpt: %s", text[:500]
        )
    raise ValueError("Could not parse JSON from LLM output.")


def _xml_element_to_dict(el: ET.Element) -> Any:
    """Recursively convert an XML element to a dict/list/str.

    Handles:
    - Elements with only attributes (no text, no children) → dict of attributes
    - Elements with text → text content (with bool coercion)
    - Elements with children → dict grouping children by tag
    - Multiple children with the same tag → list
    """
    children = list(el)
    has_text = el.text and el.text.strip()

    # Leaf with attributes only (e.g. <variable id="X" story_name="..."/>)
    if not children and not has_text and el.attrib:
        result = dict(el.attrib)
        # Coerce boolean strings
        for k, v in result.items():
            if v.lower() in ("true", "false"):
                result[k] = v.lower() == "true"
        return result

    # Leaf with only text
    if not children and has_text and not el.attrib:
        txt = el.text.strip()
        if txt.lower() in ("true", "false"):
            return txt.lower() == "true"
        return txt

    # Leaf with both attributes and text
    if not children and has_text and el.attrib:
        result = dict(el.attrib)
        result["_text"] = el.text.strip()
        return result

    # Empty leaf with no attributes
    if not children and not has_text and not el.attrib:
        return None

    # Has children: group by tag
    result = dict(el.attrib) if el.attrib else {}
    tag_groups: Dict[str, list] = {}
    for child in children:
        tag_groups.setdefault(child.tag, []).append(child)

    for tag, group in tag_groups.items():
        if len(group) == 1:
            child_val = _xml_element_to_dict(group[0])
            # Single child that maps to a simple value
            result[tag] = child_val
        else:
            result[tag] = [_xml_element_to_dict(c) for c in group]

    if has_text:
        result["_text"] = el.text.strip()

    return result


def _unwrap_xml_single_child(d: Any) -> Any:
    """Recursively unwrap single-child XML wrapper dicts to match JSON structure.

    XML parsing creates wrapper layers that JSON doesn't have:
    - <violations><violation>...</violation></violations>
      → {"violations": {"violation": [...]}} instead of {"violations": [...]}
    - <pair><node>X</node><node>Y</node></pair>
      → {"pair": {"node": ["X", "Y"]}} instead of {"pair": ["X", "Y"]}

    This function recursively unwraps single-key wrapper dicts so the result
    matches the flat structure that JSON parsing produces.
    """
    if isinstance(d, list):
        return [_unwrap_xml_single_child(item) for item in d]

    if not isinstance(d, dict):
        return d

    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively normalize inner dict first
            value = _unwrap_xml_single_child(value)
            # Then unwrap single-key wrappers
            if len(value) == 1:
                inner_key = next(iter(value))
                inner_val = value[inner_key]
                if isinstance(inner_val, list):
                    # {"violation": [{...}, {...}]} → [{...}, {...}]
                    result[key] = inner_val
                elif isinstance(inner_val, dict):
                    # {"violation": {...}} → [{...}] (single item → list)
                    result[key] = [inner_val]
                else:
                    # {"_text": "value"} or {"node": "X"} → "value" / "X"
                    result[key] = inner_val
            else:
                result[key] = value
        elif isinstance(value, list):
            result[key] = [_unwrap_xml_single_child(item) for item in value]
        else:
            result[key] = value

    return result


# Fields that must always be lists in parsed LLM output
_XML_LIST_FIELDS = {
    "variable_mapping",
    "violations",
    "non_edge_attestations",
    "problematic_pairs",
}


def _normalize_xml_result(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an XML-parsed LLM result to match the JSON structure.

    Applies recursive single-child unwrapping, then ensures known list fields
    are always lists (not None or a bare dict).
    """
    d = _unwrap_xml_single_child(d)

    # Ensure known list fields are always lists
    for field in _XML_LIST_FIELDS:
        val = d.get(field)
        if val is None:
            d[field] = []
        elif isinstance(val, dict):
            d[field] = [val]
        elif not isinstance(val, list):
            d[field] = []

    return d


def parse_xml_from_llm(text: str, silent: bool = False) -> Dict[str, Any]:
    """Parse XML from LLM output into a dict.

    Strategy (mirrors parse_json_from_llm's robustness):
    1. Extract from code blocks (```xml ... ```, ``` ... ```)
    2. Find raw XML by regex for <mapping>...</mapping> or any root element
    3. Parse with ElementTree and convert to dict
    4. Fallback: try parse_json_from_llm if XML parsing fails
    """
    text = (text or "").strip()
    if not silent:
        logger.debug("Parsing XML from LLM output (length: %d)", len(text))

    if not text:
        raise ValueError("Could not parse XML from LLM output.")

    def try_parse_xml(s: str) -> Optional[Dict[str, Any]]:
        try:
            root = ET.fromstring(s)
            result = _xml_element_to_dict(root)
            if isinstance(result, dict):
                return _normalize_xml_result(result)
            return {root.tag: result}
        except ET.ParseError:
            return None

    # 1. Try code blocks
    candidates: List[str] = []
    patterns = [
        r"```xml\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if match and match.strip() and match not in candidates:
                candidates.append(match.strip())

    # 2. Try raw XML with common root elements
    for root_tag in ("mapping", "audit", "pre_audit", "causal_graph"):
        pattern = rf"<{root_tag}[\s>][\s\S]*?</{root_tag}>"
        for match in re.findall(pattern, text):
            if match not in candidates:
                candidates.append(match)

    # Also try any XML-looking block starting with <
    xml_block = re.search(r"(<[a-zA-Z_][\s\S]*>)\s*$", text)
    if xml_block:
        candidate = xml_block.group(1)
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        result = try_parse_xml(candidate)
        if result is not None:
            if not silent:
                logger.debug("Successfully parsed XML from candidate")
            return result

    # 3. Try full text as XML
    result = try_parse_xml(text)
    if result is not None:
        if not silent:
            logger.debug("Successfully parsed XML from full text")
        return result

    # 4. Fallback: try JSON parser (LLM may have ignored XML instruction)
    try:
        result = parse_json_from_llm(text, silent=True)
        if not silent:
            logger.debug("XML parse failed, fell back to JSON parser successfully")
        return result
    except ValueError:
        pass

    if not silent:
        logger.error(
            "Failed to parse XML from LLM output. Text excerpt: %s", text[:500]
        )
    raise ValueError("Could not parse XML from LLM output.")


def parse_llm_output(
    text: str, output_format: str = "json", silent: bool = False
) -> Dict[str, Any]:
    """Dispatch to the right parser based on output_format.

    Args:
        text: Raw LLM response text
        output_format: "json" or "xml"
        silent: Suppress debug logging

    Returns:
        Parsed dict
    """
    if output_format == "xml":
        return parse_xml_from_llm(text, silent=silent)
    return parse_json_from_llm(text, silent=silent)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softsign(x):
    return x / (1.0 + np.abs(x))


def tanh(x):
    return np.tanh(x)


def softplus(x):
    return np.log1p(np.exp(x))


def ensure_list(x):
    if x is None:
        return []
    return list(x)


def random_sign(rng: np.random.RandomState):
    return rng.choice([-1.0, 1.0])


###############################################################################
# API Retry Utilities
###############################################################################


def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_sleep: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Execute a function with exponential backoff on transient errors.

    Args:
        fn: Callable to execute
        max_retries: Maximum number of retry attempts
        base_sleep: Base sleep time in seconds (doubles each retry)
        exceptions: Tuple of exception types to catch and retry

    Returns:
        The return value of fn()

    Raises:
        The last exception encountered if all retries fail
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except exceptions as e:
            last_error = e
            if attempt < max_retries - 1:
                sleep_time = base_sleep * (2**attempt)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    str(e)[:100],
                    sleep_time,
                )
                time.sleep(sleep_time)
            else:
                logger.error(
                    "All %d attempts failed. Last error: %s",
                    max_retries,
                    str(e)[:200],
                )
    raise last_error


###############################################################################
# Variable Mapping Utilities
###############################################################################


def normalize_mapping_rows(
    mapping: List[Dict[str, Any]],
    *,
    default_observed: Optional[bool] = True,
) -> List[Dict[str, Any]]:
    """Normalize mapping rows from LLM output to standard format.

    Handles various field name variants that LLMs might use.

    Args:
        mapping: List of mapping dicts from LLM
        default_observed: Default observed value when missing/invalid.
            Use None to preserve missing values.

    Returns:
        Normalized list of dicts with standard field names
    """
    normalized = []
    for row in mapping:
        if not isinstance(row, dict):
            continue

        vid = (
            row.get("id")
            or row.get("var_id")
            or row.get("variable_id")
            or row.get("node_id")
            or row.get("variable")
            or ""
        )
        story_name = row.get("story_name") or row.get("name") or row.get("label") or ""

        obs = row.get("observed")
        if isinstance(obs, str):
            obs_value: Optional[bool] = obs.lower() in ("true", "yes", "1", "observed")
        elif isinstance(obs, bool):
            obs_value = obs
        elif obs is None:
            obs_value = default_observed
        else:
            obs_value = bool(obs) if default_observed is not None else None

        normalized_row = {
            "id": str(vid).strip(),
            "story_name": str(story_name).strip(),
            "observed": obs_value,
            "type": row.get("type") or None,
            "unit": row.get("unit") or None,
        }

        if normalized_row["id"] or normalized_row["story_name"]:
            normalized.append(normalized_row)

    return normalized


def mapping_rows_incomplete(mapping_rows: Any) -> bool:
    """Return True when normalized mapping rows are missing usable assignments."""
    if not isinstance(mapping_rows, list) or not mapping_rows:
        return True

    for row in mapping_rows:
        if not isinstance(row, dict):
            continue
        if (row.get("id") or "").strip() or (row.get("story_name") or "").strip():
            return False
    return True
