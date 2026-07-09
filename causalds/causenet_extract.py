import bz2
import hashlib
import json
import logging
import pickle
import random
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_MEMORY_INDEX_CACHE: Dict[
    Tuple[str, int, Optional[str]],
    Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[Tuple[str, str], "CNEdge"]],
] = {}
_MEMORY_INDEX_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class CNEdge:
    cause: str
    effect: str
    support: Optional[int] = None
    sources: Optional[List[dict]] = None


def iter_edges(path: str) -> Iterable[CNEdge]:
    logger.info("Loading CauseNet edges from: %s", path)
    opener = bz2.open if path.endswith(".bz2") else open
    edge_count = 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cr = rec.get("causal_relation", {})
            a = (cr.get("cause", {}) or {}).get("concept")
            b = (cr.get("effect", {}) or {}).get("concept")
            if a and b and a != b:
                edge_count += 1
                yield CNEdge(
                    a.strip(), b.strip(), rec.get("support"), rec.get("sources")
                )
    logger.info("Loaded %d valid CauseNet edges", edge_count)


def _cache_path(
    source_path: str, min_support: int, domain_regex: Optional[str]
) -> Path:
    """Generate a unique cache filename based on source and parameters."""
    source_file = Path(source_path)
    # Create hash of parameters for cache key
    params = f"{source_file.stem}_{min_support}_{domain_regex or 'none'}"
    param_hash = hashlib.md5(params.encode()).hexdigest()[:8]

    cache_dir = source_file.parent
    cache_name = f"{source_file.stem}_index_{param_hash}.pkl"
    cache_path = cache_dir / cache_name
    logger.debug("Cache path: %s", cache_path)
    return cache_path


def build_index(
    path: str,
    min_support: int = 2,
    domain_regex: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[Tuple[str, str], CNEdge]]:
    """children[a] = {b}, parents[b] = {a}, and info[(a,b)] = CNEdge.

    Args:
        path: Path to CauseNet data file
        min_support: Minimum support threshold
        domain_regex: Optional regex filter for domain concepts
            # NOTE: this is very rough since it only performs the check on the concept itself
        use_cache: If True, load from/save to cache
    """
    resolved_path = str(Path(path).resolve())
    memory_cache_key = (resolved_path, int(min_support), domain_regex)

    if use_cache:
        with _MEMORY_INDEX_CACHE_LOCK:
            cached_index = _MEMORY_INDEX_CACHE.get(memory_cache_key)
            if cached_index is not None:
                logger.info("Reusing CauseNet index from process cache: %s", resolved_path)
                return cached_index

    cache_file = _cache_path(path, min_support, domain_regex)

    if use_cache:
        with _MEMORY_INDEX_CACHE_LOCK:
            cached_index = _MEMORY_INDEX_CACHE.get(memory_cache_key)
            if cached_index is not None:
                logger.info("Reusing CauseNet index from process cache: %s", resolved_path)
                return cached_index

            # Hold the lock while loading/building so concurrent batch-worker
            # threads do not deserialize the same large index multiple times.
            if cache_file.exists():
                logger.info("Loading CauseNet index from cache: %s", cache_file)
                try:
                    with open(cache_file, "rb") as f:
                        cached = pickle.load(f)
                    ret = (cached["children"], cached["parents"], cached["info"])
                    _MEMORY_INDEX_CACHE[memory_cache_key] = ret
                    logger.info("Cache loaded successfully")
                    return ret
                except Exception as e:
                    logger.warning("Failed to load cache (%s), rebuilding index", e)

            ret = _build_index_fresh(
                path=path,
                min_support=min_support,
                domain_regex=domain_regex,
            )

            try:
                logger.info("Saving index to cache: %s", cache_file)
                with open(cache_file, "wb") as f:
                    pickle.dump(
                        {
                            "children": dict(ret[0]),
                            "parents": dict(ret[1]),
                            "info": ret[2],
                        },
                        f,
                    )
                logger.info("Cache saved successfully")
            except Exception as e:
                logger.warning("Failed to save cache: %s", e)

            _MEMORY_INDEX_CACHE[memory_cache_key] = ret
            return ret

    return _build_index_fresh(
        path=path,
        min_support=min_support,
        domain_regex=domain_regex,
    )


def _build_index_fresh(
    *,
    path: str,
    min_support: int,
    domain_regex: Optional[str],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[Tuple[str, str], CNEdge]]:
    """Build a CauseNet index from source data without consulting caches."""
    logger.info(
        "Building CauseNet index (min_support=%d, domain_regex=%s)",
        min_support,
        domain_regex if domain_regex else "None",
    )
    children, parents = defaultdict(set), defaultdict(set)
    info: Dict[Tuple[str, str], CNEdge] = {}
    rgx = re.compile(domain_regex, re.I) if domain_regex else None

    filtered_count = 0
    for e in iter_edges(path):
        if e.support is not None and e.support < min_support:
            filtered_count += 1
            continue
        if rgx and not (rgx.search(e.cause) or rgx.search(e.effect)):
            filtered_count += 1
            continue
        children[e.cause].add(e.effect)
        parents[e.effect].add(e.cause)
        info[(e.cause, e.effect)] = e

    logger.info(
        "CauseNet index built: %d concepts, %d edges (filtered %d edges)",
        len(set(children.keys()) | set(parents.keys())),
        len(info),
        filtered_count,
    )
    return children, parents, info


########################################################################################
# NL-FOCUSED PROVENANCE EXTRACTION
########################################################################################

_NL_PAYLOAD_KEYS = (
    # Sentence sources
    "sentence",
    "surface",
    "wikipedia_page_title",
    "sentence_section_heading",
    # Wikipedia list sources
    "list_toc_parent_title",
    "list_toc_section_heading",
    # Wikipedia infobox sources
    "infobox_title",
    "infobox_template",
    "infobox_argument",
)

_URL_PAYLOAD_KEYS = (
    "clueweb12_page_reference",
    "wikipedia_page_url",
    "url",
)


def extract_nl_provenance_fields(
    sources: Optional[List[dict]],
    *,
    include_url: bool = True,
    include_path_pattern: bool = False,
) -> List[Dict[str, Any]]:
    """Extract compact, NL-focused fields from CauseNet provenance sources.

    Intended for prompting/verbalization. Keeps only human-readable context:
    sentences, page titles, headings, and infobox/list metadata.
    """
    out: List[Dict[str, Any]] = []
    if not sources:
        logger.debug("No sources provided for NL provenance extraction")
        return out

    for src in sources:
        if not isinstance(src, dict):
            continue
        stype = src.get("type")
        payload = src.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        item: Dict[str, Any] = {"source_type": stype}

        for k in _NL_PAYLOAD_KEYS:
            v = payload.get(k)
            if isinstance(v, str):
                v = v.strip()
                if v:
                    item[k] = v

        if include_url:
            for k in _URL_PAYLOAD_KEYS:
                v = payload.get(k)
                if isinstance(v, str):
                    v = v.strip()
                    if v:
                        item["url"] = v
                        break

        if include_path_pattern:
            pp = payload.get("path_pattern")
            if isinstance(pp, str):
                pp = pp.strip()
                if pp:
                    item["path_pattern"] = pp

        if len(item) > 1:
            out.append(item)

    logger.debug(
        "Extracted %d NL provenance entries from %d sources", len(out), len(sources)
    )
    return out


def choose_best_nl_source(
    nl_sources: List[Dict[str, Any]], shuffle=True, include_length=False
) -> Optional[Dict[str, Any]]:
    """Pick one best NL provenance entry (prefer sentence evidence)."""

    if not nl_sources:
        logger.debug("No NL sources to choose from")
        return None

    def score(s: Dict[str, Any]) -> Tuple[int, int, int, int]:
        st = s.get("source_type") or ""
        has_sentence = 1 if isinstance(s.get("sentence"), str) else 0
        has_surface = 1 if isinstance(s.get("surface"), str) else 0
        has_title = (
            1
            if (
                isinstance(s.get("wikipedia_page_title"), str)
                or isinstance(s.get("infobox_title"), str)
            )
            else 0
        )
        is_wiki_sentence = 1 if st == "wikipedia_sentence" else 0
        is_web_sentence = 1 if st == "clueweb12_sentence" else 0
        is_wiki_other = 1 if isinstance(st, str) and st.startswith("wikipedia_") else 0

        sentence_len = (
            (len(s.get("sentence", "")) if isinstance(s.get("sentence"), str) else 0)
            if include_length
            else 0
        )

        return (
            is_wiki_sentence,
            is_web_sentence,
            is_wiki_other,
            has_sentence + has_surface + has_title,
            sentence_len,  # tiebreaker: prefer longer sentences
        )

    if shuffle:
        shuffled = nl_sources.copy()
        random.shuffle(shuffled)

        result = max(shuffled, key=score)
        logger.debug(
            "Selected best NL source: type=%s, has_sentence=%s",
            result.get("source_type"),
            "sentence" in result,
        )
        return result

    result = max(nl_sources, key=score)
    logger.debug(
        "Selected best NL source: type=%s, has_sentence=%s",
        result.get("source_type"),
        "sentence" in result,
    )
    return result
