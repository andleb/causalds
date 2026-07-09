"""
Web search backend abstraction.

The agent-facing tool contract remains split into search and open, but the
backend can be swapped. The default backend is Tavily. Alternative backend
modes include a generic HTTP document-search service and a command that
accepts a minimal JSON contract:

    {"query": "...", "k": 5, "return_fulltext": false}

That command can itself be a thin MCP client or any other wrapper.
"""

import json
import logging
import os
import shlex
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

_HTTP_DOC_URL_PREFIX = "web-search://doc/"
_RUNTIME_WEB_SEARCH_CONFIG: Dict[str, Any] = {}

try:
    # Package name: tavily-python; module import: tavily
    from tavily import TavilyClient  # type: ignore
except Exception:
    logger.warning("tavily-python package not found. Tavily web search will not work.")
    TavilyClient = None  # type: ignore


class WebSearchError(Exception):
    pass


class TavilyError(WebSearchError):
    pass


def configure_web_search_backend(
    *,
    backend: Optional[str] = None,
    base_url: Optional[str] = None,
    command: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> None:
    """Configure provider selection for the current process and clear the cache."""
    _RUNTIME_WEB_SEARCH_CONFIG.clear()
    _RUNTIME_WEB_SEARCH_CONFIG.update(
        {
            "backend": (
                str(backend).strip().lower()
                if backend is not None and str(backend).strip()
                else None
            ),
            "base_url": (
                str(base_url).strip()
                if base_url is not None and str(base_url).strip()
                else None
            ),
            "command": (
                str(command).strip()
                if command is not None and str(command).strip()
                else None
            ),
            "timeout_sec": float(timeout_sec) if timeout_sec is not None else None,
        }
    )
    get_web_search_provider.cache_clear()


def _resolve_setting(
    runtime_key: str,
    env_name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    runtime_value = _RUNTIME_WEB_SEARCH_CONFIG.get(runtime_key)
    if runtime_value is not None:
        return str(runtime_value)
    env_value = os.getenv(env_name)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()
    return default


def _resolve_timeout_setting(default: float = 60.0) -> float:
    runtime_value = _RUNTIME_WEB_SEARCH_CONFIG.get("timeout_sec")
    if runtime_value is not None:
        return float(runtime_value)
    env_value = os.getenv("CAUSALDS_WEB_SEARCH_TIMEOUT_SEC")
    if env_value is not None and str(env_value).strip():
        return float(env_value)
    return float(default)


def _coerce_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _extract_results(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)]
    return []


def _normalize_search_results(payload: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(_extract_results(payload)):
        out.append(
            {
                "title": _coerce_str(item.get("title"), f"Result {i + 1}"),
                "url": _coerce_str(item.get("url")),
                "snippet": _coerce_str(
                    item.get("snippet")
                    or item.get("content")
                    or item.get("chunk")
                    or item.get("text")
                    or item.get("raw_content")
                ),
            }
        )
    return out


def _normalize_open_result(payload: Any, url: str) -> Dict[str, Any]:
    first: Dict[str, Any] = {}
    results = _extract_results(payload)
    if results:
        first = results[0]
    elif isinstance(payload, dict):
        first = payload

    images = first.get("images")
    if not isinstance(images, list):
        images = []

    return {
        "url": _coerce_str(first.get("url"), url),
        "title": _coerce_str(first.get("title")),
        "raw_content": _coerce_str(
            first.get("raw_content")
            or first.get("fulltext")
            or first.get("content")
            or first.get("text")
        ),
        "images": images,
    }


def _build_http_doc_url(doc_id: int) -> str:
    return f"{_HTTP_DOC_URL_PREFIX}{int(doc_id)}"


def _parse_http_doc_url(url: str) -> Optional[int]:
    if not isinstance(url, str) or not url.strip():
        return None
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme != "web-search" or parsed.netloc != "doc":
        return None
    try:
        return int(parsed.path.lstrip("/"))
    except (TypeError, ValueError):
        return None


def _build_http_doc_title(doc_file: Any, position: Any) -> str:
    file_part = _coerce_str(doc_file, "HTTP search document")
    if position is None:
        return file_part
    return f"{file_part} #{position}"


class BaseWebSearchProvider(ABC):
    """Backend-neutral search/open interface used by the LLM tool handlers."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def open(self, url: str) -> Dict[str, Any]:
        raise NotImplementedError


class TavilyWebSearchProvider(BaseWebSearchProvider):
    """Default provider backed by Tavily search + extract."""

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return tavily_search(query=query, max_results=k)

    def open(self, url: str) -> Dict[str, Any]:
        payload = tavily_extract(
            urls=url,
            extract_depth="basic",
            format="markdown",
            include_images=False,
        )
        return _normalize_open_result(payload, url)


class CommandWebSearchProvider(BaseWebSearchProvider):
    """
    Provider that shells out to an external command over stdio.

    The command must accept a JSON request on stdin with keys:
      - query: string
      - k: integer
      - return_fulltext: boolean

    And return JSON on stdout. Expected shapes:
      - {"results": [{title, url, snippet}, ...]}
      - {"results": [{title, url, raw_content}, ...]}

    This is intentionally small so the command can wrap an MCP server, an HTTP
    endpoint, or any custom search stack.
    """

    def __init__(self, command: Sequence[str], timeout: float = 60.0):
        if not command:
            raise WebSearchError("Command backend requires a non-empty command.")
        self.command = list(command)
        self.timeout = float(timeout)

    def _invoke(
        self, *, query: str, k: int, return_fulltext: bool
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        request = {
            "query": query,
            "k": int(k),
            "return_fulltext": bool(return_fulltext),
        }
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise WebSearchError(
                f"Configured web search command not found: {self.command[0]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise WebSearchError(
                f"Web search command timed out after {self.timeout:.1f}s."
            ) from exc
        except Exception as exc:
            raise WebSearchError(f"Web search command failed: {exc}") from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise WebSearchError(
                f"Web search command exited with code {completed.returncode}: {stderr}"
            )

        stdout = (completed.stdout or "").strip()
        if not stdout:
            raise WebSearchError("Web search command returned empty stdout.")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise WebSearchError(
                f"Web search command returned invalid JSON: {stdout[:200]}"
            ) from exc

        if isinstance(payload, dict) and payload.get("error"):
            raise WebSearchError(str(payload["error"]))

        return payload

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        payload = self._invoke(query=query, k=k, return_fulltext=False)
        return _normalize_search_results(payload)

    def open(self, url: str) -> Dict[str, Any]:
        payload = self._invoke(query=url, k=1, return_fulltext=True)
        return _normalize_open_result(payload, url)


class HttpWebSearchProvider(BaseWebSearchProvider):
    """Provider backed by a generic HTTP document-search service."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        base_url = str(base_url or "").strip()
        if not base_url:
            raise WebSearchError("HTTP web-search backend requires a non-empty base URL.")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise WebSearchError(
                f"HTTP web-search backend returned HTTP {exc.code} for {path}: {detail[:300]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise WebSearchError(
                f"HTTP web-search backend request failed for {path}: {exc.reason}"
            ) from exc
        except Exception as exc:
            raise WebSearchError(
                f"HTTP web-search backend request failed for {path}: {exc}"
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise WebSearchError(
                f"HTTP web-search backend returned invalid JSON for {path}: {body[:300]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise WebSearchError(
                f"HTTP web-search backend returned unexpected payload type for {path}: "
                f"{type(parsed).__name__}"
            )
        return parsed

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        payload = self._post_json("/search", {"query": query, "k": int(k)})
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(_extract_results(payload)):
            raw_doc_id = item.get("doc_id")
            try:
                doc_id = int(raw_doc_id) if raw_doc_id is not None else None
            except (TypeError, ValueError):
                doc_id = None
            normalized: Dict[str, Any] = {
                "title": _coerce_str(
                    item.get("title")
                    or _build_http_doc_title(
                        item.get("doc_file"),
                        item.get("doc_position", item.get("passage_position")),
                    )
                ),
                "url": _coerce_str(
                    item.get("url"),
                    _build_http_doc_url(doc_id) if doc_id is not None else "",
                ),
                "snippet": _coerce_str(
                    item.get("snippet")
                    or item.get("content")
                    or item.get("chunk")
                    or item.get("text")
                    or item.get("passage_text"),
                    f"Result {i + 1}",
                ),
            }
            for key in (
                "doc_id",
                "passage_id",
                "doc_file",
                "doc_position",
                "passage_position",
                "score",
                "rerank_score",
            ):
                if key in item:
                    normalized[key] = item.get(key)
            out.append(normalized)
        return out

    def open(self, url: str) -> Dict[str, Any]:
        doc_id = _parse_http_doc_url(url)
        if doc_id is None:
            payload = self._post_json("/open", {"url": url})
            return _normalize_open_result(payload, url)

        payload = self._post_json("/get_document", {"doc_id": doc_id})
        doc_url = _build_http_doc_url(doc_id)
        return {
            "url": doc_url,
            "title": _build_http_doc_title(
                payload.get("doc_file"), payload.get("doc_position")
            ),
            "raw_content": _coerce_str(
                payload.get("doc_text")
                or payload.get("raw_content")
                or payload.get("fulltext")
                or payload.get("content")
                or payload.get("text")
            ),
            "images": [],
            "doc_id": doc_id,
            "doc_file": payload.get("doc_file"),
            "doc_position": payload.get("doc_position"),
        }


@lru_cache(maxsize=1)
def get_web_search_provider() -> BaseWebSearchProvider:
    """
    Construct the configured web-search provider.

    Environment variables:
      - CAUSALDS_WEB_SEARCH_BACKEND=tavily|http|command
      - CAUSALDS_WEB_SEARCH_BASE_URL=http://host:8000
      - CAUSALDS_WEB_SEARCH_COMMAND="python path/to/adapter.py"
      - CAUSALDS_WEB_SEARCH_TIMEOUT_SEC=60
    """
    backend = (
        _resolve_setting("backend", "CAUSALDS_WEB_SEARCH_BACKEND", "tavily")
        or "tavily"
    ).strip().lower()

    if backend == "tavily":
        return TavilyWebSearchProvider()

    if backend == "http":
        base_url = _resolve_setting("base_url", "CAUSALDS_WEB_SEARCH_BASE_URL")
        if not base_url:
            raise WebSearchError(
                "CAUSALDS_WEB_SEARCH_BASE_URL must be set when "
                "CAUSALDS_WEB_SEARCH_BACKEND=http."
            )
        timeout = _resolve_timeout_setting(60.0)
        return HttpWebSearchProvider(base_url=base_url, timeout=timeout)

    if backend in {"command", "stdio"}:
        raw_command = (
            _resolve_setting("command", "CAUSALDS_WEB_SEARCH_COMMAND", "") or ""
        ).strip()
        if not raw_command:
            raise WebSearchError(
                "CAUSALDS_WEB_SEARCH_COMMAND must be set when "
                "CAUSALDS_WEB_SEARCH_BACKEND=command."
            )
        timeout = _resolve_timeout_setting(60.0)
        return CommandWebSearchProvider(
            command=shlex.split(raw_command),
            timeout=timeout,
        )

    raise WebSearchError(f"Unknown web search backend: {backend}")


def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Wrapper around Tavily Python client's search method.
    Returns a list of {title, url, snippet} dicts (simplified).

    Args:
        query: Search query string
        max_results: Maximum number of results (default 5)
        search_depth: "basic" or "advanced"
        include_domains: Optional list of domains to include
        exclude_domains: Optional list of domains to exclude
        days: Optional recency filter (results from last N days)
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; Tavily search will fail")
        raise TavilyError("Missing TAVILY_API_KEY")

    if not TavilyClient:
        raise TavilyError(
            "TavilyClient not installed; install with: pip install tavily-python"
        )

    logger.info(
        "Tavily search: query=%r, max_results=%d, depth=%s, domains_in=%s, domains_out=%s, days=%s",
        query[:100],
        max_results,
        search_depth,
        include_domains,
        exclude_domains,
        days,
    )

    client = TavilyClient(api_key=api_key)

    # Build params matching Tavily's official client.search signature
    params: Dict[str, Any] = {
        "query": query,
        "search_depth": search_depth,
        "max_results": int(max_results),
        "include_answer": False,
    }

    if include_domains:
        params["include_domains"] = include_domains
    if exclude_domains:
        params["exclude_domains"] = exclude_domains
    if days is not None:
        params["days"] = int(days)

    try:
        logger.debug("Calling Tavily API with params: %s", params)
        js: Dict[str, Any] = client.search(**params)  # type: ignore[arg-type]
        logger.debug("Tavily API returned: %d results", len(js.get("results", [])))
    except Exception as exc:
        logger.exception("Tavily client error for query=%r", query[:100])
        raise TavilyError(f"Tavily client error: {exc}") from exc

    return _normalize_search_results(js)


def tavily_extract(
    urls: Union[str, List[str]],
    *,
    include_images: bool = False,
    include_favicon: bool = False,
    include_usage: bool = False,
    extract_depth: str = "basic",
    format: str = "markdown",
    query: Optional[str] = None,
    chunks_per_source: int = 3,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Wrapper around TavilyClient.extract - to follow up on search results.
    Returns the raw Tavily response dict; downstream code can pick out
    results[0]['raw_content'].
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise TavilyError("Missing TAVILY_API_KEY")

    if not TavilyClient:
        raise TavilyError("TavilyClient not installed; pip install tavily-python")

    client = TavilyClient(api_key=api_key)
    if not hasattr(client, "extract"):
        raise TavilyError("TavilyClient.extract not available; upgrade tavily-python")

    if isinstance(urls, list) and len(urls) > 20:
        raise TavilyError("Too many URLs for Tavily extract (max 20).")

    kwargs: Dict[str, Any] = {"include_images": bool(include_images)}
    if include_favicon:
        kwargs["include_favicon"] = bool(include_favicon)
    if include_usage:
        kwargs["include_usage"] = bool(include_usage)
    if extract_depth:
        kwargs["extract_depth"] = str(extract_depth)
    if format:
        kwargs["format"] = str(format)
    if query:
        kwargs["query"] = str(query)
        kwargs["chunks_per_source"] = int(chunks_per_source)
    if timeout is not None:
        kwargs["timeout"] = float(timeout)

    try:
        # Newer SDKs: urls=...
        try:
            return client.extract(urls=urls, **kwargs)  # type: ignore[arg-type]
        except TypeError:
            # Older SDKs: positional
            return client.extract(urls, **kwargs)  # type: ignore[misc]
    except Exception as exc:
        raise TavilyError(f"Tavily extract error: {exc}") from exc
