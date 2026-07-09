# llm_client.py
# Unified, concise LLM client for CYAML verbalization with structured output,
# optional web search/open tools, and full tool-call tracing.

import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import httpx
from openai import BadRequestError, OpenAI

from .schemas import (
    VERBALIZATION_JSON_SCHEMA,
    WEB_SEARCH_TOOL_INSTRUCTION,
    _submit_tool_spec,
    _web_open_tool_spec,
    _web_search_tool_spec,
    build_user_prompt_single_shot,
)
from .utils import mapping_rows_incomplete, normalize_mapping_rows, parse_json_from_llm

# NOTE: this is to get around embedding variable mapping in a follow-up prompt
try:
    from .schemas import (
        MAPPING_ONLY_JSON_SCHEMA,
        build_user_prompt_passA_mapping,
        build_user_prompt_passB_story,
    )

    _HAS_MULTIPASS = True
except Exception:
    MAPPING_ONLY_JSON_SCHEMA = None  # type: ignore
    build_user_prompt_passA_mapping = None  # type: ignore
    build_user_prompt_passB_story = None  # type: ignore
    _HAS_MULTIPASS = False

try:
    from .web_search import WebSearchError, get_web_search_provider

    _HAS_WEB = True
except Exception:
    try:
        from web_search import WebSearchError, get_web_search_provider

        _HAS_WEB = True
    except Exception:
        get_web_search_provider = None
        WebSearchError = Exception
        _HAS_WEB = False

logger = logging.getLogger(__name__)
_DEFAULT_CONNECT_TIMEOUT_SEC = 5.0


# --------------------------
# Utilities
# --------------------------
def _safe_parse_json(text: str) -> Dict[str, Any]:
    """Parse strict JSON; attempt minimal repair if the model added stray text."""
    try:
        return parse_json_from_llm(text, silent=True)
    except ValueError:
        logger.debug("JSON parse failed even with lenient parser")
        return {"_parse_error": "Could not parse JSON", "_raw": text}


def _repair_tool_arguments(raw_args: str) -> Dict[str, Any]:
    """
    Attempt to repair malformed tool arguments from LLM output.

    Common issues:
    - Missing opening brace: query": "foo" -> {"query": "foo"}
    - Missing closing brace: {"query": "foo" -> {"query": "foo"}
    - Truncated strings: {"query": "foo -> {"query": "foo"}
    - Extra whitespace
    """
    if not raw_args or not raw_args.strip():
        return {}

    s = raw_args.strip()

    # Try direct parse first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Repair: missing opening brace
    if not s.startswith("{") and (":" in s or '"' in s):
        # Check if it looks like "key": "value" or key": "value"
        s_fixed = "{" + s
        if not s_fixed.rstrip().endswith("}"):
            s_fixed = s_fixed.rstrip().rstrip(",") + "}"
        try:
            return json.loads(s_fixed)
        except json.JSONDecodeError:
            pass

    # Repair: missing closing brace
    if s.startswith("{") and not s.rstrip().endswith("}"):
        s_fixed = s.rstrip().rstrip(",") + "}"
        try:
            return json.loads(s_fixed)
        except json.JSONDecodeError:
            pass

    # Repair: truncated string value - try to close it
    # Pattern: {"key": "value without closing quote
    if s.startswith("{"):
        # Count quotes - if odd, add one
        quote_count = s.count('"')
        if quote_count % 2 == 1:
            s_fixed = s + '"'
            if not s_fixed.rstrip().endswith("}"):
                s_fixed = s_fixed.rstrip().rstrip(",") + "}"
            try:
                return json.loads(s_fixed)
            except json.JSONDecodeError:
                pass

    # Last resort: try to extract query parameter with regex
    # Handles cases like: query": "meditation depth scale..."
    import re

    query_match = re.search(r'"?query"?\s*:\s*"([^"]*)"?', s)
    if query_match:
        query_value = query_match.group(1)
        logger.debug("Extracted query from malformed args: %s", query_value[:50])
        return {"query": query_value}

    # Give up
    logger.warning("Could not repair tool arguments: %s", s[:100])
    return {}


def _build_client_timeout(
    request_timeout_sec: Optional[float],
) -> Optional[httpx.Timeout]:
    """Translate a simple timeout value into the OpenAI/httpx timeout shape."""
    if request_timeout_sec is None:
        return None

    timeout_sec = float(request_timeout_sec)
    if timeout_sec <= 0:
        raise ValueError("request_timeout_sec must be positive")

    return httpx.Timeout(
        connect=_DEFAULT_CONNECT_TIMEOUT_SEC,
        read=timeout_sec,
        write=timeout_sec,
        pool=timeout_sec,
    )


def _normalize_variable_mapping(vm):
    """Backward-compatible shim around the shared normalization helper."""
    return normalize_mapping_rows(
        vm if isinstance(vm, list) else [], default_observed=None
    )


def _mapping_incomplete(parsed: dict) -> bool:
    if not isinstance(parsed, dict):
        return True
    normalized = _normalize_variable_mapping(parsed.get("variable_mapping", []))
    return mapping_rows_incomplete(normalized)


def _strip_reasoning_payload(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove reasoning payload from stored history to avoid context bloat."""
    cleaned: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        if msg.get("reasoning_details") or msg.get("reasoning"):
            msg_copy = msg.copy()
            msg_copy.pop("reasoning_details", None)
            msg_copy.pop("reasoning", None)
            cleaned.append(msg_copy)
        else:
            cleaned.append(msg)
    return cleaned


# NOTE:
# --------------------------
# Tool handlers
# --------------------------
def _handle_web_search(
    tc, args: Dict[str, Any], msgs: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Handle web_search tool call; returns (log_entry, updated_msgs).
    log_entry is just for diagnostics.
    """

    query = args.get("query", "")
    # Validate query is not empty
    if not query or not query.strip():
        log_entry = {
            "id": tc.id,
            "name": "web_search",
            "error": "Empty query provided",
            "args": args,
        }
        msgs = msgs + [
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(
                    {"error": "Query cannot be empty. Please provide a search query."}
                ),
            }
        ]
        return log_entry, msgs

    log_entry: Dict[str, Any] = {
        "tool_call_id": getattr(tc, "id", None),
        "name": "web_search",
        "args": args,
    }
    try:
        if not _HAS_WEB or get_web_search_provider is None:
            raise WebSearchError(
                "web_search not available (missing dependency or import)."
            )

        k = args.get("k", args.get("max_results", args.get("top_n", 5)))
        logger.info(
            "Calling web_search with query=%r, k=%d",
            args.get("query", ""),
            k,
        )

        # cursor is 1 + number of prior web_search calls
        cursor = 1 + len(_collect_web_search_calls(msgs))

        provider = get_web_search_provider()
        results = provider.search(query=args.get("query", ""), k=int(k))

        logger.info("web_search returned %d results", len(results))

        results_with_ids = []
        for i, r in enumerate(results):
            rr = dict(r)
            rr.setdefault("id", i + 1)
            results_with_ids.append(rr)

        content = json.dumps({"cursor": cursor, "results": results_with_ids})

        log_entry["results"] = results
        log_entry["error"] = None

    except Exception as ex:
        logger.warning("web_search tool error: %s", ex, exc_info=True)
        content = json.dumps({"error": str(ex), "results": []})
        log_entry["results"] = []
        log_entry["error"] = str(ex)

    # Build the tool response message
    tool_msg = {
        "role": "tool",
        "tool_call_id": tc.id,
        "content": content,
    }

    updated_msgs = msgs + [tool_msg]
    logger.debug("Added tool message with id=%s, content_len=%d", tc.id, len(content))
    return log_entry, updated_msgs


# NOTE: These are so web search results can be followed on
def _build_tool_call_id_to_name(msgs: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Tool response messages include only tool_call_id; recover tool name by scanning assistant tool_calls.
    """
    out: Dict[str, str] = {}
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tool_calls = m.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function")
            if not tc_id or not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if isinstance(name, str) and name:
                out[str(tc_id)] = name
    return out


def _collect_web_search_calls(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collect prior web_search tool outputs from the running message list.
    Returns list of dicts: {cursor, tool_call_id, results}
    """
    id_to_name = _build_tool_call_id_to_name(msgs)
    calls: List[Dict[str, Any]] = []

    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        tc_id = m.get("tool_call_id")
        if not tc_id or id_to_name.get(str(tc_id)) != "web_search":
            continue

        raw = m.get("content") or ""
        parsed = _safe_parse_json(raw)

        # Backward compatible: allow tool content to be either {"results":[...]} or just [...]
        results: List[Any] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            results = parsed["results"]
        elif isinstance(parsed, list):
            results = parsed

        calls.append(
            {
                "cursor": len(calls) + 1,
                "tool_call_id": str(tc_id),
                "results": results,
            }
        )

    return calls


def _resolve_web_open_url(
    args: Dict[str, Any], msgs: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Resolve URL for web_open from either:
      - args['url'], or
      - args['cursor'] + args['id'] referencing prior web_search outputs
    """
    url = args.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()

    # Cursor/id reference
    try:
        cursor = int(args.get("cursor"))
        idx = int(args.get("id"))
    except Exception:
        return None

    calls = _collect_web_search_calls(msgs)
    if cursor < 1 or cursor > len(calls):
        return None

    results = calls[cursor - 1].get("results")
    if not isinstance(results, list) or not results:
        return None

    if idx < 1 or idx > len(results):
        return None

    r = results[idx - 1]
    if not isinstance(r, dict):
        return None

    resolved = r.get("url")
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()

    return None


def _handle_web_open(
    tc, args: Dict[str, Any], msgs: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Basic web_open implementation using the configured web-search provider.
    Returns tool message content as JSON: {url, raw_content, images}
    """
    log_entry: Dict[str, Any] = {
        "tool_call_id": getattr(tc, "id", None),
        "name": "web_open",
        "args": args,
    }

    try:
        if not _HAS_WEB or get_web_search_provider is None:
            raise WebSearchError("web_open not available (missing dependency/import).")

        url = _resolve_web_open_url(args, msgs)
        if not url:
            raise WebSearchError(
                "Could not resolve URL. Provide url=... or (cursor,id)."
            )

        logger.info("Calling web_open on url=%s", url)

        provider = get_web_search_provider()
        opened = provider.open(url)
        raw_content = opened.get("raw_content") or ""
        images = opened.get("images") or []

        # Truncate to avoid blowing up context windows
        max_chars = int(args.get("max_chars", 8000))
        max_chars = max(500, min(max_chars, 50000))
        if isinstance(raw_content, str) and len(raw_content) > max_chars:
            raw_content = raw_content[:max_chars] + "\n\n[TRUNCATED]"

        payload = {
            "url": opened.get("url") or url,
            "title": opened.get("title"),
            "raw_content": raw_content,
            "images": images,
        }
        content = json.dumps(payload)

        log_entry["error"] = None
        log_entry["url"] = payload["url"]
        log_entry["raw_content_len"] = len(raw_content)

    except Exception as ex:
        content = json.dumps({"error": str(ex), "url": None, "raw_content": ""})
        log_entry["error"] = str(ex)

    tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": content}
    return log_entry, msgs + [tool_msg]


# NOTE: this is mostly a fallback for structured output extraction
def _handle_submit(
    tc, args: Dict[str, Any], msgs: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Handle the 'submit' tool call by extracting structured output.
    Returns (log_entry, updated_messages) to match other handlers.

    Gets used if:
    - **Provider/model doesn't accept `response_format`** (or you switched it off mid-run): letting the model "finalize" via a tool call keeps you schema-aligned anyway.
    - **Mid-run fallback** when you're already doing tool use: some loops prefer a clear "I'm done" signal; having the model call `submit` gives you a deterministic termination point and a structured payload in `function.arguments`
    - **Strict schema enforcement through tools**: Structured Outputs also exists in the **function-calling form** (set `strict: true` on the tool schema). In that flavor, a "submit" tool *is* the structured-output mechanism, not a fallback.
    """
    content = tc.function.arguments
    log_entry = {
        "name": "submit",
        "args": args,
        "content": content,
    }

    # NOTE: Do not include "name" on the tool message. (Name lives on the assistant's tool call; the tool message pairs by id, not name.
    # Add tool response message (even though we'll break after)
    tool_msg = {
        "role": "tool",
        "tool_call_id": tc.id,
        # "name"        : "submit",
        "content": "Submission received.",
    }

    updated_msgs = msgs + [tool_msg]

    # Store parsed content in log for extraction
    log_entry["parsed_content"] = content

    return log_entry, updated_msgs


def _handle_unknown_tool(
    tc, args: Dict[str, Any], name: str, msgs: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Handle unknown tool call; returns (log_entry, updated_msgs)."""
    logger.warning("Unknown tool requested: %s", name)
    log_entry: Dict[str, Any] = {
        "tool_call_id": getattr(tc, "id", None),
        "name": name,
        "args": args,
        "results": [],
        "error": f"Unknown tool: {name}",
    }
    updated_msgs = msgs + [
        {
            "role": "tool",
            "tool_call_id": tc.id,
            # "name": name,
            "content": json.dumps({"error": f"Unknown tool {name}"}),
        }
    ]
    return log_entry, updated_msgs


# Tool registry: maps tool name to handler function
_TOOL_HANDLERS = {
    "web_search": _handle_web_search,
    "web_open": _handle_web_open,
    "submit": _handle_submit,
}


# --------------------------
# Client and Session classes
# --------------------------


class ChatSession:
    """
    Holds conversation state (messages) and provides structured verb methods.

    Public methods:
      - chat(...)  # Now supports optional tool calling and reasoning
      - chat_structured_verbalization(...)
      - chat_structured_multipass(...)
    """

    def __init__(
        self,
        client: "LLMClient",
        session_id: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.client = client
        self.session_id = session_id
        self.default_model = model or client.default_model
        self.messages: List[Dict[str, str]] = []

        if system_prompt:
            self.system_prompt = system_prompt
            self.messages.append({"role": "system", "content": self.system_prompt})

        logger.debug(
            "ChatSession created (id=%s, model=%s, has_system=%s)",
            session_id,
            self.default_model,
            bool(system_prompt),
        )

    # --------------------------
    # Public: Basics
    # --------------------------

    # NOTE: expand/remove as needed
    def chat(
        self,
        user_prompt: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]] | bool] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None,
        max_tool_loops: int = 5,
        reasoning: Optional[Dict[str, Any] | bool] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Plain chat with optional tool calling and reasoning support.

        Args:
            user_prompt: User message content
            model: Model to use (falls back to session default)
            tools: Tool usage mode:
                  - False/None: No tools
                  - True: Use built-in tools from _TOOL_HANDLERS
                  - List[Dict]: Custom tool specifications (OpenAI format)
            tool_handlers: Optional dict mapping tool name -> handler function.
                          Handler signature: (tool_call, args, messages) -> (log_entry, updated_messages)
                          If not provided and tools=True, uses built-in handlers (_TOOL_HANDLERS).
                          Required if tools is a list of custom specifications.
            max_tool_loops: Maximum number of tool call iterations (default 5)
            reasoning: Optional reasoning configuration. Can be:
                      - True: enable with defaults
                      - dict: e.g., {"max_tokens": 1000} or {"effort": "high"}
                      - None/False: disabled (default)
            **kwargs: Additional arguments passed to the API (e.g., temperature)

        Returns:
            Tuple of (assistant_text, raw_response_dict)
            raw_response_dict includes:
              - 'tool_trace' if tools were called
              - 'reasoning' if reasoning was enabled and returned
              - 'reasoning_trace' list of reasoning from each completion call
        """
        msgs = self.messages + [{"role": "user", "content": user_prompt}]
        use_model = model or self.default_model

        logger.debug(
            "ChatSession.chat: model=%s, messages=%d, tools=%s, reasoning=%s",
            use_model,
            len(msgs),
            type(tools).__name__ if tools is not None else "None",
            bool(reasoning),
        )

        # Build request kwargs
        request_kwargs = dict(kwargs)

        # Determine tool configuration
        actual_tools: Optional[List[Dict[str, Any]]] = None
        actual_handlers: Dict[str, Callable] = {}

        if tools is True:
            # Use built-in tools (excluding 'submit' which is for structured outputs only)
            actual_tools = [_web_search_tool_spec(), _web_open_tool_spec()]
            actual_handlers = {
                "web_search": _TOOL_HANDLERS["web_search"],
                "web_open": _TOOL_HANDLERS["web_open"],
            }
        elif isinstance(tools, list):
            # Use custom tools
            actual_tools = tools
            if tool_handlers:
                actual_handlers = tool_handlers
            else:
                # Fall back to built-in handlers for any matching tool names
                actual_handlers = _TOOL_HANDLERS.copy()
        # else: tools is False/None, no tools

        if actual_tools:
            request_kwargs["tools"] = actual_tools
            request_kwargs.setdefault("tool_choice", "auto")

        # Handle reasoning configuration
        if reasoning:
            if reasoning is True:
                # Default reasoning config
                request_kwargs.setdefault("extra_body", {})["reasoning"] = {
                    "max_tokens": 1000
                }
            elif isinstance(reasoning, dict):
                request_kwargs.setdefault("extra_body", {})["reasoning"] = reasoning

        # Initial completion
        resp = self.client.create_completion(
            model=use_model, messages=msgs, **request_kwargs
        )

        tool_trace: List[Dict[str, Any]] = []
        reasoning_trace: List[Optional[str]] = []
        loops = 0
        ghost_tool_call_retry = False

        # Some providers signal tool_calls without providing any tool_calls.
        # If that happens and the content is empty, retry once without tools.
        if actual_tools:
            choice = resp.choices[0]
            msg = choice.message
            finish_reason = getattr(choice, "finish_reason", None)
            tool_calls = getattr(msg, "tool_calls", None) or []
            if (
                finish_reason == "tool_calls"
                and not tool_calls
                and not (msg.content or "").strip()
            ):
                logger.warning(
                    "Tool calls requested but none provided; retrying without tools."
                )
                ghost_tool_call_retry = True
                # Add a small hint to proceed without tools
                msgs = msgs + [
                    {
                        "role": "user",
                        "content": "Tools are unavailable for this request. Return ONLY the JSON object.",
                    }
                ]
                request_kwargs_no_tools = dict(request_kwargs)
                request_kwargs_no_tools.pop("tools", None)
                request_kwargs_no_tools.pop("tool_choice", None)
                resp = self.client.create_completion(
                    model=use_model, messages=msgs, **request_kwargs_no_tools
                )
                actual_tools = None

        # Tool loop (only if tools are enabled)
        while actual_tools:
            choice = resp.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None) or []

            # Capture reasoning from this response
            msg_reasoning = getattr(msg, "reasoning", None)
            msg_reasoning_details = getattr(msg, "reasoning_details", None)
            if msg_reasoning:
                reasoning_trace.append(msg_reasoning)

            # No tool calls or max loops reached - we're done
            if not tool_calls or loops >= max_tool_loops:
                break

            logger.debug("Tool loop %d: %d tool_calls", loops + 1, len(tool_calls))
            loops += 1

            # Build assistant message with tool_calls (and reasoning_details if present)
            assistant_turn: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }

            # Preserve reasoning_details in context for multi-turn coherence
            if msg_reasoning_details:
                assistant_turn["reasoning_details"] = msg_reasoning_details

            # Preserve raw reasoning in context for multi-turn coherence (mid-tool-use)
            if msg_reasoning:
                assistant_turn["reasoning"] = msg_reasoning

            if tool_calls:
                tool_calls_payload = []
                for tc in tool_calls:
                    fn = tc.function
                    tool_calls_payload.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": fn.name,
                                "arguments": fn.arguments,
                            },
                        }
                    )
                assistant_turn["tool_calls"] = tool_calls_payload

            msgs = msgs + [assistant_turn]

            # Process each tool call
            for tc in tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception as e:
                    # Use robust repair function for malformed tool arguments
                    args = _repair_tool_arguments(raw_args)
                    if not args:
                        logger.warning(
                            "Failed to parse tool arguments: %s; raw=%r",
                            e,
                            tc.function.arguments,
                        )

                logger.info("Executing tool: %s with args: %s", name, args)

                if name in actual_handlers:
                    log_entry, msgs = actual_handlers[name](tc, args, msgs)
                    tool_trace.append(log_entry)
                else:
                    # Unknown tool - add error response
                    log_entry, msgs = _handle_unknown_tool(tc, args, name, msgs)
                    tool_trace.append(log_entry)

            # Re-query the model with tool results
            resp = self.client.create_completion(
                model=use_model, messages=msgs, **request_kwargs
            )

        # Extract final response details
        final_msg = resp.choices[0].message
        text = final_msg.content or ""
        final_reasoning = getattr(final_msg, "reasoning", None)

        # Capture final reasoning if present
        if final_reasoning and (
            not reasoning_trace or reasoning_trace[-1] != final_reasoning
        ):
            reasoning_trace.append(final_reasoning)

        # Build final assistant message for context
        final_assistant_turn: Dict[str, Any] = {"role": "assistant", "content": text}

        # Update session messages
        # Strip reasoning_details and raw reasoning from intermediate messages to avoid bloating history
        self.messages = _strip_reasoning_payload(msgs) + [final_assistant_turn]

        logger.debug(
            "ChatSession.chat: received %d chars, tool_loops=%d, has_reasoning=%s",
            len(text),
            loops,
            bool(final_reasoning),
        )

        # Build raw response dict
        raw = resp.to_dict() if hasattr(resp, "to_dict") else {"_raw_object": str(resp)}

        # Add tool info
        if tool_trace:
            raw["tool_trace"] = tool_trace
            raw["used_tools"] = True
        else:
            raw["used_tools"] = False

        if ghost_tool_call_retry:
            raw["ghost_tool_call_retry"] = True

        # Add reasoning info
        if final_reasoning:
            raw["reasoning"] = final_reasoning
        if reasoning_trace:
            raw["reasoning_trace"] = reasoning_trace

        return text, raw

    def stream_chat(
        self, user_message: str, model: Optional[str] = None, **kwargs
    ) -> Generator[str, None, None]:
        self.messages.append({"role": "user", "content": user_message})
        use_model = model or self.default_model
        if not use_model:
            raise ValueError(
                "Model must be specified either when creating the session or when calling stream_chat()."
            )
        logger.debug("ChatSession.stream_chat: model=%s", use_model)
        accumulator = ""
        for chunk in self.client.stream_chat(
            messages=self.messages, model=use_model, **kwargs
        ):
            accumulator += chunk
            yield chunk
        if accumulator:
            self.messages.append({"role": "assistant", "content": accumulator})
        logger.debug("ChatSession.stream_chat: streamed %d chars", len(accumulator))

    def get_messages(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def clear(self, keep_system: bool = True) -> None:
        if keep_system:
            system_msgs = [m for m in self.messages if m.get("role") == "system"]
            self.messages = system_msgs
        else:
            self.messages = []

    # --------------------------
    # Public: One-shot
    # --------------------------
    def chat_structured_verbalization(
        self,
        cyaml: Optional[str] = None,
        user_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        enable_web: bool = True,
        use_json_schema: bool = True,
        max_tool_loops: int = 3,
        reasoning: Optional[Dict[str, Any] | bool] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        One-shot: naming + story + causal justifications => VERBALIZATION_JSON_SCHEMA.

        Args:
            cyaml: CYAML graph representation
            user_prompt: Custom user prompt (if None, built from cyaml)
            model: Model to use
            temperature: Sampling temperature (if None, uses API default)
            enable_web: Enable web search tool
            use_json_schema: Use JSON schema response format
            max_tool_loops: Maximum tool call iterations
            reasoning: Optional reasoning configuration:
                      - True: enable with defaults
                      - dict: e.g., {"max_tokens": 1000} or {"effort": "high"}
                      - None/False: disabled (default)

        Returns:
            Tuple of (parsed_json, raw_response_with_trace)
        """
        logger.info(
            "chat_structured_verbalization (web=%s, json_schema=%s, reasoning=%s)",
            enable_web,
            use_json_schema,
            bool(reasoning),
        )
        user = build_user_prompt_single_shot(cyaml) if not user_prompt else user_prompt
        # NOTE: not needed - the enable_web flag is sufficient to inform the model
        # extra_hint = (
        #     "You may call the `web_search` tool to gather background ideas; do not copy; "
        #     "return only the structured JSON when done."
        #     if enable_web
        #     else None
        # )

        return self._complete_with_schema_and_tools(
            user_prompt=user,
            schema=VERBALIZATION_JSON_SCHEMA,
            model=model,
            temperature=temperature,
            enable_web=enable_web,
            use_json_schema=use_json_schema,
            max_tool_loops=max_tool_loops,
            submit_name="submit",
            # extra_user_hint_for_tools=extra_hint,
            reasoning=reasoning,
        )

    # --------------------------
    # Public: Multipass
    # --------------------------
    def chat_variable_mapping(
        self,
        cyaml: Optional[str] = None,
        user_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        enable_web: bool = True,
        use_json_schema: bool = True,
        max_tool_loops: int = 3,
        reasoning: Optional[Dict[str, Any] | bool] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Pass A: Generate variable mapping only (names + optional proposed_domain).

        Returns (mapping_json, raw_response_with_trace)

        Args:
            cyaml: CYAML graph representation
            user_prompt: Custom user prompt (if None, built from cyaml)
            model: Model to use
            temperature: Sampling temperature (if None, uses API default)
            enable_web: Enable web search tool
            use_json_schema: Use JSON schema response format
            max_tool_loops: Maximum tool call iterations
            reasoning: Optional reasoning configuration:
                      - True: enable with defaults
                      - dict: e.g., {"max_tokens": 1000} or {"effort": "high"}
                      - None/False: disabled (default)

        Requires schemas.py to provide:
          - MAPPING_ONLY_JSON_SCHEMA
          - build_user_prompt_passA_mapping(cyaml)
        """
        if not _HAS_MULTIPASS or MAPPING_ONLY_JSON_SCHEMA is None:
            raise RuntimeError(
                "Variable mapping requires MAPPING_ONLY_JSON_SCHEMA in schemas.py"
            )

        logger.info(
            "chat_variable_mapping start (web=%s, json_schema=%s, reasoning=%s)",
            enable_web,
            use_json_schema,
            bool(reasoning),
        )

        if user_prompt is None:
            if cyaml is None:
                raise ValueError("Either 'cyaml' or 'user_prompt' must be provided")
            user_prompt = build_user_prompt_passA_mapping(cyaml)

        mapping_json, raw = self._complete_with_schema_and_tools(
            user_prompt=user_prompt,
            schema=MAPPING_ONLY_JSON_SCHEMA,
            model=model,
            temperature=temperature,
            enable_web=enable_web,
            use_json_schema=use_json_schema,
            max_tool_loops=max_tool_loops,
            submit_name="submit",
            reasoning=reasoning,
        )

        logger.info(
            "chat_variable_mapping done (mapping_keys=%s)",
            list(mapping_json.keys()) if isinstance(mapping_json, dict) else None,
        )

        return mapping_json, raw

    def chat_story_given_mapping(
        self,
        cyaml: Optional[str] = None,
        mapping_json: Optional[Dict[str, Any]] = None,
        user_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        enable_web: bool = True,
        use_json_schema: bool = True,
        max_tool_loops: int = 3,
        submit_name: str = "submit",
        reasoning: Optional[Dict[str, Any] | bool] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        DEPRECATED: Use verbalization_story.run_story_generation() instead.

        Pass B: Generate full verbalization (story + mapping + justifications) given a mapping.
        Returns (verbalization_json, raw_response_with_trace)

        Requires OpenRouter structured outputs.
        """
        from .verbalization_story import chat_story_given_mapping

        return chat_story_given_mapping(
            session=self,
            cyaml=cyaml,
            mapping_json=mapping_json,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            enable_web=enable_web,
            use_json_schema=use_json_schema,
            max_tool_loops=max_tool_loops,
            submit_name=submit_name,
            reasoning=reasoning,
        )

    def chat_structured_multipass(
        self,
        cyaml: Optional[str] = None,
        user_promptA: Optional[str] = None,
        user_promptB: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        enable_web: bool = True,
        use_json_schema: bool = True,
        max_tool_loops: int = 3,
        submit_name: str = "submit",
        reasoning: Optional[Dict[str, Any] | bool] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Two-step pipeline:
          Pass A: return mapping-only JSON (names + optional proposed_domain)
          Pass B: return full VERBALIZATION_JSON_SCHEMA (story + mapping + justifications)

        Returns (passA_json, passB_json, passB_raw_with_trace)

        Uses OpenRouter structured outputs.
        """

        logger.info(
            "chat_structured_multipass start (web=%s, json_schema=%s, reasoning=%s)",
            enable_web,
            use_json_schema,
            bool(reasoning),
        )

        # Pass A: mapping only (uses structured outputs)
        passA_json, _ = self.chat_variable_mapping(
            cyaml=cyaml,
            user_prompt=user_promptA,
            model=model,
            temperature=temperature,
            enable_web=enable_web,
            use_json_schema=use_json_schema,
            max_tool_loops=max_tool_loops,
            reasoning=reasoning,
        )

        # Pass B: full verbalization given mapping (uses structured outputs)
        passB_json, passB_raw = self.chat_story_given_mapping(
            cyaml=cyaml,
            mapping_json=passA_json,
            user_prompt=user_promptB,
            model=model,
            temperature=temperature,
            enable_web=enable_web,
            use_json_schema=use_json_schema,
            max_tool_loops=max_tool_loops,
            submit_name=submit_name,
            reasoning=reasoning,
        )

        logger.info(
            "chat_structured_multipass done (passA_keys=%s, used_web=%s)",
            list(passA_json.keys()) if isinstance(passA_json, dict) else None,
            passB_raw.get("used_web") if isinstance(passB_raw, dict) else None,
        )

        return passA_json, passB_json, passB_raw

    # Private:
    # --------------------------
    # Unified engine
    # --------------------------
    def _complete_with_schema_and_tools(
        self,
        *,
        user_prompt,
        schema,
        model=None,
        temperature=None,
        enable_web=False,
        use_json_schema=True,
        max_tool_loops=3,
        submit_name="submit",
        extra_user_hint_for_tools=None,
        reasoning: Optional[Dict[str, Any] | bool] = None,
        # internal: allow soft fallback to add submit when using JSON-Schema
        add_submit_even_if_json_schema: bool = False,
        _is_soft_retry: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Core routine used by both one-shot and multipass flows.
        - Tries JSON Schema + (optional) tools first.
        - Falls back to function schema + tools if provider rejects response_format.
        - Supports optional reasoning with context preservation.
        - Returns (parsed_json, raw_response_dict_with_trace).
        """
        # Optional line to remind the model that tools are available

        if enable_web:
            user_prompt = f"{user_prompt}\n\n{WEB_SEARCH_TOOL_INSTRUCTION}"

        if extra_user_hint_for_tools:
            user_prompt = f"{user_prompt}\n\n{extra_user_hint_for_tools}"

        user_prompt += " Return only the structured JSON when done."

        msgs = self.messages + [{"role": "user", "content": user_prompt}]
        tool_trace: List[Dict[str, Any]] = []
        reasoning_trace: List[Optional[str]] = []

        logger.info(
            "Structured completion start (schema=%s, web=%s, json_schema=%s, temp=%s, reasoning=%s)",
            schema.get("title", "") or schema.get("name", ""),
            enable_web,
            use_json_schema,
            temperature if temperature is not None else "default",
            bool(reasoning),
        )

        # --- First attempt: JSON-Schema only (no submit tool) ---
        try:
            kwargs = self._compose_kwargs(
                schema,
                use_json_schema,
                enable_web,
                temperature,
                submit_name,
                add_submit_even_if_json_schema=add_submit_even_if_json_schema,
                reasoning=reasoning,
            )
            resp = self.client.create_completion(
                model=model or self.default_model,
                messages=msgs,
                **kwargs,
            )
            used_json_schema = bool(use_json_schema)
            logger.debug(
                "First completion call succeeded (used_json_schema=%s)",
                used_json_schema,
            )
        except BadRequestError as e:
            # If provider rejects JSON schema, fallback to function schema + tools
            if (
                "invalid_json_schema" in str(e).lower()
                or "text.format.schema" in str(e).lower()
            ):
                logger.warning(
                    "JSON schema rejected by provider; falling back to function-calling schema: %s",
                    e,
                )
                kwargs = self._compose_kwargs(
                    schema=schema,
                    use_json_schema=False,
                    enable_web=enable_web,
                    temperature=temperature,
                    submit_name=submit_name,
                    reasoning=reasoning,
                )
                resp = self.client.create_completion(
                    model=model or self.default_model,
                    messages=msgs,
                    **kwargs,
                )
                used_json_schema = False
                logger.warning("Couldn't use json schema")
            else:
                logger.exception("Completion call failed")
                raise

        # ---- Tool loop
        loops = 0
        while True:
            choice = resp.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None) or []

            # Capture reasoning from this response
            msg_reasoning = getattr(msg, "reasoning", None)
            msg_reasoning_details = getattr(msg, "reasoning_details", None)
            if msg_reasoning:
                reasoning_trace.append(msg_reasoning)

            if not tool_calls or loops >= max_tool_loops:
                break

            logger.debug("Tool loop %d: %d tool_calls", loops + 1, len(tool_calls))
            loops += 1

            # NOTE: Assistant tool calls
            # Build assistant turn with tool_calls and reasoning_details
            assistant_turn: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }

            # Preserve reasoning_details in context for multi-turn coherence
            if msg_reasoning_details:
                assistant_turn["reasoning_details"] = msg_reasoning_details

            # Preserve raw reasoning in context for multi-turn coherence (mid-tool-use)
            if msg_reasoning:
                assistant_turn["reasoning"] = msg_reasoning

            if getattr(msg, "tool_calls", None):
                tool_calls_payload = []
                for _tc in msg.tool_calls:
                    fn = _tc.function
                    tool_calls_payload.append(
                        {
                            "id": _tc.id,
                            "type": "function",
                            "function": {
                                "name": fn.name,
                                "arguments": fn.arguments,  # already a JSON string from the provider
                            },
                        }
                    )
                assistant_turn["tool_calls"] = tool_calls_payload

            msgs = msgs + [assistant_turn]
            logger.debug(
                "Added assistant turn with tool_calls: %s, has_reasoning_details: %s",
                [
                    {"id": tc["id"], "name": tc["function"]["name"]}
                    for tc in assistant_turn.get("tool_calls", [])
                ],
                bool(msg_reasoning_details),
            )

            should_break = False

            for tc in tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception as e:
                    # Use robust repair function for malformed tool arguments
                    args = _repair_tool_arguments(raw_args)
                    if not args:
                        logger.warning("Failed to parse tool arguments: %s", e)

                logger.info("Executing tool: %s with args: %s", name, args)

                # NOTE: this is for structured output if json_schema fails
                # (function-calling mode)
                if name == "submit":
                    log_entry, msgs = _TOOL_HANDLERS[name](tc, args, msgs)
                    tool_trace.append(log_entry)
                    # IMPORTANT: handlers must receive the running message list and return an updated list
                    logger.debug("Submit tool called, breaking loop")
                    should_break = True
                    break
                elif name in _TOOL_HANDLERS:
                    # Regular tool (e.g., web_search) updates the message list.
                    log_entry, msgs = _TOOL_HANDLERS[name](tc, args, msgs)
                    tool_trace.append(log_entry)
                    logger.debug("Tool %s executed successfully", name)
                else:
                    # Unknown tool -> record a tool output so the model can recover
                    log_entry, msgs = _handle_unknown_tool(tc, args, name, msgs)
                    tool_trace.append(log_entry)
                    logger.warning("Unknown tool called: %s", name)

            if should_break:
                break

            # Re-ask the model with tool outputs
            # NOTE: Further tool calls enabled here
            logger.debug(
                "Re-asking model with %d messages (including tool responses)", len(msgs)
            )
            try:
                resp = self.client.create_completion(
                    model=model or self.default_model,
                    messages=msgs,
                    **self._compose_kwargs(
                        schema,
                        use_json_schema,
                        enable_web,
                        temperature,
                        submit_name,
                        reasoning=reasoning,
                    ),
                )
                logger.debug(
                    "Re-ask completion succeeded, finish_reason=%s",
                    resp.choices[0].finish_reason if resp.choices else "unknown",
                )
            except BadRequestError as e:
                # If schema now fails mid-run, drop back to function schema while preserving tools
                if used_json_schema and (
                    "invalid_json_schema" in str(e).lower()
                    or "text.format.schema" in str(e).lower()
                ):
                    logger.warning(
                        "Schema rejected mid-run; switching to function-calling schema"
                    )
                    used_json_schema = False
                    resp = self.client.create_completion(
                        model=model or self.default_model,
                        messages=msgs,
                        **self._compose_kwargs(
                            schema,
                            use_json_schema,
                            enable_web,
                            temperature,
                            submit_name,
                            reasoning=reasoning,
                        ),
                    )
                else:
                    logger.exception("Completion after tool outputs failed")
                    raise

        # ---- Extract final response details
        final_msg = resp.choices[0].message
        final_reasoning = getattr(final_msg, "reasoning", None)

        # Capture final reasoning if present and not already captured
        if final_reasoning and (
            not reasoning_trace or reasoning_trace[-1] != final_reasoning
        ):
            reasoning_trace.append(final_reasoning)

        # ---- Parse final content
        # Abusing try/except a bit for flow control - first case is no JSON schema
        # second is the mainstream or the fallback
        try:
            submit_entry = next(t for t in tool_trace if t.get("name") == "submit")
            parsed = _safe_parse_json(submit_entry["parsed_content"])
        except:
            # Fallback to content or the final message is already JSON schema
            content = resp.choices[0].message.content or ""
            parsed = _safe_parse_json(content)

        # Ensure required top-level keys exist (won't hurt if present)
        if isinstance(parsed, dict):
            parsed.setdefault("story", "")
            parsed.setdefault("variable_mapping", [])
            parsed.setdefault("causal_justifications", {})
            parsed["variable_mapping"] = _normalize_variable_mapping(
                parsed["variable_mapping"]
            )

        # Build a raw dict and attach trace
        raw = resp.to_dict() if hasattr(resp, "to_dict") else {"_raw_object": str(resp)}
        raw["tool_trace"] = tool_trace
        raw["used_web"] = any(
            t.get("name") == "web_search" and not t.get("error") for t in tool_trace
        )

        # Add reasoning info to raw response
        if final_reasoning:
            raw["reasoning"] = final_reasoning
        if reasoning_trace:
            raw["reasoning_trace"] = reasoning_trace

        # Optional external hook
        if callable(self.client.on_tool_trace):
            try:
                self.client.on_tool_trace(tool_trace)
            except Exception:
                logger.debug("on_tool_trace callback raised; ignoring", exc_info=True)

        # Build final assistant message for context with reasoning_details
        final_assistant_turn: Dict[str, Any] = {
            "role": "assistant",
            "content": json.dumps(parsed),
        }
        # NOTE: We do not preserve reasoning_details in the final message for history
        # if final_reasoning_details:
        #     final_assistant_turn["reasoning_details"] = final_reasoning_details

        # NOTE: Keep continuity (assistant turn)
        # Strip reasoning_details and raw reasoning from intermediate messages to avoid bloating history
        self.messages = _strip_reasoning_payload(msgs) + [final_assistant_turn]

        logger.info(
            "Structured completion done (used_web=%s, used_json_schema=%s, has_reasoning=%s)",
            raw["used_web"],
            used_json_schema,
            bool(final_reasoning),
        )

        # Soft fallback: only when JSON-Schema is enabled, mapping is required by schema,
        # mapping is incomplete, and we haven't retried yet.
        title = (schema.get("title") or "").lower()
        needs_mapping = (
            "verbalization" in title
            or "mappingonly" in title
            or "mapping_only" in title
        )
        if (
            use_json_schema
            and needs_mapping
            and _mapping_incomplete(parsed)
            and not _is_soft_retry
        ):
            hint = "If a tool named `submit` is available, please return the final JSON by calling it; otherwise return a strictly valid JSON object that satisfies the schema."
            # Second attempt: still JSON-Schema, but make submit available and add a brief hint.
            return self._complete_with_schema_and_tools(
                user_prompt=user_prompt,
                schema=schema,
                model=model,
                temperature=temperature,
                enable_web=enable_web,
                use_json_schema=use_json_schema,
                max_tool_loops=max_tool_loops,
                submit_name=submit_name,
                extra_user_hint_for_tools=hint,
                reasoning=reasoning,
                add_submit_even_if_json_schema=True,
                _is_soft_retry=True,
            )

        return parsed, raw

    # NOTE: pure helper, no default args
    @staticmethod
    def _compose_kwargs(
        schema,
        use_json_schema,
        enable_web,
        temperature,
        submit_name,
        *,
        add_submit_even_if_json_schema=False,
        reasoning: Optional[Dict[str, Any] | bool] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}

        # Only add temperature if explicitly provided
        if temperature is not None:
            kwargs["temperature"] = temperature

        tools: List[Dict[str, Any]] = []

        if enable_web:
            tools.append(_web_search_tool_spec())
            tools.append(_web_open_tool_spec())

        if use_json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "StructuredOutput",
                    "schema": schema,
                    "strict": True,
                },
            }
            if add_submit_even_if_json_schema:
                tools.append(_submit_tool_spec(schema=schema, submit_name=submit_name))
            if tools:
                kwargs["tools"] = tools
                # NOTE: allegedly some routers behave better when that flag is present alongside tools.
                kwargs["tool_choice"] = "auto"
                # NOTE: allegedly sequential tool use can reduce confusion for models that might otherwise fire multiple empty calls.
                kwargs["parallel_tool_calls"] = False
        else:
            # tool/function-driven schema with optional web tool
            tools.append(_submit_tool_spec(schema=schema, submit_name=submit_name))
            kwargs["tools"] = tools

        # Handle reasoning configuration
        if reasoning:
            if reasoning is True:
                # Default reasoning config
                kwargs.setdefault("extra_body", {})["reasoning"] = {"max_tokens": 1000}
            elif isinstance(reasoning, dict):
                kwargs.setdefault("extra_body", {})["reasoning"] = reasoning

        return kwargs


class LLMClient:
    """
    Lightweight wrapper around OpenAI/Router clients that supports:
      - JSON Schema + tool calling
      - auto-fallback to function-calling schema
      - session reuse with a system prompt
    """

    def __init__(
        self,
        provider: str = "openrouter",  # or "openai"
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        request_timeout_sec: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):

        if provider == "openrouter":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
        elif provider == "openai":
            api_key = api_key or os.environ["OPENAI_API_KEY"]
            base_url = base_url or None  # default OpenAI
        else:
            raise ValueError(f"Unknown provider: {provider}")

        timeout = _build_client_timeout(request_timeout_sec)

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.api_key = api_key  # Store for downstream use (e.g., audit)
        self.default_model = (
            default_model or os.getenv("OPENAI_MODEL") or "openai/gpt-oss-120b"
        )
        self.request_timeout_sec = (
            None if request_timeout_sec is None else float(request_timeout_sec)
        )

        self.extra_headers = extra_headers or {}
        self.sessions: Dict[str, ChatSession] = {}
        logger.info(
            "LLMClient initialized (provider=%s, base_url=%s, default_model=%s, request_timeout_sec=%s)",
            provider,
            base_url or "default",
            self.default_model,
            (
                f"{self.request_timeout_sec:g}"
                if self.request_timeout_sec is not None
                else "sdk-default"
            ),
        )

        # Optional callback to stream traces externally (e.g., logging)
        self.on_tool_trace: Optional[Callable[[List[Dict[str, Any]]], None]] = None
        logger.info(
            "LLMClient initialized (model=%s, base_url=%s, request_timeout_sec=%s)",
            self.default_model,
            base_url or "default",
            (
                f"{self.request_timeout_sec:g}"
                if self.request_timeout_sec is not None
                else "sdk-default"
            ),
        )

    def create_completion(
        self, *, model: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Any:
        """Single transport path for non-streaming chat completions."""
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=self.extra_headers,
            **kwargs,
        )

    # Low-level methods (return clean content and raw response)
    def chat(self, messages: List[Dict], model: str, **kwargs) -> Tuple[str, object]:
        # Non-streaming convenience
        logger.debug("LLMClient.chat: model=%s, n_msgs=%d", model, len(messages))
        resp = self.create_completion(model=model, messages=messages, **kwargs)
        content = resp.choices[0].message.content or ""
        return content, resp

    def stream_chat(
        self, messages: List[Dict], model: str, **kwargs
    ) -> Generator[str, None, None]:
        logger.debug("LLMClient.stream_chat: model=%s, n_msgs=%d", model, len(messages))
        # Generator yielding text pieces
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            extra_headers=self.extra_headers,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.get("content"):
                yield delta["content"]

    # New: create/get sessions
    def create_session(
        self,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatSession:
        """
        Create or return a session object. Usage:
            sess = client.create_session(model="openai/gpt-5.5"); sess.chat("Hello")
        """
        sid = session_id or str(uuid.uuid4())
        if sid in self.sessions:
            sess = self.sessions[sid]
            if model and not sess.default_model:
                sess.default_model = model
            if system_prompt and not any(
                m.get("role") == "system" for m in sess.messages
            ):
                sess.messages.insert(0, {"role": "system", "content": system_prompt})
            logger.debug(
                "LLMClient.create_session: id=%s, model=%s, has_system=%s",
                sid,
                model,
                bool(system_prompt),
            )
            return sess
        sess = ChatSession(
            client=self, session_id=sid, model=model, system_prompt=system_prompt
        )
        self.sessions[sid] = sess

        logger.info("New session created: %s", sid)

        return sess

    def new_session(
        self, model: Optional[str] = None, system_prompt: Optional[str] = None
    ) -> ChatSession:
        """Convenience to always create a fresh session with a random id."""
        return self.create_session(
            session_id=None, model=model, system_prompt=system_prompt
        )

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("Session deleted: %s", session_id)
            return True
        return False


# --------------------------
# Convenience one-liners (DEPRECATED)
# --------------------------
def verbalize_cyaml_one_shot(
    client: LLMClient,
    cyaml: str,
    model: Optional[str] = None,
    use_json_schema: bool = True,
    enable_web: bool = True,
    temperature: Optional[float] = None,
    max_tool_loops: int = 3,
):
    """
    DEPRECATED: Use var_mapping.run_variable_mapping() + verbalization_story.run_story_generation() instead.

    Convenience wrapper for one-shot structured verbalization.
    Returns (parsed_json, raw_response_with_tool_trace).

    Requires OpenRouter structured outputs.
    """
    from .verbalization_story import verbalize_cyaml_one_shot

    return verbalize_cyaml_one_shot(
        client=client,
        cyaml=cyaml,
        model=model,
        use_json_schema=use_json_schema,
        enable_web=enable_web,
        temperature=temperature,
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
):
    """
    DEPRECATED: Use var_mapping.run_variable_mapping() + verbalization_story.run_story_generation() instead.

    Convenience wrapper for multi-shot structured verbalization.
    Returns (parsed_json, raw_response_with_tool_trace).

    Requires OpenRouter structured outputs.
    """
    from .verbalization_story import verbalize_cyaml_multi_shot

    return verbalize_cyaml_multi_shot(
        client=client,
        cyaml=cyaml,
        model=model,
        use_json_schema=use_json_schema,
        enable_web=enable_web,
        temperature=temperature,
        max_tool_loops=max_tool_loops,
    )
