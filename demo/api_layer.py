"""Shared bookkeeping for the record / replay layer.

The demo never edits the original vBBQ scripts. Instead, every call they make to
OpenAI (``openai``) or the web (``requests.get``) goes through a thin layer:

* ``record_demo.py`` (run once, locally, with real API keys) lets each call go
  through and saves the request/response to a per-step "tape".
* In the browser, stub ``openai`` / ``requests`` packages (``replay_stubs/``)
  answer each call from that tape, so the demo costs no API credits.

Both sides log a short summary of each call into ``LOG`` so the UI can show it.
"""

import hashlib

STEP = None
LOG = []
_begin_hooks = []


def on_begin_step(fn):
    _begin_hooks.append(fn)
    return fn


def begin_step(name):
    global STEP
    STEP = name
    LOG.clear()
    for fn in _begin_hooks:
        fn(name)


def _summarize_content(content):
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        kind = part.get("type")
        if kind in ("text", "input_text"):
            parts.append(part.get("text", ""))
        elif kind in ("image_url", "input_image"):
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url", "")
            url = url or ""
            digest = hashlib.sha1(url.encode()).hexdigest()[:10]
            parts.append(f"[image: {len(url) * 3 // 4 // 1024} KB base64, sha1 {digest}]")
        else:
            parts.append(f"[{kind}]")
    return parts


def _tool_name(tool):
    if isinstance(tool, dict):
        return (tool.get("function") or tool).get("name")
    return str(tool)


def summarize_openai_request(kind, kwargs):
    fmt = kwargs.get("response_format")
    return {
        "api": f"chat.completions.{kind}",
        "model": kwargs.get("model"),
        "messages": [
            {"role": m.get("role"), "content": _summarize_content(m.get("content"))}
            for m in kwargs.get("messages", [])
        ],
        "tools": [_tool_name(t) for t in kwargs.get("tools") or kwargs.get("functions") or []],
        "response_format": getattr(fmt, "__name__", None),
    }


def summarize_openai_response(dump):
    msg = (dump.get("choices") or [{}])[0].get("message") or {}
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    if msg.get("function_call"):
        fc = msg["function_call"]
        calls.append({"name": fc.get("name"), "arguments": fc.get("arguments")})
    return {
        "model": dump.get("model"),
        "content": msg.get("content"),
        "parsed": msg.get("parsed"),
        "refusal": msg.get("refusal"),
        "tool_calls": calls,
    }


def log_openai(kind, kwargs, dump):
    LOG.append({
        "ch": "openai",
        "request": summarize_openai_request(kind, kwargs),
        "response": summarize_openai_response(dump),
    })


SECRET_PARAMS = {"key", "cx"}


def sanitize_params(params):
    """Drop credentials (Google API key / search engine id) from query params."""
    return {k: v for k, v in (params or {}).items() if k not in SECRET_PARAMS}


def log_http(url, params, stream, status, ctype, nbytes):
    LOG.append({"ch": "http", "url": url, "params": params, "stream": stream, "status": status,
                "content_type": ctype, "bytes": nbytes})
