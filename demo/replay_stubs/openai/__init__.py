"""Replay stand-in for the `openai` package (browser demo only).

Answers each chat.completions call with the response recorded by
demo/record_demo.py, so the original scripts run unchanged without API credits.
"""
from types import SimpleNamespace

import api_layer
import replay_tape

api_key = None


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def _replay(kind, kwargs):
    ev = replay_tape.next_event("openai")
    if "error" in ev:
        raise Exception(ev["error"])
    dump = ev["resp"]
    api_layer.log_openai(kind, kwargs, dump)
    resp = _ns(dump)
    fmt = kwargs.get("response_format")
    for choice, raw in zip(resp.choices, dump["choices"]):
        parsed = raw["message"].get("parsed")
        if parsed is not None and hasattr(fmt, "model_validate"):
            choice.message.parsed = fmt.model_validate(parsed)
    return resp


class _Completions:
    def create(self, **kwargs):
        return _replay("create", kwargs)

    def parse(self, **kwargs):
        return _replay("parse", kwargs)


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class OpenAI:
    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.api_key = api_key
        self.chat = _Chat()
        self.beta = SimpleNamespace(chat=self.chat)


chat = _Chat()   # module-level client, as used by get_tags.py (openai.chat.completions.create)
