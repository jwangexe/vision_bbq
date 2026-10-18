"""Replay stand-in for `requests.get` (browser demo only).

Serves the Google Custom Search responses and image downloads recorded by
demo/record_demo.py.
"""
import json as _json

import api_layer
import replay_tape


class RequestException(Exception):
    pass


class _Headers(dict):
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


class Response:
    def __init__(self, ev):
        self.status_code = ev["status"]
        self.headers = _Headers({"Content-Type": ev.get("ctype", "")})
        if "text" in ev:
            self.text = ev["text"]
            self.content = ev["text"].encode()
        elif "blob" in ev:
            self.content = replay_tape.blob(ev["blob"])
            self.text = ""
        else:
            self.content, self.text = b"", ""

    def json(self):
        return _json.loads(self.text)


def get(url, params=None, headers=None, timeout=None, stream=False, **kwargs):
    ev = replay_tape.next_event("http", url)
    if "error" in ev:
        raise RequestException(ev["error"])
    r = Response(ev)
    api_layer.log_http(url, api_layer.sanitize_params(params), bool(stream), r.status_code,
                       r.headers.get("Content-Type"), len(r.content))
    return r
