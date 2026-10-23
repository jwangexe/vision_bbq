"""Loads the recorded API tape for the current demo step (browser side)."""
import json
import os

import api_layer

TAPE_DIR = "./_replay/tapes"
BLOB_DIR = "./_replay/blobs"
_queue = []


@api_layer.on_begin_step
def _load(step):
    path = os.path.join(TAPE_DIR, f"{step}.json")
    _queue[:] = json.load(open(path)) if os.path.exists(path) else []


def next_event(ch, url=None):
    if not _queue:
        raise RuntimeError(f"replay tape for step '{api_layer.STEP}' has no more recorded {ch} calls")
    ev = _queue.pop(0)
    if ev["ch"] != ch or (url is not None and ev.get("url") != url):
        raise RuntimeError(f"replay mismatch in step '{api_layer.STEP}': expected {ev['ch']} "
                           f"{ev.get('url', '')}, got {ch} {url or ''}")
    return ev


def blob(sha):
    with open(os.path.join(BLOB_DIR, sha), "rb") as f:
        return f.read()
