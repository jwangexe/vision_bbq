"""One-time recording run for the Hugging Face Space demo.

Runs every step in demo_steps.py against the REAL APIs (OpenAI + Google Custom
Search), using the original, unmodified vBBQ scripts, and saves each API
request/response to demo/recordings/. The Space then replays these recordings
in the browser, so visitors never spend API credits.

Usage (from the repo root, keys in ./api_keys/ as the original scripts expect):
    python demo/record_demo.py --data-root "/path/to/original/code"

--data-root must contain data/Sexual_orientation.jsonl, data/Sexual_orientation_prompts.csv,
data/vocabulary_proper_names.csv and results/Sexual_orientation{A..F}.csv.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_DIR = DEMO_DIR.parent
DATA_FILES = [
    "data/Sexual_orientation.jsonl",
    "data/Sexual_orientation_prompts.csv",
    "data/vocabulary_proper_names.csv",
] + [f"results/Sexual_orientation{e}.csv" for e in "ABCDEF"]
MAX_BLOB_SIDE = 1024   # downloaded web images are stored at most this size to keep the Space small


def prepare_workdir(work, data_root, keys_dir):
    if work.exists():
        shutil.rmtree(work)
    for rel in DATA_FILES:
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(data_root) / rel, work / rel)
    shutil.copy(REPO_DIR / "bias_classes.txt", work / "bias_classes.txt")
    os.symlink(Path(keys_dir).resolve(), work / "api_keys")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--keys-dir", default=str(REPO_DIR / "api_keys"))
    ap.add_argument("--out", default=str(DEMO_DIR / "recordings"))
    ap.add_argument("--steps", default=None, help="comma-separated subset of steps (for dry runs)")
    args = ap.parse_args()

    keys = Path(args.keys_dir)
    openai_key = (keys / "openai/api_key.txt").read_text().strip()
    secrets = [openai_key] + [(keys / f).read_text().strip() for f in ("google/api_key.txt", "google/cse_id.txt")
                              if (keys / f).exists()]
    os.environ["OPENAI_API_KEY"] = openai_key   # predict_gpt5.py / default client
    os.environ["API_KEY"] = openai_key          # predict.py and toolbox.py read API_KEY

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "tapes").mkdir(parents=True)
    (out / "blobs").mkdir()
    (out / "results").mkdir()

    work = DEMO_DIR / "_work"
    prepare_workdir(work, args.data_root, keys)
    os.chdir(work)
    sys.path[:0] = [str(REPO_DIR), str(DEMO_DIR)]

    import api_layer
    tapes = {}

    def scrub(text):
        for s in secrets:
            text = text.replace(s, "REDACTED")
        return text

    def tape():
        return tapes.setdefault(api_layer.STEP, [])

    # ----- record OpenAI calls -----
    from openai.resources.chat import Completions
    depth = [0]

    def recorder(kind, orig):
        def wrapper(self, *a, **kw):
            depth[0] += 1
            try:
                resp = orig(self, *a, **kw)
            except Exception as e:
                if depth[0] == 1:
                    tape().append({"ch": "openai", "kind": kind, "error": scrub(str(e))})
                raise
            finally:
                depth[0] -= 1
            if depth[0] == 0:
                dump = json.loads(scrub(json.dumps(resp.model_dump(mode="json", warnings=False))))
                tape().append({"ch": "openai", "kind": kind, "resp": dump})
                api_layer.log_openai(kind, kw, dump)
            return resp
        return wrapper

    Completions.create = recorder("create", Completions.create)
    Completions.parse = recorder("parse", Completions.parse)

    # ----- record HTTP GETs -----
    import requests
    from PIL import Image
    real_get = requests.get

    def store_blob(body, ctype):
        if ctype.startswith("image/"):
            try:
                img = Image.open(io.BytesIO(body))
                if max(img.size) > MAX_BLOB_SIDE:
                    img = img.convert("RGB")
                    img.thumbnail((MAX_BLOB_SIDE, MAX_BLOB_SIDE))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    body = buf.getvalue()
            except Exception:
                pass
        sha = hashlib.sha1(body).hexdigest()
        (out / "blobs" / sha).write_bytes(body)
        return sha

    def get(url, params=None, **kw):
        stream = bool(kw.get("stream"))
        ev = {"ch": "http", "url": url, "params": api_layer.sanitize_params(params), "stream": stream}
        try:
            r = real_get(url, params=params, **kw)
        except Exception as e:
            ev["error"] = scrub(f"{type(e).__name__}: {e}")
            tape().append(ev)
            raise
        ctype = r.headers.get("Content-Type", "")
        ev.update(status=r.status_code, ctype=ctype)
        nbytes = 0
        if not stream:
            body = r.content
            nbytes = len(body)
            if "json" in ctype or ctype.startswith("text/"):
                ev["text"] = scrub(r.text)
            else:
                ev["blob"] = store_blob(body, ctype)
        tape().append(ev)
        api_layer.log_http(url, ev["params"], stream, r.status_code, ctype, nbytes)
        return r

    requests.get = get

    # ----- run every step, exactly as the browser will -----
    import demo_steps
    steps = args.steps.split(",") if args.steps else list(demo_steps.STEPS)
    for name in steps:
        print(f"\n===== recording step: {name} =====", flush=True)
        t = time.time()
        res = demo_steps.run_step(name)
        (out / "tapes" / f"{name}.json").write_text(json.dumps(tapes.get(name, []), indent=1))
        (out / "results" / f"{name}.json").write_text(scrub(res))
        status = json.loads(res)
        print(f"-> {'ok' if status['ok'] else 'FAILED'} in {time.time() - t:.1f}s, "
              f"{len(tapes.get(name, []))} API calls recorded")
        if not status["ok"]:
            print(status["error"])
            sys.exit(1)

    models = sorted({ev["resp"].get("model") for t in tapes.values() for ev in t
                     if ev["ch"] == "openai" and "resp" in ev})
    (out / "meta.json").write_text(json.dumps({
        "recorded_at": time.strftime("%Y-%m-%d"),
        "models": models,
        "featured_index": demo_steps.FEATURED_INDEX,
        "n_images": demo_steps.N_IMAGES,
    }, indent=1))

    # never ship a credential
    for f in out.rglob("*"):
        if f.is_file():
            data = f.read_bytes()
            for s in secrets:
                assert s.encode() not in data, f"secret leaked into {f}"
    print(f"\nRecordings saved to {out}")


if __name__ == "__main__":
    main()
