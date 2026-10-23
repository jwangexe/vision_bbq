"""Assemble the static Hugging Face Space in demo/dist/.

    python demo/build_space.py --data-root "/path/to/original/code"

Copies, byte for byte, the original vBBQ scripts the demo imports, plus the data
files, the recordings made by record_demo.py, and the web front end.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parent
ORIGINAL_SCRIPTS = [
    "generate_file_structure.py", "get_tags.py", "search_google_images_to_files.py",
    "image_evaluation.py", "toolbox.py", "utils.py", "scenario_obfuscator.py",
    "predict.py", "bias_eval.py", "bias_classes.txt",
]
DATA_FILES = [
    "data/Sexual_orientation.jsonl", "data/Sexual_orientation_prompts.csv",
    "data/vocabulary_proper_names.csv",
] + [f"results/Sexual_orientation{e}.csv" for e in "ABCDEF"]
DUMMY_KEYS = ["api_keys/openai/api_key.txt", "api_keys/google/api_key.txt", "api_keys/google/cse_id.txt"]

SPACE_README = """---
title: Vision-BBQ Walkthrough
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
short_description: Step-by-step demo of the vBBQ bias benchmark and SAIC
pinned: false
license: mit
---

# Vision-BBQ (vBBQ): live walkthrough

Runs the original [vision_bbq](https://github.com/jwangexe/vision_bbq) Python code in the browser
(Pyodide) on one BBQ question, from dataset curation (entity tags, image search, GPT image scoring)
to model prediction with the SAIC bias-mitigation tools, and the final bias metrics.

API calls (GPT-4.1, GPT-4o, Google Custom Search) were recorded once and are replayed, so the demo
uses no API credits. `py/` holds the unmodified project scripts (`py/*.py`), the replay layer
(`py/api_layer.py`, `py/demo_steps.py`, `py/replay_stubs/`) and the recordings (`py/_replay/`).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--recordings", default=str(DEMO / "recordings"))
    ap.add_argument("--out", default=str(DEMO / "dist"))
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    py = out / "py"
    py.mkdir(parents=True)

    def put(src, rel):
        dst = py / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    for f in ORIGINAL_SCRIPTS:
        put(REPO / f, f)
        assert (REPO / f).read_bytes() == (py / f).read_bytes()
    for f in DATA_FILES:
        put(Path(args.data_root) / f, f)
    put(DEMO / "api_layer.py", "api_layer.py")
    put(DEMO / "demo_steps.py", "demo_steps.py")
    for f in (DEMO / "replay_stubs").rglob("*.py"):
        put(f, Path("replay_stubs") / f.relative_to(DEMO / "replay_stubs"))
    rec = Path(args.recordings)
    for f in list((rec / "tapes").glob("*.json")) + list((rec / "blobs").iterdir()):
        put(f, Path("_replay") / f.relative_to(rec))
    put(rec / "meta.json", "_replay/meta.json")
    for f in DUMMY_KEYS:   # the scripts read key files at import; the replay layer never uses them
        (py / f).parent.mkdir(parents=True, exist_ok=True)
        (py / f).write_text("replayed-no-key")

    files = sorted(str(p.relative_to(py)) for p in py.rglob("*") if p.is_file())
    (out / "manifest.json").write_text(json.dumps({"files": files}, indent=0))
    for f in (DEMO / "web").iterdir():
        shutil.copy(f, out / f.name)
    (out / "README.md").write_text(SPACE_README)

    print(f"Built {out} ({len(files)} runtime files)")
    for f in ORIGINAL_SCRIPTS:
        print(f"  {f}: sha1 {hashlib.sha1((py / f).read_bytes()).hexdigest()[:12]} (identical to repo)")


if __name__ == "__main__":
    main()
