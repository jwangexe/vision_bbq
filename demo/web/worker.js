// Runs the original vBBQ Python code in a Web Worker via Pyodide.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js");

const ROOT = "/home/pyodide/vbbq";
let pyodide = null;

const post = (msg) => self.postMessage(msg);

function mkdirs(FS, path) {
  const parts = path.split("/").slice(0, -1);
  let cur = "";
  for (const p of parts) {
    cur += (cur || path.startsWith("/") ? "/" : "") + p;
    if (!cur || cur === "/") continue;
    try { FS.mkdir(cur); } catch (e) { /* exists */ }
  }
}

async function fetchAll(files, onProgress) {
  let done = 0;
  const out = new Array(files.length);
  const queue = files.map((f, i) => [f, i]);
  async function worker() {
    while (queue.length) {
      const [f, i] = queue.shift();
      const res = await fetch("py/" + f.split("/").map(encodeURIComponent).join("/"));
      if (!res.ok) throw new Error(`could not fetch ${f}: ${res.status}`);
      out[i] = new Uint8Array(await res.arrayBuffer());
      onProgress(++done, files.length);
    }
  }
  await Promise.all(Array.from({ length: 8 }, worker));
  return out;
}

async function init() {
  post({ type: "status", state: "loading", text: "Loading Python runtime (Pyodide)…" });
  pyodide = await loadPyodide();
  post({ type: "status", state: "loading", text: "Loading pandas, Pillow, pydantic…" });
  await pyodide.loadPackage(["pandas", "pillow", "pydantic"]);

  const manifest = await (await fetch("manifest.json")).json();
  const blobs = await fetchAll(manifest.files, (n, total) =>
    post({ type: "status", state: "loading", text: `Loading project files and recordings ${n}/${total}…` }));
  manifest.files.forEach((f, i) => {
    const path = `${ROOT}/${f}`;
    mkdirs(pyodide.FS, path);
    pyodide.FS.writeFile(path, blobs[i]);
  });

  pyodide.setStdout({ batched: (text) => post({ type: "stdout", text }) });
  pyodide.setStderr({ batched: (text) => post({ type: "stdout", text }) });
  pyodide.runPython(`
import os, sys
os.chdir("${ROOT}")
sys.path[:0] = ["${ROOT}/replay_stubs", "${ROOT}"]
import demo_steps
`);
  post({ type: "status", state: "ready", text: `Python ${pyodide.runPython("import sys; sys.version.split()[0]")} ready` });
}

const ready = init().catch((e) => post({ type: "status", state: "error", text: String(e) }));

self.onmessage = async (event) => {
  const { step } = event.data;
  await ready;
  try {
    const out = pyodide.runPython(`demo_steps.run_step(${JSON.stringify(step)})`);
    post({ type: "result", step, data: JSON.parse(out) });
  } catch (e) {
    post({ type: "result", step, data: { ok: false, error: String(e) } });
  }
};
