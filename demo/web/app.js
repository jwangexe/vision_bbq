// UI for the vBBQ walkthrough. All computation happens in worker.js (Pyodide).
const GH = "https://github.com/jwangexe/vision_bbq/blob/main/";
const LETTERS = ["A", "B", "C"];
const PAPER = { // Table 9 of the paper: GPT-4o, Sexual orientation (full run)
  A: { overall_acc: 0.8958, ambig_acc: 1.0, disambig_acc: 0.7917, ambig_bias: null, disambig_bias: -0.0526 },
  B: { overall_acc: 0.9896, ambig_acc: 1.0, disambig_acc: 0.9792, ambig_bias: null, disambig_bias: -0.0213 },
  C: { overall_acc: 0.8698, ambig_acc: 1.0, disambig_acc: 0.7396, ambig_bias: null, disambig_bias: -0.0833 },
  D: { overall_acc: 0.974, ambig_acc: 1.0, disambig_acc: 0.9479, ambig_bias: null, disambig_bias: -0.011 },
};

const STEPS = [
  {
    id: "find_case", title: "Find a working case",
    desc: "Scan the archived GPT-4o predictions for all six experiments on the Sexual-orientation category, keeping only questions where the plain image run (Base-vBBQ) answered wrongly and SAIC-vBBQ answered correctly.",
    code: [["results/*.csv", null], ["question_index % 4 == 0 sampling, as in predict.py", "predict.py"]], live: true,
  },
  {
    id: "entities", title: "Extract the two people (entities)",
    desc: "Each answer choice that names a person becomes an entity with a unique bbq_id. \"Unknown\"-style answers are dropped, and human names would be replaced by their race and gender.",
    code: [["generate_file_structure.get_entity_dataframe", "generate_file_structure.py"], ["utils.get_entity", "utils.py"]], live: true,
  },
  {
    id: "tags", title: "Describe each entity with visible tags",
    desc: "GPT-4.1 turns each entity name into 3–5 visible traits. Tags work better than raw labels: the image judge may refuse \"the black man\" but accepts \"black skin\".",
    code: [["get_tags.query", "get_tags.py"]], ai: "GPT-4.1, replayed",
  },
  {
    id: "search", title: "Search the web for candidate photos",
    desc: "The name and tags become an image-search query. Each result URL is checked for accessibility, then downloaded. The paper collects 100 photos per entity from Google; this demo collects 10.",
    code: [["search_google_images_to_files.get_n_images_per_entity", "search_google_images_to_files.py"], ["utils.is_image_accessible", "utils.py"], ["utils.download_image_as_jpg", "utils.py"]], ai: "Openverse stand-in, replayed",
  },
  {
    id: "evaluate", title: "Score the photos and pick one",
    desc: "GPT-4.1 vision scores the photos 0–10, in batches of 5, for accuracy and clarity against the tags. The first photo scoring 10/10 is chosen. If none does, the Imagen-generated picture is used instead.",
    code: [["image_evaluation.process_folders", "image_evaluation.py"], ["image_evaluation.ask_gpt_vision_from_files", "image_evaluation.py"]], ai: "GPT-4.1 vision, replayed",
  },
  {
    id: "transforms", title: "SAIC: the interventions the model can choose",
    desc: "SAIC gives the model tools that hide sensitive attributes: grayscale (ITU-R BT.601 luminance), a pencil sketch (invert, Gaussian blur, colour dodge), and entity anonymization. This step runs the image tools on the dataset's photos of the two people. The anonymization tool is an LLM call, so it appears in the next step whenever the model asks for it.",
    code: [["toolbox.remove_color_base64_image", "toolbox.py"], ["toolbox.sketch_base64_image", "toolbox.py"], ["toolbox.resize_base64_jpg", "toolbox.py"], ["scenario_obfuscator.replace_groups_with_images", "scenario_obfuscator.py"]], live: true,
  },
  {
    id: "predict", title: "Ask GPT-4o, with and without SAIC",
    desc: "The question is posed under all six experiment settings. When tools are offered, the first call returns the tools the model wants. Those run locally and replace the prompt and images, and a second call gives the final A/B/C answer.",
    code: [["predict.process_row", "predict.py"], ["predict.get_model_answer_no_self_feedback", "predict.py"], ["toolbox.debias_prompt", "toolbox.py"]], ai: "GPT-4o, replayed",
  },
  {
    id: "analyse", title: "Measure accuracy and bias across the category",
    desc: "Repeat for every question and compute the BBQ metrics: accuracy, plus a bias score, the share of non-unknown answers that follow the stereotype rescaled to [-1, 1]. Shown here for the archived GPT-4o run on all sampled Sexual-orientation questions.",
    code: [["bias_eval.get_stats", "bias_eval.py"]], live: true,
  },
];

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => (v == null ? "N/A" : (v * 100).toFixed(1) + "%");
const num = (v) => (v == null ? "N/A" : v.toFixed(3));
const state = { ready: false, running: false, done: new Set(), results: {}, console: {} };
const worker = new Worker("worker.js");
let pending = null;

// ---------- layout ----------
function buildSteps() {
  const ol = $("#steps");
  ol.innerHTML = STEPS.map((s, i) => `
    <li class="step" id="step-${s.id}">
      <div class="step-head">
        <div class="num">${i}</div>
        <div>
          <h3>${esc(s.title)}</h3>
          <p class="desc">${esc(s.desc)}</p>
          <div class="chips">
            ${s.ai ? `<span class="chip ai">${esc(s.ai)}</span>` : `<span class="chip live">runs live</span>`}
            ${s.code.map(([label, file]) => file
              ? `<a class="chip" href="${GH}${file}" target="_blank" rel="noopener">${esc(label)}</a>`
              : `<span class="chip">${esc(label)}</span>`).join("")}
          </div>
        </div>
        <button class="btn run" data-step="${s.id}" disabled>Run</button>
      </div>
      <div class="step-body"></div>
    </li>`).join("");
  ol.addEventListener("click", (e) => {
    const b = e.target.closest("button.run");
    if (b) runUpTo(b.dataset.step);
  });
}

function refreshButtons() {
  STEPS.forEach((s, i) => {
    const b = $(`#step-${s.id} button.run`);
    const prevDone = i === 0 || state.done.has(STEPS[i - 1].id);
    // the original scripts keep state between calls (caches, downloaded folders), so a step
    // can't be replayed twice in one session; "Start over" reloads instead
    b.disabled = !state.ready || state.running || state.done.has(s.id);
    b.textContent = state.done.has(s.id) ? "Done" : prevDone ? "Run" : "Run up to here";
  });
  const all = $("#run-all");
  all.disabled = !state.ready || state.running;
  all.textContent = state.done.size ? "Start over" : "Run all steps";
}

// ---------- running ----------
function runStep(id) {
  return new Promise((resolve) => {
    state.console[id] = "";
    const el = $(`#step-${id}`);
    el.classList.add("running");
    pending = { id, resolve };
    worker.postMessage({ step: id });
  });
}

async function runUpTo(id) {
  if (state.running) return;
  state.running = true; refreshButtons();
  const target = STEPS.findIndex((s) => s.id === id);
  for (let i = 0; i <= target; i++) {
    const s = STEPS[i];
    if (i < target && state.done.has(s.id)) continue;
    const ok = await runStep(s.id);
    if (!ok) break;
  }
  state.running = false; refreshButtons();
}

worker.onmessage = ({ data: msg }) => {
  if (msg.type === "status") {
    const st = $("#status");
    st.className = "status " + msg.state;
    $("#status-text").textContent = msg.text;
    state.ready = msg.state === "ready";
    refreshButtons();
  } else if (msg.type === "stdout") {
    if (pending) state.console[pending.id] += msg.text + "\n";
  } else if (msg.type === "result") {
    const { id, resolve } = pending; pending = null;
    const el = $(`#step-${id}`);
    el.classList.remove("running");
    el.classList.add("has-output");
    const body = $(".step-body", el);
    if (msg.data.ok) {
      state.done.add(id);
      el.classList.add("done");
      state.results[id] = msg.data.result;
      body.innerHTML = RENDER[id](msg.data.result) + extras(id, msg.data.result);
    } else {
      body.innerHTML = `<div class="error">${esc(msg.data.error)}</div>`;
    }
    resolve(msg.data.ok);
  }
};

// ---------- shared bits ----------
function extras(id, r) {
  const calls = (r.api || []).filter((c) => c.ch === "openai");
  const http = (r.api || []).filter((c) => c.ch === "http");
  let html = "";
  if (calls.length) html += `<details><summary>Replayed OpenAI calls (${calls.length})</summary>${calls.map(callHtml).join("")}</details>`;
  if (http.length) html += `<details><summary>Replayed web requests (${http.length})</summary><pre class="console">${esc(http.map((h) =>
    `${h.status ?? "ERR"} ${h.stream ? "[check] " : ""}${h.url}${h.params && Object.keys(h.params).length ? "?" + new URLSearchParams(h.params) : ""}`).join("\n"))}</pre></details>`;
  const out = state.console[id];
  if (out && out.trim()) html += `<details><summary>Console output from the original code</summary><pre class="console">${esc(out)}</pre></details>`;
  return html;
}

function callHtml(c) {
  const req = c.request, res = c.response;
  const msgs = req.messages.map((m) => `[${m.role}]\n${Array.isArray(m.content) ? m.content.join("\n") : m.content}`).join("\n\n");
  const answer = res.tool_calls.length
    ? res.tool_calls.map((t) => `${t.name}(${t.arguments || ""})`).join("\n")
    : res.parsed ? JSON.stringify(res.parsed) : res.content ?? res.refusal ?? "";
  return `<div class="call">
    <div class="hdr"><span>${esc(req.api)}</span><span class="k">model ${esc(res.model || req.model)}</span>
      ${req.tools.length ? `<span class="k">tools offered: ${req.tools.map((t) => `<span class="tool">${esc(t)}</span>`).join("")}</span>` : ""}
      ${req.response_format ? `<span class="k">structured output: ${esc(req.response_format)}</span>` : ""}</div>
    <details><summary>Request</summary><pre>${esc(msgs)}</pre></details>
    <div class="k" style="margin-top:6px">Response${res.tool_calls.length ? " (tool calls)" : ""}</div><pre>${esc(answer)}</pre>
  </div>`;
}

function caseHtml(c, names, labelsByExpt) {
  return `<div class="case">
    <p class="ctx">${esc(c.context)}</p>
    <p class="q">${esc(c.question)}</p>
    <div class="choices">${c.answers.map((a, i) => `
      <div class="choice ${i === c.gold ? "gold" : ""}"><span class="letter">${LETTERS[i]}</span><span>${esc(a)}</span>
        ${i === c.gold ? `<span class="pick ok">correct answer</span>` : ""}
        <span class="picks">${Object.entries(labelsByExpt).filter(([, l]) => l === i).map(([e]) =>
          `<span class="pick ${i === c.gold ? "ok" : "no"}" title="${esc(names[e])}">${e}</span>`).join("")}</span>
      </div>`).join("")}</div>
  </div>`;
}

function creditText(c) {
  return `"${c.title || "Untitled"}" by ${c.creator || "unknown"}, ${(c.license || "").toUpperCase()} ${c.license_version || ""}`.trim();
}
function creditHtml(c) {
  return `<li><a href="${esc(c.foreign_landing_url)}" target="_blank" rel="noopener">${esc(creditText(c))}</a> (${esc(c.source)})</li>`;
}
function tile(im, cls = "", badge = null) {
  const title = im.credit ? creditText(im.credit) : im.file;
  return `<div class="tile ${cls}" title="${esc(title)}"><img src="${im.src}" alt="${esc(title)}" loading="lazy"><span class="fname">${esc(im.file)}</span>
    ${badge ? `<span class="badge">${esc(badge)}</span>` : ""}</div>`;
}

// ---------- renderers ----------
const RENDER = {
  find_case(r) {
    const f = r.candidates.find((c) => c.index === r.featured);
    const strong = r.candidates.filter((c) => c.every_saic_variant_fixes_it).length;
    return `
      <div class="kpi-row">
        <div class="kpi"><div class="v">${r.sampled}</div><div class="l">sampled questions scanned</div></div>
        <div class="kpi"><div class="v">${r.candidates.length}</div><div class="l">Base-vBBQ ✗ → SAIC-vBBQ ✓</div></div>
        <div class="kpi"><div class="v">${strong}</div><div class="l">fixed by every SAIC variant</div></div>
      </div>
      <p><b>Featured case</b> (row ${f.index}, ${esc(f.context_condition)}uous context): letters show which answer each experiment chose.</p>
      ${caseHtml(f, r.expt_names, f.labels)}
      <p class="note">${Object.entries(r.expt_names).map(([e, n]) => `<b>${e}</b> ${esc(n)}`).join(" · ")}</p>
      <p class="note">Without help, GPT-4o picked the wrong person from the text alone (A), then refused to commit ("Can't answer") once the photos were added (C), even though the context says who chatted about network security. Every SAIC variant got it right.</p>
      <details><summary>All ${r.candidates.length} working cases</summary><div class="table-wrap"><table>
        <tr><th>Row</th><th>Question</th>${"ABCDEF".split("").map((e) => `<th>${e}</th>`).join("")}</tr>
        ${r.candidates.map((c) => `<tr class="${c.index === r.featured ? "featured" : ""}"><td>${c.index}</td><td>${esc(c.question)}<br><span class="note">${esc(c.context.slice(0, 120))}…</span></td>
          ${"ABCDEF".split("").map((e) => `<td>${c.labels[e] === c.gold ? '<span class="mark-ok">✓</span>' : '<span class="mark-no">✗</span>'}</td>`).join("")}</tr>`).join("")}
      </table></div></details>`;
  },

  entities(r) {
    return `<div class="table-wrap"><table>
      <tr><th>bbq_id</th><th>name</th><th>tags</th><th>imgpath</th></tr>
      ${r.entities.map((e) => `<tr><td class="mono">${e.bbq_id}</td><td>${esc(e.name)}</td><td class="note">(next step)</td><td class="note">(step 4)</td></tr>`).join("")}
    </table></div>
    <p class="note">bbq_id = (1,000,000 × file id ${r.fileid} + row) × 10 + answer slot, as in the paper's metadata section. Answer info: <code>${esc(JSON.stringify(r.answer_info))}</code></p>`;
  },

  tags(r) {
    return r.entities.map((e) => `<div class="ent-block"><h4>${esc(e.name)}</h4>
      <div class="tags">${e.tags.split(",").map((t) => `<span class="tag">${esc(t.trim())}</span>`).join("")}</div></div>`).join("")
      + `<p class="note">Model: <code>${esc(r.model)}</code>, temperature 0.</p>`;
  },

  search(r) {
    return `<div class="callout warn"><b>Stand-in search engine.</b> The paper used the Google Custom Search JSON API, which no longer accepts new projects.
        For this demo, only the function that calls Google (<code>google_image_search</code>) is swapped for a search of
        <a href="https://openverse.org" target="_blank" rel="noopener">Openverse</a>, which returns openly licensed (CC BY / BY-SA / CC0) photos.
        Openverse requires every query word to match, so it searches the first tag rather than the full Google query.
        The batching loop, URL checks and downloads are still the original code.</div>
      <div class="kpi-row">
        <div class="kpi"><div class="v">${r.search_requests}</div><div class="l">search API requests</div></div>
        <div class="kpi"><div class="v">${r.urls_checked}</div><div class="l">URLs checked for access</div></div>
        <div class="kpi"><div class="v">${r.entities.reduce((n, e) => n + e.images.length, 0)}</div><div class="l">photos downloaded</div></div>
      </div>` + r.entities.map((e) => `<div class="ent-block"><h4>${esc(e.name)}</h4>
        <p class="note" style="margin:0 0 2px">Paper's Google query: <span class="mono">${esc(e.google_query)}</span></p>
        <p class="note" style="margin:0 0 8px">Openverse query used here: <span class="mono">${esc(e.openverse_query)}</span></p>
        <div class="grid">${e.images.map((im) => tile(im)).join("")}</div></div>`).join("")
      + `<p class="note">Asked for ${r.n_requested} per entity. The script keeps every accessible URL from each batch of 10, so it can return slightly more.</p>
      <details><summary>Photo credits</summary><ul class="note">${r.entities.flatMap((e) => e.images).filter((im) => im.credit).map((im) => creditHtml(im.credit)).join("")}</ul></details>`;
  },

  evaluate(r) {
    const search = state.results.search;
    return r.chosen.map((c) => {
      const ent = search.entities.find((e) => e.name === c.name);
      const scores = {};
      r.batches.filter((b) => b.entity === c.name).forEach((b) => b.images.forEach((f, i) => (scores[f] = b.scores[i])));
      const chosenFile = c.fallback ? null : c.imgpath.split("/").pop();
      return `<div class="ent-block"><h4>${esc(c.name)}</h4>
        <div class="grid">${ent.images.map((im) => {
          const s = scores[im.file];
          const cls = im.file === chosenFile ? "chosen" : s === undefined ? "skipped" : "";
          return tile(im, cls, s === undefined ? "not scored" : s + "/10");
        }).join("")}</div>
        <p class="note">${c.fallback
          ? `No photo reached ${r.threshold}/10, so the code falls back to the Imagen 4.0 picture (<code>${esc(c.imgpath)}</code>, made by generate_images.py, which this demo doesn't re-run). This happened for 414 of the dataset's 1,092 entities. In the published dataset, the full 100-photo Google search did find a 10/10 photo for this entity; it is the one used in steps 5 and 6.`
          : `Chosen: <code>${esc(c.imgpath)}</code>, the first photo to score ${r.threshold}/10. Photos after it are never sent for scoring.`}</p></div>`;
    }).join("") + `<p class="note">Model: <code>${esc(r.model)}</code>, ${r.batch_size} images per request, structured output <code>ScoreResponse</code>.</p>`;
  },

  transforms(r) {
    const hl = (text) => esc(text).replace(/Image[12]/g, (m) => `<mark>${m}</mark>`);
    return `<div class="cols">${r.images.map((im) => `<div><h4 style="margin:0 0 8px">${esc(im.name)} <span class="note">(dataset photo)</span></h4>
        <div class="trio">
          <figure><img src="${im.original}" alt="original"><figcaption>original (≤512 px)</figcaption></figure>
          <figure><img src="${im.grayscale}" alt="grayscale"><figcaption>grayscale</figcaption></figure>
          <figure><img src="${im.sketch}" alt="sketch"><figcaption>sketch</figcaption></figure>
        </div></div>`).join("")}</div>
      <h4 style="margin:20px 0 6px">Text: manual entity masking (experiments E and F)</h4>
      <p class="note" style="margin:0 0 6px">The groups "${esc(r.groups.join('" and "'))}" are replaced with references to the images:</p>
      <div class="case diff"><p class="ctx">${hl(r.masked_context)}</p>
        <p class="note">Answers: ${r.masked_answers.map(hl).join(" · ")}</p></div>
      <p class="note">${r.masked_context === r.dataset_masked_context ? "✓ Matches" : "✗ Differs from"} the dataset's masked prompt (data/Sexual_orientation_prompts.csv).</p>`;
  },

  predict(r) {
    const c = state.results.find_case.candidates.find((x) => x.index === state.results.find_case.featured);
    const labels = Object.fromEntries(r.results.map((x) => [x.expt, x.label]));
    const names = Object.fromEntries(r.results.map((x) => [x.expt, x.name]));
    const rows = r.results.map((x) => {
      const ok = x.label === x.gold;
      return `<tr>
        <td><b>${x.expt}</b> ${esc(x.name)}</td>
        <td class="icons">${[["text", true], ["images", x.images], ["masked names", x.masked], ["anonymize tool", x.text_tool], ["image tools", x.image_tools]]
          .filter(([, on]) => on).map(([l]) => `<span class="on">${l}</span>`).join("")}</td>
        <td>${x.tool_calls.length ? x.tool_calls.map((t) => `<span class="tool">${esc(t.name)}(${esc(Object.entries(JSON.parse(t.arguments || "{}")).map(([k, v]) => `${k}=${v}`).join(", "))})</span>`).join("") : '<span class="note">—</span>'}</td>
        <td>${x.error ? `<span class="mark-no">error</span>` : x.label == null ? "—" : `${LETTERS[x.label]}. ${esc(x.answers[x.label])}`}</td>
        <td>${x.error ? "" : ok ? '<span class="mark-ok">✓</span>' : '<span class="mark-no">✗</span>'}</td>
        <td class="note">${x.archived_label == null ? "—" : LETTERS[x.archived_label] + (x.archived_label === x.gold ? " ✓" : " ✗")}</td>
      </tr>`;
    }).join("");
    const errs = r.results.filter((x) => x.error).map((x) => `<div class="error">${x.expt}: ${esc(x.error)}</div>`).join("");
    return `${caseHtml(c, names, labels)}
      <div class="table-wrap" style="margin-top:14px"><table>
        <tr><th>Experiment</th><th>Inputs</th><th>Tools the model called</th><th>Answer</th><th></th><th>Archived run</th></tr>${rows}
      </table></div>${errs}
      <p class="note">Answers come from the GPT-4o calls recorded for this demo. "Archived run" is the answer the same experiment gave in the original 2025 run used in step 0.</p>
      ${r.results.map((x) => x.calls.filter((k) => k.ch === "openai").length
        ? `<details><summary>${x.expt}: replayed calls</summary>${x.calls.filter((k) => k.ch === "openai").map(callHtml).join("")}</details>` : "").join("")}`;
  },

  analyse(r) {
    const metric = (k) => r.rows.map((x) => x[k]);
    const best = (vals) => Math.max(...vals.filter((v) => v != null));
    const dis = metric("disambig_acc"), bestDis = best(dis);
    return `<div class="table-wrap"><table>
      <tr><th>Experiment</th><th class="num-cell">Overall acc.</th><th class="num-cell">Ambig. acc.</th><th class="num-cell">Disambig. acc.</th><th class="num-cell">Ambig. bias</th><th class="num-cell">Disambig. bias</th><th class="num-cell">Paper: disambig. acc.</th></tr>
      ${r.rows.map((x) => `<tr><td><b>${x.expt}</b> ${esc(x.name)}</td>
        <td class="num-cell">${pct(x.overall_acc)}</td><td class="num-cell">${pct(x.ambig_acc)}</td>
        <td class="num-cell" style="${x.disambig_acc === bestDis ? "font-weight:700" : ""}">${pct(x.disambig_acc)}</td>
        <td class="num-cell">${num(x.ambig_bias)}</td><td class="num-cell">${num(x.disambig_bias)}</td>
        <td class="num-cell note">${PAPER[x.expt] ? pct(PAPER[x.expt].disambig_acc) : "—"}</td></tr>`).join("")}
    </table></div>
    <p class="note">Ambiguous contexts: GPT-4o always answered "unknown", which is correct, so the ambiguous bias score is undefined (N/A, as in the paper).
      <code>bias_eval.get_stats</code> divides by zero in that case, so the demo recomputes those rows with the same formulas plus an N/A guard
      (${r.rows.filter((x) => x.computed_by !== "bias_eval.get_stats").length}/6 rows). The paper column is Table 9 (GPT-4o, full run). E and F are extra ablations not reported there.</p>`;
  },
};

// ---------- boot ----------
buildSteps();
refreshButtons();
$("#run-all").addEventListener("click", () => {
  if (state.done.size) location.reload();
  else runUpTo(STEPS[STEPS.length - 1].id);
});
fetch("py/_replay/meta.json").then((r) => r.json()).then((m) => {
  $("#rec-date").textContent = ` on ${m.recorded_at}`;
  $("#foot-meta").textContent = `API responses recorded ${m.recorded_at} (${m.models.join(", ")}). Re-record with demo/record_demo.py.`;
}).catch(() => {});
