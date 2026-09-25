"use strict";
const { lineChart, heatmap, groupedBars, sparkPair, fmt, css } = window.Charts;
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 0) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`);
const SEED_COLORS = ["#2446c7", "#c23a6b", "#1f8a70", "#b7791f", "#6b3fb3", "#5f6b7a", "#0e7c9e", "#a3472f", "#7a8b1e", "#9a4fb8", "#3d6b4f", "#b85c8a"];

const S = { runs: [], run: null, family: null, job: null, hubHistory: [] };

// ------------------------------------------------------------------ plumbing
async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts, body: opts.body ? JSON.stringify(opts.body) : undefined });
  let data = null;
  try { data = await r.json(); } catch { /* empty */ }
  if (!r.ok) {
    const d = data && data.detail;
    throw new Error(Array.isArray(d) ? d.map((x) => `${x.loc?.slice(-1)[0]}: ${x.msg}`).join("; ") : d || `Request failed (${r.status}).`);
  }
  return data;
}
let toastTimer;
function toast(msg, error = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (error ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), error ? 7000 : 3500);
}
function panel(title, sub) {
  const p = document.createElement("div");
  p.className = "panel";
  p.innerHTML = `<h3>${title}</h3>${sub ? `<p class="sub">${sub}</p>` : ""}`;
  const body = document.createElement("div");
  p.appendChild(body);
  return [p, body];
}

// Submit work; the server either answers from cache or hands back a job to poll.
async function runJob(path, body, onDone, onLive) {
  if (S.job && ["queued", "running"].includes(S.job.state)) {
    toast("Another job is running. Wait for it or stop it first.", true);
    return;
  }
  let res;
  try { res = await api(path, { method: "POST", body }); } catch (e) { toast(e.message, true); return; }
  if (res.result) return onDone(res.result);
  const id = res.job;
  showJob({ id, title: "Starting", message: "", progress: 0, state: "queued" });
  const tick = async () => {
    let j;
    try { j = await api(`/api/jobs/${id}`); } catch (e) { hideJob(); toast(e.message, true); return; }
    showJob(j);
    if (j.live && onLive) onLive(j.live);
    if (j.state === "done") { hideJob(); onDone(j.result); if (window.Picker) window.Picker.loadSaved(); }
    else if (j.state === "error" || j.state === "stopped") { hideJob(); toast(j.error || "The job stopped.", j.state === "error"); }
    else setTimeout(tick, 700);
  };
  tick();
}
function showJob(j) {
  S.job = j;
  $("#jobbar").hidden = false;
  $("#job-title").textContent = j.title;
  $("#job-msg").textContent = j.message;
  $("#job-meter").style.width = `${Math.round(j.progress * 100)}%`;
}
// Presets only fill in the model boxes; make that visible and point at the next step.
function presetFilled(inputs, buttons, msg) {
  for (const sel of inputs) {
    const el = $(sel);
    if (!el) continue;
    el.dispatchEvent(new Event("change"));
    el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  }
  for (const sel of buttons) {
    const b = $(sel);
    if (!b) continue;
    b.classList.remove("pulse"); void b.offsetWidth; b.classList.add("pulse");
  }
  const first = $(inputs[0]);
  if (first) first.scrollIntoView({ block: "nearest", behavior: "smooth" });
  toast(msg);
}
function hideJob() { S.job = null; $("#jobbar").hidden = true; }
$("#job-stop").onclick = async () => { if (S.job) { await api(`/api/jobs/${S.job.id}/stop`, { method: "POST" }).catch(() => {}); toast("Stopping after the current step."); } };

// ------------------------------------------------------------------ tabs
document.querySelectorAll("nav button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll("nav button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `view-${b.dataset.view}`));
}));

// ------------------------------------------------------------------ lab: runs
async function loadRuns(selectId) {
  S.runs = await api("/api/lab/runs").catch(() => []);
  const ul = $("#runs");
  ul.innerHTML = S.runs.length ? "" : `<li class="empty">No runs yet. Train a family to start.</li>`;
  for (const r of S.runs) {
    const c = r.config;
    const li = document.createElement("li");
    li.innerHTML = `<button data-id="${esc(r.run_id)}" class="${S.run === r.run_id ? "active" : ""}">
      <span class="t1">p = ${c.p}, ${r.trained_seeds.length} of ${c.seeds.length} seeds</span>
      <span class="t2">${esc(r.run_id.slice(0, 15).replace(/(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/, "$1-$2-$3 $4:$5"))}${r.complete ? "" : ", incomplete"}</span></button>`;
    li.querySelector("button").onclick = () => openRun(r.run_id);
    ul.appendChild(li);
  }
  if (selectId) openRun(selectId);
}

$("#train-form").onsubmit = (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const body = Object.fromEntries(f.entries());
  body.seeds = String(body.seeds).split(/[\s,]+/).filter(Boolean).map(Number);
  for (const k of ["p", "steps", "d_model", "n_heads", "d_mlp"]) body[k] = parseInt(body[k], 10);
  for (const k of ["train_frac", "lr", "wd"]) body[k] = parseFloat(body[k]);
  const stage = $("#lab-stage");
  stage.innerHTML = `<div class="head"><h1>Training</h1><span class="meta">Seeds ${body.seeds.join(", ")} on (a + b) mod ${body.p}</span></div>
    <p class="finding">Watch for the jump in test accuracy: that is the moment a model stops memorising and finds the general algorithm.</p>`;
  const [p, host] = panel("Current seed", "Accuracy on the training pairs and on held-out pairs it never saw.");
  stage.appendChild(p);
  runJob("/api/lab/train", body, (res) => { toast("Training finished."); loadRuns(res.run_id); }, (live) => {
    if (!live || !live.history) return;
    const h = live.history;
    p.querySelector("h3").textContent = `Seed ${live.seed}`;
    lineChart(host, {
      series: [
        { name: "Train accuracy", color: css("--muted"), points: h.step.map((s, i) => [s, h.train_acc[i]]), dash: "4 3" },
        { name: "Test accuracy", color: css("--ab"), points: h.step.map((s, i) => [s, h.test_acc[i]]) },
      ], yMin: 0, yMax: 1, xLabel: "Step", yFmt: (v) => pct(v), xMin: 0, xMax: body.steps,
    });
  });
};

async function openRun(id) {
  S.run = id;
  document.querySelectorAll("#runs button").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  const stage = $("#lab-stage");
  stage.innerHTML = `<p class="spinner">Surveying the family…</p>`;
  runJob("/api/lab/family", { run_id: id }, (fam) => { S.family = fam; renderFamily(fam); });
}

// ------------------------------------------------------------------ lab: family
function renderFamily(f) {
  const stage = $("#lab-stage");
  const c = f.config, ms = f.models;
  const learned = ms.filter((m) => m.test_acc >= 0.95);
  const pending = ms.filter((m) => m.test_acc < 0.95);
  const nk = ms.map((m) => m.key_freqs.length);
  let finding = `${learned.length} of ${ms.length} models learned the rule for pairs they never saw. `;
  if (learned.length) {
    finding += `Each leans on a handful of frequencies (${Math.min(...nk)} to ${Math.max(...nk)}), yet only <b class="cab">${f.shared_freqs.length}</b> ${f.shared_freqs.length === 1 ? "frequency is" : "frequencies are"} shared by every model. `;
    const worst = Math.min(...learned.map((m) => m.compressed_acc));
    finding += `Keeping only those few directions of the number embedding, the learned models still score at least <b>${pct(worst)}</b>.`;
  }
  stage.innerHTML = `<div class="head"><h1>Family ${esc(c.run_id || S.run)}</h1>
    <span class="meta">(a + b) mod ${c.p}, ${ms.length} seeds, ${c.steps} steps each, ${pct(c.train_frac)} of pairs used for training</span>
    <button class="link" id="del-run">Delete run</button></div>
    <p class="finding">${finding}</p>`;
  if (pending.length) stage.insertAdjacentHTML("beforeend", `<p class="note">${pending.length > 1 ? "Seeds" : "Seed"} ${pending.map((m) => m.seed).join(pending.length > 2 ? ", " : " and ")} ${pending.length > 1 ? "have" : "has"} not generalised yet (below 95% test accuracy). Comparisons with ${pending.length > 1 ? "them" : "it"} are noisier; train longer for cleaner results.</p>`);
  $("#del-run").onclick = async () => {
    if (!confirm("Delete this run and its models?")) return;
    await api(`/api/lab/runs/${S.run}`, { method: "DELETE" }).catch((e) => toast(e.message, true));
    S.run = null; stage.innerHTML = ""; loadRuns();
  };

  // Pair picker first: it is the main action.
  const [pp, pb] = panel("Compare two members", "Pick any two seeds. The comparison removes symmetries step by step and tests whether the models can swap parts.");
  const opts = ms.map((m) => `<option value="${m.seed}">Seed ${m.seed} (${pct(m.test_acc)})</option>`).join("");
  pb.innerHTML = `<div class="pair-picker"><span class="tag a">A</span><select id="pa">${opts}</select>
    <span class="tag b">B</span><select id="pb">${opts}</select><button class="primary" id="go-pair">Compare</button></div>`;
  stage.appendChild(pp);
  const ranked = [...ms].sort((a, b) => b.test_acc - a.test_acc);
  $("#pa").value = ranked[0].seed;
  $("#pb").value = (ranked[1] || ranked[0]).seed;
  const pairHost = document.createElement("div");
  pairHost.id = "pair-host";
  stage.appendChild(pairHost);
  $("#go-pair").onclick = () => {
    const a = +$("#pa").value, b = +$("#pb").value;
    if (a === b) return toast("Pick two different seeds.", true);
    pairHost.innerHTML = `<p class="spinner">Aligning seed ${b} to seed ${a}…</p>`;
    runJob("/api/lab/pair", { run_id: S.run, a, b }, (r) => renderPair(pairHost, r));
  };

  const seededHost = document.createElement("div");
  stage.appendChild(seededHost);
  seededPanel(seededHost, f);

  const cols = document.createElement("div");
  cols.className = "stack-gap";
  stage.appendChild(cols);

  // Frequency barcode: the fingerprint of each solution.
  const nf = ms[0].energy.length;
  const [bp, bb] = panel("Frequency fingerprints",
    "Each row is one model's number embedding, split into Fourier frequencies. Dark cells hold the energy. A model that has learned the task concentrates on a few frequencies; which ones is decided by the seed.");
  const grid = document.createElement("div");
  grid.className = "barcode";
  grid.style.gridTemplateColumns = `64px repeat(${nf}, minmax(6px, 1fr))`;
  const c1 = css("--ab");
  for (const m of ms) {
    grid.insertAdjacentHTML("beforeend", `<div class="rowlab">Seed ${m.seed}</div>`);
    const mx = Math.max(...m.energy);
    m.energy.forEach((e, k) => {
      const t = e / mx;
      const key = m.key_freqs.includes(k + 1);
      grid.insertAdjacentHTML("beforeend", `<div class="cell" title="Seed ${m.seed}, frequency ${k + 1}: ${pct(e, 1)} of energy" style="background:${Charts.mix(css("--soft").startsWith("#") ? css("--soft") : "#e9edf2", c1, t)};${key ? `box-shadow: inset 0 -3px 0 ${css("--ink")}` : ""}"></div>`);
    });
  }
  grid.insertAdjacentHTML("beforeend", `<div class="rowlab">Frequency</div>`);
  for (let k = 1; k <= nf; k++) grid.insertAdjacentHTML("beforeend", `<div class="axis">${k === 1 || k % 5 === 0 ? k : ""}</div>`);
  bb.appendChild(grid);
  bb.insertAdjacentHTML("beforeend", `<p class="sub" style="margin-top:10px">Underlined cells are each model's key frequencies. Shared by all: ${f.shared_freqs.length ? f.shared_freqs.join(", ") : "none"}.</p>`);
  cols.appendChild(bp);

  const [tp, tb] = panel("How little of each model is the solution",
    "Project each model's number embedding onto its key frequencies only, then measure accuracy again. The ratio of kept directions is a direct estimate of how much of the embedding is doing the work.");
  tb.className = "scroll";
  tb.innerHTML = `<table><thead><tr><th>Seed</th><th class="num">Test accuracy</th><th>Key frequencies</th><th class="num">Directions kept</th><th class="num">Accuracy with only those</th><th class="num">Neurons tuned to them</th></tr></thead><tbody>${
    ms.map((m) => `<tr><td>${m.seed}</td><td class="num">${pct(m.test_acc, 1)}</td><td>${m.key_freqs.join(", ")}</td>
      <td class="num">${m.compressed_dims} of ${m.full_dims}</td><td class="num">${pct(m.compressed_acc, 1)}</td><td class="num">${pct(m.neurons_on_key_freqs)}</td></tr>`).join("")}</tbody></table>`;
  cols.appendChild(tp);

  const row = document.createElement("div");
  row.className = "cols";
  const [lp, lb] = panel("Learning curves", "Test accuracy for each seed. The late jump is grokking.");
  const series = Object.entries(f.history).map(([s, h], i) => ({ name: `Seed ${s}`, color: SEED_COLORS[i % SEED_COLORS.length], points: h.step.map((x, j) => [x, h.test_acc[j]]) }));
  row.appendChild(lp);
  const labels = ms.map((m) => `Seed ${m.seed}`);
  const [cp, cb] = panel("Representation similarity between seeds", "Linear CKA of the MLP neuron activations over every input, before any relabelling. Low values here are expected when seeds use different frequencies.");
  row.appendChild(cp);
  cols.appendChild(row);
  lineChart(lb, { series, yMin: 0, yMax: 1, xLabel: "Step", yFmt: (v) => pct(v) });
  heatmap(cb, { matrix: f.cka.mlp_post, rows: labels, cols: labels, values: true, rowLabelWidth: 60, colLabelHeight: 50 });
}

// ------------------------------------------------------------------ lab: pattern-seeded training
const SEED_CONDS = [
  ["random", "Ordinary random start (control)", true],
  ["fourier", "Waves predicted by the task's symmetry", true],
  ["fourier_frozen", "Waves, never updated", false],
  ["transplant", "Embedding copied from a trained model", true],
  ["transplant_frozen", "Copied embedding, never updated", false],
  ["shuffled", "Copied embedding with rows shuffled (control)", true],
];
const COND_COLORS = { random: "#5f6b7a", fourier: "#6b3fb3", fourier_frozen: "#b595f0", transplant: "#2446c7", transplant_frozen: "#7d97ff", shuffled: "#c23a6b" };

function seededPanel(host, f) {
  const c = f.config;
  const [p, b] = panel("Does a known pattern save compute?",
    "Train fresh models from different starting points and count the steps until they generalise. The pattern can come from theory (waves at a few frequencies, which the task's symmetry predicts) or from a trained member of this family. The shuffled transplant keeps the same numbers but destroys the pattern, so it shows whether the pattern itself is what helps.");
  const learned = f.models.filter((m) => m.test_acc >= 0.95);
  b.innerHTML = `<div class="seedform">
    <div class="conds">${SEED_CONDS.map(([k, label, on]) => `<label class="check"><input type="checkbox" value="${k}" ${on ? "checked" : ""} ${k === "random" ? "disabled" : ""}> <span style="color:${COND_COLORS[k]}">■</span> ${label}</label>`).join("")}</div>
    <div class="grid-form" style="max-width:520px">
      <label>Fresh seeds<input id="sd-seeds" value="21, 22, 23"></label>
      <label>Step limit<input id="sd-max" type="number" value="${Math.round(c.steps * 1.5)}" step="100"></label>
      <label>Target test accuracy<input id="sd-target" type="number" value="0.95" step="0.01" min="0.5" max="1"></label>
      <label>Donor for transplants<select id="sd-donor">${(learned.length ? learned : f.models).map((m) => `<option value="${m.seed}">Seed ${m.seed} (${pct(m.test_acc)})</option>`).join("")}</select></label>
    </div>
    <button class="primary" id="sd-go" style="margin-top:12px">Run the experiment</button>
    <p class="sub" style="margin-top:8px">Each condition runs once per seed and stops as soon as it generalises. With the defaults this is about four times the cost of training one seed per condition.</p></div>
    <div id="sd-out"></div>`;
  host.appendChild(p);
  const out = $("#sd-out");
  api(`/api/lab/seeded/${S.run}`).then((prev) => { if (prev.length) renderSeeded(out, prev[0]); }).catch(() => {});
  $("#sd-go").onclick = () => {
    const conditions = [...b.querySelectorAll(".conds input:checked")].map((i) => i.value);
    const body = { run_id: S.run, conditions: ["random", ...conditions.filter((x) => x !== "random")],
      seeds: $("#sd-seeds").value.split(/[\s,]+/).filter(Boolean).map(Number), max_steps: parseInt($("#sd-max").value, 10),
      target: parseFloat($("#sd-target").value), donor_seed: parseInt($("#sd-donor").value, 10) };
    out.innerHTML = `<p class="spinner">Training…</p>`;
    runJob("/api/lab/seeded", body, (r) => renderSeeded(out, r), (live) => {
      if (live && live.done && live.done.length) renderSeeded(out, { live: true, results: live.done, conditions: body.conditions, target: body.target, summary: [] });
    });
  };
}

function renderSeeded(out, r) {
  out.innerHTML = "";
  const byCond = {};
  for (const x of r.results) (byCond[x.condition] ||= []).push(x);
  if (!r.live && r.summary.length) {
    const base = r.summary.find((s) => s.condition === "random");
    const best = [...r.summary].filter((s) => s.condition !== "random" && s.step_saving !== null).sort((a, b) => b.step_saving - a.step_saving)[0];
    let f = base && base.median_steps !== null
      ? `From an ordinary random start, models needed a median of <b>${base.median_steps.toLocaleString()}</b> steps to reach ${pct(r.target)} test accuracy. `
      : `The random-start models did not reach ${pct(r.target)} within ${r.max_steps.toLocaleString()} steps, so savings are not defined; raise the step limit. `;
    if (best && base && base.median_steps !== null) {
      f += best.step_saving > 0.05
        ? `The biggest saving came from <b class="cab">${esc(best.label.toLowerCase())}</b>: ${best.median_steps.toLocaleString()} steps, <b class="cab">${pct(best.step_saving)} fewer</b>${best.compute_saving !== null ? ` and about ${pct(best.compute_saving)} less compute` : ""}. `
        : `No starting pattern cut the steps by more than 5%. `;
      const fo = r.summary.find((s) => s.condition === "fourier");
      if (fo && fo !== best && fo.median_steps !== null && fo.step_saving > 0.05)
        f += `Waves predicted by the task's symmetry alone, with no trained model involved, needed ${fo.median_steps.toLocaleString()} steps (${pct(fo.step_saving)} fewer). `;
      const sh = r.summary.find((s) => s.condition === "shuffled"), tr = r.summary.find((s) => s.condition === "transplant");
      if (sh && tr && sh.reached === 0 && tr.median_steps !== null)
        f += "The shuffled control never generalised within the limit even though it holds exactly the same numbers, so it is the arrangement, the pattern, that carries the benefit. ";
      else if (sh && tr && sh.median_steps !== null && tr.median_steps !== null)
        f += tr.median_steps < sh.median_steps * 0.9 ? "The shuffled control was slower than the real transplant, so the arrangement of the numbers, not just their scale, is what helped. " : "The shuffled control did about as well as the real transplant, so the benefit may come from the scale of the numbers rather than the pattern. ";
    }
    f += ` With ${r.seeds.length} seed${r.seeds.length === 1 ? "" : "s"} per condition, treat differences under about 20% as noise.`;
    out.insertAdjacentHTML("beforeend", `<p class="finding" style="font-size:18px;margin:18px 0">${f}</p>`);
  }
  const chart = document.createElement("div");
  out.appendChild(chart);
  const series = [];
  for (const [c, rs] of Object.entries(byCond)) rs.forEach((x, i) => series.push({ name: i === 0 ? (SEED_CONDS.find((s) => s[0] === c) || [c, c])[1] : "", color: COND_COLORS[c] || "#888", width: 1.6, opacity: 0.85, points: x.history.step.map((s, j) => [s, x.history.test_acc[j]]) }));
  lineChart(chart, { series, yMin: 0, yMax: 1, xMin: 0, xLabel: "Training step", yFmt: (v) => pct(v) });
  if (!r.live && r.summary.length) {
    out.insertAdjacentHTML("beforeend", `<div class="scroll" style="margin-top:12px"><table><thead><tr><th>Starting point</th><th class="num">Generalised</th><th class="num">Median steps</th><th class="num">Fewer steps</th><th class="num">Less compute</th><th>Per seed</th></tr></thead><tbody>${
      r.summary.map((s) => `<tr><td><span style="color:${COND_COLORS[s.condition]}">■</span> ${esc(s.label)}</td><td class="num">${s.reached} of ${s.runs}</td><td class="num">${s.median_steps === null ? "–" : s.median_steps.toLocaleString()}</td>
        <td class="num">${s.condition === "random" ? "baseline" : s.step_saving === null ? "–" : pct(s.step_saving)}</td><td class="num">${s.condition === "random" ? "baseline" : s.compute_saving === null ? "–" : pct(s.compute_saving)}</td>
        <td class="muted">${byCond[s.condition].map((x) => (x.steps_to_target === null ? `>${x.steps_run}` : x.steps_to_target)).join(", ")}</td></tr>`).join("")
    }</tbody></table></div><p class="sub" style="margin-top:8px">Compute counts a forward pass over all parameters and a backward pass over the trainable ones for every step. A frozen embedding saves a little per step; the large savings come from needing fewer steps.</p>`);
  }
}

// ------------------------------------------------------------------ lab: pair
function renderPair(host, r) {
  const circ = r.circuits.pairs;
  const avg = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const rel = avg(circ.map((c) => c.relabeled)), raw = avg(circ.map((c) => c.raw)), nul = avg(circ.map((c) => c.null));
  const A = `<b class="ca">seed ${r.a}</b>`, B = `<b class="cb">seed ${r.b}</b>`;
  let finding = `Stored as they are, ${A} and ${B} look unrelated: weight cosine <b>${fmt(r.overall.raw)}</b>. `;
  finding += `Rewriting ${B} in ${A}'s coordinates, which leaves its outputs unchanged, lifts that to <b class="cab">${fmt(r.overall.aligned)}</b>. `;
  if (rel !== null) {
    finding += rel > nul + 0.2
      ? `Then comes the part the raw numbers hide: once ${B}'s inputs are relabelled by the symmetry of the problem, its circuits match ${A}'s at <b class="cab">${fmt(rel)}</b> average neuron correlation, against <b>${fmt(nul)}</b> for relabellings that should not work. The two models are built from the same parts, tuned to different frequencies.`
      : `Relabelling by the problem's symmetry does not produce a clear match (${fmt(rel)} against a null of ${fmt(nul)}), so these two models may genuinely differ, or one has not finished learning.`;
  }
  host.innerHTML = `<div class="head" style="margin-top:28px"><h1>Seed ${r.a} and seed ${r.b}</h1></div><p class="finding">${finding}</p>`;

  const lad = document.createElement("div");
  lad.className = "ladder";
  const rung = (step, big, what, fill) => `<div class="rung"><div class="step">${step}</div><div class="big">${big}</div><div class="what">${what}</div><div class="fill" style="width:${Math.max(0, Math.min(1, fill)) * 100}%"></div></div>`;
  lad.innerHTML = rung("1. Raw weights", fmt(r.overall.raw), "Cosine of all parameters as stored.", r.overall.raw)
    + rung("2. Weight symmetries removed", fmt(r.overall.aligned), `Rotation, head and neuron order fixed. B's logits moved by at most ${fmt(r.drift, 5)}, so nothing about its behaviour changed.`, r.overall.aligned)
    + rung("3. Task symmetry removed", rel === null ? "–" : fmt(rel), `Average matched-neuron correlation per circuit after relabelling numbers by u. Null: ${fmt(nul)}.`, rel || 0);
  host.appendChild(lad);

  const [cp, cb] = panel("Circuit by circuit",
    "Each key frequency drives its own clock circuit. For every pair of circuits, B's inputs are relabelled by the u that maps its frequency onto A's, and the circuits' neurons are matched. Raw is the same matching without relabelling; null uses relabellings that should not line up.");
  cb.className = "scroll";
  const bar = (v, color) => `<div class="bar"><i style="width:${Math.max(0, v) * 100}%;background:${color}"></i></div>`;
  cb.innerHTML = circ.length ? `<table><thead><tr><th>A frequency</th><th>B frequency</th><th class="num">Relabel u</th><th class="num">Neurons A / B</th><th class="num">Raw</th><th class="num">Null</th><th>After relabelling</th><th class="num"></th></tr></thead><tbody>${
    circ.map((c) => `<tr><td>${c.ka}</td><td>${c.kb}</td><td class="num">× ${c.u}</td><td class="num">${c.n_a} / ${c.n_b}</td><td class="num">${fmt(c.raw)}</td><td class="num">${fmt(c.null)}</td><td>${bar(c.relabeled, "var(--ab)")}</td><td class="num">${fmt(c.relabeled)}</td></tr>`).join("")
  }</tbody></table><p class="sub" style="margin-top:10px">One relabelling for the whole model (u = ${r.circuits.global_best_u}) raises MLP CKA from ${fmt(r.circuits.global_identity_cka)} to ${fmt(r.circuits.global_best_cka)}. A single u can only fix every circuit at once when the two frequency sets are multiples of each other, which is why the per-circuit view is the fairer test.</p>`
    : `<p class="muted">No circuits had enough tuned neurons to match. At least one model probably has not learned the task.</p>`;
  host.appendChild(cp);

  const row = document.createElement("div");
  row.className = "cols";
  const [ip, ib] = panel("Walking from A's weights to B's",
    `Accuracy along the straight line between the two parameter sets. Loss barrier: ${fmt(r.interp.naive_barrier)} raw, ${fmt(r.interp.aligned_barrier)} after alignment.`);
  row.appendChild(ip);
  const [np_, nb] = panel("Neuron twins", `Correlation of each A neuron with its matched B neuron, against random pairs. Mean ${fmt(r.neurons.matched_mean)} matched, ${fmt(r.neurons.random_mean)} random; ${pct(r.neurons.above_0_8)} of neurons have a twin above 0.8.`);
  row.appendChild(np_);
  host.appendChild(row);
  lineChart(ib, {
    series: [
      { name: "Raw path", color: css("--muted"), dash: "5 4", points: r.interp.naive.map((o) => [o.t, o.acc]), dots: true },
      { name: "After weight symmetries", color: css("--ab"), points: r.interp.aligned.map((o) => [o.t, o.acc]), dots: true },
    ], yMin: 0, yMax: 1, xMin: 0, xMax: 1, xLabel: "Position between A (0) and B (1)", yFmt: (v) => pct(v),
  });
  const edges = r.neurons.edges;
  groupedBars(nb, {
    labels: edges.slice(0, -1).map((e) => fmt(e + 0.05, 2)),
    series: [{ name: "Matched pairs", color: css("--ab"), values: r.neurons.hist_matched }, { name: "Random pairs", color: css("--line"), values: r.neurons.hist_random }],
    yLabel: "Neurons",
  });

  const row2 = document.createElement("div");
  row2.className = "cols";
  const [sp, sb] = panel("Stitching: can B finish A's work?",
    "A's state at a point is mapped into B's space, then B completes the forward pass. Accuracy on held-out pairs. Mapping the token embeddings is expected to work, since a few dozen tokens can be mapped anywhere in a wide space; the later sites are the real test.");
  sb.className = "scroll";
  const site = { resid_pre: "Token embeddings", resid_mid: "After attention", resid_post: "After MLP" };
  sb.innerHTML = `<table><thead><tr><th>Stitch point</th><th class="num">Plugged in raw</th><th class="num">Rotation only</th><th class="num">Affine map</th></tr></thead><tbody>${
    r.stitch.rows.map((x) => `<tr><td>${site[x.site]}</td><td class="num">${pct(x.identity, 1)}</td><td class="num">${pct(x.orthogonal, 1)}</td><td class="num"><b>${pct(x.affine, 1)}</b></td></tr>`).join("")
  }</tbody></table><p class="sub" style="margin-top:10px">For reference, A alone scores ${pct(r.stitch.acc_a, 1)} and B alone ${pct(r.stitch.acc_b, 1)}. When the two use different frequencies, no linear map can turn one set of clocks into the other after attention, so mid-network stitching failing is itself evidence of different routes.</p>`;
  row2.appendChild(sp);
  const [kp, kb] = panel("Where representations agree", "Linear CKA between every site in A (rows) and every site in B (columns).");
  row2.appendChild(kp);
  host.appendChild(row2);
  heatmap(kb, { matrix: r.cka.matrix, rows: r.cka.labels, cols: r.cka.labels, values: true, rowLabelWidth: 150, colLabelHeight: 110, rowTitle: `Seed ${r.a}`, colTitle: `Seed ${r.b}` });

  const row3 = document.createElement("div");
  row3.className = "cols";
  const [fp, fb] = panel("Frequency energy side by side", `Key frequencies: A uses ${r.fourier.key_a.join(", ")}; B uses ${r.fourier.key_b.join(", ")}.`);
  row3.appendChild(fp);
  const [tp, tb] = panel("Parameter by parameter", "Cosine similarity per tensor, before and after removing weight symmetries.");
  tb.className = "scroll";
  tb.innerHTML = `<table><thead><tr><th>Tensor</th><th>Shape</th><th class="num">Raw</th><th>After alignment</th><th class="num"></th></tr></thead><tbody>${
    r.tensors.map((t) => `<tr><td>${t.name}</td><td class="muted">${t.shape.join(" × ")}</td><td class="num">${fmt(t.raw)}</td><td>${bar(t.aligned, "var(--ab)")}</td><td class="num">${fmt(t.aligned)}</td></tr>`).join("")
  }</tbody></table><p class="sub" style="margin-top:10px">Heads matched A→B: ${r.heads.map((h) => `${h.a}→${h.b}`).join(", ")}. The bias b_in is not rotated, so its raw and aligned values are similar.</p>`;
  row3.appendChild(tp);
  host.appendChild(row3);
  groupedBars(fb, {
    labels: r.fourier.a.map((_, i) => String(i + 1)),
    series: [{ name: `Seed ${r.a}`, color: css("--a"), values: r.fourier.a }, { name: `Seed ${r.b}`, color: css("--b"), values: r.fourier.b }],
    yLabel: "Share of energy", yFmt: (v) => pct(v),
  });
  host.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ------------------------------------------------------------------ hub
const PRESETS = [
  { t: "Tiny smoke test", d: "Two 14M PolyPythia seeds. Downloads in seconds.", a: "EleutherAI/pythia-14m-seed1", b: "EleutherAI/pythia-14m-seed2" },
  { t: "Same recipe, different seed", d: "Pythia 70M, original run vs PolyPythia seed 1.", a: "EleutherAI/pythia-70m", b: "EleutherAI/pythia-70m-seed1" },
  { t: "Same init, different data order", d: "Weights start identical; only the order of training data differs.", a: "EleutherAI/pythia-160m-data-seed1", b: "EleutherAI/pythia-160m-data-seed2" },
  { t: "Same data, different init", d: "Data order fixed; only the initial weights differ.", a: "EleutherAI/pythia-160m-weight-seed1", b: "EleutherAI/pythia-160m-weight-seed2" },
  { t: "Early vs finished", d: "Pythia 70M at step 3000 vs the final checkpoint.", a: "EleutherAI/pythia-70m@step3000", b: "EleutherAI/pythia-70m" },
  { t: "Different families", d: "GPT-2 small vs Pythia 160M. Different tokenizers, so sentences are compared.", a: "openai-community/gpt2", b: "EleutherAI/pythia-160m" },
  { t: "Laya: base vs fine-tune", d: "Weights only; about 1.7 GB each. Is the fine-tune a low-rank update?", a: "convaiinnovations/laya", b: "convaiinnovations/laya::typed-decisions" },
];
function renderPresets() {
  const ul = $("#presets");
  for (const p of PRESETS) {
    const li = document.createElement("li");
    li.innerHTML = `<button><span class="t1">${esc(p.t)}</span><span class="t2">${esc(p.d)}</span></button>`;
    li.querySelector("button").onclick = () => { $("#hub-a").value = p.a; $("#hub-b").value = p.b; presetFilled(["#hub-a", "#hub-b"], ["#hub-weights", "#hub-acts"], `A and B are now ${p.a.split("/").pop()} and ${p.b.split("/").pop()}. Press Compare weights or Compare activations to run.`); };
    ul.appendChild(li);
  }
}
async function loadHubHistory() {
  S.hubHistory = await api("/api/hub/history").catch(() => []);
  const ul = $("#hub-history");
  ul.innerHTML = S.hubHistory.length ? "" : `<li class="empty">Results you run appear here.</li>`;
  for (const h of S.hubHistory) {
    const li = document.createElement("li");
    const short = (s) => s.split("/").pop();
    li.innerHTML = `<button><span class="t1">${esc(short(h.a))} vs ${esc(short(h.b))}</span><span class="t2">${h.kind === "weights" ? "Weights" : "Activations"}, ${new Date(h.created * 1000).toLocaleString()}</span></button>`;
    li.querySelector("button").onclick = async () => renderHub(await api(`/api/hub/history/${h.id}`));
    ul.appendChild(li);
  }
}
function hubBody() {
  const a = $("#hub-a").value.trim(), b = $("#hub-b").value.trim();
  if (!a || !b) { toast("Enter two model names.", true); return null; }
  const raw = $("#hub-texts").value.trim();
  const texts = raw ? raw.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean) : null;
  return { a, b, texts, max_tokens: parseInt($("#hub-tokens").value, 10) || 1500 };
}
$("#hub-weights").onclick = () => {
  const body = hubBody(); if (!body) return;
  $("#hub-stage").innerHTML = `<p class="spinner">Downloading and comparing weights. Large checkpoints take a while the first time; later runs use the local cache.</p>`;
  runJob("/api/hub/weights", body, (blob) => { renderHub(blob); loadHubHistory(); });
};
$("#hub-acts").onclick = () => {
  const body = hubBody(); if (!body) return;
  if (body.texts && body.texts.length < 8) return toast("Give at least 8 texts, or leave the box empty for the built-in corpus.", true);
  $("#hub-stage").innerHTML = `<p class="spinner">Loading both models and running them on the probe texts…</p>`;
  runJob("/api/hub/activations", body, (blob) => { renderHub(blob); loadHubHistory(); });
};

function renderHub(blob) {
  const stage = $("#hub-stage");
  stage.innerHTML = `<div class="head"><h1><span style="color:var(--a)">${esc(blob.a)}</span> <span class="muted">and</span> <span style="color:var(--b)">${esc(blob.b)}</span></h1>
    <span class="meta">${blob.kind === "weights" ? "Weight comparison" : "Activation comparison"}, ${new Date(blob.created * 1000).toLocaleString()}</span></div>`;
  if (blob.kind === "weights") renderWeights(stage, blob.result);
  else renderActs(stage, blob.result);
}

function renderWeights(stage, r) {
  const s = r.summary;
  stage.insertAdjacentHTML("beforeend", `<p class="finding">${esc(s.verdict)}</p>`);
  const [kp, kb] = panel("Summary");
  kb.className = "kv";
  kb.innerHTML = [
    ["Tensors paired", `${s.paired}`], ["Parameters compared", Number(s.params_compared).toLocaleString()],
    ["Weight-matrix cosine", fmt(s.overall_cosine, 3)], ["Relative difference", fmt(s.overall_rel_diff, 3)],
    ["Median spectral distance", s.median_spec_dist === null ? "–" : fmt(s.median_spec_dist, 3)],
    ["Directions holding 90% of the change", s.median_delta_rank_frac === null ? "–" : pct(s.median_delta_rank_frac, 1)],
  ].map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  stage.appendChild(kp);

  if (r.by_layer.length) {
    const row = document.createElement("div");
    row.className = "cols";
    const [lp, lb] = panel("Layer by layer", "Raw cosine and relative difference, weighted by tensor size.");
    const [sp, sb] = panel("Singular-value shape by layer", "Distance between normalised singular-value profiles. 0 means the matrices stretch space the same way, whatever their basis. This is the fair comparison across seeds.");
    row.append(lp, sp);
    stage.appendChild(row);
    const L = r.by_layer;
    lineChart(lb, { series: [
      { name: "Cosine", color: css("--ab"), points: L.map((x) => [x.layer, x.cosine]), dots: true },
      { name: "Relative difference", color: css("--muted"), dash: "4 3", points: L.map((x) => [x.layer, x.rel_diff]), dots: true },
    ], xLabel: "Layer" });
    lineChart(sb, { series: [{ name: "Spectral distance", color: css("--ab"), points: L.filter((x) => x.spec_dist !== null).map((x) => [x.layer, x.spec_dist]), dots: true }], yMin: 0, xLabel: "Layer" });
  }

  const [tp, tb] = panel("Matrices", "Spectra are drawn on a log scale; solid is A, dashed is B. The last column shows what share of directions carries 90% of the difference B − A: small values mean the difference is low-rank.");
  tb.className = "scroll";
  const mats = r.tensors.filter((t) => t.sv_a);
  tb.innerHTML = `<table><thead><tr><th>Tensor</th><th>Shape</th><th>Spectra</th><th class="num">Cosine</th><th class="num">Relative diff</th><th class="num">Spectral distance</th><th class="num">Change rank (90%)</th></tr></thead><tbody>${
    mats.slice(0, 200).map((t) => `<tr><td>${esc(t.name)}</td><td class="muted">${t.shape.join(" × ")}</td><td>${sparkPair(t.sv_a, t.sv_b)}</td><td class="num">${fmt(t.cosine, 3)}</td><td class="num">${fmt(t.rel_diff, 3)}</td><td class="num">${fmt(t.spec_dist, 3)}</td><td class="num">${t.delta_rank90} of ${t.full_rank}</td></tr>`).join("")
  }</tbody></table>${mats.length > 200 ? `<p class="sub">Showing 200 of ${mats.length} matrices.</p>` : ""}`;
  stage.appendChild(tp);
}

function renderActs(stage, r) {
  const d = r.diag;
  const mid = d.slice(1, -1);
  const avg = (xs) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
  let finding = `At matched depth the two models organise ${r.mode === "token" ? "tokens" : "sentences"} with an average CKA of <b class="cab">${fmt(avg(mid.map((x) => x.cka)))}</b> in the middle layers, and share <b>${pct(avg(mid.map((x) => x.knn)))}</b> of their nearest neighbours. `;
  if (r.neurons.length) {
    const m = avg(r.neurons.map((n) => n.matched_mean)), rn = avg(r.neurons.map((n) => n.random_mean)), hi = avg(r.neurons.map((n) => n.above_0_5));
    finding += `Matched one to one, MLP neurons correlate at <b class="cab">${fmt(m)}</b> on average (random pairs: ${fmt(rn)}), and <b>${pct(hi)}</b> have a twin above 0.5. `;
  }
  if (r.agreement) finding += `They pick the same next token <b>${pct(r.agreement.top1)}</b> of the time.`;
  stage.insertAdjacentHTML("beforeend", `<p class="finding">${finding}</p>`);
  for (const n of r.notes) stage.insertAdjacentHTML("beforeend", `<p class="note">${esc(n)}</p>`);

  const row = document.createElement("div");
  row.className = "cols";
  const [hp, hb] = panel("Layer against layer", `Linear CKA over ${r.rows} ${r.mode === "token" ? "tokens" : "sentences"}. A bright diagonal means the two models build their representations in the same order.`);
  const [dp, db] = panel("Agreement with depth", "CKA and mutual nearest-neighbour overlap at matched depth.");
  row.append(hp, dp);
  stage.appendChild(row);
  heatmap(hb, { matrix: r.cka, rows: r.cka.map((_, i) => (i === 0 ? "Embed" : `Layer ${i}`)), cols: r.cka[0].map((_, j) => (j === 0 ? "Embed" : `Layer ${j}`)), values: r.cka.length <= 13, rowLabelWidth: 64, colLabelHeight: 56, rowTitle: "A", colTitle: "B" });
  lineChart(db, { series: [
    { name: "CKA", color: css("--ab"), points: d.map((x) => [x.a, x.cka]), dots: true },
    { name: "Shared nearest neighbours", color: css("--muted"), dash: "4 3", points: d.map((x) => [x.a, x.knn]), dots: true },
  ], yMin: 0, yMax: 1, xLabel: "Layer of A (0 = embeddings)" });

  if (r.neurons.length || r.agreement) {
    const row2 = document.createElement("div");
    row2.className = "cols";
    if (r.neurons.length) {
      const [np_, nb] = panel("Neuron twins by layer", "Mean correlation of each sampled A neuron with its one-to-one match in B, its best match allowing reuse, and a random B neuron.");
      row2.appendChild(np_);
      groupedBars(nb, {
        labels: r.neurons.map((n) => `L${n.a}`),
        series: [{ name: "Matched", color: css("--ab"), values: r.neurons.map((n) => n.matched_mean) },
          { name: "Best match", color: css("--a"), values: r.neurons.map((n) => n.best_mean) },
          { name: "Random", color: css("--line"), values: r.neurons.map((n) => n.random_mean) }],
        yMin: 0, yMax: 1,
      });
    }
    if (r.agreement) {
      const [ap, ab] = panel("Next-token behaviour", `Measured over ${r.agreement.tokens.toLocaleString()} tokens.`);
      ab.className = "kv";
      ab.innerHTML = [["Same top prediction", pct(r.agreement.top1, 1)], ["KL(A ‖ B), nats", fmt(r.agreement.kl_a_b, 3)], ["Loss of A", fmt(r.agreement.loss_a, 3)], ["Loss of B", fmt(r.agreement.loss_b, 3)]]
        .map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
      row2.appendChild(ap);
    }
    stage.appendChild(row2);
  }

  if (r.stitch.length) {
    const [sp, sb] = panel("Stitching: B finishing from A's state",
      "A's hidden state is mapped into B's space (fit on half the texts) and injected into B at the same depth; B's next-token loss is measured on the other half. Close to B's own loss means the two models carry interchangeable information at that layer. Plugging A's state in unmapped shows how much the coordinates differ.");
    stage.appendChild(sp);
    groupedBars(sb, {
      labels: r.stitch.map((x) => `A${x.a_layer}→B${x.b_layer}`),
      series: [{ name: "B on its own", color: css("--b"), values: r.stitch.map((x) => x.own) },
        { name: "Affine map", color: css("--ab"), values: r.stitch.map((x) => x.affine) },
        { name: "Rotation only", color: css("--a"), values: r.stitch.map((x) => x.orthogonal ?? null) },
        { name: "Unmapped", color: css("--line"), values: r.stitch.map((x) => x.identity ?? null) }],
      yLabel: "Loss (lower is better)", yMin: 0,
    });
  }
}

// ------------------------------------------------------------------ boot
(async function boot() {
  try {
    const h = await api("/api/health");
    $("#env").textContent = `v${h.version}, torch ${h.torch.split("+")[0]}, ${h.device}`;
  } catch { $("#env").textContent = "Server not reachable"; }
  renderPresets();
  await loadRuns();
  loadHubHistory();
  const jobs = await api("/api/jobs").catch(() => []);
  const active = jobs.find((j) => ["queued", "running"].includes(j.state));
  if (active) toast(`A job is still running on the server: ${active.title}.`);
})();
