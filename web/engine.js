"use strict";
// Engine: rules mined from measurements, experiments that test them, blueprints, world view.
(function () {
  const E = { L: null, meta: null, tab: "autopilot", focus: null, pre: null, ap: null, timer: null };
  const STATUS = { holds: ["holds", "var(--ab)"], rejected: ["rejected", "var(--b)"], mixed: ["mixed", "var(--warn)"], inconclusive: ["inconclusive", "var(--warn)"],
    observed: ["observed, untested", "var(--muted)"], "needs more data": ["needs more data", "var(--muted)"] };
  const KIND = { invariant: "Invariant", "near-invariant": "Near-invariant", absence: "Absence", ordering: "Ordering", location: "Location",
    redundancy: "Redundancy", sparsity: "Sparsity", "low-rank": "Low rank", convergence: "Convergence", scaling: "Scaling", lab: "Lab result", lever: "Lever" };
  const LEVER_OF_KIND = { redundancy: "share", sparsity: "topk", "low-rank": "rank", lab: "seed" };
  const badge = (st) => { const [t, c] = STATUS[st] || [st, "var(--muted)"]; return `<span class="badge" style="--c:${c}">${esc(t)}</span>`; };
  const num = (n) => (n === null || n === undefined ? "–" : n >= 1e12 ? `${(n / 1e12).toFixed(1)}T` : n >= 1e9 ? `${(n / 1e9).toFixed(1)}G` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(Math.round(n)));

  async function refresh(mine = false) {
    try {
      E.meta = E.meta || (await api("/api/engine/meta"));
      E.L = mine ? await api("/api/engine/mine", { method: "POST" }) : await api("/api/engine/ledger");
    } catch (e) { toast(e.message, true); return; }
    const R = Object.values(E.L.rules);
    const count = (f) => R.filter(f).length;
    $("#en-counts").innerHTML = [["Rules", R.length], ["Tested", count((r) => ["holds", "rejected", "mixed"].includes(r.status))],
      ["Hold", count((r) => r.status === "holds")], ["Experiments", Object.keys(E.L.experiments).length]]
      .map(([k, v]) => `<div><div class="k">${k}</div><div class="v" style="font-size:18px">${v}</div></div>`).join("");
    $("#en-updated").textContent = E.L.updated ? `Ledger updated ${new Date(E.L.updated * 1000).toLocaleString()}; ${E.L.models || 0} profiled models.` : "Nothing mined yet.";
    keyState();
    render();
  }
  async function keyState() {
    const m = await api("/api/engine/meta").catch(() => null);
    if (m) { E.meta = m; $("#en-keystate").textContent = m.has_anthropic_key ? `A key is saved. The world view will be written by ${m.default_model}.` : "No key saved."; }
  }
  $("#en-mine").onclick = () => refresh(true).then(() => toast("Rules mined from the library and the lab."));
  $("#en-keysave").onclick = async () => {
    const k = $("#en-key").value.trim();
    await api("/api/settings/anthropic", { method: "POST", body: { key: k || null } }).catch((e) => toast(e.message, true));
    $("#en-key").value = "";
    toast(k ? "Key saved." : "Key removed.");
    keyState();
  };

  function render() {
    const st = $("#en-stage");
    st.innerHTML = `<div class="head"><h1>Engine</h1><span class="meta">From measurements to rules to tested designs</span></div>
      <div class="subtabs" role="tablist" id="en-tabs"></div><div id="en-body"></div>`;
    const bar = $("#en-tabs");
    if (E.timer) { clearTimeout(E.timer); E.timer = null; }
    for (const [k, label] of [["autopilot", "Autopilot"], ["rules", "Rules"], ["experiments", "Experiments"], ["blueprint", "Blueprint"], ["world", "World view"]]) {
      const b = document.createElement("button");
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", E.tab === k);
      b.textContent = label;
      b.onclick = () => { E.tab = k; render(); };
      bar.appendChild(b);
    }
    ({ autopilot, rules, experiments, blueprint, world })[E.tab]($("#en-body"));
  }

  // ------------------------------------------------------------ autopilot
  const dur = (sec) => { const tot = Math.max(0, Math.round(sec / 60)); const h = Math.floor(tot / 60), m = tot % 60; return h ? `${h} h ${m} min` : `${m} min`; };
  const stepLabel = (st) => st.type === "profile" ? `Profile ${st.spec}` : st.type === "experiment" ? `Test ${E.meta.levers[st.lever].toLowerCase()} on ${E.meta.tasks[st.task].split(",")[0].toLowerCase()}${st.rule ? ` (${st.rule})` : ""}`
    : ({ mine: "Mine rules", plan: "Choose the most informative experiments", blueprint: "Build a blueprint from what held", narrate: "Write the world view" })[st.type] || st.type;

  async function autopilot(body) {
    let ap;
    try { ap = await api("/api/autopilot"); } catch (e) { body.innerHTML = `<p class="warn">${esc(e.message)}</p>`; return; }
    E.ap = ap;
    if (E.tab !== "autopilot") return;
    body.innerHTML = "";
    const active = ["running", "paused", "stopping"].includes(ap.status);
    if (active) apProgress(body, ap);
    else apForm(body, ap);
    if (ap.report) apReport(body, ap.report, active ? "Previous campaign" : "Latest campaign report");
    if (active) E.timer = setTimeout(() => { if (E.tab === "autopilot" && document.querySelector("#en-body")) { autopilot($("#en-body")); refreshCounts(); } }, 5000);
  }

  async function refreshCounts() {
    try { E.L = await api("/api/engine/ledger"); } catch { return; }
    const R = Object.values(E.L.rules);
    const count = (f) => R.filter(f).length;
    $("#en-counts").innerHTML = [["Rules", R.length], ["Tested", count((r) => ["holds", "rejected", "mixed"].includes(r.status))],
      ["Hold", count((r) => r.status === "holds")], ["Experiments", Object.keys(E.L.experiments).length]]
      .map(([k, v]) => `<div><div class="k">${k}</div><div class="v" style="font-size:18px">${v}</div></div>`).join("");
  }

  function apForm(body, ap) {
    const [p, b] = panel("Run the loop overnight",
      "Autopilot profiles the models you pick, mines rules, chooses the experiments that should teach the most (rules with weak support but a large possible saving come first, and every lever gets tried on every task at least once), builds a blueprint from whatever held, and writes the world view. It keeps the Mac awake while it works, continues after a restart, and leaves a report here.");
    const mk = (c) => `<label class="check"><input type="checkbox" data-model="${esc(c.spec)}" ${c.default && !c.profiled ? "checked" : ""}> ${esc(c.spec)} <span class="muted">${esc(c.note || "")}</span>${c.profiled ? ` <span class="badge" style="--c:var(--ab)">in library</span>` : ""}</label>`;
    b.innerHTML = `<h3 style="margin-top:4px">Models to profile</h3><div class="conds">${ap.curated.map(mk).join("")}</div>
      <label class="check"><input type="checkbox" id="ap-traj"> Add Pythia 70M at three training checkpoints (steps 1,000, 8,000 and 33,000) to see which patterns form first</label>
      <label>More models, one per line (names, links or folders)<textarea id="ap-extra" rows="2" placeholder="owner/name"></textarea></label>
      <h3>Experiments</h3>
      <div class="grid-form" style="grid-template-columns:repeat(4,1fr);max-width:900px">
        <label>Most experiments<input id="ap-max" type="number" value="8" min="0" max="40"></label>
        <label>Seeds each<input id="ap-seeds" value="1, 2, 3"></label>
        <label>Training length (× default)<input id="ap-scale" type="number" value="1" step="0.25" min="0.25" max="4"></label>
        <label>Time budget (hours)<input id="ap-hours" type="number" value="8" step="0.5" min="0.5" max="72"></label>
      </div>
      <label class="check"><input type="checkbox" id="ap-ts" checked> Include TinyStories language-model tests (slower, closest to real language)</label>
      <label class="check"><input type="checkbox" id="ap-bp" checked> Build a blueprint from the levers that hold</label>
      <label class="check"><input type="checkbox" id="ap-claude" ${E.meta && E.meta.has_anthropic_key ? "checked" : "disabled"}> Have Claude write the world view${E.meta && E.meta.has_anthropic_key ? "" : " (add an API key on the left first; otherwise the built-in digest is used)"}</label>
      <label class="check"><input type="checkbox" id="ap-re"> Profile models again even if they are already in the library</label>
      <button class="primary" id="ap-go" style="margin-top:12px">Start the campaign</button>
      <p class="sub" id="ap-est" style="margin-top:8px"></p>`;
    body.appendChild(p);
    const estimate = () => {
      const n = b.querySelectorAll("[data-model]:checked").length + ($("#ap-traj").checked ? 3 : 0) + $("#ap-extra").value.split("\n").filter((x) => x.trim()).length;
      const ex = parseInt($("#ap-max").value, 10) || 0, seeds = $("#ap-seeds").value.split(/[\s,]+/).filter(Boolean).length || 1, sc = parseFloat($("#ap-scale").value) || 1;
      const mins = n * 4 + ex * seeds * 2 * 1.5 * sc * ($("#ap-ts").checked ? 2 : 1) + ($("#ap-bp").checked ? 20 * sc : 0);
      $("#ap-est").textContent = `Rough estimate on an Apple-silicon Mac: ${dur(mins * 60)} (profiling about 4 min per small model; larger models and TinyStories take longer). The budget stops new tests when time runs out and still writes the report.`;
    };
    b.querySelectorAll("input,textarea").forEach((el) => el.addEventListener("input", estimate));
    estimate();
    $("#ap-go").onclick = async () => {
      const models = [...b.querySelectorAll("[data-model]:checked")].map((c) => c.dataset.model).concat($("#ap-extra").value.split("\n").map((x) => x.trim()).filter(Boolean));
      const cfg = { models, trajectory: $("#ap-traj").checked, reprofile: $("#ap-re").checked, max_experiments: parseInt($("#ap-max").value, 10) || 0,
        seeds: $("#ap-seeds").value.split(/[\s,]+/).filter(Boolean).map(Number), scale: parseFloat($("#ap-scale").value) || 1,
        hours: parseFloat($("#ap-hours").value) || 8, tinystories: $("#ap-ts").checked, blueprint: $("#ap-bp").checked, claude: $("#ap-claude").checked };
      try { await api("/api/autopilot/start", { method: "POST", body: cfg }); toast("Campaign started. You can close this page; it keeps running."); render(); }
      catch (e) { toast(e.message, true); }
    };
  }

  function apProgress(body, ap) {
    const done = ap.done.length, left = ap.queue.length + (ap.current ? 1 : 0);
    const el = (Date.now() / 1000 - ap.started), budget = ap.deadline - Date.now() / 1000;
    const [p, b] = panel(`Campaign ${ap.status === "paused" ? "paused" : ap.status === "stopping" ? "stopping" : "running"}`,
      `Started ${new Date(ap.started * 1000).toLocaleString()}, ${dur(el)} ago. ${budget > 0 ? `${dur(budget)} of the time budget left.` : "Time budget used up; wrapping up."} It keeps running with this page closed.`);
    b.innerHTML = `<div class="kv"><div><div class="k">Steps done</div><div class="v">${done}</div></div><div><div class="k">Steps left</div><div class="v">${left}</div></div>
      <div><div class="k">Failed</div><div class="v">${ap.done.filter((d) => !d.ok).length}</div></div></div>
      <div class="meter" style="width:100%;margin:14px 0"><div style="width:${(done / Math.max(1, done + left)) * 100}%"></div></div>
      <p><b>Now:</b> ${ap.current ? `${esc(stepLabel(ap.current))}, for ${dur(Date.now() / 1000 - ap.current.started)}` : ap.status === "paused" ? "paused between steps" : "starting the next step"}</p>
      <div class="pk-row">${ap.status === "paused" ? `<button class="primary" data-a="resume">Resume</button>` : ap.status === "running" ? `<button class="ghost" data-a="pause">Pause after this step</button>` : ""}
      ${ap.status !== "stopping" ? `<button class="ghost" data-a="stop">Stop and write the report</button>` : ""}</div>
      <div class="two" style="margin-top:14px"><div><h3>Next</h3><ol class="aplist">${ap.queue.slice(0, 14).map((q) => `<li>${esc(stepLabel(q))}</li>`).join("") || "<li class='muted'>Nothing queued</li>"}${ap.queue.length > 14 ? `<li class="muted">and ${ap.queue.length - 14} more</li>` : ""}</ol></div>
      <div><h3>Done</h3><ul class="aplist">${ap.done.slice().reverse().slice(0, 14).map((d) => `<li><span style="color:${d.ok ? "var(--ab)" : "var(--b)"}">${d.ok ? "✓" : "✗"}</span> ${esc(stepLabel(d))}<div class="sub" style="margin:0">${esc((d.info || "").slice(0, 160))}</div></li>`).join("") || "<li class='muted'>Nothing yet</li>"}</ul></div></div>
      <details><summary>Log</summary><div class="hex" style="max-height:240px;overflow:auto">${ap.log.slice().reverse().map((l) => `${new Date(l.t * 1000).toLocaleTimeString()}  ${esc(l.msg)}`).join("\n")}</div></details>`;
    b.querySelectorAll("[data-a]").forEach((x) => (x.onclick = async () => { await api(`/api/autopilot/${x.dataset.a}`, { method: "POST" }).catch((e) => toast(e.message, true)); render(); }));
    body.appendChild(p);
  }

  function apReport(body, r, title) {
    const [p, b] = panel(`${title}: ${r.how === "done" ? "finished" : r.how}`, `${new Date(r.started * 1000).toLocaleString()} to ${new Date(r.finished * 1000).toLocaleString()} (${dur(r.hours * 3600)}).`);
    p.style.marginTop = "18px";
    let f = `Profiled ${r.profiled.length} model${r.profiled.length === 1 ? "" : "s"}${r.skipped_profiles.length ? ` (${r.skipped_profiles.length} already in the library)` : ""}, found ${r.new_rules.length} new rule${r.new_rules.length === 1 ? "" : "s"}, and ran ${r.experiments.length} experiment${r.experiments.length === 1 ? "" : "s"}: <b class="cab">${r.holds} held</b>, ${r.rejected} rejected, ${r.experiments.length - r.holds - r.rejected} inconclusive.`;
    if (r.blueprints.length) {
      const best = r.blueprints.flatMap((bp) => bp.results).filter((x) => x.cpc).sort((a, c) => c.cpc - a.cpc)[0];
      if (best) f += ` The blueprint's best result was ${fmt(best.cpc, 2)}× capability per unit of compute on ${esc(best.task.split(",")[0].toLowerCase())}.`;
    }
    if (r.failed.length) f += ` ${r.failed.length} step${r.failed.length === 1 ? "" : "s"} failed; see below.`;
    b.innerHTML = `<p class="finding" style="font-size:18px;margin-top:0">${f}</p>
      ${r.experiments.length ? `<h3>Experiments</h3><table><tbody>${r.experiments.map((x) => `<tr><td>${badge(x.verdict)}</td><td>${esc(x.label)}<div class="sub" style="margin:0">${esc(x.reason)}</div></td><td class="rid">${x.id}</td></tr>`).join("")}</tbody></table>` : ""}
      ${r.changed.length ? `<h3 style="margin-top:14px">Rules that changed status</h3><ul class="aplist">${r.changed.map((x) => `<li><span class="rid">${x.id}</span> ${esc(x.statement)}: ${esc(x.before)} → <b>${esc(x.after)}</b></li>`).join("")}</ul>` : ""}
      ${r.new_rules.length ? `<details><summary>New rules (${r.new_rules.length})</summary><ul class="aplist">${r.new_rules.map((x) => `<li><span class="rid">${x.id}</span> ${esc(x.statement)}</li>`).join("")}</ul></details>` : ""}
      ${r.failed.length ? `<h3 style="margin-top:14px">Failed steps</h3><ul class="aplist">${r.failed.map((x) => `<li>${esc(x.step)}<div class="sub warn" style="margin:0">${esc(x.info)}</div></li>`).join("")}</ul>` : ""}
      <div class="pk-row" style="margin-top:12px"><button class="ghost" data-go="rules">Open the rules</button>${r.world_view ? `<button class="ghost" data-go="world">Read the world view</button>` : ""}</div>`;
    b.querySelectorAll("[data-go]").forEach((x) => (x.onclick = () => { E.tab = x.dataset.go; refresh(false); }));
    body.appendChild(p);
  }

  // ------------------------------------------------------------ rules
  function rules(body) {
    const R = Object.values(E.L.rules);
    if (!R.length) {
      body.innerHTML = `<p class="finding">No rules yet. Profile a few models in the Pattern library (and run the seeding experiment in the lab), then press <b>Mine rules now</b>.</p>`;
      return;
    }
    const tested = R.filter((r) => ["holds", "rejected", "mixed"].includes(r.status));
    const testable = R.filter((r) => !tested.includes(r) && (r.lever || LEVER_OF_KIND[r.kind]) && r.kind !== "absence");
    const unknown = R.filter((r) => !tested.includes(r) && !testable.includes(r) && (r.kind === "absence" || (r.counter && r.counter.length) || r.status === "needs more data"));
    const other = R.filter((r) => !tested.includes(r) && !testable.includes(r) && !unknown.includes(r));
    body.insertAdjacentHTML("beforeend", `<p class="finding">${R.length} rules from ${E.L.models || 0} profiled models and the lab. ${tested.length} tested (${tested.filter((r) => r.status === "holds").length} hold), ${testable.length} ready to test, ${unknown.length} more point at unknown territory.</p>`);
    const section = (title, sub, list) => {
      if (!list.length) return;
      const [p, b] = panel(title, sub);
      b.className = "scroll";
      b.innerHTML = `<table class="rules"><thead><tr><th>Rule</th><th>Kind</th><th>Status</th><th class="num">For / against</th><th class="num">Effect</th><th></th></tr></thead><tbody>${
        list.map((r) => `<tr id="row-${r.id}" class="${E.focus === r.id ? "focus" : ""}"><td><span class="rid">${r.id}</span> ${esc(r.statement)}${r.about ? `<div class="sub" style="margin:2px 0 0">${esc(r.about)}</div>` : ""}
          <details><summary>Evidence (${r.evidence.length})</summary><div class="scroll">${evidenceTable(r.evidence)}</div></details></td>
          <td class="muted">${KIND[r.kind] || r.kind}<div class="sub" style="margin:0">${esc(r.confidence)}</div></td><td>${badge(r.status)}</td>
          <td class="num">${(r.support || []).length} / ${(r.counter || []).length}</td><td class="num">${r.effect === null || r.effect === undefined ? "–" : fmt(r.effect, 2)}${r.unit ? `<div class="sub" style="margin:0">${esc(r.unit)}</div>` : ""}</td>
          <td>${(r.lever || LEVER_OF_KIND[r.kind]) && r.kind !== "absence" ? `<button class="ghost small" data-test="${r.id}">Test</button>` : ""}</td></tr>`).join("")
      }</tbody></table>`;
      b.querySelectorAll("[data-test]").forEach((x) => (x.onclick = () => { const r = E.L.rules[x.dataset.test]; E.pre = prefill(r); E.tab = "experiments"; render(); }));
      body.appendChild(p);
    };
    section("Tested", "Rules an experiment has confirmed or rejected.", tested);
    section("Ready to test", "Each of these suggests a saving. Test runs a controlled experiment and records the verdict here.", testable);
    section("Unknown territory", "Patterns missing everywhere, rules with counterexamples, and results resting on too little data. These are where new probes, bigger models or new experiments are needed.", unknown);
    section("Other observations", "Regularities worth keeping, with no direct efficiency lever yet.", other);
    if (E.focus) { const row = $(`#row-${E.focus}`); if (row) row.scrollIntoView({ block: "center" }); }
  }

  function evidenceTable(ev) {
    if (!ev || !ev.length) return `<p class="muted">No evidence rows.</p>`;
    const keys = [...new Set(ev.flatMap((e) => Object.keys(e)))].slice(0, 7);
    const cell = (v) => (typeof v === "number" ? fmt(v, 3) : typeof v === "boolean" ? (v ? "yes" : "no") : esc(String(v ?? "–")));
    return `<table><thead><tr>${keys.map((k) => `<th>${esc(k.replace(/_/g, " "))}</th>`).join("")}</tr></thead><tbody>${ev.slice(0, 20).map((e) => `<tr>${keys.map((k) => `<td>${cell(e[k])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }

  function prefill(r) {
    const lever = (r.lever && r.lever.lever) || LEVER_OF_KIND[r.kind] || "share";
    const args = {};
    if (lever === "topk" && r.lever && r.lever.k) args.k = r.lever.k;
    if (lever === "rank" && r.lever && r.lever.frac) args.frac = r.lever.frac;
    return { lever, args, rule: r.id, task: lever === "seed" ? "modadd" : "sort" };
  }

  // ------------------------------------------------------------ experiments
  function experiments(body) {
    const m = E.meta, pre = E.pre || { lever: "share", args: {}, task: "sort", rule: "" };
    const rulesWithLever = Object.values(E.L.rules).filter((r) => r.kind !== "absence" && (r.lever || LEVER_OF_KIND[r.kind]));
    const [fp, fb] = panel("Run a controlled experiment", "The baseline and the variant train on the same task with the same seeds; only the lever differs. A lever holds if the variant reaches the same capability, within 1.5× the steps (or 3% of the loss), while saving at least 10% of parameters or compute.");
    fb.innerHTML = `<div class="grid-form" style="max-width:760px;grid-template-columns:repeat(3,1fr)">
      <label>Task<select id="ex-task">${Object.entries(m.tasks).map(([k, v]) => `<option value="${k}" ${k === pre.task ? "selected" : ""}>${esc(v)}</option>`).join("")}</select></label>
      <label>Lever<select id="ex-lever">${Object.entries(m.levers).map(([k, v]) => `<option value="${k}" ${k === pre.lever ? "selected" : ""}>${esc(v)}</option>`).join("")}</select></label>
      <label id="ex-arg-wrap">Setting<input id="ex-arg"></label>
      <label>Seeds<input id="ex-seeds" value="1, 2, 3"></label>
      <label>Step limit<input id="ex-steps" type="number" step="100"></label>
      <label>Tests rule<select id="ex-rule"><option value="">None</option>${rulesWithLever.map((r) => `<option value="${r.id}" ${r.id === pre.rule ? "selected" : ""}>${r.id}: ${esc(r.statement.slice(0, 60))}</option>`).join("")}</select></label>
    </div><button class="primary" id="ex-go" style="margin-top:12px">Run experiment</button><p class="sub" id="ex-note" style="margin-top:8px"></p>`;
    body.appendChild(fp);
    const setArg = () => {
      const lv = $("#ex-lever").value, t = $("#ex-task").value, arch = m.arch[t];
      const lab = { share: `Blocks shared by ${arch.layers} layers`, topk: "Share of neurons kept (0 to 1)", rank: "Rank as a share of width", seed: "No setting" }[lv];
      const def = { share: Math.max(1, Math.floor(arch.layers / 2)), topk: 0.25, rank: 0.25, seed: "" }[lv];
      $("#ex-arg-wrap").firstChild.textContent = lab;
      const pv = pre.lever === lv ? (pre.args.n ?? pre.args.k ?? pre.args.frac) : undefined;
      $("#ex-arg").value = pv ?? def;
      $("#ex-arg").disabled = lv === "seed";
      $("#ex-steps").value = m.opt[t].steps;
      $("#ex-note").textContent = lv === "seed" && t !== "modadd" ? "The seeding lever only applies to modular addition." : lv === "share" && arch.layers < 2 ? "This task's model has one layer, so there is nothing to share; pick sort, copy or TinyStories." : t === "tinystories" ? "TinyStories downloads about 19 MB the first time and trains a byte-level model; expect several minutes per run." : "";
    };
    $("#ex-lever").onchange = setArg;
    $("#ex-task").onchange = setArg;
    setArg();
    E.pre = null;
    $("#ex-go").onclick = () => {
      const lever = $("#ex-lever").value, v = $("#ex-arg").value;
      const args = lever === "share" ? { n: parseInt(v, 10) } : lever === "topk" ? { k: parseFloat(v) } : lever === "rank" ? { frac: parseFloat(v) } : {};
      const bodyReq = { task: $("#ex-task").value, lever, args, seeds: $("#ex-seeds").value.split(/[\s,]+/).filter(Boolean).map(Number),
        steps: parseInt($("#ex-steps").value, 10), rule_id: $("#ex-rule").value || null };
      runJob("/api/engine/experiment", bodyReq, async (exp) => { toast(`Verdict: ${exp.verdict}.`); await refresh(false); E.tab = "experiments"; render(); });
    };

    const X = Object.values(E.L.experiments).sort((a, b) => b.created - a.created);
    if (!X.length) { body.insertAdjacentHTML("beforeend", `<p class="muted">No experiments yet.</p>`); return; }
    for (const x of X.slice(0, 12)) {
      const s = x.summary;
      const [p, b] = panel(`${badge(x.verdict)} ${esc(x.lever_label)} on ${esc(x.task_label)}`, `${x.id}${x.rule ? `, testing ${x.rule}` : ""}, ${x.seeds.length} seed${x.seeds.length > 1 ? "s" : ""}, ${new Date(x.created * 1000).toLocaleString()}. ${esc(x.reason)}${x.source ? ` Data: ${esc(x.source)}.` : ""}`);
      p.style.marginTop = "18px";
      const kv = [["Parameters", `${num(s.params_var)} vs ${num(s.params_base)}`], ["Compute per token", `${num(s.fpt_var)} vs ${num(s.fpt_base)}`]];
      if (s.steps_base !== undefined) kv.push(["Steps to target", `${s.steps_var ?? "not reached"} vs ${s.steps_base ?? "not reached"}`], ["Training compute to target", `${num(s.flops_var)} vs ${num(s.flops_base)}`]);
      else kv.push(["Final score", `${fmt(s.final_var, 3)} vs ${fmt(s.final_base, 3)}`]);
      b.innerHTML = `<div class="kv">${kv.map(([k, v]) => `<div><div class="k">${k} (variant vs baseline)</div><div class="v" style="font-size:16px">${v}</div></div>`).join("")}</div><div class="curve"></div>`;
      const series = [];
      x.runs.baseline.forEach((r, i) => series.push({ name: i ? "" : "Baseline", color: css("--muted"), dash: "4 3", width: 1.5, points: r.history.step.map((st, j) => [st, r.history.score[j]]) }));
      x.runs.variant.forEach((r, i) => series.push({ name: i ? "" : "Variant", color: css("--ab"), width: 1.8, points: r.history.step.map((st, j) => [st, r.history.score[j]]) }));
      body.appendChild(p);
      lineChart(b.querySelector(".curve"), { series, xLabel: "Training step", yLabel: x.task === "tinystories" ? "Bits per character" : "Accuracy", yMin: x.task === "tinystories" ? undefined : 0, yMax: x.task === "tinystories" ? undefined : 1, height: 200 });
    }
  }

  // ------------------------------------------------------------ blueprint
  function blueprint(body) {
    const m = E.meta, R = Object.values(E.L.rules);
    const held = {};
    for (const r of R) if (r.kind === "lever" && r.status === "holds" && r.lever) held[r.lever.lever] = held[r.lever.lever] || r;
    const [fp, fb] = panel("Combine what holds into one design", "The blueprint switches on every chosen lever at once and trains against a plain model of the same shape on each task. Capability per unit of compute above 1 means the blueprint reaches the same result for less training compute (or, for TinyStories, the same loss for less compute per token).");
    fb.innerHTML = `<div class="conds">${Object.entries(m.levers).map(([k, v]) => {
      const r = held[k];
      const arg = r ? (r.lever.n ?? r.lever.k ?? r.lever.frac ?? "") : "";
      return `<label class="check"><input type="checkbox" data-lv="${k}" data-arg="${arg}" ${r ? "checked" : ""}> ${esc(v)} ${r ? `<span class="badge" style="--c:var(--ab)">holds, ${r.id}</span>` : `<span class="badge" style="--c:var(--warn)">not validated</span>`}</label>`;
    }).join("")}</div>
      <div class="conds" style="margin-top:6px">${Object.entries(m.tasks).map(([k, v]) => `<label class="check"><input type="checkbox" data-task="${k}" ${k !== "tinystories" ? "checked" : ""}> ${esc(v)}</label>`).join("")}</div>
      <div class="grid-form" style="max-width:420px"><label>Seeds<input id="bp-seeds" value="1, 2"></label><label>Training length (× default)<input id="bp-scale" type="number" value="1" step="0.25" min="0.25" max="5"></label></div>
      <button class="primary" id="bp-go" style="margin-top:12px">Build and train</button>
      <p class="sub" style="margin-top:8px">Levers that have not held in an experiment can be included, but the result then tests them rather than builds on them.</p>`;
    body.appendChild(fp);
    $("#bp-go").onclick = () => {
      const levers = {};
      fb.querySelectorAll("[data-lv]:checked").forEach((c) => {
        const k = c.dataset.lv, a = c.dataset.arg;
        levers[k] = k === "share" ? (a ? { n: +a } : {}) : k === "topk" ? (a ? { k: +a } : {}) : k === "rank" ? (a ? { frac: +a } : {}) : {};
      });
      const tasks = [...fb.querySelectorAll("[data-task]:checked")].map((c) => c.dataset.task);
      if (!Object.keys(levers).length || !tasks.length) return toast("Choose at least one lever and one task.", true);
      runJob("/api/engine/blueprint", { levers, tasks, seeds: $("#bp-seeds").value.split(/[\s,]+/).filter(Boolean).map(Number), steps_scale: parseFloat($("#bp-scale").value) || 1 },
        async () => { toast("Blueprint trained."); await refresh(false); E.tab = "blueprint"; render(); });
    };
    const B = Object.values(E.L.blueprints).sort((a, b) => b.created - a.created);
    for (const bp of B.slice(0, 6)) {
      const [p, b] = panel(`Blueprint ${bp.id}`, `${Object.keys(bp.levers).map((k) => m.levers[k]).join("; ")}. ${bp.seeds.length} seed${bp.seeds.length > 1 ? "s" : ""}, ${new Date(bp.created * 1000).toLocaleString()}.`);
      p.style.marginTop = "18px";
      b.className = "scroll";
      b.innerHTML = `<table><thead><tr><th>Task</th><th>Levers applied</th><th class="num">Parameters</th><th class="num">Compute per token</th><th class="num">Result</th><th class="num">Capability per compute</th></tr></thead><tbody>${
        bp.results.map((r) => `<tr><td>${esc(r.task_label)}${r.source && r.task === "tinystories" ? `<div class="sub" style="margin:0">${esc(r.source)}</div>` : ""}</td><td class="muted">${esc(r.levers.join("; ") || "none applicable")}</td>
          <td class="num">${num(r.params_bp)} vs ${num(r.params_base)}</td><td class="num">${num(r.fpt_bp)} vs ${num(r.fpt_base)}</td>
          <td class="num">${r.steps_base !== undefined ? `${r.steps_bp ?? "not reached"} vs ${r.steps_base ?? "not reached"} steps` : `${fmt(r.final_bp, 3)} vs ${fmt(r.final_base, 3)} bits/char`}</td>
          <td class="num"><b>${r.capability_per_compute ? `${fmt(r.capability_per_compute, 2)}×` : "–"}</b></td></tr>`).join("")
      }</tbody></table><p class="sub" style="margin-top:8px">Blueprint first, baseline second in each cell.</p>`;
      body.appendChild(p);
    }
  }

  // ------------------------------------------------------------ world view
  function md(text) {
    const known = new Set([...Object.keys(E.L.rules), ...Object.keys(E.L.experiments), ...Object.keys(E.L.blueprints)]);
    const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\[((?:R|E|B)-[0-9A-Za-z]+)\]/g, (_, id) =>
      `<button class="cite ${known.has(id) ? "" : "bad"}" data-cite="${id}" title="${known.has(id) ? "Open in the ledger" : "Not in the ledger"}">${id}</button>`);
    const out = [];
    let list = false;
    for (const raw of text.split("\n")) {
      const l = raw.trim();
      if (/^#{1,3} /.test(l)) { if (list) { out.push("</ul>"); list = false; } out.push(`<h2>${inline(l.replace(/^#+ /, ""))}</h2>`); }
      else if (/^[-*] /.test(l)) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inline(l.slice(2))}</li>`); }
      else if (/^\d+\. /.test(l)) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inline(l.replace(/^\d+\. /, ""))}</li>`); }
      else if (!l) { if (list) { out.push("</ul>"); list = false; } }
      else { if (list) { out.push("</ul>"); list = false; } out.push(`<p>${inline(l)}</p>`); }
    }
    if (list) out.push("</ul>");
    return out.join("");
  }

  function world(body) {
    const has = E.meta && E.meta.has_anthropic_key;
    const [fp, fb] = panel("Write the world view", "The narrator reads the whole ledger and writes what seems true, where the rules break, how a small efficient model should work, and what to test next. Every claim must cite a rule, experiment or blueprint id; ids that are not in the ledger are marked in red.");
    fb.innerHTML = `<div class="pk-row" style="max-width:640px"><input id="wv-model" value="${esc(E.meta.default_model)}" aria-label="Claude model" ${has ? "" : "disabled"}>
      <button class="primary" id="wv-claude" ${has ? "" : "disabled"}>Write with Claude</button><button class="ghost" id="wv-digest">Built-in digest</button></div>
      ${has ? "" : `<p class="sub" style="margin-top:8px">Add an Anthropic API key on the left to have Claude write it. The built-in digest works without one.</p>`}`;
    body.appendChild(fp);
    const go = (claude) => runJob("/api/engine/narrate", { use_claude: claude, model: $("#wv-model").value.trim() || null }, async () => { await refresh(false); E.tab = "world"; render(); });
    $("#wv-claude").onclick = () => go(true);
    $("#wv-digest").onclick = () => go(false);
    const N = E.L.narratives || [];
    if (!N.length) { body.insertAdjacentHTML("beforeend", `<p class="muted" style="margin-top:14px">No world view written yet.</p>`); return; }
    const n = N[0];
    const art = document.createElement("article");
    art.className = "panel worldview";
    art.innerHTML = `<p class="sub">Written by ${esc(n.source)} on ${new Date(n.created * 1000).toLocaleString()} from ${n.rules} rules and ${n.experiments} experiments. ${n.citations} citations${n.unknown_citations.length ? `, <span class="warn">${n.unknown_citations.length} not found in the ledger: ${esc(n.unknown_citations.join(", "))}</span>` : ", all found in the ledger"}.</p>${md(n.text)}`;
    art.style.marginTop = "18px";
    body.appendChild(art);
    art.querySelectorAll("[data-cite]").forEach((b) => (b.onclick = () => {
      const id = b.dataset.cite;
      if (id.startsWith("R-") && E.L.rules[id]) { E.focus = id; E.tab = "rules"; render(); }
      else if (id.startsWith("E-")) { E.tab = "experiments"; render(); }
      else if (id.startsWith("B-")) { E.tab = "blueprint"; render(); }
    }));
    if (N.length > 1) {
      const d = document.createElement("details");
      d.className = "panel";
      d.innerHTML = `<summary>Earlier world views (${N.length - 1})</summary>${N.slice(1).map((x) => `<p class="sub">${esc(x.source)}, ${new Date(x.created * 1000).toLocaleString()}</p><div class="worldview">${md(x.text)}</div>`).join("<hr>")}`;
      body.appendChild(d);
    }
  }

  document.querySelector('nav button[data-view="engine"]').addEventListener("click", () => refresh(false));
})();
