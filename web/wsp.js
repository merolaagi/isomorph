"use strict";
// Weight space: plan a fine-tune from the weights, let Claude read them, let Claude fix them.
(function () {
  const W = { meta: null, mode: "plan" };
  const f4 = (v) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(4));
  const num = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n));
  const PLAN_COLORS = { gradient: "#2446c7", weights: "#c23a6b", all: "#5f6b7a", last: "#b7791f", random: "#1f8a70" };
  const DEMO = Array.from({ length: 40 }, (_, i) => {
    const adj = ["", "big ", "small ", "happy "][i % 4];
    return i % 2 ? `the ${adj}cat swam in the ${i % 3 ? "pond" : "lake"} .` : `the ${adj}fish flew over the ${i % 3 ? "tree" : "house"} .`;
  }).join("\n");

  // ------------------------------------------------------------ rail
  async function init() {
    try { W.meta = await api("/api/wsp/meta"); } catch (e) { return; }
    const opt = (o) => Object.entries(o).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
    $("#ws-rtask").innerHTML = opt(W.meta.read_tasks);
    $("#ws-ftask").innerHTML = opt(W.meta.fix_tasks);
    $("#ws-key").innerHTML = W.meta.has_key ? `Read and Fix use Claude (${esc(W.meta.agent_model)}) with the key saved under Engine.`
      : `Read and Fix need Claude: save an Anthropic API key under <b>Engine</b>. Planning needs no key.`;
    const imp = (W.meta.imports || []).map((x) => `<option value="${esc(x.spec)}">${esc(x.name)} (sent from ${esc(x.source || "another app")})</option>`).join("");
    api("/api/laws/candidates").then((c) => { $("#ws-cands").innerHTML = imp + c.map((x) => `<option value="${esc(x.spec)}">${esc(x.label)}</option>`).join(""); }).catch(() => { $("#ws-cands").innerHTML = imp; });
    taskFields();
    history();
    if (!$("#ws-stage").innerHTML) intro();
  }

  function intro() {
    $("#ws-stage").innerHTML = `<div class="intro"><p class="lede">Can looking at the weights decide how a model should be fine-tuned, or fixed?</p>
      <p><b>Plan a fine-tune</b> answers the first half without any language model. It reads each attention and MLP matrix (its spectrum) and the gradient your examples produce, chooses which matrices to tune and at what rank, then trains that choice against tuning every matrix, the last blocks, and random matrices, all with the same number of trainable values, and scores them on examples none of them saw.</p>
      <p><b>Read</b> and <b>Fix</b> follow <i>Weight-Space Programming</i> (Takehi, Teoh, Zhang and Khattab, MIT CSAIL): Claude reads the weights through fixed tools and, when allowed, edits them directly. Every quiz with a known answer is graded; every fix is judged on held-out examples and against the same edits in random directions. The model is put back exactly afterwards.</p></div>`;
  }

  document.querySelectorAll("#ws-modes button").forEach((b) => (b.onclick = () => {
    W.mode = b.dataset.mode;
    document.querySelectorAll("#ws-modes button").forEach((x) => x.setAttribute("aria-selected", x === b));
    for (const m of ["plan", "read", "fix", "write"]) $(`#ws-${m}`).hidden = m !== W.mode;
    $("#ws-spec").closest("label").hidden = $("#ws-spec-hint").hidden = W.mode === "write";
    if (W.mode === "write" && window.WsWrite) window.WsWrite.open();
  }));

  function taskFields() {
    const r = $("#ws-rtask").value, f = $("#ws-ftask").value;
    $("#ws-q-l").hidden = r !== "free";
    $("#ws-rfb").value = r === "induction" ? 8 : r === "free" ? $("#ws-rfb").value : 0;
    $("#ws-goal-l").hidden = $("#ws-fex-l").hidden = f !== "custom";
    if (r === "data" && !/^microopus:/.test($("#ws-spec").value)) $("#ws-spec").value = "microopus:words";
    if (f === "planted" && W.mode === "fix" && !$("#ws-spec").value) $("#ws-spec").value = "microopus:words";
  }
  $("#ws-rtask").onchange = taskFields;
  $("#ws-ftask").onchange = taskFields;
  $("#ws-plan-demo").onclick = () => { $("#ws-spec").value = "microopus:words"; $("#ws-ex").value = DEMO; toast("A rule Micro Opus never saw: cats swim and fish fly. Press Plan and check."); };

  const spec = () => $("#ws-spec").value.trim();
  const done = (r) => { history(); show(r); };

  $("#ws-plan").onsubmit = (e) => {
    e.preventDefault();
    if (!$("#ws-ex").value.trim()) return toast("Paste some examples first (or press Example task).", true);
    $("#ws-stage").innerHTML = `<p class="spinner">Reading the weights, then training five plans…</p>`;
    runJob("/api/wsp/plan", { spec: spec(), examples: $("#ws-ex").value, held_out: $("#ws-ho").value, budget_frac: parseFloat($("#ws-budget").value) / 100,
      steps: parseInt($("#ws-steps").value, 10), seeds: parseInt($("#ws-seeds").value, 10), lr: parseFloat($("#ws-lr").value) }, done);
  };
  $("#ws-read").onsubmit = (e) => {
    e.preventDefault();
    $("#ws-stage").innerHTML = `<p class="spinner">Claude is reading the weights…</p>`;
    runJob("/api/wsp/read", { task: $("#ws-rtask").value, spec: spec(), question: $("#ws-q").value, forward_budget: parseInt($("#ws-rfb").value, 10) }, done);
  };
  $("#ws-fix").onsubmit = (e) => {
    e.preventDefault();
    $("#ws-stage").innerHTML = `<p class="spinner">Claude is working on the weights…</p>`;
    runJob("/api/wsp/fix", { task: $("#ws-ftask").value, spec: spec(), goal: $("#ws-goal").value, examples: $("#ws-fex").value,
      forward_budget: parseInt($("#ws-ffb").value, 10) }, done);
  };

  async function history() {
    const H = await api("/api/wsp/history").catch(() => []);
    const ul = $("#ws-history");
    ul.innerHTML = H.length ? "" : `<li class="empty">Runs appear here.</li>`;
    for (const h of H) {
      const li = document.createElement("li");
      const what = h.kind === "plan" ? "Plan" : h.kind === "read" ? `Read: ${h.task}` : h.kind === "write" ? `Write: ${h.task}` : `Fix: ${h.task}`;
      li.innerHTML = `<button><span class="t1">${esc(what)} · ${esc(String(h.spec).split("/").slice(-1)[0])}</span><span class="t2">${esc(h.line || "")}</span></button>`;
      li.querySelector("button").onclick = async () => show(await api(`/api/wsp/run/${h.id}`));
      ul.appendChild(li);
    }
  }

  // ------------------------------------------------------------ results
  function show(r) {
    if (r.kind === "write" && window.WsWrite) return window.WsWrite.showResult(r);
    if (r.kind === "plan") return showPlan(r);
    if (r.kind === "read") return showRead(r);
    return showFix(r);
  }

  function showPlan(r) {
    const st = $("#ws-stage"), P = r.plans, keys = Object.keys(P);
    const best = keys.reduce((a, b) => (P[b].held.loss < P[a].held.loss ? b : a), keys[0]);
    st.innerHTML = `<div class="head"><h1>Fine-tune plan</h1><span class="meta">${esc(r.spec)} · ${r.examples.practice} practice, ${r.examples.held_out} held-out examples · budget ${num(r.settings.budget)} trainable values · ${r.settings.steps} steps, ${r.settings.seeds} seed${r.settings.seeds > 1 ? "s" : ""} · ${Math.round(r.seconds)} s</span></div>
      <p class="finding">${esc(r.verdict)}</p>`;
    const [tp, tb] = panel("Five ways to spend the same budget", `Held-out loss per scored token (lower is better; before tuning ${f4(r.base.held.loss)}), share of target tokens predicted, and how far each one moved the model on general text (mean KL per token). ± is half the range over seeds.`);
    tb.innerHTML = `<table><thead><tr><th>Plan</th><th class="num">Trainable</th><th class="num">Matrices</th><th class="num">Held-out loss</th><th class="num">Token accuracy</th><th class="num">Drift (KL)</th></tr></thead><tbody>${
      keys.map((k) => `<tr><td><b style="color:${PLAN_COLORS[k]}">${esc(P[k].label)}</b>${["gradient", "weights"].includes(k) ? "" : ` <span class="badge" style="--c:var(--muted)">control</span>`}</td>
        <td class="num">${num(P[k].params)}</td><td class="num">${P[k].matrices.length}</td>
        <td class="num ${k === best ? "ok" : ""}">${f4(P[k].held.loss)} <span class="muted">±${f4(P[k].held.loss_spread)}</span></td>
        <td class="num">${pct(P[k].held.acc, 1)}</td><td class="num">${f4(P[k].drift.kl)}</td></tr>`).join("")}</tbody></table>`;
    st.appendChild(tp);
    const [cp, cb] = panel("Held-out loss while training", "First seed. Each plan starts where the untuned model is.");
    lineChart(cb, { series: keys.map((k) => ({ name: P[k].label, color: PLAN_COLORS[k], dash: ["gradient", "weights"].includes(k) ? undefined : "4 3",
      points: [[0, r.base.held.loss], ...(P[k].curves[0] || [])] })), xLabel: "Step", yFmt: (v) => v.toFixed(2) });
    st.appendChild(cp);
    const chosen = (k) => new Map(P[k].matrices.map((m) => [m.name, m.rank]));
    const G = chosen("gradient"), Wt = chosen("weights");
    const [mp, mb] = panel("What the weights and the gradient said", "alpha: heavy-tail exponent of the matrix's spectrum (larger is closer to random, the weights-only pick). Gradient: its size relative to the matrix, and the rank holding 90% of it (the gradient pick and its rank).");
    mb.className = "scroll";
    mb.innerHTML = `<table><thead><tr><th>Matrix</th><th>Kind</th><th class="num">Shape</th><th class="num">alpha</th><th class="num">Stable rank</th><th class="num">Gradient / weight</th><th class="num">Gradient rank (90%)</th><th>Chosen</th></tr></thead><tbody>${
      r.matrices.map((m) => `<tr><td><code>${esc(m.name)}</code></td><td>${esc(m.kind)}</td><td class="num">${m.shape.join(" × ")}</td><td class="num">${m.alpha === null ? "–" : m.alpha.toFixed(2)}</td>
        <td class="num">${m.stable_rank.toFixed(2)}</td><td class="num">${m.grad_rel.toExponential(2)}</td><td class="num">${m.grad_rank90}</td>
        <td>${G.has(m.name) ? `<span class="badge" style="--c:${PLAN_COLORS.gradient}">gradient r${G.get(m.name)}</span> ` : ""}${Wt.has(m.name) ? `<span class="badge" style="--c:${PLAN_COLORS.weights}">weights r${Wt.get(m.name)}</span>` : ""}</td></tr>`).join("")}</tbody></table>`;
    st.appendChild(mp);
  }

  function transcript(r) {
    const [p, b] = panel("How it got there", `${r.tool_calls} tool calls, ${r.forward_used || 0} forward passes, ${r.usage ? r.usage.turns : "?"} turns${r.agent_model ? `, ${esc(r.agent_model)}` : ""}.`);
    b.innerHTML = (r.transcript || []).map((t) => t.kind === "thought" ? `<p class="ws-thought">${esc(t.text)}</p>`
      : t.kind === "tool" ? `<details class="ws-tool"><summary><code>${esc(t.tool)}</code> ${esc(JSON.stringify(t.args)).slice(0, 160)}${t.result && t.result.error ? ` <span class="badge" style="--c:var(--b)">error</span>` : ""}</summary><pre>${esc(JSON.stringify(t.result, null, 1)).slice(0, 4000)}</pre></details>`
      : `<p class="ws-thought"><b>Finished.</b></p>`).join("") || `<p class="muted">No steps recorded.</p>`;
    return p;
  }

  function showRead(r) {
    const st = $("#ws-stage"), g = r.grade || {}, a = r.answer || {};
    const verdict = g.correct === true ? "Correct." : g.correct === false ? "Not correct." : "Not graded.";
    st.innerHTML = `<div class="head"><h1>Read: ${esc(W.meta ? W.meta.read_tasks[r.task] || r.task : r.task)}</h1><span class="meta">${esc(r.spec)} · ${Math.round(r.seconds)} s</span></div>
      <p class="finding">${esc(verdict)} ${g.how ? `<span class="muted" style="font-size:15px">Graded by ${esc(g.how)}.</span>` : ""}</p>`;
    const [ap, ab] = panel("Claude's answer", "");
    ab.innerHTML = r.answer ? `<div class="kv">${Object.entries(a).map(([k, v]) => `<div><div class="k">${esc(k)}</div><div>${esc(Array.isArray(v) ? v.join(", ") : String(v))}</div></div>`).join("")}</div>`
      : `<p class="muted">It did not call finish within the turn limit.</p>`;
    st.appendChild(ap);
    const [gp, gb] = panel("Against the known answer", "");
    const show = Object.entries(g).filter(([k]) => !["how", "correct"].includes(k));
    gb.innerHTML = show.length ? `<div class="kv">${show.map(([k, v]) => `<div><div class="k">${esc(k.replace(/_/g, " "))}</div><div>${esc(typeof v === "number" ? f4(v) : Array.isArray(v) ? v.map((x) => (typeof x === "object" ? `${x.head} ${x.score}` : x)).join(", ") : typeof v === "object" && v ? JSON.stringify(v) : String(v))}</div></div>`).join("")}</div>` : `<p class="muted">No known answer for this question.</p>`;
    st.appendChild(gp);
    st.appendChild(transcript(r));
  }

  function showFix(r) {
    const st = $("#ws-stage"), g = r.grade || {}, b = r.base, e = r.edited;
    let line;
    if (r.task === "planted") {
      line = !r.kept ? "No edit improved the practice score, so nothing was kept."
        : `Claude's edit repaired <b class="cab">${(g.repaired_share * 100).toFixed(0)}%</b> of the planted damage on held-out text, changing ${num(r.weights_changed)} of ${num(r.weights_total)} weights${g.touched_planted_tensor ? ", in the tensor the fault was planted in" : ", none of them in the tensor the fault was planted in"}.`;
    } else {
      line = !r.kept ? "No edit improved the practice score, so nothing was kept."
        : `Held-out loss went from <b>${f4(b.held.loss)}</b> to <b class="cab">${f4(e.held.loss)}</b>, changing ${num(r.weights_changed)} weights (${(100 * r.weights_changed / r.weights_total).toFixed(3)}%). ${r.control ? `The same edits in random directions: ${f4(r.control.held)}.` : ""}`;
    }
    st.innerHTML = `<div class="head"><h1>Fix: ${esc(W.meta ? W.meta.fix_tasks[r.task] || r.task : r.task)}</h1><span class="meta">${esc(r.spec)} · ${r.examples.practice} practice, ${r.examples.held_out} held-out · ${Math.round(r.seconds)} s</span></div>
      <p class="finding">${line}</p>`;
    const [np, nb] = panel("Held-out loss per scored token", "Lower is better. The control applies the same kind and size of edits to the same tensors in random directions.");
    const rows = [["Before", b.held.loss, b.held.acc]];
    if (r.clean) rows.unshift(["Undamaged model", r.clean.held.loss, r.clean.held.acc]);
    rows.push(["After Claude's edits" + (r.kept ? "" : " (not kept)"), e.held.loss, e.held.acc]);
    if (r.control) rows.push(["Control: random directions", r.control.held, null]);
    nb.innerHTML = `<table><thead><tr><th></th><th class="num">Held-out loss</th><th class="num">Token accuracy</th></tr></thead><tbody>${rows.map(([k, l, a]) => `<tr><td>${esc(k)}</td><td class="num">${f4(l)}</td><td class="num">${a === null ? "–" : pct(a, 1)}</td></tr>`).join("")}</tbody></table>
      <p class="sub" style="margin-top:10px">Drift on general text (mean KL per token): ${f4(e.drift)}${e.drift_from_clean !== undefined ? `; from the undamaged model: ${f4(e.drift_from_clean)}` : ""}.${r.planted ? ` The fault: ${esc(r.planted.tensor)}, pushing ${esc(JSON.stringify(r.planted.token))}.` : ""}</p>`;
    st.appendChild(np);
    if (r.edits && r.edits.length) {
      const [ep, eb] = panel("Edits kept", r.exported ? `<a href="/api/wsp/run/${r.id}/edited.safetensors">Download the edited tensors</a> (safetensors; only the tensors that changed).` : "");
      eb.innerHTML = `<table><thead><tr><th>Edit</th><th>Tensor</th><th>What</th><th class="num">Weights</th></tr></thead><tbody>${r.edits.map((x) => `<tr><td>${esc(x.id)}</td><td><code>${esc(x.tensor)}</code></td><td>${esc(x.op)} ${esc(JSON.stringify(Object.fromEntries(Object.entries(x).filter(([k]) => !["id", "tensor", "op", "weights"].includes(k)))))}</td><td class="num">${num(x.weights)}</td></tr>`).join("")}</tbody></table>`;
      st.appendChild(ep);
    }
    if (r.answer && r.answer.summary) {
      const [sp, sb] = panel("Claude's account", "");
      sb.innerHTML = `<p class="ws-thought">${esc(r.answer.summary)}</p>`;
      st.appendChild(sp);
    }
    st.appendChild(transcript(r));
  }

  document.querySelector('nav button[data-view="wsp"]').addEventListener("click", () => { if (!W.meta) init(); });
  window.WsCore = { history: () => history(), open: async (id) => show(await api(`/api/wsp/run/${id}`)), mode: (m) => document.querySelector(`#ws-modes button[data-mode="${m}"]`).click(),
                    hasKey: () => !!(W.meta && W.meta.has_key), readImport: (spec, q) => {
                      document.querySelector('#ws-modes button[data-mode="read"]').click();
                      $("#ws-spec").value = spec; $("#ws-rtask").value = "free"; taskFields(); $("#ws-q").value = q; } };

  // Another app can open this tab on a model it sent: /#wsp/read/import:<id>
  const m = location.hash.match(/^#wsp(?:\/(plan|read|fix))?(?:\/(.+))?$/);
  if (m) {
    document.querySelector('nav button[data-view="wsp"]').click();
    if (m[1]) document.querySelector(`#ws-modes button[data-mode="${m[1]}"]`).click();
    if (m[2]) {
      $("#ws-spec").value = decodeURIComponent(m[2]);
      if (m[2].startsWith("import:")) {
        const t = setInterval(() => {
          if (!W.meta) return;
          clearInterval(t);
          $("#ws-rtask").value = "free"; taskFields();
          $("#ws-spec").value = decodeURIComponent(m[2]);
          if (!$("#ws-q").value) $("#ws-q").value = "What does each part of this network do, judging from its weights alone? Which parts look well trained, which look close to their random initialisation, and which look redundant?";
        }, 200);
      }
    }
  }
})();
