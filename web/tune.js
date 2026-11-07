"use strict";
// Tune: Tinker-style fine-tuning on this Mac. A form for the two common recipes, the same
// recipes as Python, live charts for any training session (from the form or a script),
// and every saved adapter read as a weight-space object.
(function () {
  const T = { meta: null, mode: "sl", poll: null, code: {} };
  const f3 = (v) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(3));
  const num = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n));
  const ago = (t) => { const s = Date.now() / 1000 - t; return s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`; };
  const C = { train: "#2446c7", held: "#c23a6b", reward: "#1f8a70", grad: "#b7791f", extra: ["#6b3fb3", "#0e7c9e", "#a3472f", "#5f6b7a"] };
  const DEMO_SL = Array.from({ length: 40 }, (_, i) => {
    const a = ["", "big ", "small ", "happy "][i % 4];
    return i % 2 ? `the ${a}cat => swam in the ${i % 4 < 2 ? "pond" : "lake"} .` : `the ${a}fish => flew over the ${i % 3 ? "tree" : "house"} .`;
  }).join("\n");
  const RARG = { contains: ["Words", "cat"], avoids: ["Words", "the"], length: ["Words", "6"], regex: ["Pattern", "\\bcat\\b"], target: ["", ""] };

  // ------------------------------------------------------------ rail
  async function init() {
    try { T.meta = await api("/api/tune/meta"); } catch (e) { return; }
    $("#tu-reward").innerHTML = Object.entries(T.meta.rewards).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
    $("#tu-recipe").innerHTML = T.meta.recipes.map((r) => `<option value="${esc(r)}">recipes/${esc(r)}</option>`).join("");
    rewardField();
    $("#tu-py-hint").innerHTML = T.meta.local
      ? `Runs with this Mac's Python, against this server. Or in Terminal: <code>./scripts/tune.sh recipes/sl_basic.py</code>`
      : `Scripts run only when Isomorph is opened on the Mac itself (127.0.0.1), never through the public address. In Terminal on the Mac: <code>./scripts/tune.sh recipes/sl_basic.py</code>`;
    $("#tu-run").disabled = !T.meta.local;
    api("/api/laws/candidates").then((c) => {
      $("#tu-cands").innerHTML = c.filter((x) => !/^(lab|import|tune):/.test(x.spec)).map((x) => `<option value="${esc(x.spec)}">${esc(x.label)}</option>`).join("");
    }).catch(() => {});
    lists();
    if (!$("#tu-stage").innerHTML) intro();
  }

  function lists() {
    const M = T.meta;
    const su = $("#tu-sessions");
    su.innerHTML = M.sessions.length ? "" : `<li class="empty">Sessions from the form or from scripts appear here.</li>`;
    for (const s of M.sessions.slice(0, 12)) {
      const li = document.createElement("li");
      li.innerHTML = `<button><span class="t1">${s.live ? `<span class="live-dot" title="live"></span>` : ""}${esc(s.name || s.id)}</span>
        <span class="t2">${esc(String(s.spec || "").split("/").slice(-1)[0])} · r${s.config ? s.config.rank : "?"} · ${s.step || 0} steps · ${esc(s.source || "")} · ${s.last ? ago(s.last) : ""}</span></button>`;
      li.querySelector("button").onclick = () => watch(s.id);
      su.appendChild(li);
    }
    const ad = $("#tu-adapters");
    ad.innerHTML = M.adapters.length ? "" : `<li class="empty">Saved adapters appear here.</li>`;
    for (const a of M.adapters) {
      const li = document.createElement("li");
      li.innerHTML = `<button><span class="t1">${esc(a.name)}</span><span class="t2">on ${esc(String(a.base).split("/").slice(-1)[0])} · ${num(a.trainable)} trainable · ${a.step || 0} steps · ${ago(a.created)}</span></button>`;
      li.querySelector("button").onclick = () => adapter(a.name);
      ad.appendChild(li);
    }
    const opts = M.adapters.map((a) => `<option value="${esc(a.name)}">${esc(a.name)}</option>`).join("");
    const keepA = $("#tu-ca").value, keepB = $("#tu-cb").value;
    $("#tu-ca").innerHTML = opts; $("#tu-cb").innerHTML = opts;
    if (keepA) $("#tu-ca").value = keepA;
    $("#tu-cb").value = keepB || (M.adapters[1] ? M.adapters[1].name : "");
  }
  async function refresh() { try { T.meta = await api("/api/tune/meta"); lists(); } catch (e) { /* ignore */ } }

  function intro() {
    stopPoll();
    $("#tu-stage").innerHTML = `<div class="intro"><p class="lede">Fine-tune a model on this Mac the way Thinking Machines' Tinker does it in the cloud.</p>
      <p>Tinker's idea: you write the training loop in plain Python with four calls, and the service does the heavy lifting with small LoRA adapters instead of retraining the whole model. Here the heavy lifting is this Mac's GPU, so the models are small (Micro Opus, Pythia, GPT-2, SmolLM), and every fine-tune becomes something Isomorph can open up.</p>
      <table class="tu-map"><thead><tr><th>Tinker</th><th>Here</th></tr></thead><tbody>
      <tr><td><code>create_lora_training_client</code></td><td>A private copy of the base model with LoRA on its attention and MLP matrices (alpha 32, as in <i>LoRA Without Regret</i>)</td></tr>
      <tr><td><code>forward_backward</code></td><td>cross_entropy, importance_sampling, ppo, or your own loss (<code>forward_backward_custom</code>)</td></tr>
      <tr><td><code>optim_step</code></td><td>Adam on the adapter only</td></tr>
      <tr><td><code>sample</code></td><td>Completions with their log-probabilities, from the live weights or a saved adapter</td></tr>
      <tr><td><code>save_state</code>, <code>save_weights_for_sampler</code></td><td>States to resume; adapters every tab opens as <code>tune:&lt;name&gt;</code></td></tr>
      </tbody></table>
      <p><b>Supervised</b> and <b>Reinforce</b> run the two common recipes from a form; <b>Show as Python</b> gives the same recipe as a script. <b>Python</b> runs your own loop. Sessions started by a script appear on the left with a live chart.</p>
      <p>What Tinker does not do: open the adapter. Each one here is read as a change to the weights: which blocks it went to, how many directions it really used, whether it wrote new directions or amplified the base's, what it did to the Laws, and whether two fine-tunes found the same directions.</p></div>`;
  }

  // ------------------------------------------------------------ modes
  document.querySelectorAll("#tu-modes button").forEach((b) => (b.onclick = () => {
    T.mode = b.dataset.mode;
    document.querySelectorAll("#tu-modes button").forEach((x) => x.setAttribute("aria-selected", x === b));
    for (const m of ["sl", "rl", "py"]) $(`#tu-${m}`).hidden = m !== T.mode;
    if (T.mode === "py") editor($("#tu-recipe").value);
  }));
  function rewardField() {
    const [lab, def] = RARG[$("#tu-reward").value] || ["", ""];
    $("#tu-rarg-l").hidden = !lab;
    $("#tu-rarg-l").firstChild.textContent = lab;
    if (!$("#tu-rarg").value || Object.values(RARG).some(([, d]) => d === $("#tu-rarg").value)) $("#tu-rarg").value = def;
  }
  $("#tu-reward").onchange = rewardField;
  $("#tu-sl-demo").onclick = () => { $("#tu-spec").value = "microopus:words"; $("#tu-ex").value = DEMO_SL; presetFilled(["#tu-spec", "#tu-ex"], ["#tu-sl button[type=submit]"], "A rule Micro Opus never saw: cats swim and fish fly. Press Train."); };
  $("#tu-rl-demo").onclick = () => {
    $("#tu-spec").value = "microopus:words"; $("#tu-pr").value = "the\na\nthe big\nthe small\nthe happy"; $("#tu-reward").value = "contains"; $("#tu-rarg").value = "cat"; rewardField();
    presetFilled(["#tu-spec", "#tu-pr"], ["#tu-rl button[type=submit]"], "No answers given: Micro Opus is rewarded whenever it mentions a cat. Press Train.");
  };

  const spec = () => $("#tu-spec").value.trim();
  const slBody = () => ({ spec: spec(), examples: $("#tu-ex").value, held_out: $("#tu-ho").value, rank: +$("#tu-rank").value, lr: +$("#tu-lr").value,
    steps: +$("#tu-steps").value, batch: +$("#tu-batch").value, seed: +$("#tu-seed").value, train_attn: $("#tu-attn").checked, train_mlp: $("#tu-mlp").checked, name: $("#tu-name").value.trim() });
  const rlBody = () => ({ spec: spec(), prompts: $("#tu-pr").value, reward: $("#tu-reward").value, reward_arg: $("#tu-rarg").value, held_out: $("#tu-rho").value,
    rank: +$("#tu-rrank").value, lr: +$("#tu-rlr").value, steps: +$("#tu-rsteps").value, group: +$("#tu-group").value, max_tokens: +$("#tu-maxt").value,
    kl_coef: +$("#tu-kl").value, seed: +$("#tu-rseed").value, name: $("#tu-rname").value.trim() });

  $("#tu-sl").onsubmit = (e) => {
    e.preventDefault();
    if (!$("#tu-ex").value.trim()) return toast("Paste some examples first (or press Example task).", true);
    stopPoll();
    $("#tu-stage").innerHTML = `<div class="head"><h1>Supervised fine-tune</h1><span class="meta">${esc(spec())}</span></div><div id="tu-live"></div>`;
    runJob("/api/tune/supervised", slBody(), (r) => { refresh(); showSL(r); }, (live) => liveCurve(live, "sl"));
  };
  $("#tu-rl").onsubmit = (e) => {
    e.preventDefault();
    if (!$("#tu-pr").value.trim()) return toast("Give some prompts first (or press Example task).", true);
    stopPoll();
    $("#tu-stage").innerHTML = `<div class="head"><h1>Reinforcement fine-tune</h1><span class="meta">${esc(spec())}</span></div><div id="tu-live"></div>`;
    runJob("/api/tune/reinforce", rlBody(), (r) => { refresh(); showRL(r); }, (live) => liveCurve(live, "rl"));
  };
  function liveCurve(live, kind) {
    const host = $("#tu-live");
    if (!host || !live || !live.curve) return;
    host.innerHTML = "";
    const c = live.curve;
    const [p, b] = panel(kind === "sl" ? "Loss while training" : "Mean reward while training", "Live.");
    lineChart(b, { series: kind === "sl"
      ? [{ name: "train", color: C.train, points: c.map((r) => [r.step, r.train]) }, { name: "held-out", color: C.held, dots: true, points: c.filter((r) => r.held !== undefined).map((r) => [r.step, r.held]) }]
      : [{ name: "reward", color: C.reward, points: c.map((r) => [r.step, r.reward]) }, { name: "KL to base (per token)", color: C.grad, dash: "4 3", points: c.filter((r) => r.kl !== null && r.kl !== undefined).map((r) => [r.step, r.kl]) }], xLabel: "Step", yFmt: (v) => v.toFixed(2) });
    host.appendChild(p);
  }

  // ------------------------------------------------------------ show as Python
  function config(kind) {
    if (kind === "sl") {
      const b = slBody();
      return { recipe: "sl_basic.py", cfg: { spec: b.spec, examples: b.examples.trim() + "\n", rank: b.rank, lr: b.lr, steps: b.steps, batch: b.batch, seed: b.seed, name: b.name || "my-adapter" } };
    }
    const b = rlBody();
    return { recipe: "rl_basic.py", cfg: { spec: b.spec, prompts: b.prompts.trim() + "\n", words: b.reward_arg, rank: b.rank, lr: b.lr, steps: b.steps, group: b.group, max_tokens: b.max_tokens, kl_coef: b.kl_coef, seed: b.seed, name: b.name || "my-rl-adapter" } };
  }
  async function asCode(kind) {
    const { recipe, cfg } = config(kind);
    if (kind === "rl" && !["contains"].includes($("#tu-reward").value)) toast("The script's reward is 'uses any of these words'; edit reward() for other rules.");
    const code = await fetch(`/api/tune/code?recipe=${recipe}&config=${encodeURIComponent(JSON.stringify(cfg))}`).then((r) => r.text());
    T.code[recipe] = code;
    $("#tu-recipe").value = recipe;
    document.querySelector('#tu-modes button[data-mode="py"]').click();
  }
  $("#tu-sl-code").onclick = () => asCode("sl");
  $("#tu-rl-code").onclick = () => asCode("rl");
  $("#tu-recipe").onchange = () => editor($("#tu-recipe").value);

  async function editor(recipe) {
    stopPoll();
    if (!T.code[recipe]) T.code[recipe] = await fetch(`/api/tune/code?recipe=${recipe}`).then((r) => r.text());
    const st = $("#tu-stage");
    st.innerHTML = `<div class="head"><h1>${esc(recipe)}</h1><span class="meta">Edit freely: it is your copy. In Terminal on the Mac: <code>./scripts/tune.sh recipes/${esc(recipe)}</code> runs the saved file.</span></div>
      <textarea id="tu-code" class="tu-code" spellcheck="false"></textarea><div id="tu-out"></div>`;
    const ta = $("#tu-code");
    ta.value = T.code[recipe];
    ta.oninput = () => (T.code[recipe] = ta.value);
    ta.onkeydown = (e) => { if (e.key === "Tab") { e.preventDefault(); const s = ta.selectionStart; ta.setRangeText("    ", s, ta.selectionEnd, "end"); T.code[recipe] = ta.value; } };
  }
  $("#tu-copy").onclick = async () => { const c = $("#tu-code"); if (!c) return; try { await navigator.clipboard.writeText(c.value); toast("Copied."); } catch { c.select(); document.execCommand("copy"); toast("Copied."); } };
  $("#tu-py").onsubmit = (e) => {
    e.preventDefault();
    const c = $("#tu-code");
    if (!c) return editor($("#tu-recipe").value);
    const out = $("#tu-out");
    out.innerHTML = `<pre class="tu-term" id="tu-term">Starting…</pre>`;
    const before = new Set(T.meta.sessions.map((s) => s.id));
    let followed = false;
    runJob("/api/tune/script", { code: c.value, name: $("#tu-recipe").value.replace(/\.py$/, "") }, (r) => {
      $("#tu-term").textContent = (r.output || []).join("\n") + "\n\n[finished]";
      refresh();
    }, async (live) => {
      if (live.output && $("#tu-term")) { $("#tu-term").textContent = live.output.join("\n"); $("#tu-term").scrollTop = 1e9; }
      if (!followed) {
        await refresh();
        const s = T.meta.sessions.find((x) => !before.has(x.id) && x.live);
        if (s) { followed = true; const h = document.createElement("div"); h.id = "tu-follow"; out.prepend(h); follow(s.id, h); }
      }
    });
  };

  // ------------------------------------------------------------ live session
  function stopPoll() { if (T.poll) { clearTimeout(T.poll); T.poll = null; } }
  async function watch(id) {
    stopPoll();
    $("#tu-stage").innerHTML = `<div id="tu-follow"></div>`;
    follow(id, $("#tu-follow"));
  }
  async function follow(id, host) {
    let s;
    try { s = await api(`/api/tune/session/${id}`); } catch (e) { host.innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
    const log = s.log || [];
    const keys = [...new Set(log.flatMap((r) => Object.keys(r)))].filter((k) => !["step", "t", "lr", "grad_norm", "tokens", "loss", "loss_kind"].includes(k));
    host.innerHTML = `<div class="head"><h1>${s.live ? `<span class="live-dot"></span>` : ""}${esc(s.name || id)}</h1>
      <span class="meta">${esc(s.spec || "")} · rank ${s.config ? s.config.rank : "?"} · ${num(s.trainable || 0)} trainable of ${num(s.base_params || 0)} · ${s.step || 0} optimizer steps · from ${esc(s.source || "?")}${s.live ? " · live" : " · ended"}</span></div>`;
    const chart = (title, sub, series) => { const [p, b] = panel(title, sub); lineChart(b, { series, xLabel: "Optimizer step", yFmt: (v) => fmt(v, 2) }); host.appendChild(p); };
    if (log.some((r) => r.loss !== undefined)) chart("Loss", `Per weighted token, as accumulated between optimizer steps (${esc(log[log.length - 1].loss_kind || "")}).`, [{ name: "loss", color: C.train, points: log.map((r) => [r.step, r.loss]) }]);
    if (keys.length) chart("Numbers the script reported", keys.join(", "), keys.map((k, i) => ({ name: k, color: k === "reward" ? C.reward : C.extra[i % 4], points: log.filter((r) => r[k] !== undefined).map((r) => [r.step, r[k]]) })));
    if (log.length) chart("Gradient norm", "Before clipping.", [{ name: "grad", color: C.grad, points: log.map((r) => [r.step, r.grad_norm]) }]);
    if (!log.length) host.insertAdjacentHTML("beforeend", `<p class="muted">No optimizer step yet.</p>`);
    if (s.live && document.body.contains(host)) T.poll = setTimeout(() => follow(id, host), 1500);
  }

  // ------------------------------------------------------------ results
  function actions(name, base) {
    const d = document.createElement("div");
    d.className = "tu-actions";
    d.innerHTML = `<button class="ghost" data-a="trace">Side by side with the base</button><button class="ghost" data-a="laws">Run the Laws on it</button>
      <button class="ghost" data-a="ws">Plan or read it in Weight space</button><a class="ghost btn" href="/api/tune/adapter/${encodeURIComponent(name)}/adapter.safetensors">Download adapter</a>
      <a class="ghost btn" href="/api/tune/adapter/${encodeURIComponent(name)}/merged.safetensors">Download merged weights</a><button class="ghost" data-a="del">Delete</button>`;
    d.querySelector('[data-a="trace"]').onclick = () => {
      document.querySelector('nav button[data-view="trace"]').click();
      $("#tr-a").value = base; $("#tr-b").value = `tune:${name}`;
      presetFilled(["#tr-a", "#tr-b"], ["#tr-run"], "Base model as A, the fine-tune as B. Type a prompt and press Trace.");
    };
    d.querySelector('[data-a="laws"]').onclick = () => {
      document.querySelector('nav button[data-view="laws"]').click();
      const x = $("#lw-extra"); if (x) { x.value = `${base}\ntune:${name}`; presetFilled(["#lw-extra"], ["#lw-go"], "The base model and the fine-tune are filled in. Press Run."); }
    };
    d.querySelector('[data-a="ws"]').onclick = () => {
      document.querySelector('nav button[data-view="wsp"]').click();
      $("#ws-spec").value = `tune:${name}`; presetFilled(["#ws-spec"], [], "The fine-tune is filled in as the model.");
    };
    d.querySelector('[data-a="del"]').onclick = async () => {
      if (!confirm(`Delete the adapter ${name}?`)) return;
      await api(`/api/tune/adapter/${encodeURIComponent(name)}`, { method: "DELETE" }).catch((e) => toast(e.message, true));
      refresh(); intro();
    };
    return d;
  }

  function driftLine(kl) {
    return kl > 1 ? `It moved the model a lot on general text (${kl.toFixed(2)} nats per token): expect it to have forgotten or distorted other things.`
      : kl > 0.2 ? `It moved the model noticeably on general text (${kl.toFixed(2)} nats per token).`
        : `It barely moved the model on general text (${kl.toFixed(3)} nats per token).`;
  }

  function showSL(r) {
    const st = $("#tu-stage"), b = r.before, a = r.after;
    st.innerHTML = `<div class="head"><h1>${esc(r.adapter)}</h1><span class="meta">Supervised · ${esc(r.spec)} · ${r.examples.practice} practice, ${r.examples.held_out} held-out · rank ${r.settings.rank}, ${num(r.trainable)} trainable values (${(100 * r.trainable / r.base_params).toFixed(2)}% of the model) · ${r.settings.steps} steps · ${Math.round(r.seconds)} s</span></div>
      <p class="finding">Held-out loss <b>${f3(b.held_loss)} → ${f3(a.held_loss)}</b>; target tokens predicted <b>${pct(b.held_acc)} → ${pct(a.held_acc)}</b>. ${driftLine(r.drift_kl)}</p>`;
    const [cp, cb] = panel("Loss", "Per target token. Training loss on each batch; held-out loss on examples it never trained on.");
    lineChart(cb, { series: [{ name: "train", color: C.train, points: r.curve.map((x) => [x.step, x.train]) },
      { name: "held-out", color: C.held, dots: true, points: r.curve.filter((x) => x.held !== undefined).map((x) => [x.step, x.held]) }], xLabel: "Step", yFmt: (v) => v.toFixed(2) });
    st.appendChild(cp);
    if (r.samples.length) {
      const [sp, sb] = panel("Held-out prompts, before and after", "Greedy completions (temperature 0).");
      sb.innerHTML = `<table><thead><tr><th>Prompt</th><th>Wanted</th><th>Before</th><th>After</th></tr></thead><tbody>${r.samples.map((x) =>
        `<tr><td>${esc(x.prompt)}</td><td><b>${esc(x.target)}</b></td><td class="muted">${esc(x.before)}</td><td class="${x.after.trim().startsWith(x.target) ? "ok" : ""}">${esc(x.after)}</td></tr>`).join("")}</tbody></table>`;
      st.appendChild(sp);
    }
    st.appendChild(actions(r.adapter, r.spec));
    if (r.anatomy) anatomyPanels(st, r.anatomy);
  }

  function showRL(r) {
    const st = $("#tu-stage");
    st.innerHTML = `<div class="head"><h1>${esc(r.adapter)}</h1><span class="meta">Reinforce · ${esc(r.spec)} · reward: ${esc(r.reward_label)}${r.settings.reward_arg ? ` (${esc(r.settings.reward_arg)})` : ""} · group ${r.settings.group} · rank ${r.settings.rank}, ${num(r.trainable)} trainable · ${r.settings.steps} steps · ${Math.round(r.seconds)} s</span></div>
      <p class="finding">Mean reward on held-out prompts <b>${f3(r.before.reward)} → ${f3(r.after.reward)}</b>, same number of samples before and after. ${driftLine(r.drift_kl)}</p>`;
    const [cp, cb] = panel("Mean reward while training", `Over the samples drawn at each step, from the weights as they were then. Dashed: how far the sampled tokens are from the base model (KL penalty ${r.settings.kl_coef}).`);
    lineChart(cb, { series: [{ name: "reward", color: C.reward, points: r.curve.map((x) => [x.step, x.reward]) }, { name: "KL to base", color: C.grad, dash: "4 3", points: r.curve.filter((x) => x.kl !== null && x.kl !== undefined).map((x) => [x.step, x.kl]) }], xLabel: "Step", yFmt: (v) => v.toFixed(2) });
    st.appendChild(cp);
    const [sp, sb] = panel("Samples, before and after", "Temperature 1. The number is each sample's reward.");
    sb.innerHTML = `<table><thead><tr><th>Prompt</th><th>Before</th><th>After</th></tr></thead><tbody>${r.samples.map((x) =>
      `<tr><td>${esc(x.prompt)}</td><td class="muted">${x.before.map((t, i) => `${esc(t)} <span class="badge" style="--c:var(--muted)">${x.before_r[i].toFixed(1)}</span>`).join("<br>")}</td>
       <td>${x.after.map((t, i) => `${esc(t)} <span class="badge" style="--c:${x.after_r[i] > 0.5 ? C.reward : "var(--muted)"}">${x.after_r[i].toFixed(1)}</span>`).join("<br>")}</td></tr>`).join("")}</tbody></table>`;
    st.appendChild(sp);
    st.appendChild(actions(r.adapter, r.spec));
    if (r.anatomy) anatomyPanels(st, r.anatomy);
  }

  async function adapter(name) {
    stopPoll();
    let m;
    try { m = await api(`/api/tune/adapter/${encodeURIComponent(name)}`); } catch (e) { return toast(e.message, true); }
    const st = $("#tu-stage");
    st.innerHTML = `<div class="head"><h1>${esc(m.name)}</h1><span class="meta">on ${esc(m.base)} · ${m.matrices.length} matrices, rank ${m.config ? m.config.rank : m.matrices[0].rank}, alpha ${m.config ? m.config.alpha : "?"} · ${num(m.trainable)} trainable values · ${m.step || 0} steps · open anywhere as <code>tune:${esc(m.name)}</code></span></div>`;
    const log = m.log || [];
    if (log.some((r) => r.loss !== undefined || r.reward !== undefined)) {
      const [p, b] = panel("How it was trained", "From the session's log.");
      const series = [];
      if (log.some((r) => r.loss !== undefined)) series.push({ name: "loss", color: C.train, points: log.filter((r) => r.loss !== undefined).map((r) => [r.step, r.loss]) });
      if (log.some((r) => r.reward !== undefined)) series.push({ name: "reward", color: C.reward, points: log.filter((r) => r.reward !== undefined).map((r) => [r.step, r.reward]) });
      lineChart(b, { series, xLabel: "Optimizer step", yFmt: (v) => v.toFixed(2) });
      st.appendChild(p);
    }
    st.appendChild(actions(m.name, m.base));
    if (m.anatomy) anatomyPanels(st, m.anatomy);
    else {
      const b = document.createElement("button");
      b.className = "primary"; b.textContent = "Where did the fine-tune go?";
      b.onclick = () => runJob(`/api/tune/adapter/${encodeURIComponent(name)}/anatomy`, {}, () => adapter(name));
      st.appendChild(b);
    }
  }

  function anatomyPanels(st, A) {
    const s = A.summary;
    const [p, b] = panel("Where did the fine-tune go?", "The adapter read as a change to the weights, dW = (alpha/r) U Vᵀ per matrix, against a random change of the same size and rank.");
    b.innerHTML = `<ul class="tu-lines">${s.lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>
      <div class="kv"><div><div class="k">Matrices changed</div><div class="v">${s.changed} / ${s.matrices}</div></div>
      <div><div class="k">Directions used (90%)</div><div class="v">${s.median_rank90} of ${s.rank}</div></div>
      <div><div class="k">Intruder directions</div><div class="v">${s.intruders} <span class="muted" style="font-size:13px">vs ${s.intruders_random.toFixed(1)} random</span></div></div>
      <div><div class="k">In the base's top directions</div><div class="v">${pct(s.in_top)} <span class="muted" style="font-size:13px">vs ${pct(s.in_top_random)}</span></div></div></div>`;
    st.appendChild(p);
    const [lp, lb] = panel("Share of the change by block", "Of the total squared size of every dW.");
    const mx = Math.max(...s.by_layer.map((x) => x.share), 1e-9);
    lb.innerHTML = s.by_layer.map((x) => `<div class="tu-bar"><span>${x.layer >= 1e6 ? "unembed" : `block ${x.layer}`}</span><i style="width:${(100 * x.share / mx).toFixed(1)}%"></i><b>${pct(x.share)}</b></div>`).join("")
      + `<p class="hint">By kind: ${Object.entries(s.by_kind).map(([k, v]) => `${esc(k)} ${pct(v)}`).join(", ")}</p>`;
    st.appendChild(lp);
    const [mp, mb] = panel("Every matrix", "Size relative to the matrix; directions holding 90% of the change; intruders among the top 10 directions of W + dW (random change in brackets); share of output in W's top 10% of directions (chance in brackets); heavy-tail exponent and eigenvalues beyond the noise edge, before → after.");
    mb.className = "scroll";
    mb.innerHTML = `<table><thead><tr><th>Matrix</th><th class="num">Shape</th><th class="num">Share</th><th class="num">|dW| / |W|</th><th class="num">Directions</th><th class="num">Intruders</th><th class="num">In top</th><th class="num">alpha</th><th class="num">Spikes</th></tr></thead><tbody>${
      A.matrices.map((m) => `<tr><td><code>${esc(m.name)}</code></td><td class="num">${m.shape.join(" × ")}</td><td class="num">${pct(m.share, 1)}</td><td class="num">${pct(m.rel, 2)}</td>
      <td class="num">${m.rank90} / ${m.rank}</td><td class="num">${m.intruders} <span class="muted">(${(m.intruders_random || 0).toFixed(1)})</span></td>
      <td class="num">${m.in_top === null || m.in_top === undefined ? "–" : pct(m.in_top)} <span class="muted">(${m.in_top_random === undefined || m.in_top_random === null ? "–" : pct(m.in_top_random)})</span></td>
      <td class="num">${m.laws ? `${f2(m.laws.alpha_before)} → ${f2(m.laws.alpha_after)}` : "–"}</td><td class="num">${m.laws ? `${m.laws.spikes_before ?? "–"} → ${m.laws.spikes_after ?? "–"}` : "–"}</td></tr>`).join("")}</tbody></table>`;
    st.appendChild(mp);
  }
  const f2 = (v) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(2));

  $("#tu-compare").onclick = () => {
    const a = $("#tu-ca").value, b = $("#tu-cb").value;
    if (!a || !b || a === b) return toast("Choose two different adapters of the same base model.", true);
    stopPoll();
    runJob("/api/tune/compare", { a, b }, showCompare);
  };
  function showCompare(r) {
    const st = $("#tu-stage");
    st.innerHTML = `<div class="head"><h1>${esc(r.a)} vs ${esc(r.b)}</h1><span class="meta">both on ${esc(r.base)}</span></div><p class="finding">${esc(r.verdict)}</p>`;
    const [p, b] = panel("Matrix by matrix", "Overlap: share of the directions holding 90% of one change that lie in the other's (output side, then input side). Random: the same for a random subspace of the same size. Cosine: of the two changes as whole matrices.");
    b.className = "scroll";
    b.innerHTML = `<table><thead><tr><th>Matrix</th><th class="num">Directions</th><th class="num">Overlap out</th><th class="num">Overlap in</th><th class="num">Random</th><th class="num">Cosine</th></tr></thead><tbody>${
      r.rows.map((x) => `<tr><td><code>${esc(x.name)}</code></td><td class="num">${x.k_a} / ${x.k_b}</td><td class="num ${x.overlap_out > 3 * x.random ? "ok" : ""}">${pct(x.overlap_out)}</td>
      <td class="num">${pct(x.overlap_in)}</td><td class="num">${pct(x.random, 1)}</td><td class="num">${x.cosine.toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
    st.appendChild(p);
  }

  // ------------------------------------------------------------ open the tab
  document.querySelector('nav button[data-view="tune"]').addEventListener("click", init);
  window.Tune = { init, adapter, watch };
  const m = location.hash.match(/^#tune(?:\/(adapter|session)\/(.+))?$/);
  if (m) setTimeout(() => { document.querySelector('nav button[data-view="tune"]').click(); if (m[1] === "adapter") adapter(decodeURIComponent(m[2])); else if (m[1]) watch(m[2]); }, 50);
})();
