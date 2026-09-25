"use strict";
// Side-by-side dashboard: one prompt, two models. Uses helpers from app.js and charts.js.
(function () {
  const T = { data: null, pos: 0, layer: 0, timer: null, tab: "flow", files: {}, weights: null, anat: {}, anSide: "a", anLayer: 0, anNode: "attn" };
  const tokLabel = (t) => String(t).replace(/\n/g, "↵").replace(/^ /, "␣").replace(/ $/, "␣") || "∅";
  const depthB = (i, la, lb) => Math.round((i * lb) / Math.max(1, la));

  const PRESETS = [
    { t: "Two seeds, tiny", d: "Pythia 14M, seed 1 vs seed 2. Downloads in seconds.", a: "EleutherAI/pythia-14m-seed1", b: "EleutherAI/pythia-14m-seed2" },
    { t: "Two seeds, 70M", d: "Same recipe and data, different random start.", a: "EleutherAI/pythia-70m", b: "EleutherAI/pythia-70m-seed1" },
    { t: "Small vs bigger sibling", d: "Pythia 70M vs 160M: same data and tokenizer, twice the depth.", a: "EleutherAI/pythia-70m", b: "EleutherAI/pythia-160m" },
    { t: "Teacher vs distilled student", d: "GPT-2 small vs DistilGPT-2, trained to imitate it with half the layers.", a: "openai-community/gpt2", b: "distilbert/distilgpt2" },
    { t: "Early vs finished", d: "Pythia 70M at step 3000 vs the final checkpoint.", a: "EleutherAI/pythia-70m@step3000", b: "EleutherAI/pythia-70m" },
  ];

  function presets() {
    const ul = $("#tr-presets");
    for (const p of PRESETS) {
      const li = document.createElement("li");
      li.innerHTML = `<button><span class="t1">${esc(p.t)}</span><span class="t2">${esc(p.d)}</span></button>`;
      li.querySelector("button").onclick = () => { $("#tr-a").value = p.a; $("#tr-b").value = p.b; presetFilled(["#tr-a", "#tr-b"], ["#tr-run"], `A and B are now ${p.a.split("/").pop()} and ${p.b.split("/").pop()}. Press Trace prompt to run.`); };
      ul.appendChild(li);
    }
  }

  async function history() {
    const items = await api("/api/trace/history").catch(() => []);
    const ul = $("#tr-history");
    ul.innerHTML = items.length ? "" : `<li class="empty">Traces you run appear here.</li>`;
    const short = (s) => s.split("/").pop();
    for (const h of items) {
      const li = document.createElement("li");
      li.innerHTML = `<button><span class="t1">${esc(short(h.a))} vs ${esc(short(h.b))}</span><span class="t2">${esc(h.prompt)}</span></button>`;
      li.querySelector("button").onclick = async () => show(await api(`/api/trace/history/${h.id}`));
      ul.appendChild(li);
    }
  }

  const specs = () => ({ a: $("#tr-a").value.trim(), b: $("#tr-b").value.trim() });

  $("#tr-run").onclick = () => {
    const { a, b } = specs();
    const prompt = $("#tr-prompt").value.trim();
    if (!a || !b) return toast("Enter two model names.", true);
    if (!prompt) return toast("Enter a prompt to trace.", true);
    $("#tr-stage").innerHTML = `<p class="spinner">Loading both models and tracing the prompt. The first run downloads the checkpoints.</p>`;
    runJob("/api/trace/run", { a, b, prompt, learning: $("#tr-learn").checked }, (blob) => { show(blob); history(); });
  };
  $("#tr-files").onclick = () => { ensureShell(); T.tab = "files"; renderTabs(); };
  $("#tr-weights").onclick = () => { ensureShell(); T.tab = "weights"; renderTabs(); };

  // A result shell exists even before a trace, so Files and Weights work on their own.
  function ensureShell() {
    const { a, b } = specs();
    if (T.data && T.data.a === a && T.data.b === b) return;
    if (!T.data || T.data.a !== a || T.data.b !== b) {
      T.data = { a, b, prompt: null, result: null };
      T.files = {}; T.weights = null; T.anat = {};
    }
    frame();
  }

  function show(blob) {
    stop();
    const r = blob.result;
    const prevFiles = T.data && T.data.a === blob.a && T.data.b === blob.b ? T.files : {};
    const prevWeights = T.data && T.data.a === blob.a && T.data.b === blob.b ? T.weights : null;
    const prevAnat = T.data && T.data.a === blob.a && T.data.b === blob.b && T.data.prompt === blob.prompt ? T.anat : {};
    T.data = blob; T.files = prevFiles; T.weights = prevWeights; T.anat = prevAnat;
    T.pos = r.A.tokens.length - 1;
    T.layer = r.A.layers;
    T.tab = "flow";
    $("#tr-a").value = blob.a; $("#tr-b").value = blob.b; $("#tr-prompt").value = blob.prompt;
    frame();
  }

  function frame() {
    const d = T.data;
    const st = $("#tr-stage");
    st.innerHTML = `<div class="head"><h1><span class="headA">${esc(d.a)}</span> <span class="muted">and</span> <span class="headB">${esc(d.b)}</span></h1></div>
      ${d.result ? `<p class="finding" id="tr-finding"></p>` : `<p class="finding">Pick a tab below. Trace a prompt to fill in the workflow views.</p>`}
      <div class="subtabs" role="tablist" id="tr-tabs"></div><div id="tr-body"></div>`;
    if (d.result) $("#tr-finding").innerHTML = finding(d.result);
    renderTabs();
  }

  function finding(r) {
    const A = r.A, B = r.B, last = A.tokens.length - 1;
    const topA = A.lens[A.layers].top[last][0], topB = B.lens[B.layers].top[B.tokens.length - 1][0];
    let s = `After “${esc(r.prompt.slice(-60))}”, <b class="ca">A</b> predicts <b class="ca">“${esc(tokLabel(topA.t))}”</b> (${pct(topA.p)}) and <b class="cb">B</b> predicts <b class="cb">“${esc(tokLabel(topB.t))}”</b> (${pct(topB.p)}). `;
    if (r.divergence.length) {
      const js = r.divergence.map((x, i) => (i === 0 ? -1 : x.js.reduce((a, b) => a + b, 0) / x.js.length));
      const peak = js.indexOf(Math.max(...js));
      s += `Their layer-by-layer guesses are furthest apart at <b>layer ${r.divergence[peak].a}</b> of A (JS divergence ${fmt(js[peak])}) `;
      s += r.summary && r.summary.settle_layer !== null
        ? `and settle on the same answer from <b class="cab">layer ${r.summary.settle_layer}</b> onward.`
        : `and never settle on the same final answer.`;
      const al = r.divergence.filter((x) => x.align_err);
      if (al.length) {
        const worst = al.reduce((m, x) => { const v = x.align_err[last] / (x.baseline || 1); return v > m.v ? { v, a: x.a } : m; }, { v: -1, a: 0 });
        s += ` Once coordinates are aligned, their internal states differ most at layer ${worst.a}, where the gap is ${fmt(worst.v, 1)}× what is typical on ordinary text.`;
      }
    } else {
      s += "They tokenise the prompt differently, so the comparison below is layer by layer rather than token by token.";
    }
    return s;
  }

  const TABS = [["flow", "Workflow"], ["arch", "Architecture"], ["attention", "Attention"], ["neurons", "Neurons"], ["learning", "If trained on this"], ["files", "Weight files"], ["weights", "Weight patterns"]];
  function renderTabs() {
    const bar = $("#tr-tabs");
    if (!bar) return;
    const hasTrace = !!T.data.result;
    bar.innerHTML = "";
    for (const [k, label] of TABS) {
      const b = document.createElement("button");
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", T.tab === k);
      b.textContent = label;
      b.onclick = () => { stop(); T.tab = k; renderTabs(); };
      bar.appendChild(b);
    }
    const body = $("#tr-body");
    body.innerHTML = "";
    const needTrace = ["flow", "attention", "neurons", "learning"].includes(T.tab);
    if (needTrace && !hasTrace) { body.innerHTML = `<p class="muted">Trace a prompt to see this view.</p>`; return; }
    ({ flow, arch, attention, neurons, learning, files, weights })[T.tab](body);
  }

  function tokenChips(host, onPick) {
    const r = T.data.result;
    const div = document.createElement("div");
    div.className = "tokens";
    const last = r.divergence.length ? r.divergence[r.divergence.length - 1] : null;
    r.A.tokens.forEach((t, i) => {
      const b = document.createElement("button");
      b.textContent = tokLabel(t);
      b.setAttribute("aria-pressed", i === T.pos);
      if (last && !last.agree[i]) { b.classList.add("split"); b.title = "The two models predict different next tokens here"; }
      b.onclick = () => { T.pos = i; onPick(); };
      div.appendChild(b);
    });
    host.appendChild(div);
  }

  // ---------------------------------------------------------------- workflow
  function flow(body) {
    const r = T.data.result, A = r.A, B = r.B;
    body.insertAdjacentHTML("beforeend", `<p class="sub">Pick the token to follow. The models are asked what comes next after it. Underlined tokens are where their final answers differ.</p>`);
    tokenChips(body, () => { const t = T.tab; renderTabs(); T.tab = t; });

    const player = document.createElement("div");
    player.className = "player";
    player.innerHTML = `<button class="primary" id="tr-play">Play</button><input type="range" id="tr-layer" min="0" max="${A.layers}" value="${T.layer}" aria-label="Layer"><span class="muted" id="tr-layer-lab"></span>`;
    body.appendChild(player);

    const grid = document.createElement("div");
    grid.className = "flow";
    grid.innerHTML = `<div class="hdr headA">A: ${esc(T.data.a)}</div><div class="hdr mid">Apart</div><div class="hdr headB">B: ${esc(T.data.b)}</div>`;
    const cells = [];
    const posB = r.same_tokens ? T.pos : B.tokens.length - 1;
    for (let i = 0; i <= A.layers; i++) {
      const j = depthB(i, A.layers, B.layers);
      const dv = r.divergence[i];
      const guesses = (side, l, p) => side.lens[l].top[p].map((g) => `<div class="guess"><span class="w">${esc(tokLabel(g.t))}</span><div class="bar"><i style="width:${g.p * 100}%;background:${side === A ? "var(--a)" : "var(--b)"}"></i></div><span class="p">${pct(g.p, 1)}</span></div>`).join("");
      const lname = (side, l, p) => `<div class="lname"><span>${l === 0 ? "Embedding" : l === side.layers ? `Layer ${l}, output` : `Layer ${l}`}</span><span>norm ${fmt(side.hidden_norm[l][p], 1)}${side.mlp[l - 1] && l > 0 ? `, ${pct(side.mlp[l - 1].share[p], 1)} of neurons carry 90%` : ""}</span></div>`;
      const ca = document.createElement("div"); ca.className = "cellA"; ca.innerHTML = lname(A, i, T.pos) + guesses(A, i, T.pos);
      const cb = document.createElement("div"); cb.className = "cellB"; cb.innerHTML = lname(B, j, posB) + guesses(B, j, posB);
      const cm = document.createElement("div"); cm.className = "cellM";
      if (dv) {
        const ratio = dv.align_err ? dv.align_err[T.pos] / (dv.baseline || 1) : null;
        cm.innerHTML = `<span>${dv.agree[T.pos] ? "Same top guess" : "Different guess"}</span>
          <span title="Jensen-Shannon divergence of the two predictions, 0 to 1">JS ${fmt(dv.js[T.pos])}</span><div class="gauge"><i style="width:${dv.js[T.pos] * 100}%"></i></div>
          ${ratio !== null ? `<span title="Aligned state gap relative to what is typical on ordinary text">State gap ${fmt(ratio, 1)}×</span><div class="gauge al"><i style="width:${Math.min(1, ratio / 3) * 100}%"></i></div>` : ""}`;
      } else cm.innerHTML = `<span>Layer ${i} ↔ ${j}</span>`;
      grid.append(ca, cm, cb);
      cells.push([ca, cm, cb]);
    }
    body.appendChild(grid);

    const setLayer = (L) => {
      T.layer = L;
      $("#tr-layer").value = L;
      $("#tr-layer-lab").textContent = `Signal at layer ${L} of ${A.layers}`;
      cells.forEach((row, i) => row.forEach((c) => { c.classList.toggle("future", i > L); c.classList.toggle("current", i === L); }));
    };
    setLayer(T.layer);
    $("#tr-layer").oninput = (e) => { stop(); setLayer(+e.target.value); };
    $("#tr-play").onclick = () => {
      if (T.timer) return stop();
      $("#tr-play").textContent = "Pause";
      let L = T.layer >= A.layers ? 0 : T.layer;
      setLayer(L);
      T.timer = setInterval(() => {
        L += 1;
        if (L > A.layers) return stop();
        setLayer(L);
        cells[L][0].scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      }, 900);
    };

    if (r.divergence.length) {
      const row = document.createElement("div");
      row.className = "cols";
      row.style.marginTop = "18px";
      const [p1, b1] = panel("Where the predictions part ways", "Jensen-Shannon divergence between the two models' layer-by-layer guesses: at the selected token, and averaged over the prompt.");
      const [p2, b2] = panel("Where the internal states part ways", "Gap between A's state (mapped into B's space with an affine map fitted on other text) and B's actual state, divided by the gap typical on ordinary text. 1× is normal; higher means this prompt pushes them apart.");
      row.append(p1, p2);
      body.appendChild(row);
      const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
      lineChart(b1, { series: [
        { name: "Selected token", color: css("--b"), points: r.divergence.map((x) => [x.a, x.js[T.pos]]), dots: true },
        { name: "Prompt average", color: css("--muted"), dash: "4 3", points: r.divergence.map((x) => [x.a, mean(x.js)]) },
      ], yMin: 0, yMax: 1, xLabel: "Layer of A" });
      if (r.divergence[0].align_err) {
        lineChart(b2, { series: [
          { name: "Selected token", color: css("--ab"), points: r.divergence.map((x) => [x.a, x.align_err[T.pos] / (x.baseline || 1)]), dots: true },
          { name: "Prompt average", color: css("--muted"), dash: "4 3", points: r.divergence.map((x) => [x.a, mean(x.align_err) / (x.baseline || 1)]) },
        ], yMin: 0, xLabel: "Layer of A" });
      } else b2.textContent = "Not available for this pair.";

      const toks = A.tokens.map(tokLabel);
      const lay = r.divergence.map((x) => `L${x.a}`);
      const [p3, b3] = panel("Divergence map", "Every layer (rows) against every token (columns). Bright cells are where the two models disagree about what comes next.");
      body.appendChild(p3);
      heatmap(b3, { matrix: r.divergence.map((x) => x.js), rows: lay, cols: toks, min: 0, max: 1, color: css("--b"), rowLabelWidth: 44, colLabelHeight: 70 });
    }

    const two = document.createElement("div");
    two.className = "two";
    two.style.marginTop = "18px";
    const [pa, ba] = panel(`<span class="headA">A</span>: when does it know the next token?`, "Probability each layer already gives to the token that actually follows.");
    const [pb, bb] = panel(`<span class="headB">B</span>: when does it know the next token?`, "Same measure for model B.");
    two.append(pa, pb);
    body.appendChild(two);
    const tp = (S_) => S_.lens.map((l) => l.true_p.slice(0, -1));
    heatmap(ba, { matrix: tp(A), rows: A.lens.map((_, i) => (i ? `L${i}` : "Emb")), cols: A.tokens.slice(0, -1).map(tokLabel), min: 0, max: 1, color: css("--a"), rowLabelWidth: 44, colLabelHeight: 70 });
    heatmap(bb, { matrix: tp(B), rows: B.lens.map((_, i) => (i ? `L${i}` : "Emb")), cols: B.tokens.slice(0, -1).map(tokLabel), min: 0, max: 1, color: css("--b"), rowLabelWidth: 44, colLabelHeight: 70 });
    for (const n of r.notes) body.insertAdjacentHTML("afterbegin", `<p class="note">${esc(n)}</p>`);
  }

  function stop() {
    if (T.timer) clearInterval(T.timer);
    T.timer = null;
    const b = $("#tr-play");
    if (b) b.textContent = "Play";
  }

  // ---------------------------------------------------------------- attention
  function attention(body) {
    const r = T.data.result, A = r.A, B = r.B;
    if (!A.attention || !B.attention) { body.innerHTML = `<p class="muted">This model did not return attention patterns.</p>`; return; }
    const la = A.layers;
    let L = Math.max(1, Math.min(la, T.attL || Math.ceil(la / 2)));
    let hA = T.attH || 0;
    body.insertAdjacentHTML("beforeend", `<p class="sub">Each grid shows, for every token (row), how much it looks at each earlier token (column). Heads have no fixed order, so for each head of A the view picks B's most similar head at the same depth.</p>
      <div class="player"><label style="flex-direction:row;align-items:center;gap:8px">Layer of A <input type="range" id="at-l" min="1" max="${la}" value="${L}"></label><span class="muted" id="at-lab"></span>
      <label style="flex-direction:row;align-items:center;gap:8px">Head of A <select id="at-h">${A.attention[0].map((_, i) => `<option value="${i}">${i}</option>`).join("")}</select></label></div>`);
    const two = document.createElement("div");
    two.className = "two";
    const [pa, ba] = panel(`<span class="headA">A</span>`);
    const [pb, bb] = panel(`<span class="headB">B</span>`);
    two.append(pa, pb);
    body.appendChild(two);
    const [pc, bc] = panel("How alike are the heads, layer by layer?", "Average similarity between each head of A and its best one-to-one match in B, on this prompt, after removing the plain look-at-everything-before pattern that every head shares. 1 means identical, 0 unrelated.");
    pc.style.marginTop = "18px";
    body.appendChild(pc);

    const draw = () => {
      T.attL = L; T.attH = hA;
      const j = depthB(L, la, B.layers);
      const dv = r.divergence[L];
      const hB = dv && dv.head_match ? dv.head_match[hA] : Math.min(hA, B.attention[0].length - 1);
      const sim = dv && dv.head_sim ? dv.head_sim[hA] : null;
      $("#at-lab").textContent = `B layer ${j}, head ${hB}${sim !== null ? `, similarity ${fmt(sim)}` : ""}`;
      pa.querySelector("h3").innerHTML = `<span class="headA">A</span>, layer ${L}, head ${hA}`;
      pb.querySelector("h3").innerHTML = `<span class="headB">B</span>, layer ${j}, head ${hB}`;
      heatmap(ba, { matrix: A.attention[L - 1][hA], rows: A.tokens.map(tokLabel), cols: A.tokens.map(tokLabel), color: css("--a"), rowLabelWidth: 70, colLabelHeight: 60 });
      heatmap(bb, { matrix: B.attention[j - 1][hB], rows: B.tokens.map(tokLabel), cols: B.tokens.map(tokLabel), color: css("--b"), rowLabelWidth: 70, colLabelHeight: 60 });
    };
    $("#at-l").oninput = (e) => { L = +e.target.value; draw(); };
    $("#at-h").value = hA;
    $("#at-h").onchange = (e) => { hA = +e.target.value; draw(); };
    draw();
    const rows = r.divergence.filter((x) => x.head_sim);
    if (rows.length) {
      lineChart(bc, { series: [{ name: "Matched head similarity", color: css("--ab"), points: rows.map((x) => [x.a, x.head_sim.reduce((a, b) => a + b, 0) / x.head_sim.length]), dots: true }], yMin: -0.2, yMax: 1, xLabel: "Layer of A" });
    } else bc.textContent = "Head matching needs both models to split the prompt into the same tokens.";
  }

  // ---------------------------------------------------------------- neurons
  function neurons(body) {
    const r = T.data.result, A = r.A, B = r.B;
    body.insertAdjacentHTML("beforeend", `<p class="sub">The prompt does not change any weight, but it decides which weights matter. For each layer, this is the share of MLP neurons that together carry 90% of the activity. Small shares mean the input is routed through a narrow slice of the network.</p>`);
    tokenChips(body, () => { const t = T.tab; renderTabs(); T.tab = t; });
    const has = (S_) => S_.mlp.some((m) => m);
    if (!has(A) && !has(B)) { body.insertAdjacentHTML("beforeend", `<p class="muted">No MLP activation modules were recognised in these architectures.</p>`); return; }
    const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
    const [p1, b1] = panel("Share of neurons doing the work", "By relative depth, so models with different numbers of layers line up.");
    body.appendChild(p1);
    const ser = (S_, color, name) => ({ name, color, dots: true, points: S_.mlp.map((m, i) => (m ? [(i + 1) / S_.layers, mean(m.share)] : null)).filter(Boolean) });
    lineChart(b1, { series: [ser(A, css("--a"), "A, prompt average"), ser(B, css("--b"), "B, prompt average")], yMin: 0, xMin: 0, xMax: 1, xLabel: "Relative depth", yFmt: (v) => pct(v) });
    const two = document.createElement("div");
    two.className = "two";
    two.style.marginTop = "18px";
    const pB = r.same_tokens ? T.pos : B.tokens.length - 1;
    for (const [S_, name, cls, p] of [[A, "A", "headA", T.pos], [B, "B", "headB", pB]]) {
      const [pp, bb] = panel(`<span class="${cls}">${name}</span>: strongest neurons after “${esc(tokLabel(S_.tokens[p]))}”`, "Top five MLP neurons per layer, with their activation.");
      bb.className = "scroll";
      bb.innerHTML = `<table><thead><tr><th>Layer</th><th class="num">Neurons</th><th class="num">Share for 90%</th><th>Top neurons (index: activation)</th></tr></thead><tbody>${
        S_.mlp.map((m, i) => (m ? `<tr><td>${i + 1}</td><td class="num">${m.units}</td><td class="num">${pct(m.share[p], 1)}</td><td>${m.top[p].idx.map((ix, k) => `${ix}: ${fmt(m.top[p].val[k], 2)}`).join(", ")}</td></tr>` : "")).join("")
      }</tbody></table>`;
      two.appendChild(pp);
    }
    body.appendChild(two);
    const two2 = document.createElement("div");
    two2.className = "two";
    two2.style.marginTop = "18px";
    for (const [S_, name, cls, col] of [[A, "A", "headA", css("--a")], [B, "B", "headB", css("--b")]]) {
      const [pp, bb] = panel(`<span class="${cls}">${name}</span>: share per layer and token`, "Darker means more neurons are involved; the scale runs from this model's lowest to highest share.");
      two2.appendChild(pp);
      const rows = S_.mlp.map((m, i) => (m ? { l: `L${i + 1}`, v: m.share } : null)).filter(Boolean);
      heatmap(bb, { matrix: rows.map((x) => x.v), rows: rows.map((x) => x.l), cols: S_.tokens.map(tokLabel), min: Math.min(...rows.flatMap((x) => x.v)), max: Math.max(...rows.flatMap((x) => x.v)), color: col, rowLabelWidth: 44, colLabelHeight: 70 });
    }
    body.appendChild(two2);
  }

  // ---------------------------------------------------------------- learning
  function learning(body) {
    const r = T.data.result;
    if (!r.learning) { body.innerHTML = `<p class="muted">This trace was run without the learning probe. Tick the box on the left and trace again.</p>`; return; }
    const a = r.learning.a, b = r.learning.b;
    const line = (x, n, c) => `<b class="${c}">${n}</b>'s loss on this prompt falls from ${fmt(x.loss_before, 3)} to ${fmt(x.loss_after, 3)}${x.pred_before !== x.pred_after ? `, and its final guess changes from “${esc(tokLabel(x.pred_before_t))}” to “${esc(tokLabel(x.pred_after_t))}”` : `; its final guess stays “${esc(tokLabel(x.pred_before_t))}”`}.`;
    body.insertAdjacentHTML("beforeend", `<p class="finding">Reading a prompt never changes a weight. Training on it does: here each model takes one gradient step on this prompt, sized so every model moves by 0.1% of its total weight norm. ${line(a, "A", "ca")} ${line(b, "B", "cb")}</p>`);
    const [p1, b1] = panel("Where the change lands", "Share of the total gradient that falls on each layer, by relative depth. Parameters outside the layers (embeddings, final norm) take the rest.");
    body.appendChild(p1);
    const ser = (x, n, color, L) => ({ name: n, color, dots: true, points: x.by_layer.map((y) => [(y.layer + 1) / L, y.share]) });
    lineChart(b1, { series: [ser(a, `A (outside layers: ${pct(a.non_layer_share)})`, css("--a"), r.A.layers), ser(b, `B (outside layers: ${pct(b.non_layer_share)})`, css("--b"), r.B.layers)], yMin: 0, xMin: 0, xMax: 1, xLabel: "Relative depth", yFmt: (v) => pct(v) });
    const two = document.createElement("div");
    two.className = "two";
    two.style.marginTop = "18px";
    for (const [x, n, cls] of [[a, "A", "headA"], [b, "B", "headB"]]) {
      const [pp, bb] = panel(`<span class="${cls}">${n}</span>: tensors that would move most`,
        `Rank shows how many directions hold 90% of a matrix's gradient. One prompt of ${x.tokens} tokens can only push a matrix in about ${x.tokens} directions, which is why fine-tuning on a few examples is naturally low-rank.`);
      bb.className = "scroll";
      bb.innerHTML = `<table><thead><tr><th>Tensor</th><th class="num">Share</th><th class="num">Relative size</th><th class="num">Rank (90%)</th></tr></thead><tbody>${
        x.tensors.slice(0, 14).map((t) => `<tr><td>${esc(t.name)}</td><td class="num">${pct(t.share, 1)}</td><td class="num">${t.rel === null ? "–" : fmt(t.rel, 3)}</td><td class="num">${t.rank90 ? `${t.rank90} of ${t.full_rank}` : "–"}</td></tr>`).join("")
      }</tbody></table>`;
      two.appendChild(pp);
    }
    body.appendChild(two);
  }

  // ---------------------------------------------------------------- files
  function files(body) {
    const { a, b } = T.data;
    body.insertAdjacentHTML("beforeend", `<p class="sub">A checkpoint is a long run of numbers plus an index that says where each tensor starts. This view reads the files directly.</p>`);
    const two = document.createElement("div");
    two.className = "two";
    body.appendChild(two);
    const hostA = document.createElement("div"), hostB = document.createElement("div");
    two.append(hostA, hostB);
    const load = (spec, host, side, next) => {
      if (T.files[side]) { fileView(host, T.files[side], side); return next && next(); }
      host.innerHTML = `<p class="spinner">Reading ${esc(spec)}…</p>`;
      runJob("/api/trace/storage", { spec }, (r) => { T.files[side] = r; if (T.tab === "files") fileView(host, r, side); next && next(); });
    };
    load(a, hostA, "a", () => load(b, hostB, "b"));
  }

  function fileView(host, r, side) {
    const cls = side === "a" ? "headA" : "headB";
    const col = side === "a" ? "var(--a)" : "var(--b)";
    host.innerHTML = "";
    const [p0, b0] = panel(`<span class="${cls}">${side.toUpperCase()}</span>: ${esc(r.spec)}`, esc(r.source));
    b0.className = "kv";
    b0.innerHTML = [["Parameters", Number(r.total_params).toLocaleString()], ["Tensor bytes", bytes(r.total_bytes)], ["Format", r.format === "safetensors" ? "safetensors" : "PyTorch zip (.bin)"], ["Number types", Object.keys(r.dtypes).join(", ")]]
      .map(([k, v]) => `<div><div class="k">${k}</div><div class="v" style="font-size:16px">${v}</div></div>`).join("");
    host.appendChild(p0);

    const [p1, b1] = panel("Anatomy of the file", r.format === "safetensors"
      ? "First 8 bytes give the length of a JSON index; the index lists every tensor's type, shape and byte range; the rest is raw numbers packed end to end."
      : "A zip archive: a pickled index (data.pkl) plus one raw data blob per tensor.");
    p1.style.marginTop = "18px";
    const palette = { embedding: "#2446c7", attention: "#6b3fb3", mlp: "#c23a6b", norm: "#1f8a70", unembedding: "#b7791f", head: "#b7791f", other: "#5f6b7a" };
    let html = "";
    if (r.format === "safetensors" && r.preview) {
      const tot = r.total_bytes + r.header_bytes + 8;
      html += `<div class="anatomy"><div style="flex:0 0 3px;background:#18212f" title="8-byte length"></div><div style="flex:${Math.max(r.header_bytes / tot, 0.02)};background:#5f6b7a" title="JSON index">index</div>${
        r.by_kind.map((k) => `<div style="flex:${k.bytes / tot};background:${palette[k.kind] || "#5f6b7a"}" title="${k.kind}: ${bytes(k.bytes)}">${k.bytes / tot > 0.08 ? k.kind : ""}</div>`).join("")}</div>`;
      html += `<p class="sub">Legend: ${r.by_kind.map((k) => `<span style="color:${palette[k.kind]}">■</span> ${k.kind} ${pct(k.bytes / r.total_bytes)}`).join(", ")}</p>`;
      html += `<p class="sub" style="margin:10px 0 4px">First 8 bytes (index length ${r.preview.header_len.toLocaleString()} bytes):</p><div class="hex">${esc(r.preview.header_prefix)}</div>`;
      html += `<p class="sub" style="margin:10px 0 4px">Index entry for <b>${esc(r.preview.tensor)}</b>:</p><div class="hex">${esc(r.preview.header_json)}</div>`;
      html += `<p class="sub" style="margin:10px 0 4px">Its first bytes at offset ${r.preview.offset.toLocaleString()}, and what they mean as ${r.preview.dtype} numbers:</p><div class="hex">${esc(r.preview.hex)}\n→ ${r.preview.values.map((v) => v.toPrecision(5)).join(", ")}</div>`;
    } else if (r.zip_entries.length) {
      html += `<div class="hex">${r.zip_entries.slice(0, 14).map((z) => `${esc(z.name)}  ${bytes(z.bytes)}`).join("\n")}</div>`;
      if (r.preview) html += `<p class="sub" style="margin:10px 0 4px">First values of <b>${esc(r.preview.tensor)}</b>:</p><div class="hex">${r.preview.values.map((v) => v.toPrecision(5)).join(", ")}</div>`;
    }
    if (r.preview && r.preview.bits) {
      const bt = r.preview.bits;
      html += `<p class="sub" style="margin:10px 0 4px">One weight, ${r.preview.values[0].toPrecision(6)}, bit by bit (<span style="color:var(--b)">sign</span>, <span style="color:var(--a)">exponent</span>, <span style="color:var(--ab)">mantissa</span>):</p>
        <div class="bits"><span class="s">${bt.sign}</span> <span class="e">${bt.exponent}</span> <span class="m">${bt.mantissa}</span></div>`;
    }
    b1.innerHTML = html;
    host.appendChild(p1);

    const [p2, b2] = panel("What the numbers look like", "Distribution of learned values by kind of tensor, over ±4 standard deviations."
      + (r.buffers && r.buffers.length ? ` Left out: ${r.buffers.length} stored constants that are not learned (causal masks, masking fillers, rotary frequencies), such as ${esc(r.buffers[0])}.` : "")
      + (r.nonfinite ? ` ${r.nonfinite.toLocaleString()} non-finite values were skipped.` : ""));
    p2.style.marginTop = "18px";
    host.appendChild(p2);
    for (const h of r.histograms.filter((h) => h.kind !== "other" || r.histograms.length < 4)) {
      const d = document.createElement("div");
      d.innerHTML = `<p class="sub" style="margin:8px 0 0">${esc(h.kind)}: mean ${fmt(h.mean, 4)}, std ${fmt(h.std, 4)}, range ${fmt(h.min, 3)} to ${fmt(h.max, 3)}</p>`;
      const c = document.createElement("div");
      d.appendChild(c);
      b2.appendChild(d);
      const step = (h.hi - h.lo) / h.hist.length;
      groupedBars(c, { labels: h.hist.map((_, i) => fmt(h.lo + (i + 0.5) * step, 2)), series: [{ name: "", color: col, values: h.hist }], legend: false, height: 150, yFmt: (v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(v)) });
    }

    const [p3, b3] = panel("Index", `All ${r.tensors.length} tensors in storage order.`);
    p3.style.marginTop = "18px";
    b3.className = "scroll";
    b3.style.maxHeight = "420px";
    b3.innerHTML = `<table><thead><tr><th>Tensor</th><th>Type</th><th>Shape</th><th class="num">Bytes</th></tr></thead><tbody>${
      r.tensors.map((t) => `<tr><td>${esc(t.name)}</td><td class="muted">${t.dtype}</td><td class="muted">${t.shape.join(" × ")}</td><td class="num">${bytes(t.bytes)}</td></tr>`).join("")
    }</tbody></table>`;
    host.appendChild(p3);
  }

  function bytes(n) {
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)} kB`;
    return `${n} B`;
  }

  // ---------------------------------------------------------------- weights
  function weights(body) {
    const { a, b } = T.data;
    const host = document.createElement("div");
    body.appendChild(host);
    if (T.weights) return renderWeights(host, T.weights);
    host.innerHTML = `<p class="sub">Compares the stored weights directly: raw similarity, singular-value shape per matrix (which ignores neuron order and rotations), and whether the difference is low-rank.</p><button class="primary" id="w-go">Compare weights of A and B</button>`;
    $("#w-go").onclick = () => {
      host.innerHTML = `<p class="spinner">Comparing weights…</p>`;
      runJob("/api/hub/weights", { a, b }, (blob) => { T.weights = blob.result; if (T.tab === "weights") { host.innerHTML = ""; renderWeights(host, blob.result); } });
    };
  }

  // ---------------------------------------------------------------- architecture
  function arch(body) {
    const d = T.data;
    const bar = document.createElement("div");
    bar.className = "player";
    bar.innerHTML = `<span class="muted">Model</span>
      <button class="ghost small" data-s="a" aria-pressed="${T.anSide === "a"}"><span class="headA">A</span> ${esc(d.a.split("/").pop())}</button>
      <button class="ghost small" data-s="b" aria-pressed="${T.anSide === "b"}"><span class="headB">B</span> ${esc(d.b.split("/").pop())}</button>`;
    bar.querySelectorAll("button").forEach((b) => (b.onclick = () => { T.anSide = b.dataset.s; T.anLayer = 0; renderTabs(); }));
    body.appendChild(bar);
    const host = document.createElement("div");
    body.appendChild(host);
    const side = T.anSide;
    if (T.anat[side]) return archView(host, T.anat[side], side);
    const spec = side === "a" ? d.a : d.b;
    const prompt = d.prompt || $("#tr-prompt").value.trim();
    host.innerHTML = `<p class="spinner">Reading the code and structure of ${esc(spec)}…</p>`;
    runJob("/api/trace/anatomy", { spec, prompt }, (r) => { T.anat[side] = r; if (T.tab === "arch" && T.anSide === side) archView(host, r, side); });
  }

  const PARAMS = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k` : String(n));
  const sumW = (ws) => (ws || []).reduce((a, w) => a + w.params, 0);

  function archView(host, r, side) {
    const I = r.info, imp = r.impact;
    const col = side === "a" ? "var(--a)" : "var(--b)";
    const L = Math.min(T.anLayer, I.layers - 1);
    host.innerHTML = "";
    let lede = `${esc(I.class)}: ${I.layers} blocks, ${I.d_model}-dimensional residual stream, ${I.heads} attention heads of ${I.d_head} dims, ${I.d_ff ? `${I.d_ff} MLP neurons per block` : "MLP"}, ${PARAMS(I.total_params)} parameters. `;
    lede += I.parallel_residual ? "Attention and MLP run side by side in each block and are added to the stream together. " : "Attention runs first, then the MLP, each adding to the stream in turn. ";
    lede += I.rotary ? "Position is injected by rotating queries and keys, not by a learned vector. " : I.learned_positions ? "Position comes from a learned vector added at the start. " : "";
    if (imp) {
      const worst = imp.layers.flatMap((l) => [["attention", l.layer, l.attn_dloss], ["MLP", l.layer, l.mlp_dloss]]).filter((x) => x[2] !== undefined).sort((a, b) => b[2] - a[2])[0];
      if (worst) lede += `On your prompt, the single most important step is the <b>${worst[0]} of layer ${worst[1]}</b>: removing it raises the loss by ${fmt(worst[2], 3)}.`;
    }
    host.insertAdjacentHTML("beforeend", `<p class="finding" style="font-size:18px">${lede}</p>`);

    // where the parameters live
    const G = I.param_groups, tot = Object.values(G).reduce((a, b) => a + b, 0) || 1;
    const gcol = { embedding: "#2446c7", attention: "#6b3fb3", mlp: "#c23a6b", norms: "#1f8a70", unembedding: "#b7791f", other: "#5f6b7a" };
    host.insertAdjacentHTML("beforeend", `<div class="anatomy" title="Where the parameters live">${Object.entries(G).filter(([, v]) => v > 0).map(([k, v]) =>
      `<div style="flex:${v / tot};background:${gcol[k]}" title="${k}: ${PARAMS(v)}">${v / tot > 0.08 ? `${k} ${pct(v / tot)}` : ""}</div>`).join("")}</div>
      <p class="sub">Where the ${PARAMS(I.total_params)} parameters live: ${Object.entries(G).filter(([, v]) => v > 0).map(([k, v]) => `<span style="color:${gcol[k]}">■</span> ${k} ${PARAMS(v)}`).join(", ")}${I.tied_embeddings ? ". The output layer reuses the embedding table." : "."}</p>`);

    const wrap = document.createElement("div");
    wrap.className = "archwrap";
    host.appendChild(wrap);
    const chart = document.createElement("div");
    chart.className = "chartcol";
    const detail = document.createElement("div");
    detail.className = "detailcol";
    wrap.append(chart, detail);

    const nodes = r.nodes;
    const block = nodes.find((n) => n.id === "block");
    const byId = {};
    for (const n of nodes) byId[n.id] = n;
    for (const c of block.children) byId[c.id] = c;
    const row = imp ? imp.layers[L] : null;

    const card = (n, extra = "") => {
      const w = sumW(n.weights);
      const badge = n.has_weights ? `<span class="wbadge">${w ? `${PARAMS(w)} weights` : "weights"}${n.kind === "attn" || n.kind === "mlp" || n.kind === "norm" ? " per block" : ""}</span>` : `<span class="nobadge">no weights</span>`;
      return `<button class="fnode ${n.has_weights ? "hasw" : "now"} k-${n.kind} ${T.anNode === n.id ? "sel" : ""}" data-id="${n.id}" style="--c:${col}">
        <span class="fl">${esc(n.label)}</span>${badge}${extra}</button>`;
    };
    const arrow = `<div class="farrow" aria-hidden="true"></div>`;
    const imps = (kind) => {
      if (!row || row[`${kind}_norm`] === undefined) return "";
      const dl = row[`${kind}_dloss`];
      return `<span class="fimp">output ${fmt(row[`${kind}_norm`], 1)} vs stream ${fmt(row.resid_in, 1)}; removing it: loss ${dl >= 0 ? "+" : ""}${fmt(dl, 3)}${row[`${kind}_flips`] ? ", changes the answer" : ""}</span>`;
    };
    let html = card(byId.tokens) + arrow + card(byId.embed) + arrow;
    html += `<div class="fblock"><div class="fbhead"><button class="linkbtn" data-id="block">${esc(block.label)}</button>
      <label class="fbl">Layer <input type="range" id="an-layer" min="0" max="${I.layers - 1}" value="${L}"> <b id="an-lab">${L}</b></label></div>
      <div class="fstream">residual stream in: h<sup>${L}</sup></div>`;
    if (block.parallel) {
      html += `<div class="fpar"><div>${card(byId.ln1)}${arrow}${card(byId.attn, imps("attn"))}</div><div>${card(byId.ln2)}${arrow}${card(byId.mlp, imps("mlp"))}</div></div>${arrow}${card(byId.add)}`;
    } else {
      html += `${card(byId.ln1)}${arrow}${card(byId.attn, imps("attn"))}${arrow}${card(byId.add)}${arrow}${card(byId.ln2)}${arrow}${card(byId.mlp, imps("mlp"))}${arrow}${card(byId.add)}`;
    }
    html += `<div class="fstream">residual stream out: h<sup>${L + 1}</sup>, repeated for all ${I.layers} blocks</div></div>`;
    html += arrow + (byId.fnorm ? card(byId.fnorm) + arrow : "") + card(byId.unembed) + arrow + card(byId.softmax, imp ? `<span class="fimp">predicts “${esc(tokLabel(imp.prediction))}” at ${pct(imp.prediction_p)}</span>` : "");
    chart.innerHTML = html;
    chart.querySelectorAll("[data-id]").forEach((b) => (b.onclick = () => { T.anNode = b.dataset.id; archView(host, r, side); }));
    chart.querySelector("#an-layer").oninput = (e) => { T.anLayer = +e.target.value; archView(host, r, side); };

    // detail panel
    const n = byId[T.anNode] || byId.attn;
    let dh = `<h3>${esc(n.label)}${n.id === "block" ? "" : ["ln1", "attn", "ln2", "mlp", "add"].includes(n.id) ? `, layer ${L}` : ""}</h3>`;
    if (n.note) dh += `<p class="sub">${esc(n.note)}</p>`;
    if (n.math && n.math.length) dh += `<div class="math">${n.math.map(esc).join("<br>")}</div>`;
    if (n.steps) {
      dh += `<ol class="steps">${n.steps.map((st) => `<li class="${st.has_weights ? "hasw" : "now"}" style="--c:${col}"><div class="sl">${esc(st.label)} ${st.has_weights ? `<span class="wbadge">${PARAMS(sumW(st.weights))} weights</span>` : `<span class="nobadge">no weights</span>`}</div>
        <div class="math">${st.math.map(esc).join("<br>")}</div>${st.weights && st.weights.length ? `<div class="wlist">${st.weights.map((w) => `${esc(w.name)} <span class="muted">${w.shape.join(" × ")}</span>`).join("<br>")}</div>` : ""}</li>`).join("")}</ol>`;
    }
    if (n.id === "block") dh += `<p class="sub">${PARAMS(n.params_per_block)} parameters per block, ${I.layers} blocks.</p>`;
    if (row && (n.id === "attn" || n.id === "mlp")) {
      const k = n.id;
      dh += `<div class="kv" style="margin:12px 0"><div><div class="k">Output size, layer ${L}</div><div class="v" style="font-size:17px">${fmt(row[`${k}_norm`], 2)}</div></div>
        <div><div class="k">Stream size going in</div><div class="v" style="font-size:17px">${fmt(row.resid_in, 2)}</div></div>
        <div><div class="k">Loss change if removed</div><div class="v" style="font-size:17px">${row[`${k}_dloss`] >= 0 ? "+" : ""}${fmt(row[`${k}_dloss`], 3)}</div></div>
        <div><div class="k">Change in the answer's probability</div><div class="v" style="font-size:17px">${row[`${k}_dprob`] >= 0 ? "+" : ""}${pct(row[`${k}_dprob`], 1)}</div></div></div>`;
    }
    if (n.weights && n.weights.length && !n.steps) {
      dh += `<table style="margin-top:10px"><thead><tr><th>Weight tensor</th><th>Shape</th><th class="num">Parameters</th></tr></thead><tbody>${
        n.weights.map((w) => `<tr><td>${esc(w.name)}</td><td class="muted">${w.shape.join(" × ")}</td><td class="num">${w.params.toLocaleString()}</td></tr>`).join("")}</tbody></table>`;
    }
    if (n.code) dh += `<p class="sub" style="margin:14px 0 4px">The code that runs this step: <b>${esc(n.code.cls)}.forward</b> in ${esc(n.code.file)}, line ${n.code.line}</p><pre class="code">${esc(n.code.code)}</pre>`;
    detail.innerHTML = `<div class="panel">${dh}</div>`;

    // impact over depth
    if (imp) {
      const [pp, pb] = panel("How much each layer matters on this prompt",
        `Loss change when one step's output is removed (higher means the model relied on it). Baseline loss ${fmt(imp.loss, 3)} over ${imp.tokens} tokens.`);
      pp.style.marginTop = "18px";
      host.appendChild(pp);
      groupedBars(pb, { labels: imp.layers.map((l) => `L${l.layer}`), series: [
        { name: "Remove attention", color: "#6b3fb3", values: imp.layers.map((l) => l.attn_dloss ?? null) },
        { name: "Remove MLP", color: "#c23a6b", values: imp.layers.map((l) => l.mlp_dloss ?? null) }], yLabel: "Loss change" });
      const [qp, qb] = panel("How loudly each step writes", "Average size of each step's output compared with the residual stream it adds to.");
      qp.style.marginTop = "18px";
      host.appendChild(qp);
      lineChart(qb, { series: [
        { name: "Residual stream", color: css("--muted"), dash: "4 3", points: imp.layers.map((l) => [l.layer, l.resid_in]), dots: true },
        { name: "Attention output", color: "#6b3fb3", points: imp.layers.map((l) => [l.layer, l.attn_norm]).filter((p) => p[1] !== undefined), dots: true },
        { name: "MLP output", color: "#c23a6b", points: imp.layers.map((l) => [l.layer, l.mlp_norm]).filter((p) => p[1] !== undefined), dots: true }], xLabel: "Layer", yMin: 0 });
    }
    if (I.model_code) host.insertAdjacentHTML("beforeend", `<details class="panel" style="margin-top:18px"><summary>Top-level model code: ${esc(I.model_code.cls)}.forward (${esc(I.model_code.file)}, line ${I.model_code.line})</summary><pre class="code">${esc(I.model_code.code)}</pre></details>`);
  }

  presets();
  history();
})();
