"use strict";
// Write: Weightspace Lab's playground (a 2 -> H -> 1 network edited by hand) inside Weight space,
// plus Claude writing or repairing the weights, graded on the same 41 x 41 grid.
(function () {
  const P = { meta: null, pattern: "diagonal", net: null, base: null, layer: "output", from: "broken" };
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const pctf = (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);

  function predict(n, x, y) {
    let z = n.bias;
    for (let i = 0; i < n.W.length; i++) z += n.v[i] * Math.max(0, n.W[i][0] * x + n.W[i][1] * y + n.b[i]);
    return sig(z);
  }
  function agreement(n, pattern) {
    const t = P.meta.patterns[pattern].target;
    let ok = 0;
    for (let yy = 0; yy < 41; yy++) for (let xx = 0; xx < 41; xx++) {
      const k = yy * 41 + xx;
      if ((predict(n, (xx - 20) / 20, (yy - 20) / 20) > 0.5) === (t[k] === 1)) ok++;
    }
    return ok / 1681;
  }
  const flat = (n) => [...n.W.flat(), ...n.b, ...n.v, n.bias];
  const zeros = (H) => ({ W: Array.from({ length: H }, () => [0, 0]), b: Array(H).fill(0), v: Array(H).fill(0), bias: 0 });
  function resize(n, H) {
    const z = zeros(H);
    for (let i = 0; i < Math.min(H, n.W.length); i++) { z.W[i] = [...n.W[i]]; z.b[i] = n.b[i]; z.v[i] = n.v[i]; }
    z.bias = n.bias;
    return z;
  }
  function broken(p) {
    const c = P.meta.patterns[p].crafted;
    if (!c) return zeros(P.meta.patterns[p].hidden);
    const n = clone(c); n.v[0] = -n.v[0]; return n;
  }

  // ------------------------------------------------------------ drawing
  function draw(canvas, n, pattern, marks = true) {
    const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height, im = ctx.createImageData(w, h);
    const c0 = [100, 83, 153], c1 = [175, 205, 118];
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      const p = predict(n, (i / (w - 1)) * 2 - 1, 1 - (j / (h - 1)) * 2), k = (j * w + i) * 4;
      for (let c = 0; c < 3; c++) im.data[k + c] = c0[c] * (1 - p) + c1[c] * p;
      im.data[k + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    ctx.strokeStyle = "rgba(240,244,230,.35)"; ctx.setLineDash([3, 5]);
    ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke(); ctx.setLineDash([]);
    if (!marks) return;
    const t = P.meta.patterns[pattern].target;
    ctx.fillStyle = "rgba(255,255,255,.95)";
    for (let yy = 0; yy < 41; yy++) for (let xx = 0; xx < 41; xx++) {      // a white dot where the network disagrees with the target
      const x = (xx - 20) / 20, y = (yy - 20) / 20;
      if ((predict(n, x, y) > 0.5) !== (t[yy * 41 + xx] === 1)) {
        ctx.beginPath(); ctx.arc(((x + 1) / 2) * (w - 1), ((1 - y) / 2) * (h - 1), 2, 0, 7); ctx.fill();
      }
    }
  }

  // ------------------------------------------------------------ playground
  async function open() {
    if (!P.meta) {
      try { P.meta = await api("/api/wsp/write/meta"); } catch (e) { toast(e.message, true); return; }
      $("#ww-pattern").innerHTML = Object.entries(P.meta.patterns).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join("");
      $("#ww-pattern").onchange = () => { P.pattern = $("#ww-pattern").value; $("#ww-hidden").value = P.meta.patterns[P.pattern].hidden; load("broken"); };
      $("#ww-mode").onchange = () => { $("#ww-hidden").disabled = $("#ww-mode").value === "repair"; };
      $("#ww-hidden").onchange = () => {
        const H = Math.max(1, Math.min(16, parseInt($("#ww-hidden").value, 10) || 4));
        if (P.net && H !== P.net.W.length) { P.net = resize(P.net, H); P.from = "custom"; playground(); }
      };
    }
    if (!P.net) load("broken"); else playground();
  }

  function load(from, net) {
    P.from = from;
    P.base = broken(P.pattern);
    P.net = net ? clone(net) : from === "crafted" && P.meta.patterns[P.pattern].crafted ? clone(P.meta.patterns[P.pattern].crafted) : clone(P.base);
    playground();
  }

  function playground() {
    const m = P.meta.patterns[P.pattern], H = P.net.W.length;
    const st = $("#ws-stage");
    st.innerHTML = `<div class="head"><h1>Write: ${esc(m.label)}</h1><span class="meta">Target: ${esc(m.rule)} · 2 → ${H} → 1 · ReLU, then sigmoid · runs in this browser</span></div>
      <p class="finding" id="ww-finding"></p>
      <div class="ww-grid">
        <div class="panel"><h3>Decision landscape</h3><p class="sub">Green is class 1, violet class 0; every pixel is a forward pass through these weights. White dots mark the 41 × 41 grid points where the network disagrees with the target.</p>
          <canvas id="ww-plot" width="420" height="420" class="ww-canvas" aria-label="Decision landscape of the network"></canvas><div class="sub" id="ww-probe">Hover to read the output at a point.</div></div>
        <div class="panel"><h3>Behaviour</h3>
          <div class="kv"><div><div class="k">Agreement with the target</div><div class="v" id="ww-acc">–</div></div>
            <div><div class="k">Weights changed</div><div class="v" id="ww-changed">–</div></div></div>
          <div class="row2" style="margin-top:14px"><button class="ghost" id="ww-crafted" ${m.crafted ? "" : "disabled"}>Apply crafted weights</button><button class="ghost" id="ww-broken">Reset to the broken network</button></div>
          <div class="row2"><button class="ghost" id="ww-export">Export weights (JSON)</button><label class="ghost ww-file">Load a JSON export<input type="file" id="ww-file" accept="application/json,.json" hidden></label></div>
          <button class="primary wide" id="ww-read">Read this network with Claude</button>
          <p class="sub" style="margin-top:8px">The broken network is the crafted one with the first output weight's sign flipped (${pctf(m.broken_agreement)} agreement; crafted ${pctf(m.crafted_agreement)}). Exports use Weightspace Lab's format, so files move both ways.</p></div>
      </div>
      <div class="panel"><h3>The weights</h3><p class="sub">h<sub>i</sub> = relu(W[i,0]·x + W[i,1]·y + b[i]); p = sigmoid(Σ v[i]·h<sub>i</sub> + bias). No optimiser: you are the training loop.</p>
        <div class="subtabs" id="ww-layers"><button data-layer="output" aria-selected="${P.layer === "output"}">Output layer (${H + 1})</button><button data-layer="hidden" aria-selected="${P.layer === "hidden"}">Hidden layer (${3 * H})</button></div>
        <div id="ww-sliders" class="ww-sliders"></div></div>`;
    sliders();
    paint();
    $("#ww-layers").querySelectorAll("button").forEach((b) => (b.onclick = () => { P.layer = b.dataset.layer; playground(); }));
    $("#ww-crafted").onclick = () => load("crafted");
    $("#ww-broken").onclick = () => load("broken");
    $("#ww-export").onclick = () => {
      const data = { format: "weightspace-lab-v1", pattern: P.pattern, architecture: [2, P.net.W.length, 1], activations: ["relu", "sigmoid"], weights: P.net };
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
      a.download = `weightspace-${P.pattern}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    };
    $("#ww-file").onchange = async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      try {
        const d = JSON.parse(await f.text()), n = d.weights || d;
        if (!n.W || !n.b || !n.v || n.bias === undefined) throw new Error("not a network");
        if (d.pattern && P.meta.patterns[d.pattern]) { P.pattern = d.pattern; $("#ww-pattern").value = d.pattern; }
        load("file", n); toast("Loaded the exported network.");
      } catch (err) { toast(`Could not read that file: ${err.message}`, true); }
    };
    $("#ww-read").onclick = async () => {
      let r;
      try { r = await api("/api/wsp/write/import", { method: "POST", body: { net: P.net, pattern: P.pattern, name: `${P.pattern} (written)` } }); }
      catch (e) { return toast(e.message, true); }
      window.WsCore.readImport(r.spec, "What does each hidden unit of this network compute, and does the network compute the pattern it was meant to? If not, which weights are wrong?");
      toast("Sent to Read. Press Read to ask Claude.");
    };
    const cv = $("#ww-plot");
    cv.onpointermove = (e) => {
      const r = cv.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * 2 - 1, y = 1 - ((e.clientY - r.top) / r.height) * 2;
      $("#ww-probe").textContent = `(${x.toFixed(2)}, ${y.toFixed(2)}) → p = ${predict(P.net, x, y).toFixed(3)}`;
    };
  }

  function sliders() {
    const n = P.net, rows = [];
    if (P.layer === "output") {
      n.v.forEach((v, i) => rows.push({ name: `v[${i}]`, get: () => n.v[i], set: (x) => (n.v[i] = x) }));
      rows.push({ name: "bias", get: () => n.bias, set: (x) => (n.bias = x) });
    } else {
      n.W.forEach((w, i) => { rows.push({ name: `W[${i},0]`, get: () => n.W[i][0], set: (x) => (n.W[i][0] = x) }); rows.push({ name: `W[${i},1]`, get: () => n.W[i][1], set: (x) => (n.W[i][1] = x) }); });
      n.b.forEach((v, i) => rows.push({ name: `b[${i}]`, get: () => n.b[i], set: (x) => (n.b[i] = x) }));
    }
    const host = $("#ww-sliders");
    host.innerHTML = "";
    rows.forEach((r, k) => {
      const v = r.get(), lim = Math.max(12, Math.ceil(Math.abs(v)));
      const row = document.createElement("div");
      row.className = "ww-row";
      row.innerHTML = `<label for="ww-s${k}"><code>${r.name}</code></label><input id="ww-s${k}" type="range" min="${-lim}" max="${lim}" step="0.1" value="${v}"><input class="ww-num" type="number" step="0.1" value="${+v.toFixed(3)}" aria-label="${r.name}">`;
      const [range, num] = row.querySelectorAll("input");
      range.oninput = () => { r.set(+range.value); num.value = (+range.value).toFixed(1); P.from = "custom"; paint(); };
      num.onchange = () => { const x = parseFloat(num.value); if (Number.isFinite(x)) { r.set(x); range.value = x; P.from = "custom"; paint(); } };
      host.appendChild(row);
    });
  }

  function paint() {
    draw($("#ww-plot"), P.net, P.pattern);
    const a = agreement(P.net, P.pattern), fb = flat(P.base), fn = flat(P.net);
    const ch = fn.length === fb.length ? fn.filter((v, i) => Math.abs(v - fb[i]) > 1e-7).length : fn.length;
    $("#ww-acc").textContent = pctf(a);
    $("#ww-changed").innerHTML = `${ch} <span class="muted" style="font-size:13px">of ${fn.length}</span>`;
    const what = { broken: "the broken network", crafted: "the crafted weights", custom: "your weights", file: "the loaded network", claude: "Claude's network" }[P.from] || "this network";
    $("#ww-finding").innerHTML = `With ${what}, the network agrees with the target on <b class="cab">${pctf(a)}</b> of the grid.`;
  }

  // ------------------------------------------------------------ Claude
  $("#ws-write").onsubmit = (e) => {
    e.preventDefault();
    if (!P.meta) return;
    const body = { pattern: $("#ww-pattern").value, mode: $("#ww-mode").value, hidden: parseInt($("#ww-hidden").value, 10), budget: parseInt($("#ww-budget").value, 10) };
    $("#ws-stage").innerHTML = `<p class="spinner">Claude is ${body.mode === "repair" ? "looking for the fault" : "writing the weights"}… then the same network is trained the ordinary way, for comparison.</p>`;
    runJob("/api/wsp/write", body, (r) => { window.WsCore.history(); showResult(r); });
  };

  async function showResult(r) {
    if (!P.meta) { try { P.meta = await api("/api/wsp/write/meta"); } catch { return; } }
    const m = P.meta.patterns[r.pattern], st = $("#ws-stage");
    if (document.querySelector('#ws-modes button[data-mode="write"]').getAttribute("aria-selected") !== "true") window.WsCore.mode("write");
    $("#ww-pattern").value = r.pattern; $("#ww-mode").value = r.task; $("#ww-hidden").value = r.hidden;
    const a = r.agreement, tr = r.trained.mean;
    let line;
    if (!r.net) line = `Claude did not return a valid network${r.invalid ? ` (${esc(r.invalid)})` : ""}.`;
    else if (r.task === "repair") line = `Claude's repair brings agreement from ${pctf(r.start_agreement)} to <b class="cab">${pctf(a)}</b>, changing ${r.weights_changed} weight${r.weights_changed === 1 ? "" : "s"}${r.fixed_the_fault ? ", including the one that was broken" : ", but not the one that was broken"}. The crafted network scores ${pctf(r.crafted_agreement)}.`;
    else line = `Claude's hand-written network agrees with the target on <b class="cab">${pctf(a)}</b> of the grid${r.budget ? ` after scoring ${r.forward_used} draft${r.forward_used === 1 ? "" : "s"}` : ", written in one go"}; the same network trained on 400 labelled points reaches ${pctf(tr)}, and random weights ${pctf(r.random.mean)}.`;
    st.innerHTML = `<div class="head"><h1>Write: ${esc(m.label)}</h1><span class="meta">${esc(r.spec)} · ${r.task === "repair" ? "repair" : "from scratch"} · ${r.budget ? `${r.budget} drafts allowed` : "one shot"} · ${Math.round(r.seconds)} s</span></div>
      <p class="finding">${line}</p>
      <div class="ww-grid">
        <div class="panel"><h3>Claude's network</h3><p class="sub">White dots: grid points where it disagrees with the target (${esc(m.rule)}).</p>
          ${r.net ? `<canvas id="ww-res" width="420" height="420" class="ww-canvas"></canvas><button class="ghost wide" id="ww-open" style="margin-top:10px">Open it in the playground</button>` : `<p class="muted">No network to draw.</p>`}</div>
        <div class="panel"><h3>Against the controls</h3><p class="sub">Agreement on the 41 × 41 grid.</p>
          <table><tbody>
            <tr><td>Claude, by reasoning</td><td class="num ok">${pctf(a)}</td></tr>
            ${r.task === "repair" ? `<tr><td>The broken network it started from</td><td class="num">${pctf(r.start_agreement)}</td></tr><tr><td>The crafted network</td><td class="num">${pctf(r.crafted_agreement)}</td></tr>` : ""}
            <tr><td>Trained: Adam, ${r.trained.points} labelled points, ${r.trained.steps} steps, ${r.trained.seeds} seeds</td><td class="num">${pctf(tr)} <span class="muted">±${(r.trained.spread * 100).toFixed(1)}</span></td></tr>
            <tr><td>Random weights (mean of ${r.random.draws})</td><td class="num">${pctf(r.random.mean)}</td></tr>
            <tr><td>Always the more common class</td><td class="num">${pctf(r.majority)}</td></tr>
          </tbody></table>
          ${r.drafts && r.drafts.length ? `<p class="sub" style="margin-top:10px">Drafts it scored: ${r.drafts.map((d) => pctf(d.agreement)).join(" → ")}.</p>` : ""}</div>
      </div>`;
    if (r.explanation) { const [p, b] = panel("Claude's account of the weights", ""); b.innerHTML = `<p class="ws-thought">${esc(r.explanation)}</p>`; st.appendChild(p); }
    if (r.net) {
      const [p, b] = panel("The weights it wrote", "");
      b.innerHTML = `<div class="scroll"><table><thead><tr><th>Unit</th><th class="num">W[i,0]</th><th class="num">W[i,1]</th><th class="num">b[i]</th><th class="num">v[i]</th></tr></thead><tbody>${
        r.net.W.map((w, i) => `<tr><td>${i}</td><td class="num">${w[0]}</td><td class="num">${w[1]}</td><td class="num">${r.net.b[i]}</td><td class="num">${r.net.v[i]}</td></tr>`).join("")}
        <tr><td>bias</td><td></td><td></td><td></td><td class="num">${r.net.bias}</td></tr></tbody></table></div>`;
      st.appendChild(p);
      draw($("#ww-res"), r.net, r.pattern);
      $("#ww-open").onclick = () => { P.pattern = r.pattern; $("#ww-pattern").value = r.pattern; window.WsCore.mode("write"); load("claude", r.net); };
    }
    const [tp, tb] = panel("How it got there", `${r.tool_calls} evaluations, ${r.usage ? r.usage.turns : "?"} turns${r.agent_model ? `, ${esc(r.agent_model)}` : ""}.`);
    tb.innerHTML = (r.transcript || []).map((t) => t.kind === "thought" ? `<p class="ws-thought">${esc(t.text)}</p>`
      : t.kind === "tool" ? `<details class="ws-tool"><summary><code>${esc(t.tool)}</code> ${t.result && t.result.agreement !== undefined ? pctf(t.result.agreement) : ""}${t.result && t.result.error ? ` <span class="badge" style="--c:var(--b)">error</span>` : ""}</summary><pre>${esc(JSON.stringify(t.result, null, 1)).slice(0, 3000)}</pre></details>`
      : `<p class="ws-thought"><b>Finished.</b></p>`).join("") || `<p class="muted">No steps recorded.</p>`;
    st.appendChild(tp);
  }

  window.WsWrite = { open, showResult, predict };
})();
