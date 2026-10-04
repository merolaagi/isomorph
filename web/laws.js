"use strict";
// Laws: physics- and maths-informed tests on any model's weights.
(function () {
  const W = { S: null, cand: [], sel: null, law: "spikes", mat: null };
  const num = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(0)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n));
  const short = (s) => {
    const m = String(s).match(/^lab:(\d{8})-(\d{6})-(p\d+):(\d+)$/);
    if (m) return `lab ${m[3]} ${m[2].slice(0, 4)} seed ${m[4]}`;
    return String(s).replace(/^https?:\/\//, "").split("/").slice(-2).join("/");
  };
  const lawId = (spec) => spec.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(-70);
  const fv = (v, d = 3) => (v === null || v === undefined ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : fmt(v, d));

  async function loadAll(open) {
    try { [W.S, W.cand] = await Promise.all([api("/api/laws"), api("/api/laws/candidates")]); } catch (e) { toast(e.message, true); return; }
    rail();
    if (open) openModel(open);
    else if (W.sel) openModel(W.sel);
    else overview();
  }

  function rail() {
    const host = $("#lw-cands");
    const groups = {};
    for (const c of W.cand) (groups[c.group] ||= []).push(c);
    host.innerHTML = Object.entries(groups).map(([g, cs]) => `<h3 class="lw-g">${esc(g)}</h3>${cs.map((c) =>
      `<label class="check"><input type="checkbox" data-spec="${esc(c.spec)}" ${!c.tested && (g === "Micro Opus" || g === "Ground-truth lab") ? "checked" : ""}> <span class="lw-l">${esc(c.label)}</span>${c.tested ? ` <span class="badge" style="--c:var(--ab)">tested</span>` : ""}</label>`).join("")}`).join("")
      || `<p class="hint">Nothing to test yet. Train a lab family, use Micro Opus, or add a model below.</p>`;
  }

  $("#lw-go").onclick = () => {
    const specs = [...document.querySelectorAll("#lw-cands [data-spec]:checked")].map((c) => c.dataset.spec)
      .concat($("#lw-extra").value.split("\n").map((x) => x.trim()).filter(Boolean));
    if (!specs.length) return toast("Tick at least one model, or add one below.", true);
    $("#lw-stage").innerHTML = `<p class="spinner">Computing spectra, digit statistics and symmetry checks for ${specs.length} model${specs.length > 1 ? "s" : ""}…</p>`;
    runJob("/api/laws/run", { specs }, (r) => {
      if (r.failed.length) toast(`${r.failed.length} failed: ${r.failed.map((f) => `${f.spec} (${f.error})`).join("; ")}`, true);
      else toast(`Tested ${r.done.length} model${r.done.length > 1 ? "s" : ""}.`);
      W.sel = null;
      loadAll();
    });
  };
  $("#lw-all").onclick = () => document.querySelectorAll("#lw-cands [data-spec]").forEach((c) => (c.checked = !c.closest("label").querySelector(".badge")));

  // ------------------------------------------------------------ overview
  function overview() {
    const S = W.S, st = $("#lw-stage");
    W.sel = null;
    if (!S || !S.models.length) {
      st.innerHTML = `<div class="intro"><p class="lede">Do laws from physics and mathematics show up in trained weights, whatever the model?</p>
        <p>Each law is tested on every weight matrix of every model you pick, and on two nulls: random matrices with the same shape and scale, and the same weights with their positions shuffled. An <b>emergent</b> law is something training creates, so it only counts if the trained weights clearly differ from the nulls. A <b>preserved</b> law is something randomness already has, or a quantity a symmetry conserves, and it counts if training leaves it intact.</p>
        <p>Tick models on the left (Micro Opus and the lab models need no download) and press Run the laws. Add a randomly initialised model as a control if you have one: emergent laws should fail on it.</p></div>`;
      return;
    }
    const L = S.laws, M = S.models, U = S.universality;
    const ids = Object.keys(L);
    const every = ids.filter((k) => U[k].of >= 2 && U[k].have === U[k].of);
    const none = ids.filter((k) => U[k].of >= 2 && U[k].have === 0);
    let f = `${M.length} model${M.length > 1 ? "s" : ""} tested. `;
    const ev = every.filter((k) => L[k].kind === "emergent"), pv = every.filter((k) => L[k].kind === "preserved");
    if (ev.length) f += `Training produced the same law in every one of them: <b class="cab">${ev.map((k) => esc(L[k].label.toLowerCase())).join("; ")}</b>. `;
    const common = ids.filter((k) => L[k].kind === "emergent" && !every.includes(k) && U[k].of >= 2 && U[k].have / U[k].of >= 0.6).sort((a, b) => U[b].have / U[b].of - U[a].have / U[a].of);
    if (common.length) {
      const top = common[0];
      const miss = M.filter((m) => m.verdicts[top] && !m.verdicts[top].holds).map((m) => short(m.spec));
      f += `Most widespread emergent law${common.length > 1 ? "s" : ""}: ${common.slice(0, 3).map((k) => `<b class="cab">${esc(L[k].label.toLowerCase())}</b> (${U[k].have} of ${U[k].of})`).join(", ")}. The models where ${esc(L[top].label.toLowerCase())} fails: ${esc(miss.join(", "))}; if those are untrained or barely trained, that is what a law created by training should look like. `;
    }
    if (pv.length) f += `Preserved everywhere: ${pv.map((k) => esc(L[k].label.toLowerCase())).join("; ")}. `;
    if (none.length) f += `Holding nowhere: ${none.map((k) => esc(L[k].label.toLowerCase())).join("; ")}.`;
    st.innerHTML = `<div class="head"><h1>Laws</h1><span class="meta">Physics- and maths-informed tests on raw weights</span></div><p class="finding">${f}</p>`;

    const [tp, tb] = panel("Every law, every model", "A tick means the law holds for that model. Numbers are medians over the model's weight matrices; the grey number below is the Gaussian null. Click a model's name for its spectra and details.");
    tb.className = "scroll";
    tb.innerHTML = `<table class="lawtab"><thead><tr><th>Law</th><th>Kind</th><th class="num">Holds in</th>${M.map((m) => `<th class="num"><button class="link" data-open="${esc(m.id)}" title="${esc(m.spec)}">${esc(short(m.spec))}</button><div class="sub" style="margin:0">${num(m.params)}</div></th>`).join("")}</tr></thead><tbody>${
      ids.map((k) => `<tr><td><button class="link" data-law="${k}">${esc(L[k].label)}</button><div class="sub" style="margin:0">${esc(L[k].field)}</div></td>
        <td><span class="badge" style="--c:${L[k].kind === "emergent" ? "var(--ab)" : "var(--a)"}">${L[k].kind}</span></td>
        <td class="num">${U[k].of ? `${U[k].have} of ${U[k].of}` : "–"}</td>
        ${M.map((m) => { const v = m.verdicts[k]; return v ? `<td class="num ${v.holds ? "ok" : "no"}">${v.holds ? "✓" : "✗"} ${fv(v.value)}<div class="sub" style="margin:0">${v.null === undefined || v.null === null ? "" : fv(v.null)}</div></td>` : `<td class="num muted">–</td>`; }).join("")}</tr>`).join("")
    }</tbody></table>`;
    st.appendChild(tp);
    tb.querySelectorAll("[data-open]").forEach((b) => (b.onclick = () => openModel(b.dataset.open)));
    tb.querySelectorAll("[data-law]").forEach((b) => (b.onclick = () => { W.law = b.dataset.law; overview(); }));

    const k = W.law in L ? W.law : "spikes";
    const law = L[k];
    const [cp, cb] = panel(`${law.label}`, `${esc(law.field)}. ${esc(law.formula)}. ${esc(law.expect)}`);
    cp.style.marginTop = "18px";
    st.appendChild(cp);
    const pts = M.filter((m) => m.verdicts[k] && m.verdicts[k].value !== null);
    if (pts.length) {
      groupedBars(cb.appendChild(document.createElement("div")), {
        labels: pts.map((m) => short(m.spec)), rotate: pts.length > 6,
        series: [{ name: "Trained weights", color: css("--ab"), values: pts.map((m) => m.verdicts[k].value) },
          { name: "Gaussian null", color: css("--line"), values: pts.map((m) => m.verdicts[k].null ?? null) }],
      });
      cb.insertAdjacentHTML("beforeend", `<p class="sub">Measure: ${esc(pts[0].verdicts[k].unit || "")}.</p>`);
    }
    const pick = document.createElement("div");
    pick.className = "tokens";
    pick.style.marginTop = "10px";
    pick.innerHTML = ids.map((x) => `<button data-l="${x}" aria-pressed="${x === k}">${esc(L[x].label)}</button>`).join("");
    pick.querySelectorAll("[data-l]").forEach((b) => (b.onclick = () => { W.law = b.dataset.l; overview(); }));
    cb.appendChild(pick);

    const [gp, gb] = panel("What the laws mean", "");
    gp.style.marginTop = "18px";
    gb.innerHTML = ids.map((x) => `<details><summary><b>${esc(L[x].label)}</b> <span class="muted">(${esc(L[x].field)}, ${L[x].kind})</span></summary><div class="math">${esc(L[x].formula)}</div><p class="sub">${esc(L[x].expect)}</p></details>`).join("");
    st.appendChild(gp);
  }

  // ------------------------------------------------------------ one model
  async function openModel(id) {
    W.sel = id;
    const st = $("#lw-stage");
    st.innerHTML = `<p class="spinner">Loading…</p>`;
    let b;
    try { b = await api(`/api/laws/model/${id}`); } catch (e) { toast(e.message, true); return overview(); }
    const L = W.S.laws, V = b.verdicts;
    const held = Object.keys(V).filter((k) => V[k].holds);
    st.innerHTML = `<div class="head"><h1>${esc(b.spec)}</h1><span class="meta">${num(b.params)} parameters, ${b.matrices.length} weight matrices tested</span><button class="link" id="lw-back">Back to all laws</button> <button class="link" id="lw-del">Remove</button></div>
      <p class="finding">${held.length} of ${Object.keys(V).length} laws hold: ${held.map((k) => esc(L[k].label.toLowerCase())).join("; ") || "none"}.</p>`;
    $("#lw-back").onclick = overview;
    $("#lw-del").onclick = async () => { await api(`/api/laws/model/${id}`, { method: "DELETE" }); W.sel = null; loadAll(); };

    const [vp, vb] = panel("Verdicts", "Trained value against the Gaussian null and the shuffled null (same numbers, positions scrambled). A gap from the shuffled null means the arrangement of the weights matters, not only their values.");
    vb.className = "scroll";
    vb.innerHTML = `<table><thead><tr><th>Law</th><th></th><th class="num">Trained</th><th class="num">Gaussian null</th><th class="num">Shuffled</th><th>Measure</th></tr></thead><tbody>${
      Object.entries(V).map(([k, v]) => `<tr><td>${esc(L[k].label)}</td><td class="${v.holds ? "ok" : "no"}">${v.holds ? "✓ holds" : "✗"}</td><td class="num"><b>${fv(v.value)}</b></td><td class="num">${fv(v.null)}</td><td class="num">${fv(v.shuffled)}</td><td class="muted">${esc(v.unit || "")}${v.r2 !== undefined ? `, R² ${fv(v.r2, 2)}` : ""}${v.direction ? `, ${v.direction}` : ""}${v.note ? `. ${esc(v.note)}` : ""}</td></tr>`).join("")
    }</tbody></table>`;
    st.appendChild(vp);

    const withEsd = b.matrices.filter((m) => m.real.esd);
    if (withEsd.length) {
      if (!W.mat || !withEsd.find((m) => m.name === W.mat)) W.mat = withEsd[0].name;
      const m = withEsd.find((x) => x.name === W.mat);
      const e = m.real.esd;
      const [ep, eb] = panel("The spectrum, against pure noise",
        `Eigenvalues of WᵀW/N for ${esc(m.name)} (${m.shape.join(" × ")}${m.sampled.join() !== m.shape.join() ? `, sampled to ${m.sampled.join(" × ")}` : ""}). The curve is what a random matrix of the same shape and scale would give (Marchenko–Pastur); the dashed line is its edge. ${m.real.spikes} eigenvalue${m.real.spikes === 1 ? "" : "s"} lie beyond it${e.max > e.edges[e.edges.length - 1] ? `; the largest, ${fv(e.max)}, is off the chart` : ""}.`);
      ep.style.marginTop = "18px";
      const pick = document.createElement("div");
      pick.className = "tokens";
      pick.innerHTML = withEsd.map((x) => `<button data-m="${esc(x.name)}" aria-pressed="${x.name === W.mat}">${esc(x.name.split(".").slice(-3).join("."))}</button>`).join("");
      pick.querySelectorAll("[data-m]").forEach((x) => (x.onclick = () => { W.mat = x.dataset.m; openModel(id); }));
      eb.appendChild(pick);
      const ch = document.createElement("div");
      eb.appendChild(ch);
      st.appendChild(ep);
      const pts = [];
      e.counts.forEach((c, i) => { pts.push([e.edges[i], c], [e.edges[i + 1], c]); });
      const mid = e.counts.map((_, i) => (e.edges[i] + e.edges[i + 1]) / 2);
      const top = Math.max(...e.counts, ...e.mp);
      lineChart(ch, { series: [
        { name: "Trained eigenvalues", color: css("--ab"), width: 2, points: pts },
        { name: "Random-matrix prediction", color: css("--muted"), width: 1.6, points: mid.map((x, i) => [x, e.mp[i]]) },
        { name: "Edge of noise", color: css("--b"), dash: "5 4", points: [[e.edge, 0], [e.edge, top]] },
      ], xLabel: "Eigenvalue", yLabel: "Count", yMin: 0, xMin: 0 });

      if (m.real.benford_freq) {
        const [bp, bb] = panel("First digits against Benford's law", `Share of weights in ${esc(m.name)} whose first significant digit is 1 to 9. Mean absolute deviation ${fv(m.real.benford_mad)}; conformity needs under 0.012.`);
        bp.style.marginTop = "18px";
        st.appendChild(bp);
        groupedBars(bb, { labels: ["1", "2", "3", "4", "5", "6", "7", "8", "9"], series: [
          { name: "Weights", color: css("--ab"), values: m.real.benford_freq },
          { name: "Benford", color: css("--line"), values: [1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => Math.log10(1 + 1 / d)) }], yFmt: (v) => pct(v) });
      }
    }

    const two = document.createElement("div");
    two.className = "two";
    two.style.marginTop = "18px";
    const bal = b.model.balance || [];
    const [bp2, bb2] = panel("Neuron balance by layer", "Correlation between each MLP neuron's incoming and outgoing squared norm, and the same with neurons paired at random.");
    bb2.innerHTML = bal.length ? `<table><thead><tr><th>Layer</th><th class="num">Correlation</th><th class="num">Random pairing</th></tr></thead><tbody>${bal.map((x) => `<tr><td>${x.layer}</td><td class="num"><b>${fv(x.corr)}</b></td><td class="num">${fv(x.null)}</td></tr>`).join("")}</tbody></table>` : `<p class="muted">No MLP pair recognised in this architecture.</p>`;
    const qk = b.model.qk || [];
    const [qp, qb] = panel("Attention: symmetric or directed?", "Share of each head's W_Q W_Kᵀ that is symmetric. Random gives 0.5.");
    qb.innerHTML = qk.length ? `<table><thead><tr><th>Tensor</th><th>Per head</th><th class="num">Mean</th></tr></thead><tbody>${qk.map((x) => `<tr><td>${esc(x.tensor)}</td><td class="muted">${x.per_head.join(", ")}</td><td class="num"><b>${fv(x.sym)}</b></td></tr>`).join("")}</tbody></table>` : `<p class="muted">Query and key weights not recognised.</p>`;
    two.append(bp2, qp);
    st.appendChild(two);

    const [mp, mb] = panel("Every matrix", "Per-matrix measures on the trained weights, with the Gaussian null in grey.");
    mp.style.marginTop = "18px";
    mb.className = "scroll";
    const cell = (m, k, d = 2) => `<td class="num">${fv(m.real[k], d)}<div class="sub" style="margin:0">${fv(m.gaussian[k], d)}</div></td>`;
    mb.innerHTML = `<table><thead><tr><th>Matrix</th><th>Shape</th><th class="num">α</th><th class="num">Spikes</th><th class="num">⟨r⟩</th><th class="num">γ</th><th class="num">Benford MAD</th><th class="num">Kurtosis</th></tr></thead><tbody>${
      b.matrices.map((m) => `<tr><td>${esc(m.name)}</td><td class="muted">${m.shape.join(" × ")}</td>${cell(m, "alpha")}<td class="num">${m.real.spikes}<div class="sub" style="margin:0">${m.gaussian.spikes}</div></td>${cell(m, "r", 3)}${cell(m, "gamma")}${cell(m, "benford_mad", 3)}${cell(m, "kurtosis")}</tr>`).join("")
    }</tbody></table>`;
    st.appendChild(mp);
  }

  document.querySelector('nav button[data-view="laws"]').addEventListener("click", () => loadAll());
  window.Laws = { loadAll, lawId };
})();
