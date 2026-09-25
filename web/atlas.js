"use strict";
// Pattern library: profiles, shared patterns, and one model against the rest.
(function () {
  const A = { lib: null, sel: null, concept: "negation" };
  const short = (s) => String(s).replace(/^https?:\/\//, "").split("/").slice(-2).join("/");
  const num = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(0)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(n));
  const GROUP = { circuit: "Attention circuit", concept: "Concept", weights: "Weights" };

  async function load(selectId) {
    let lib;
    try { lib = await api("/api/atlas/library"); } catch (e) { toast(e.message, true); return; }
    A.lib = lib;
    const ul = $("#at-list");
    ul.innerHTML = lib.models.length ? "" : `<li class="empty">No models yet.</li>`;
    for (const m of lib.models) {
      const li = document.createElement("li");
      li.className = "saved";
      li.innerHTML = `<button class="grow ${A.sel === m.id ? "active" : ""}" data-id="${esc(m.id)}"><span class="t1">${esc(short(m.spec))}</span><span class="t2">${esc(m.arch || "")}, ${num(m.params)} params${m.has_baseline ? "" : ", no baseline"}</span></button><button class="ghost small" data-x title="Remove from the library">×</button>`;
      li.querySelector("[data-id]").onclick = () => openModel(m.id);
      li.querySelector("[data-x]").onclick = async () => {
        if (!confirm(`Remove ${m.spec} from the library?`)) return;
        await api(`/api/atlas/model/${m.id}`, { method: "DELETE" }).catch((e) => toast(e.message, true));
        if (A.sel === m.id) A.sel = null;
        load();
      };
      ul.appendChild(li);
    }
    if (selectId) openModel(selectId);
    else if (!A.sel) overview();
  }

  function profile(spec, next) {
    if (!spec) return toast("Enter a model to profile.", true);
    $("#at-stage").innerHTML = `<p class="spinner">Profiling ${esc(spec)}. Small models take a minute or two; the random baseline roughly doubles that.</p>`;
    runJob("/api/atlas/profile", { spec, baseline: $("#at-base").checked }, (meta) => { toast(`Added ${spec} to the library.`); next ? next(meta) : load(meta.id); });
  }
  $("#at-go").onclick = () => profile($("#at-spec").value.trim());
  $("#at-ab").onclick = () => {
    const a = $("#tr-a").value.trim(), b = $("#tr-b").value.trim();
    profile(a, () => profile(b, (m) => load(m.id)));
  };

  // ------------------------------------------------------------ overview
  function overview() {
    const L = A.lib, st = $("#at-stage");
    A.sel = null;
    document.querySelectorAll("#at-list [data-id]").forEach((b) => b.classList.remove("active"));
    if (!L || L.models.length === 0) return;
    const n = L.models.length;
    const prev = [...L.prevalence].sort((a, b) => b.have / b.of - a.have / a.of);
    const shared = prev.filter((p) => p.have === p.of && p.of > 0).map((p) => p.label.replace("Concept: ", ""));
    const none = prev.filter((p) => p.have === 0).map((p) => p.label.replace("Concept: ", ""));
    let f = `${n} model${n === 1 ? "" : "s"} in the library. `;
    if (n < 2) f += "Add at least one more to compare. ";
    if (shared.length) f += `Found in every model: <b class="cab">${esc(shared.join(", "))}</b>. `;
    if (none.length) f += `Not found in any: ${esc(none.join(", "))}.`;
    st.innerHTML = `<div class="head"><h1>Pattern library</h1><span class="meta">Probe set version ${L.anchor_version}</span></div><p class="finding">${f}</p>`;

    const [pp, pb] = panel("Patterns across the library", "Share of profiled models that show each pattern, judged against their own random baseline.");
    pb.className = "scroll atl";
    pb.innerHTML = `<table><thead><tr><th>Pattern</th><th>Kind</th><th>Models that have it</th><th class="num"></th><th>What it is</th></tr></thead><tbody>${
      prev.map((p) => `<tr><td>${esc(p.label.replace("Concept: ", ""))}</td><td class="muted">${GROUP[p.group]}</td><td style="min-width:120px"><div class="bar"><i style="width:${(p.have / p.of) * 100}%;background:var(--ab)"></i></div></td><td class="num">${p.have} of ${p.of}</td><td class="muted">${esc(p.about)}</td></tr>`).join("")
    }</tbody></table>`;
    st.appendChild(pp);

    if (n >= 2) {
      const row = document.createElement("div");
      row.className = "cols";
      const labels = L.models.map((m) => short(m.spec));
      const [sp, sb] = panel("How alike their representations are", "Best-match CKA between the layers of each pair, over the anchor texts. For reference, a model against its own random copy scores " + L.self_random.filter((x) => x !== null).map((x) => fmt(x)).join(", ") + ": that much similarity comes from the input text alone.");
      const [mp, mb] = panel("Map of the library", "Models placed so that distance reflects dissimilarity (classical multidimensional scaling of 1 − similarity). Click a point to open that model.");
      row.append(sp, mp);
      st.appendChild(row);
      heatmap(sb, { matrix: L.similarity, rows: labels, cols: labels, values: n <= 8, min: Math.min(...L.similarity.flat()), max: 1, rowLabelWidth: 150, colLabelHeight: 110 });
      scatter(mb, L.map, labels, L.models.map((m) => m.id));

      const [cp, cb] = panel("Agreement with the consensus, by depth",
        "At each relative depth, the consensus is the average representation of all other models in the library. Solid lines are trained models, the dashed line is the average of their random copies. The gap between them is structure that training created and that models share.");
      st.appendChild(cp);
      const cols = ["#2446c7", "#c23a6b", "#1f8a70", "#b7791f", "#6b3fb3", "#5f6b7a", "#0e7c9e", "#a3472f"];
      const series = L.consensus.map((c, i) => ({ name: labels[i], color: cols[i % cols.length], dots: true, points: c.agreement.map((v, b) => [(b + 0.5) / L.bins, v]).filter((p) => p[1] !== null) }));
      const rnd = [];
      for (let b = 0; b < L.bins; b++) {
        const v = L.consensus.map((c) => c.random[b]).filter((x) => x !== null && x !== undefined);
        if (v.length) rnd.push([(b + 0.5) / L.bins, v.reduce((a, c) => a + c, 0) / v.length]);
      }
      if (rnd.length) series.push({ name: "Random copies (average)", color: css("--muted"), dash: "5 4", points: rnd });
      lineChart(cb, { series, xMin: 0, xMax: 1, yMin: 0, yMax: 1, xLabel: "Relative depth (0 = first layer, 1 = last)" });
    }

    const concepts = [...new Set(Object.values(L.patterns).flat().filter((p) => p.group === "concept").map((p) => p.id))];
    if (concepts.length) {
      const [kp, kb] = panel("Concepts: how much more readable than in a random network", "Best-layer probe score minus the higher of the random copy and a shuffled-label control. Above about 0.15 means the model has learned the concept rather than just seeing the words.");
      st.appendChild(kp);
      heatmap(kb, { matrix: L.models.map((m) => concepts.map((c) => { const p = L.patterns[m.id].find((x) => x.id === c); return p ? p.value : 0; })),
        rows: L.models.map((m) => short(m.spec)), cols: concepts.map((c) => c.replace("concept_", "")), values: true, min: -0.1, max: 0.6, rowLabelWidth: 150, colLabelHeight: 70 });
    }
  }

  function scatter(host, pts, labels, ids) {
    const W = 520, H = 320, m = 40;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const X = (v) => m + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * m), Y = (v) => H - m - ((v - y0) / (y1 - y0 || 1)) * (H - 2 * m);
    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Map of models">${pts.map((p, i) =>
      `<g class="pt" data-id="${esc(ids[i])}" style="cursor:pointer"><circle cx="${X(p[0])}" cy="${Y(p[1])}" r="7" fill="var(--ab)" fill-opacity=".8"/><text x="${X(p[0]) + 10}" y="${Y(p[1]) + 4}">${esc(labels[i])}</text></g>`).join("")}</svg>`;
    host.querySelectorAll(".pt").forEach((g) => (g.onclick = () => openModel(g.dataset.id)));
  }

  // ------------------------------------------------------------ one model
  async function openModel(pid) {
    A.sel = pid;
    document.querySelectorAll("#at-list [data-id]").forEach((b) => b.classList.toggle("active", b.dataset.id === pid));
    const st = $("#at-stage");
    st.innerHTML = `<p class="spinner">Comparing with the library…</p>`;
    let r;
    try { r = await api(`/api/atlas/model/${pid}`); } catch (e) { st.innerHTML = ""; toast(e.message, true); return; }
    const m = r.model, P = r.patterns;
    const have = P.filter((p) => p.present), missing = P.filter((p) => !p.present && p.library_of && p.library_have / p.library_of >= 0.5), unusual = P.filter((p) => p.present && p.library_of && p.library_have / p.library_of < 0.5);
    let f = `<b>${esc(short(m.spec))}</b> shows ${have.length} of ${P.length} patterns. `;
    if (missing.length) f += `It lacks ${missing.length === 1 ? "one pattern" : `${missing.length} patterns`} that most of the library has: <b class="cb">${esc(missing.map((p) => p.label.replace("Concept: ", "")).join(", "))}</b>. `;
    if (unusual.length) f += `It has ${esc(unusual.map((p) => p.label.replace("Concept: ", "")).join(", "))}, which few others show. `;
    if (r.nearest.length) f += `Its closest relative is ${esc(short(r.nearest[0].spec))} (similarity ${fmt(r.nearest[0].similarity)}${r.self_random !== null ? `, against ${fmt(r.self_random)} for its own random copy` : ""}).`;
    st.innerHTML = `<div class="head"><h1>${esc(m.spec)}</h1><span class="meta">${esc(m.arch || "")}, ${m.layers} layers, ${num(m.params)} parameters</span><button class="link" id="at-back">Back to the library</button></div><p class="finding">${f}</p>`;
    $("#at-back").onclick = overview;

    const [pp, pb] = panel("Patterns", "Each pattern with its value in this model, the same measure in a random copy, and how common it is in the rest of the library.");
    pb.className = "scroll atl";
    pb.innerHTML = `<table><thead><tr><th>Pattern</th><th>Here</th><th class="num">Value</th><th class="num">Random copy</th><th>In the library</th><th>Detail</th></tr></thead><tbody>${
      P.map((p) => `<tr><td>${esc(p.label.replace("Concept: ", "Concept, "))}<div class="sub" style="margin:0">${esc(p.verdict || "")}</div></td>
        <td>${p.present ? `<b class="headA" style="color:var(--ab)">present</b>` : `<span class="muted">absent</span>`}</td>
        <td class="num">${fmt(p.value, 3)}</td><td class="num">${p.baseline === null || p.baseline === undefined ? "–" : fmt(p.baseline, 3)}</td>
        <td class="num">${p.library_of ? `${p.library_have} of ${p.library_of}` : "–"}</td><td class="muted">${esc(p.detail || "")}</td></tr>`).join("")
    }</tbody></table>`;
    st.appendChild(pp);

    if (r.nearest.length) {
      const row = document.createElement("div");
      row.className = "cols";
      const nb = r.nearest[0];
      const [p1, b1] = panel("Nearest models", "Best-match representation similarity over the anchor texts.");
      b1.innerHTML = `<table><tbody>${r.nearest.map((x) => `<tr><td><button class="link" data-id="${esc(x.id)}">${esc(short(x.spec))}</button></td><td style="min-width:120px"><div class="bar"><i style="width:${x.similarity * 100}%;background:var(--ab)"></i></div></td><td class="num">${fmt(x.similarity)}</td></tr>`).join("")}</tbody></table>`;
      b1.querySelectorAll("[data-id]").forEach((b) => (b.onclick = () => openModel(b.dataset.id)));
      const [p2, b2] = panel(`Layer by layer against ${esc(short(nb.spec))}`, "Linear CKA between each layer of this model (rows) and each layer of its nearest neighbour (columns). A bright diagonal means both build their representations in the same order.");
      row.append(p1, p2);
      st.appendChild(row);
      heatmap(b2, { matrix: nb.layers, rows: nb.layers.map((_, i) => (i ? `L${i}` : "Emb")), cols: nb.layers[0].map((_, j) => (j ? `L${j}` : "Emb")), values: nb.layers.length <= 13, rowLabelWidth: 44, colLabelHeight: 50 });
    }

    if (r.concepts) {
      const names = Object.keys(r.concepts);
      if (!names.includes(A.concept)) A.concept = names[0];
      const [cp, cb] = panel("Where concepts become readable", "Probe score at each layer, against the same probe on the random copy and on shuffled labels. The gap is what the model learned.");
      st.appendChild(cp);
      const pick = document.createElement("div");
      pick.className = "tokens";
      for (const n of names) {
        const b = document.createElement("button");
        b.textContent = n;
        b.setAttribute("aria-pressed", n === A.concept);
        b.onclick = () => { A.concept = n; openModel(pid); };
        pick.appendChild(b);
      }
      cb.appendChild(pick);
      const host = document.createElement("div");
      cb.appendChild(host);
      const c = r.concepts[A.concept], cr = r.concepts_random && r.concepts_random[A.concept];
      const metric = c.kind === "class" ? "Accuracy" : "R²";
      host.insertAdjacentHTML("beforeend", `<p class="sub">${esc(c.about)} ${c.n} examples, 5-fold cross-validated linear probe on the last token.</p>`);
      const chart = document.createElement("div");
      host.appendChild(chart);
      const ser = [{ name: "Trained model", color: css("--ab"), dots: true, points: c.score.map((v, i) => [i, v]) },
        { name: "Shuffled labels", color: css("--line"), dash: "3 3", points: c.control.map((v, i) => [i, v]) }];
      if (cr) ser.splice(1, 0, { name: "Random copy", color: css("--muted"), dash: "5 4", points: cr.score.map((v, i) => [i, v]) });
      lineChart(chart, { series: ser, xLabel: "Layer (0 = embeddings)", yLabel: metric, yMin: c.kind === "class" ? 0.3 : -0.2, yMax: 1 });
    }

    if (r.circuits) {
      const two = document.createElement("div");
      two.className = "two";
      two.style.marginTop = "18px";
      for (const [key, label] of [["induction", "Induction score"], ["previous", "Previous-token score"]]) {
        const [p, b] = panel(`${label} per head`, r.circuits_random ? "Top: the trained model. Bottom: its random copy, for reference." : "");
        const s = r.circuits.scores[key];
        heatmap(b, { matrix: s, rows: s.map((_, i) => `L${i}`), cols: s[0].map((_, j) => `H${j}`), min: 0, max: 1, values: s[0].length <= 12 && s.length <= 12, rowLabelWidth: 40, colLabelHeight: 36 });
        if (r.circuits_random) {
          const sub = document.createElement("div");
          sub.innerHTML = `<p class="sub" style="margin-top:10px">Random copy</p>`;
          const h2 = document.createElement("div");
          sub.appendChild(h2);
          b.appendChild(sub);
          const sr = r.circuits_random.scores[key];
          heatmap(h2, { matrix: sr, rows: sr.map((_, i) => `L${i}`), cols: sr[0].map((_, j) => `H${j}`), min: 0, max: 1, color: css("--muted"), rowLabelWidth: 40, colLabelHeight: 36 });
        }
        two.appendChild(p);
      }
      st.appendChild(two);
    }
  }

  document.querySelector('nav button[data-view="atlas"]').addEventListener("click", () => { if (!A.lib) load(); });
  window.Atlas = { load };
})();
