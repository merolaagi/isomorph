"use strict";
// Model picker: search Hugging Face, paste any link, keep a list of your own models.
(function () {
  const num = (n) => (n === null || n === undefined ? "–" : n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(n));
  const mb = (b) => (b ? (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(b / 1e6))} MB`) : "–");
  let targets = { a: "#tr-a", b: "#tr-b" };

  function describe(i) {
    if (i.error) return `<span class="warn">${esc(i.error)}</span>`;
    if (i.kind === "link") return esc(i.note);
    const bits = [];
    if (i.arch) bits.push(esc(i.arch));
    if (i.params) bits.push(`${num(i.params)} parameters`);
    if (i.bytes) bits.push(`${mb(i.bytes)} download`);
    bits.push(i.traceable ? "can be traced" : "weights only (no next-token head or tokenizer)");
    if (i.gated) bits.push("gated: needs your token");
    if (i.weights === false) bits.push(`<span class="warn">no weight files found</span>`);
    return bits.join(", ");
  }

  function use(side, spec) {
    const el = $(targets[side]);
    if (!el) return;
    el.value = spec;
    el.dispatchEvent(new Event("change"));
    toast(`Model ${side.toUpperCase()} set to ${spec}.`);
  }

  async function save(spec, meta) {
    try { await api("/api/models/saved", { method: "POST", body: { spec, meta } }); toast("Saved to My models."); loadSaved(); }
    catch (e) { toast(e.message, true); }
  }

  // ------------------------------------------------------------ saved list (rail)
  async function loadSaved() {
    const items = await api("/api/models/saved").catch(() => []);
    for (const [ul, tgt] of [["#tr-saved", { a: "#tr-a", b: "#tr-b" }], ["#hub-saved", { a: "#hub-a", b: "#hub-b" }]]) {
      const host = $(ul);
      if (!host) continue;
      host.innerHTML = items.length ? "" : `<li class="empty">Save models from Find models to keep them here.</li>`;
      for (const it of items) {
        const li = document.createElement("li");
        li.className = "saved";
        const m = it.meta || {};
        li.innerHTML = `<div class="sv"><span class="t1">${esc(it.spec)}</span><span class="t2">${m.params ? `${num(m.params)} params` : ""}${m.traceable === false ? `${m.params ? ", " : ""}weights only` : ""}</span></div>
          <button class="ghost small" data-s="a" title="Use as model A"><span class="headA">A</span></button>
          <button class="ghost small" data-s="b" title="Use as model B"><span class="headB">B</span></button>
          <button class="ghost small" data-x title="Remove from My models">×</button>`;
        li.querySelectorAll("[data-s]").forEach((b) => (b.onclick = () => { $(tgt[b.dataset.s]).value = it.spec; $(tgt[b.dataset.s]).dispatchEvent(new Event("change")); }));
        li.querySelector("[data-x]").onclick = async () => { await api(`/api/models/saved?spec=${encodeURIComponent(it.spec)}`, { method: "DELETE" }); loadSaved(); };
        host.appendChild(li);
      }
    }
  }

  // ------------------------------------------------------------ inline check under the A/B inputs
  const timers = {};
  function watch(sel, out) {
    const el = $(sel);
    if (!el) return;
    const check = async () => {
      const v = el.value.trim();
      const o = $(out);
      if (!v) { o.innerHTML = ""; return; }
      o.innerHTML = `<span class="muted">Checking…</span>`;
      try { const i = await api("/api/models/info", { method: "POST", body: { spec: v } }); if (el.value.trim() === v) o.innerHTML = describe(i); }
      catch (e) { o.innerHTML = `<span class="warn">${esc(e.message)}</span>`; }
    };
    el.addEventListener("change", () => { clearTimeout(timers[sel]); timers[sel] = setTimeout(check, 250); });
    el.addEventListener("blur", () => el.dispatchEvent(new Event("change")));
  }

  // ------------------------------------------------------------ modal
  function open(t) {
    targets = t;
    let dlg = $("#picker");
    if (!dlg) {
      dlg = document.createElement("dialog");
      dlg.id = "picker";
      dlg.innerHTML = `<div class="pk">
        <div class="pk-head"><h2>Find models</h2><button class="ghost small" id="pk-close">Close</button></div>
        <label>Paste a link or name
          <div class="pk-row"><input id="pk-link" placeholder="huggingface.co/…, modelscope.cn/models/…, https://…/model.safetensors, owner/name or a local folder" spellcheck="false">
          <button class="primary" id="pk-check">Check</button></div></label>
        <div id="pk-linkinfo" class="pk-info"></div>
        <h3>Search Hugging Face</h3>
        <div class="pk-row wrap">
          <input id="pk-q" placeholder="Name or keyword, e.g. pythia, smollm, qwen" spellcheck="false">
          <select id="pk-task"><option value="text-generation">Text generation (can be traced)</option><option value="">Any task (weights only)</option></select>
          <select id="pk-size"><option value="50M">Up to 50M</option><option value="200M" selected>Up to 200M</option><option value="500M">Up to 500M</option><option value="1B">Up to 1B</option><option value="3B">Up to 3B</option><option value="">Any size</option></select>
          <select id="pk-sort"><option value="downloads">Most downloaded</option><option value="likes">Most liked</option><option value="trending_score">Trending</option><option value="last_modified">Recently updated</option></select>
          <button class="primary" id="pk-go">Search</button>
        </div>
        <p class="sub">Small models load fastest on a Mac. Anything above a few hundred million parameters will be slow to trace and needs several GB of memory.</p>
        <div id="pk-results" class="scroll"></div>
        <details class="pk-token"><summary>Hugging Face token for gated or private models</summary>
          <p class="sub" id="pk-tokstate"></p>
          <div class="pk-row"><input id="pk-tok" type="password" placeholder="hf_…" autocomplete="off"><button class="ghost" id="pk-toksave">Save token</button><button class="ghost" id="pk-tokdel">Remove</button></div>
          <p class="sub">Stored only on this Mac, in the app's data folder.</p>
        </details></div>`;
      document.body.appendChild(dlg);
      $("#pk-close").onclick = () => dlg.close();
      dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
      $("#pk-check").onclick = checkLink;
      $("#pk-link").addEventListener("keydown", (e) => { if (e.key === "Enter") checkLink(); });
      $("#pk-go").onclick = runSearch;
      $("#pk-q").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });
      $("#pk-toksave").onclick = async () => { const v = $("#pk-tok").value.trim(); if (!v) return; await setToken(v); $("#pk-tok").value = ""; };
      $("#pk-tokdel").onclick = () => setToken(null);
    }
    dlg.showModal();
    tokenState();
    if (!$("#pk-results").innerHTML) runSearch();
  }

  async function tokenState() {
    const s = await api("/api/settings/token").catch(() => null);
    $("#pk-tokstate").textContent = !s ? "" : s.has_token ? `A token is saved${s.user ? ` for ${s.user}` : " (it could not be verified)"}.` : "No token saved. Public models work without one.";
  }
  async function setToken(t) {
    try { await api("/api/settings/token", { method: "POST", body: { token: t } }); toast(t ? "Token saved." : "Token removed."); tokenState(); }
    catch (e) { toast(e.message, true); }
  }

  function actions(spec, meta) {
    const d = document.createElement("span");
    d.className = "pk-act";
    d.innerHTML = `<button class="ghost small" data-s="a">Use as <span class="headA">A</span></button><button class="ghost small" data-s="b">Use as <span class="headB">B</span></button><button class="ghost small" data-save>Save</button>`;
    d.querySelectorAll("[data-s]").forEach((b) => (b.onclick = () => use(b.dataset.s, spec)));
    d.querySelector("[data-save]").onclick = () => save(spec, meta);
    return d;
  }

  async function checkLink() {
    const v = $("#pk-link").value.trim();
    const out = $("#pk-linkinfo");
    if (!v) return;
    out.innerHTML = `<span class="muted">Checking…</span>`;
    try {
      const i = await api("/api/models/info", { method: "POST", body: { spec: v } });
      const spec = i.spec || v;
      out.innerHTML = `<div><b>${esc(spec)}</b></div><div class="sub">${describe(i)}</div>`;
      if (!i.error) out.appendChild(actions(spec, { params: i.params, traceable: i.traceable, arch: i.arch }));
    } catch (e) { out.innerHTML = `<span class="warn">${esc(e.message)}</span>`; }
  }

  async function runSearch() {
    const res = $("#pk-results");
    res.innerHTML = `<p class="spinner">Searching…</p>`;
    const q = new URLSearchParams({ q: $("#pk-q").value.trim(), task: $("#pk-task").value, max_params: $("#pk-size").value, sort: $("#pk-sort").value, limit: "40" });
    let items;
    try { items = await api(`/api/models/search?${q}`); } catch (e) { res.innerHTML = `<p class="warn">${esc(e.message)}</p>`; return; }
    if (!items.length) { res.innerHTML = `<p class="muted">No models matched. Try a broader size or another keyword.</p>`; return; }
    const tbl = document.createElement("table");
    tbl.innerHTML = `<thead><tr><th>Model</th><th class="num">Parameters</th><th class="num">Download</th><th class="num">Downloads</th><th class="num">Likes</th><th>Updated</th><th></th></tr></thead><tbody></tbody>`;
    for (const m of items) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><a href="https://huggingface.co/${esc(m.id)}" target="_blank" rel="noopener">${esc(m.id)}</a><div class="sub" style="margin:0">${esc(m.arch || m.task || "")}${m.gated ? ", gated" : ""}${m.task === "text-generation" && !m.traceable ? ", may not trace" : ""}</div></td>
        <td class="num">${num(m.params)}</td><td class="num">${mb(m.bytes)}</td><td class="num">${num(m.downloads)}</td><td class="num">${num(m.likes)}</td><td class="muted">${m.updated || ""}</td><td></td>`;
      tr.lastElementChild.appendChild(actions(m.id, { params: m.params, traceable: m.traceable, arch: m.arch }));
      tbl.tBodies[0].appendChild(tr);
    }
    res.innerHTML = "";
    res.appendChild(tbl);
  }

  // ------------------------------------------------------------ wiring
  const fb = $("#tr-find"); if (fb) fb.onclick = () => open({ a: "#tr-a", b: "#tr-b" });
  const hb = $("#hub-find"); if (hb) hb.onclick = () => open({ a: "#hub-a", b: "#hub-b" });
  watch("#tr-a", "#tr-a-info");
  watch("#tr-b", "#tr-b-info");
  loadSaved();
  window.Picker = { open, loadSaved };
})();
