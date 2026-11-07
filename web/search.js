"use strict";
// Search, in the header: tabs and modes, sections of the page you are on, the guide, models,
// every saved result (traces, laws, Weight-space runs, lab runs, library profiles), adapters,
// the superagent's missions and its ready-made plans. Press / or s to search; arrows and Enter.
(function () {
  const header = document.querySelector("header.top");
  const env = document.getElementById("env");
  if (!header || !env) return;
  const box = document.createElement("div");
  box.className = "md-search";
  box.setAttribute("role", "search");
  box.innerHTML = `<svg class="md-search-ico" viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>
    <input id="md-q" type="search" placeholder="Search" autocomplete="off" spellcheck="false" aria-label="Search Isomorph" aria-controls="md-out" aria-expanded="false">
    <kbd>${/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘K" : "Ctrl K"}</kbd><div class="md-search-out" id="md-out" role="listbox" hidden></div>`;
  header.insertBefore(box, env);
  const scrim = document.createElement("div");
  scrim.className = "md-scrim";
  document.body.appendChild(scrim);
  scrim.onclick = () => close();
  const q = box.querySelector("#md-q"), out = box.querySelector("#md-out");
  const X = { res: [], sel: 0, seq: 0, timer: 0 };

  const ICON = { Pages: "▤", "On this page": "§", Guide: "?", Models: "◎", Results: "◆", Adapters: "≋", Missions: "✦", Plans: "▸" };
  const VIEW_ABOUT = {
    trace: "Two models, one prompt, layer by layer: predictions, attention, neurons, what training would move",
    atlas: "Profile models into a library: fingerprints, circuits, probes, weight signatures against a random copy",
    laws: "Physics and maths laws on raw weights, against Gaussian and shuffled nulls",
    wsp: "Plan a fine-tune from the weights; Claude reads, fixes and writes weights",
    tune: "Tinker-style LoRA fine-tuning on this Mac; where each fine-tune went",
    engine: "Mine rules, test levers, blueprints, Autopilot, the world view",
    lab: "Ground-truth models of (a + b) mod p: seeds compared after symmetries",
    hub: "Two checkpoints compared by weights or activations",
    about: "How a model is stored and what a prompt does to it",
  };

  // ------------------------------------------------------------ what lives in the page itself
  function localItems() {
    const items = [];
    document.querySelectorAll("header.top nav button[data-view]").forEach((b) => {
      items.push({ group: "Pages", title: b.textContent.trim(), sub: VIEW_ABOUT[b.dataset.view] || "", action: { type: "go", view: b.dataset.view } });
    });
    for (const [sel, view] of [["#ws-modes button", "wsp"], ["#tu-modes button", "tune"]]) {
      document.querySelectorAll(sel).forEach((b) => items.push({ group: "Pages", title: `${b.textContent.trim()}`, sub: view === "wsp" ? "Weight space" : "Tune",
        action: { type: "go", view, mode: b.dataset.mode, modeSel: sel.split(" ")[0] } }));
    }
    document.querySelectorAll("main > .view").forEach((v) => {
      const view = v.id.replace("view-", "");
      const tab = document.querySelector(`header.top nav button[data-view="${view}"]`);
      v.querySelectorAll(".rail h2").forEach((h) => items.push({ group: "Pages", title: h.textContent.trim(), sub: tab ? tab.textContent.trim() : view, action: { type: "rail", view, el: h } }));
    });
    const cur = [...document.querySelectorAll("main > .view")].find((v) => !v.hidden);
    if (cur) cur.querySelectorAll(".md-toc a").forEach((a) => items.push({ group: "On this page", title: a.textContent, sub: "", action: { type: "jump", id: a.dataset.id } }));
    return items;
  }
  const words = (s) => (s || "").toLowerCase().match(/[a-z0-9]+/g) || [];
  function matchLocal(items, text) {
    const toks = words(text);
    if (!toks.length) return [];
    return items.map((it) => {
      const t = words(it.title), r = words(it.sub);
      let sc = 0;
      for (const [i, k] of toks.entries()) {
        const last = i === toks.length - 1;
        const hit = (ws) => ws.some((w) => w === k || ((last || k.length > 3) && w.startsWith(k)));
        if (hit(t)) sc += 3; else if (hit(r)) sc += 1; else return null;
      }
      if (it.title.toLowerCase().startsWith(text.toLowerCase().trim())) sc += 3;
      return [sc, it];
    }).filter(Boolean).sort((a, b) => b[0] - a[0]).slice(0, 6).map((x) => x[1]);
  }

  // ------------------------------------------------------------ query and draw
  async function query() {
    const text = q.value.trim();
    const my = ++X.seq;
    if (!text) { X.res = []; draw(""); return; }
    const local = matchLocal(localItems(), text);
    X.res = local; X.sel = 0; draw(text, true);
    let remote = [];
    try { remote = await api(`/api/search?q=${encodeURIComponent(text)}&limit=30`); } catch { /* offline: local only */ }
    if (my !== X.seq) return;
    X.res = [...local, ...remote];
    X.sel = Math.min(X.sel, Math.max(0, X.res.length - 1));
    draw(text);
  }
  function hl(s, text) {
    const toks = words(text).filter((t) => t.length > 1);
    let h = esc(s || "");
    if (!toks.length) return h;
    const re = new RegExp(`\\b(${toks.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
    return h.replace(re, "<mark>$1</mark>");
  }
  function draw(text, pending) {
    if (!text) {
      out.innerHTML = `<div class="md-search-meta">Type to start searching</div>`;
      return;
    }
    if (!X.res.length) { out.innerHTML = `<div class="md-search-meta">${pending ? "Searching…" : "No matching results"}</div>`; return; }
    let html = `<div class="md-search-meta">${X.res.length}${pending ? "+" : ""} matching result${X.res.length === 1 ? "" : "s"}</div>`, g = null;
    X.res.forEach((r, i) => {
      if (r.group !== g) { g = r.group; html += `<div class="md-search-g">${esc(g)}</div>`; }
      const verbs = r.action && r.action.type === "model" ? `<span class="verbs"><button data-v="trace">Trace</button><button data-v="laws">Laws</button><button data-v="tune">Tune</button></span>`
        : r.model ? `<span class="verbs"><button data-v="trace">Trace</button><button data-v="laws">Laws</button></span>` : "";
      html += `<div class="md-search-r${i === X.sel ? " on" : ""}" role="option" data-i="${i}" aria-selected="${i === X.sel}"><span class="ico" aria-hidden="true">${ICON[r.group] || "·"}</span>
        <div><b>${hl(r.title, text)}</b>${r.sub ? `<small>${hl(r.sub, text)}</small>` : ""}</div>${verbs}</div>`;
    });
    out.innerHTML = html;
    out.querySelectorAll(".md-search-r").forEach((el) => {
      el.onmousedown = (e) => e.preventDefault();
      el.onclick = (e) => { const v = e.target.closest("[data-v]"); go(X.res[+el.dataset.i], v ? v.dataset.v : null); };
      el.onmousemove = () => { if (X.sel !== +el.dataset.i) { X.sel = +el.dataset.i; mark(); } };
    });
  }
  function mark() {
    out.querySelectorAll(".md-search-r").forEach((el) => { const on = +el.dataset.i === X.sel; el.classList.toggle("on", on); el.setAttribute("aria-selected", on); if (on) el.scrollIntoView({ block: "nearest" }); });
  }

  // ------------------------------------------------------------ open what was picked
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  async function tab(view, mode, modeSel) {
    const b = document.querySelector(`header.top nav button[data-view="${view}"]`);
    if (b && b.getAttribute("aria-selected") !== "true") b.click();
    await wait(120);
    if (mode) { const m = document.querySelector(`${modeSel || "#ws-modes"} button[data-mode="${mode}"]`); if (m && m.getAttribute("aria-selected") !== "true") m.click(); await wait(120); }
  }
  function flash(el) { if (!el) return; el.scrollIntoView({ block: "center" }); el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
  async function fill(view, id, value, then) {
    await tab(view);
    const el = document.getElementById(id);
    if (!el) return;
    el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    flash(el);
    if (then) toast(then);
  }
  async function go(r, verb) {
    if (!r) return;
    close();
    const a = r.action || {};
    const SA = window.IsoAssistant;
    const spec = a.spec || r.model;
    if (verb === "trace") return fill("trace", "tr-b", spec, "Filled in as model B. Choose model A and a prompt, then Trace.");
    if (verb === "laws") return fill("laws", "lw-extra", spec, "Filled in. Press Run the laws.");
    if (verb === "tune") { await tab("tune", "sl", "#tu-modes"); return fill("tune", "tu-spec", spec, "Filled in as the base model."); }
    switch (a.type) {
      case "go": return tab(a.view, a.mode, a.modeSel);
      case "rail": await tab(a.view); return flash(a.el);
      case "jump": return window.IsoLayout && window.IsoLayout.jump(a.id);
      case "model": return fill("trace", "tr-a", a.spec, "Filled in as model A. Or use Laws or Tune from the search result.");
      case "guide": await tab(a.view, a.mode); return SA && SA.guide();
      case "mission": return SA && SA.showMission(a.id);
      case "playbook": return SA && SA.plan({ playbook: a.id });
      default: return SA && SA.openResult(a);
    }
  }

  // ------------------------------------------------------------ open, close, keys
  function open() { box.classList.add("open"); scrim.classList.add("on"); header.classList.add("searching"); out.hidden = false; q.setAttribute("aria-expanded", "true"); if (!q.value) draw(""); }
  function close() { box.classList.remove("open"); scrim.classList.remove("on"); header.classList.remove("searching"); out.hidden = true; q.setAttribute("aria-expanded", "false"); q.blur(); }
  q.addEventListener("focus", () => { open(); if (q.value) query(); });
  q.addEventListener("blur", () => setTimeout(() => { if (document.activeElement !== q) { box.classList.remove("open"); scrim.classList.remove("on"); header.classList.remove("searching"); out.hidden = true; } }, 120));
  q.addEventListener("input", () => { clearTimeout(X.timer); X.timer = setTimeout(query, 110); });
  q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); X.sel = Math.min(X.res.length - 1, X.sel + 1); mark(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); X.sel = Math.max(0, X.sel - 1); mark(); }
    else if (e.key === "Enter") { e.preventDefault(); go(X.res[X.sel]); }
    else if (e.key === "Escape") { e.preventDefault(); q.value = ""; close(); }
  });
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    const typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
    if (!typing && !e.metaKey && !e.ctrlKey && !e.altKey && (e.key === "/" || e.key === "s")) { e.preventDefault(); q.focus(); q.select(); }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); q.focus(); q.select(); }
  });
  window.IsoSearch = { focus: () => q.focus() };
})();
