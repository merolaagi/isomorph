"use strict";
// Assistant: an overlay window that follows the page. On every tab (and every Weight space mode) it opens
// with that page's guide, what to do there and the questions people ask there; every question it sends
// carries the text of the page you are on. "Next steps" is the whole checklist for this Mac. Actions
// prepare the page (tab, mode, fields, a walkthrough) and never start a job.
(function () {
  const A = { open: false, tab: "page", data: null, chat: [], busy: false, timer: null, where: null };
  const KEY = "isomorph-assistant";
  const store = {
    get() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } },
    set(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* private window */ } },
  };
  const saved = store.get();
  A.chat = Array.isArray(saved.chat) ? saved.chat.slice(-30) : [];
  const seen = new Set(Array.isArray(saved.seen) ? saved.seen : []);
  const keep = () => store.set({ chat: A.chat.slice(-30), seen: [...seen] });
  try { if (localStorage.getItem("isomorph-guide-seen")) seen.add("tour"); } catch { /* private window */ }
  const mark = (x) => { if (!seen.has(x)) { seen.add(x); keep(); } };

  const STAGES = { trace: "#tr-stage", atlas: "#at-stage", laws: "#lw-stage", wsp: "#ws-stage", tune: "#tu-stage", engine: "#en-stage", lab: "#lab-stage", hub: "#hub-stage", about: "#view-about" };
  const view = () => { const b = document.querySelector('nav button[aria-selected="true"]'); return b ? b.dataset.view : "trace"; };
  const mode = () => { const b = document.querySelector('#ws-modes button[aria-selected="true"]'); return b ? b.dataset.mode : null; };
  const where = () => ({ view: view(), mode: view() === "wsp" ? mode() : null });
  const same = (a, b) => a && b && a.view === b.view && a.mode === b.mode;
  const md = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\n/g, "<br>");
  function screenText() {
    const w = where();
    const parts = [];
    const rail = document.querySelector(`#view-${w.view} .rail`);
    const st = document.querySelector(STAGES[w.view] || "main");
    if (rail) parts.push("LEFT PANEL (form and history):\n" + rail.innerText);
    if (st) parts.push("RESULT AREA:\n" + st.innerText);
    return parts.join("\n\n").replace(/\n{3,}/g, "\n\n").slice(0, 7000);
  }

  // ------------------------------------------------------------ the window
  const btn = document.createElement("button");
  btn.className = "as-fab";
  btn.setAttribute("aria-expanded", "false");
  const win = document.createElement("div");
  win.className = "as-win";
  win.hidden = true;
  win.setAttribute("role", "dialog");
  win.setAttribute("aria-label", "Assistant");
  win.innerHTML = `<div class="as-head"><b>Assistant</b><span class="as-sub" id="as-sub"></span><button class="as-x" aria-label="Close">×</button></div>
    <div class="subtabs as-tabs"><button data-t="page" aria-selected="true" id="as-tab-page">This page</button><button data-t="steps" aria-selected="false">Next steps</button><button data-t="ask" aria-selected="false">Ask</button></div>
    <div class="as-body" id="as-page"></div>
    <div class="as-body" id="as-steps" hidden></div>
    <div class="as-body" id="as-ask" hidden>
      <div class="as-msgs" id="as-msgs"></div>
      <div class="as-chips" id="as-chips"></div>
      <form class="as-form" id="as-form"><textarea id="as-in" rows="2" placeholder="Ask about this page, a result, or what to do next"></textarea><button class="primary" type="submit">Ask</button></form>
      <p class="as-note" id="as-note"></p>
    </div>`;
  document.body.append(btn, win);
  label();

  btn.onclick = () => toggle(!A.open);
  win.querySelector(".as-x").onclick = () => toggle(false);
  win.querySelectorAll(".as-tabs button").forEach((b) => (b.onclick = () => tab(b.dataset.t)));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && A.open && !document.querySelector(".guide.on")) toggle(false); });

  function toggle(on) {
    A.open = on;
    win.hidden = !on;
    btn.setAttribute("aria-expanded", String(on));
    btn.classList.toggle("on", on);
    clearInterval(A.timer);
    if (on) { refresh(); A.timer = setInterval(refresh, 15000); if (A.tab === "ask") $("#as-in").focus(); }
  }
  function tab(t) {
    A.tab = t;
    win.querySelectorAll(".as-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.t === t)));
    for (const k of ["page", "steps", "ask"]) $(`#as-${k}`).hidden = k !== t;
    if (t === "ask") { msgs(); $("#as-in").focus(); }
  }
  function label() {
    const p = A.data && A.data.page;
    const short = p ? p.title.split(" · ").pop() : null;
    btn.innerHTML = `<span aria-hidden="true">✦</span> ${short ? `Help with ${esc(short)}` : "Assistant"}`;
    btn.title = p ? `Assistant: ready to help with ${p.title}` : "Assistant";
  }

  // ------------------------------------------------------------ following the page
  function onPage() {
    const w = where();
    if (same(w, A.where)) return;
    A.where = w;
    refresh();
  }
  document.addEventListener("click", (e) => {
    if (!e.target.closest) return;
    if (e.target.closest('#ws-modes button[data-mode="write"]')) mark("write");
    if (e.target.closest("nav button[data-view], #ws-modes button[data-mode]")) setTimeout(onPage, 60);
  }, true);
  setTimeout(onPage, 400);

  async function refresh() {
    const w = where();
    try {
      A.data = await api(`/api/assistant?seen=${encodeURIComponent([...seen].join(","))}&view=${w.view}&mode=${w.mode || ""}`);
    } catch { return; }
    A.where = w;
    const d = A.data;
    label();
    $("#as-sub").textContent = `${d.page.title} · ${d.done} of ${d.of} steps done${d.state.job ? " · a job is running" : ""}`;
    $("#as-tab-page").textContent = d.page.title.split(" · ").pop();
    $("#as-note").textContent = d.state.has_key
      ? "Claude answers about this page: each question carries the text of the form and the result you are looking at, and the state of this Mac."
      : "No key saved: answers come from the built-in manual, starting with this page's section. Save an Anthropic key under Engine for answers about your own results.";
    page();
    steps();
    chips();
    if (A.tab === "ask") msgs();
  }

  function page() {
    const p = A.data.page, host = $("#as-page");
    host.innerHTML = `<h4 class="as-h">${esc(p.title)}</h4><p class="as-intro">${esc(p.intro)}</p>
      ${p.actions.length ? `<div class="as-sec">Do here</div><div class="as-acts">${p.actions.map((a, i) => `<button class="ghost small" data-pa="${i}">${esc(a.label)}</button>`).join("")}</div>` : ""}
      ${p.steps && p.steps.length ? `<div class="as-sec">Steps on this page</div><ol class="as-list">${p.steps.map((s) => item(s)).join("")}</ol>` : ""}
      <div class="as-sec">Ask about this page</div>
      <div class="as-qs">${p.prompts.map((q, i) => `<button data-pq="${i}">${esc(q)}</button>`).join("")}</div>
      <p class="as-note">Each question goes with the text of this page, so you can ask about the result in front of you.</p>`;
    host.querySelectorAll("[data-pa]").forEach((b) => (b.onclick = () => run(p.actions[+b.dataset.pa].action)));
    host.querySelectorAll("[data-pq]").forEach((b) => (b.onclick = () => { tab("ask"); ask(p.prompts[+b.dataset.pq]); }));
    host.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => run(p.steps.find((s) => s.id === b.dataset.step).action)));
  }
  function item(s) {
    const d = A.data;
    return `<li class="${s.done ? "done" : ""}${s.id === d.next ? " next" : ""}"><span class="as-mark" aria-hidden="true">${s.done ? "✓" : s.id === d.next ? "→" : ""}</span>
      <div><b>${esc(s.title)}</b><small>${esc(s.why)}</small></div><button class="ghost small" data-step="${s.id}">${s.done ? "Again" : "Go"}</button></li>`;
  }
  function steps() {
    const d = A.data, host = $("#as-steps");
    const pct = Math.round((100 * d.done) / d.of);
    host.innerHTML = `<div class="as-prog"><div style="width:${pct}%"></div></div>
      ${d.state.job ? `<p class="as-job">Running now: <b>${esc(d.state.job.title)}</b>. ${esc(d.state.job.message || "")}</p>` : ""}
      <ol class="as-list">${d.steps.map(item).join("")}</ol>
      <p class="as-note">Go opens the right tab and fills in what it can. Nothing runs until you press the button there.</p>`;
    host.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => run(d.steps.find((s) => s.id === b.dataset.step).action)));
  }
  function chips() {
    const p = A.data.page, host = $("#as-chips");
    const qs = [...p.prompts.slice(0, 2), "What should I do next?"];
    host.innerHTML = qs.map((q, i) => `<button data-cq="${i}">${esc(q)}</button>`).join("") + `<button data-cq="screen">Explain my screen</button>`;
    host.querySelectorAll("[data-cq]").forEach((b) => (b.onclick = () => ask(b.dataset.cq === "screen" ? "Explain what is on my screen, and what it means." : qs[+b.dataset.cq])));
  }

  // ------------------------------------------------------------ actions (page only, never a job)
  const PRESS = new Set(["ws-plan-demo", "mo-pair", "ww-crafted", "ww-broken", "lw-all"]);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  async function until(sel, ms = 2500) {
    for (let t = 0; t < ms; t += 100) { const el = document.querySelector(sel); if (el && !el.disabled) return el; await wait(100); }
    return null;
  }
  async function run(a) {
    if (!a) return;
    if (a.type === "tour") { toggle(false); window.IsoGuide && window.IsoGuide.start(a.tour); return; }
    const nb = document.querySelector(`nav button[data-view="${a.view}"]`);
    if (!nb) return;
    if (nb.getAttribute("aria-selected") !== "true") nb.click();
    await wait(250);
    if (a.mode) { const mb = await until(`#ws-modes button[data-mode="${a.mode}"]`); if (mb && mb.getAttribute("aria-selected") !== "true") mb.click(); await wait(300); }
    for (const [id, value] of Object.entries(a.fill || {})) {
      const el = await until(`#${CSS.escape(id)}`);
      if (!el) continue;
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
    }
    if (a.press && PRESS.has(a.press)) { const el = await until(`#${a.press}`); if (el) el.click(); }
    if (a.focus) { const el = await until(`#${CSS.escape(a.focus)}`); if (el) { el.focus(); el.scrollIntoView({ block: "center" }); } }
    onPage();
    if (innerWidth < 1100) toggle(false);
    toast("Ready. Check the fields, then press the button to run it.");
  }

  // ------------------------------------------------------------ ask
  function msgs() {
    const host = $("#as-msgs");
    host.innerHTML = A.chat.length ? A.chat.map((m, i) => m.role === "page" ? `<div class="as-div"><span>${esc(m.text)}</span></div>`
      : `<div class="as-m ${m.role}">${md(m.text)}${(m.actions || []).length
        ? `<div class="as-acts">${m.actions.map((x, j) => `<button class="ghost small" data-act="${i}:${j}">${esc(x.label)}</button>`).join("")}</div>` : ""}</div>`).join("")
      : `<p class="as-note">Ask about ${esc(A.data ? A.data.page.title : "this page")}: how to start, what a number means, what to try next.</p>`;
    if (A.busy) host.insertAdjacentHTML("beforeend", `<div class="as-m assistant as-typing"><i></i><i></i><i></i></div>`);
    host.querySelectorAll("[data-act]").forEach((b) => (b.onclick = () => { const [i, j] = b.dataset.act.split(":").map(Number); run(A.chat[i].actions[j].action); }));
    host.scrollTop = host.scrollHeight;
  }
  async function ask(q) {
    if (A.busy || !q.trim()) return;
    const ctx = { ...where(), screen: screenText() };
    const title = A.data ? A.data.page.title : ctx.view;
    const lastPage = [...A.chat].reverse().find((m) => m.role === "page");
    if (!lastPage || lastPage.text !== title) A.chat.push({ role: "page", text: title });
    const history = A.chat.filter((m) => m.role !== "page").map((m) => ({ role: m.role, text: m.text }));
    A.chat.push({ role: "user", text: q });
    A.busy = true; msgs(); keep();
    let r;
    try { r = await api("/api/assistant/ask", { method: "POST", body: { question: q, history, context: ctx, seen: [...seen].join(",") } }); }
    catch (e) { r = { reply: `Something went wrong: ${e.message}`, actions: [] }; }
    A.busy = false;
    A.chat.push({ role: "assistant", text: r.reply, actions: r.actions || [] });
    keep(); msgs();
  }
  $("#as-form").onsubmit = (e) => { e.preventDefault(); const q = $("#as-in").value; $("#as-in").value = ""; ask(q); };
  $("#as-in").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#as-form").requestSubmit(); } });

  // the tour leaves no file: note it here
  const hook = setInterval(() => {
    if (!window.IsoGuide || window.IsoGuide._wrapped) return;
    const s = window.IsoGuide.start;
    window.IsoGuide.start = (n) => { if (n === "tour") mark("tour"); return s(n); };
    window.IsoGuide._wrapped = true;
    clearInterval(hook);
  }, 300);

  window.IsoAssistant = { open: () => toggle(true), ask, run, where };
})();
