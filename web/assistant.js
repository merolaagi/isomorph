"use strict";
// Assistant: an overlay window with the next steps for this Mac and a chat about Isomorph.
// The checklist comes from what is already measured here; "Go" prepares the page (tab, mode,
// fields, a walkthrough) and never starts a job. The chat uses Claude when a key is saved.
(function () {
  const A = { open: false, tab: "steps", data: null, chat: [], busy: false, timer: null };
  const KEY = "isomorph-assistant";
  const store = {
    get() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } },
    set(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* private window */ } },
  };
  const saved = store.get();
  A.chat = Array.isArray(saved.chat) ? saved.chat.slice(-20) : [];
  const seen = new Set(Array.isArray(saved.seen) ? saved.seen : []);
  const keep = () => store.set({ chat: A.chat.slice(-20), seen: [...seen] });
  try { if (localStorage.getItem("isomorph-guide-seen")) seen.add("tour"); } catch { /* private window */ }
  const mark = (x) => { if (!seen.has(x)) { seen.add(x); keep(); } };

  const STAGES = { trace: "#tr-stage", atlas: "#at-stage", laws: "#lw-stage", wsp: "#ws-stage", engine: "#en-stage", lab: "#lab-stage", hub: "#hub-stage", about: "#view-about" };
  const view = () => { const b = document.querySelector('nav button[aria-selected="true"]'); return b ? b.dataset.view : "trace"; };
  const mode = () => { const b = document.querySelector('#ws-modes button[aria-selected="true"]'); return b ? b.dataset.mode : null; };
  const md = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\n/g, "<br>");

  // ------------------------------------------------------------ the window
  const btn = document.createElement("button");
  btn.className = "as-fab";
  btn.innerHTML = `<span aria-hidden="true">✦</span> Assistant`;
  btn.setAttribute("aria-expanded", "false");
  const win = document.createElement("div");
  win.className = "as-win";
  win.hidden = true;
  win.setAttribute("role", "dialog");
  win.setAttribute("aria-label", "Assistant");
  win.innerHTML = `<div class="as-head"><b>Assistant</b><span class="as-sub" id="as-sub"></span><button class="as-x" aria-label="Close">×</button></div>
    <div class="subtabs as-tabs"><button data-t="steps" aria-selected="true">Next steps</button><button data-t="ask" aria-selected="false">Ask</button></div>
    <div class="as-body" id="as-steps"></div>
    <div class="as-body" id="as-ask" hidden>
      <div class="as-msgs" id="as-msgs"></div>
      <div class="as-chips"><button data-q="How do I start?">How do I start?</button><button data-q="What should I do next?">What should I do next?</button><button data-q="Explain what is on my screen, and what it means." data-screen="1">Explain my screen</button></div>
      <form class="as-form" id="as-form"><textarea id="as-in" rows="2" placeholder="Ask about any tab, result or next step"></textarea><button class="primary" type="submit">Ask</button></form>
      <p class="as-note" id="as-note"></p>
    </div>`;
  document.body.append(btn, win);

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
    $("#as-steps").hidden = t !== "steps";
    $("#as-ask").hidden = t !== "ask";
    if (t === "ask") { msgs(); $("#as-in").focus(); }
  }

  // ------------------------------------------------------------ next steps
  async function refresh() {
    try { A.data = await api(`/api/assistant?seen=${encodeURIComponent([...seen].join(","))}`); } catch { return; }
    const d = A.data;
    $("#as-sub").textContent = `${d.done} of ${d.of} done${d.state.job ? " · a job is running" : ""}`;
    $("#as-note").textContent = d.state.has_key ? "Answers come from Claude, with the state of this Mac. Explain my screen also sends the text of the current result."
      : "No key saved: answers come from the built-in manual. Save an Anthropic key under Engine for answers about your own results.";
    steps();
  }
  function steps() {
    const d = A.data, host = $("#as-steps");
    if (!d) return;
    const pct = Math.round((100 * d.done) / d.of);
    host.innerHTML = `<div class="as-prog"><div style="width:${pct}%"></div></div>
      ${d.state.job ? `<p class="as-job">Running now: <b>${esc(d.state.job.title)}</b>. ${esc(d.state.job.message || "")}</p>` : ""}
      <ol class="as-list">${d.steps.map((s) => `<li class="${s.done ? "done" : ""}${s.id === d.next ? " next" : ""}">
        <span class="as-mark" aria-hidden="true">${s.done ? "✓" : s.id === d.next ? "→" : ""}</span>
        <div><b>${esc(s.title)}</b><small>${esc(s.why)}</small></div>
        <button class="ghost small" data-step="${s.id}">${s.done ? "Again" : "Go"}</button></li>`).join("")}</ol>
      <p class="as-note">Go opens the right tab and fills in what it can. Nothing runs until you press the button there.</p>`;
    host.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => run(d.steps.find((s) => s.id === b.dataset.step).action)));
  }

  // ------------------------------------------------------------ actions (page only, never a job)
  const PRESS = new Set(["ws-plan-demo", "mo-pair", "ww-crafted", "ww-broken"]);
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
    nb.click();
    await wait(250);
    if (a.mode) { const mb = await until(`#ws-modes button[data-mode="${a.mode}"]`); if (mb) mb.click(); await wait(300); }
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
    if (innerWidth < 1100) toggle(false);
    toast("Ready. Check the fields, then press the button to run it.");
  }

  // ------------------------------------------------------------ ask
  function msgs() {
    const host = $("#as-msgs");
    host.innerHTML = A.chat.length ? A.chat.map((m, i) => `<div class="as-m ${m.role}">${md(m.text)}${(m.actions || []).length
      ? `<div class="as-acts">${m.actions.map((x, j) => `<button class="ghost small" data-act="${i}:${j}">${esc(x.label)}</button>`).join("")}</div>` : ""}</div>`).join("")
      : `<p class="as-note">Ask how to start, what a number means, or what to try next. "Explain my screen" reads the result you are looking at.</p>`;
    if (A.busy) host.insertAdjacentHTML("beforeend", `<div class="as-m assistant as-typing"><i></i><i></i><i></i></div>`);
    host.querySelectorAll("[data-act]").forEach((b) => (b.onclick = () => { const [i, j] = b.dataset.act.split(":").map(Number); run(A.chat[i].actions[j].action); }));
    host.scrollTop = host.scrollHeight;
  }
  async function ask(q, withScreen) {
    if (A.busy || !q.trim()) return;
    const ctx = { view: view(), mode: view() === "wsp" ? mode() : null };
    if (withScreen) { const st = document.querySelector(STAGES[ctx.view] || "main"); ctx.screen = st ? st.innerText.replace(/\n{3,}/g, "\n\n").slice(0, 6000) : ""; }
    const history = A.chat.map((m) => ({ role: m.role, text: m.text }));
    A.chat.push({ role: "user", text: q });
    A.busy = true; msgs(); keep();
    let r;
    try { r = await api("/api/assistant/ask", { method: "POST", body: { question: q, history, context: ctx, seen: [...seen].join(",") } }); }
    catch (e) { r = { reply: `Something went wrong: ${e.message}`, actions: [] }; }
    A.busy = false;
    A.chat.push({ role: "assistant", text: r.reply, actions: r.actions || [] });
    keep(); msgs();
  }
  $("#as-form").onsubmit = (e) => { e.preventDefault(); const q = $("#as-in").value; $("#as-in").value = ""; ask(q, /screen|result|this page|on my screen|mean/i.test(q)); };
  $("#as-in").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#as-form").requestSubmit(); } });
  win.querySelectorAll("[data-q]").forEach((b) => (b.onclick = () => ask(b.dataset.q, !!b.dataset.screen)));

  // ------------------------------------------------------------ what this browser has done that leaves no file
  document.addEventListener("click", (e) => {
    const t = e.target.closest && e.target.closest('#ws-modes button[data-mode="write"]');
    if (t) mark("write");
  }, true);
  const hook = setInterval(() => {
    if (!window.IsoGuide || window.IsoGuide._wrapped) return;
    const s = window.IsoGuide.start;
    window.IsoGuide.start = (n) => { if (n === "tour") mark("tour"); return s(n); };
    window.IsoGuide._wrapped = true;
    clearInterval(hook);
  }, 300);

  window.IsoAssistant = { open: () => toggle(true), ask, run };
})();
