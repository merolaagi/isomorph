"use strict";
// The superagent: an overlay at the bottom right that follows the page. It guides (the page's manual,
// the checklist, a chat that can see your screen), it does the work (missions: a plan of runs you approve,
// carried out step by step while you watch or do something else), and it supervises every other agent
// (the job queue, Autopilot, Claude's Weight-space runs, Tune sessions), with alerts and the controls to act.
(function () {
  const A = { open: false, tab: "do", data: null, sa: null, mission: null, chat: [], busy: false, timer: null, where: null, seenDone: {} };
  const KEY = "isomorph-assistant";
  const store = {
    get() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } },
    set(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* private window */ } },
  };
  const saved = store.get();
  A.chat = Array.isArray(saved.chat) ? saved.chat.slice(-30) : [];
  A.follow = saved.follow !== false;
  A.each = !!saved.each;
  A.missionId = saved.mission || null;
  const seen = new Set(Array.isArray(saved.seen) ? saved.seen : []);
  const keep = () => store.set({ chat: A.chat.slice(-30), seen: [...seen], follow: A.follow, each: A.each, mission: A.missionId });
  try { if (localStorage.getItem("isomorph-guide-seen")) seen.add("tour"); } catch { /* private window */ }
  const mark = (x) => { if (!seen.has(x)) { seen.add(x); keep(); } };

  const STAGES = { trace: "#tr-stage", atlas: "#at-stage", laws: "#lw-stage", wsp: "#ws-stage", tune: "#tu-stage", engine: "#en-stage", lab: "#lab-stage", hub: "#hub-stage", about: "#view-about" };
  const view = () => { const b = document.querySelector('nav button[aria-selected="true"]'); return b ? b.dataset.view : "trace"; };
  const mode = () => { const b = document.querySelector('#ws-modes button[aria-selected="true"]'); return b ? b.dataset.mode : null; };
  const where = () => ({ view: view(), mode: view() === "wsp" ? mode() : null });
  const same = (a, b) => a && b && a.view === b.view && a.mode === b.mode;
  const md = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\n/g, "<br>");
  const LIVE = ["approved", "running", "waiting", "pausing"];
  const ICON = { pending: "○", waiting: "⏸", running: "◔", done: "✓", failed: "✗", skipped: "–" };
  const ago = (t) => { const s = Date.now() / 1000 - t; return s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`; };
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
  win.setAttribute("aria-label", "Superagent");
  win.innerHTML = `<div class="as-head"><b>Superagent</b><span class="as-sub" id="as-sub"></span><button class="as-x" aria-label="Close">×</button></div>
    <div class="subtabs as-tabs"><button data-t="do" aria-selected="true">Do</button><button data-t="agents" aria-selected="false" id="as-tab-agents">Agents</button><button data-t="page" aria-selected="false" id="as-tab-page">Guide</button><button data-t="ask" aria-selected="false">Ask</button></div>
    <div class="as-body" id="as-do"></div>
    <div class="as-body" id="as-agents" hidden></div>
    <div class="as-body" id="as-page" hidden></div>
    <div class="as-body" id="as-ask" hidden>
      <div class="as-msgs" id="as-msgs"></div>
      <div class="as-chips" id="as-chips"></div>
      <form class="as-form" id="as-form"><textarea id="as-in" rows="2" placeholder="Ask about this page or a result, or tell me what to do"></textarea><button class="primary" type="submit">Ask</button></form>
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
    if (on) { refresh(); if (A.tab === "ask") $("#as-in").focus(); }
    schedule();
  }
  function tab(t) {
    A.tab = t;
    win.querySelectorAll(".as-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.t === t)));
    for (const k of ["do", "agents", "page", "ask"]) $(`#as-${k}`).hidden = k !== t;
    if (t === "ask") { msgs(); $("#as-in").focus(); }
    if (t === "agents") agents();
    if (t === "do") doTab();
  }
  function running() { return A.mission && LIVE.includes(A.mission.status) ? A.mission : null; }
  function label() {
    const m = A.mission, p = A.data && A.data.page;
    const alerts = A.sa ? A.sa.board.alerts.length : 0;
    const short = p ? p.title.split(" · ").pop() : null;
    let html;
    if (m && LIVE.includes(m.status)) {
      const done = m.steps.filter((s) => ["done", "skipped"].includes(s.status)).length;
      const cur = m.steps.find((s) => ["running", "waiting", "pending"].includes(s.status));
      const frac = cur && cur.status === "running" && cur.progress ? cur.progress : 0;
      const pct = Math.round(100 * (done + frac) / m.steps.length);
      html = `<span class="as-ring" style="--p:${pct}" aria-hidden="true"></span> ${m.status === "waiting" ? "Waiting for you" : `Step ${Math.min(done + 1, m.steps.length)} of ${m.steps.length}`}<small>${esc(cur ? cur.label : m.title)}</small>`;
      btn.title = `Superagent: ${m.title}`;
    } else {
      html = `<span aria-hidden="true">✦</span> Superagent${short ? `<small>${esc(short)}</small>` : ""}`;
      btn.title = p ? `Superagent: guide, do and supervise. On ${p.title}` : "Superagent";
    }
    if (alerts) html += `<i class="as-badge" aria-label="${alerts} alerts">${alerts}</i>`;
    btn.innerHTML = html;
  }

  // ------------------------------------------------------------ following the page and the agents
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

  function schedule() {
    clearTimeout(A.timer);
    const fast = running() ? 1500 : A.open ? 8000 : 20000;
    A.timer = setTimeout(async () => { await poll(); schedule(); }, fast);
  }
  async function poll() {
    try { A.sa = await api(`/api/superagent?view=${where().view}`); } catch { return; }
    const want = A.missionId || (A.sa.board.missions.find((m) => LIVE.includes(m.status)) || {}).id;
    if (want) {
      try { setMission(await api(`/api/superagent/mission/${want}`)); } catch { A.missionId = null; A.mission = null; keep(); }
    }
    label();
    if (A.open) { if (A.tab === "do") doTab(); if (A.tab === "agents") agents(); }
  }
  function setMission(m) {
    const prev = A.mission && A.mission.id === m.id ? A.mission : null;
    A.mission = m;
    A.missionId = m.id;
    keep();
    // follow along: open each result as its step finishes
    for (const s of m.steps) {
      const k = `${m.id}:${s.i}`;
      const was = prev && prev.steps.find((x) => x.i === s.i);
      if (s.status === "done" && was && was.status !== "done" && !A.seenDone[k]) {
        A.seenDone[k] = true;
        if (A.follow && s.open) openResult(s.open);
        toast(`Step ${s.i} done: ${s.label}`);
      }
    }
    if (prev && prev.status !== m.status && ["done", "paused", "waiting", "failed"].includes(m.status)) {
      toast(m.status === "done" ? `Mission done: ${m.title}` : m.status === "waiting" ? "The superagent is waiting for your go-ahead." : "The mission paused.", m.status === "paused" && m.steps.some((s) => s.status === "failed"));
      if (!A.open && m.status !== "done") toggle(true), tab("do");
    }
  }

  async function refresh() {
    const w = where();
    try {
      A.data = await api(`/api/assistant?seen=${encodeURIComponent([...seen].join(","))}&view=${w.view}&mode=${w.mode || ""}`);
    } catch { return; }
    A.where = w;
    const d = A.data;
    await poll();
    $("#as-sub").textContent = `${d.page.title} · ${d.done} of ${d.of} steps done`;
    $("#as-tab-page").textContent = "Guide";
    $("#as-note").textContent = d.state.has_key
      ? "Claude answers about this page (it sees the form and the result on your screen) and can hand work to the superagent as a plan for you to approve."
      : "No key saved: answers come from the built-in manual. Save an Anthropic key under Engine for answers about your own results and for planning from your own words.";
    page();
    chips();
    if (A.tab === "ask") msgs();
    if (A.tab === "do") doTab();
  }

  // ------------------------------------------------------------ Do: missions
  function doTab() {
    const host = $("#as-do"), S = A.sa, d = A.data;
    if (!S) { host.innerHTML = `<p class="as-note">Loading…</p>`; return; }
    const focus = document.activeElement && document.activeElement.id === "as-goal";
    const draftGoal = focus ? $("#as-goal").value : (host.querySelector("#as-goal") || {}).value || "";
    const m = A.mission;
    host.innerHTML = `
      <form class="as-goalf" id="as-goalf"><textarea id="as-goal" rows="2" placeholder="${S.has_key ? "What should I do? e.g. fine-tune Micro Opus twice and tell me if the two runs agree" : "What should I do? (without a key I pick the closest ready-made plan)"}"></textarea>
        <div class="as-row"><label class="as-check"><input type="checkbox" id="as-each" ${A.each ? "checked" : ""}> Ask before each step</label><button class="primary" type="submit">Plan it</button></div></form>
      <div id="as-mission">${m ? missionCard(m) : ""}</div>
      <div class="as-sec">Ready-made${A.where ? ` for ${esc((d && d.page.title.split(" · ")[0]) || "this page")}` : ""}</div>
      <div class="as-pbs">${S.playbooks.slice(0, 6).map((p) => `<div class="as-pb${p.here ? " here" : ""}"><div><b>${esc(p.title)}</b><small>${esc(p.why)}</small>
        <small class="as-chain">${p.steps.map((s) => esc(skillLabel(s.skill))).join(" → ")}</small></div><button class="ghost small" data-pb="${p.id}">Plan</button></div>`).join("")}</div>
      ${S.playbooks.length > 6 ? `<details class="as-more"><summary>${S.playbooks.length - 6} more</summary><div class="as-pbs">${S.playbooks.slice(6).map((p) => `<div class="as-pb"><div><b>${esc(p.title)}</b><small>${esc(p.why)}</small></div><button class="ghost small" data-pb="${p.id}">Plan</button></div>`).join("")}</div></details>` : ""}
      ${d ? `<div class="as-sec">Checklist · ${d.done} of ${d.of}</div><ol class="as-list">${d.steps.map((s) => item(s, S.checks[s.id])).join("")}</ol>` : ""}
      ${S.board.missions.length ? `<div class="as-sec">Missions</div><ul class="as-ms">${S.board.missions.map((x) => `<li><button data-m="${x.id}" class="${m && m.id === x.id ? "on" : ""}"><span class="as-st ${x.status}">${esc(x.status)}</span> ${esc(x.title)}<small>${x.done}/${x.of} steps · ${ago(x.created)}</small></button></li>`).join("")}</ul>` : ""}
      <p class="as-note">Nothing runs until you approve the plan. The superagent calls the same buttons you would, through Isomorph's own API: it cannot run code, delete anything or change settings.</p>`;
    const g = $("#as-goal");
    g.value = draftGoal;
    if (focus) { g.focus(); g.setSelectionRange(g.value.length, g.value.length); }
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#as-goalf").requestSubmit(); } });
    $("#as-each").onchange = (e) => { A.each = e.target.checked; keep(); };
    $("#as-goalf").onsubmit = (e) => { e.preventDefault(); const v = g.value.trim(); if (v) plan({ goal: v }); };
    host.querySelectorAll("[data-pb]").forEach((b) => (b.onclick = () => plan({ playbook: b.dataset.pb })));
    host.querySelectorAll("[data-check]").forEach((b) => (b.onclick = () => plan({ check: b.dataset.check })));
    host.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => run(d.steps.find((s) => s.id === b.dataset.step).action)));
    host.querySelectorAll("[data-m]").forEach((b) => (b.onclick = async () => { setMission(await api(`/api/superagent/mission/${b.dataset.m}`)); doTab(); label(); }));
    wireMission(host);
  }
  const skillLabel = (n) => ((A.sa && A.sa.skills.find((s) => s.name === n)) || { label: n }).label;

  function missionCard(m) {
    const st = m.status, draft = st === "draft", live = LIVE.includes(st);
    const failed = m.steps.find((s) => s.status === "failed");
    const btns = [];
    if (draft) btns.push(`<button class="primary small" data-ma="approve">Approve and run</button>`, `<button class="ghost small" data-ma="each">${m.approve === "each" ? "Ask before each step: on" : "Ask before each step: off"}</button>`);
    if (st === "waiting") btns.push(`<button class="primary small" data-ma="approve">Go ahead</button>`);
    if (live && st !== "waiting") btns.push(`<button class="ghost small" data-ma="pause">Pause</button>`);
    if (st === "paused" && !failed) btns.push(`<button class="primary small" data-ma="resume">Resume</button>`);
    if (failed && st === "paused") btns.push(`<button class="primary small" data-ma="retry">Retry step ${failed.i}</button>`, `<button class="ghost small" data-ma="skip">Skip it</button>`);
    if (failed && st === "paused" && A.sa && A.sa.has_key) btns.push(`<button class="ghost small" data-ma="replan">Let Claude replan</button>`);
    if (live || st === "paused" || st === "waiting") btns.push(`<button class="ghost small" data-ma="stop">Stop</button>`);
    if (!live) btns.push(`<button class="ghost small" data-ma="delete">${draft ? "Discard" : "Remove"}</button>`);
    return `<div class="as-card">
      <div class="as-ct"><span class="as-st ${st}">${esc(st)}</span><b>${esc(m.title)}</b></div>
      ${m.goal && m.goal !== m.title ? `<p class="as-goaltext">${esc(m.goal)}</p>` : ""}
      ${m.note ? `<p class="as-goaltext">${esc(m.note)}</p>` : ""}
      ${m.alternatives && m.alternatives.length && draft ? `<p class="as-goaltext">Or: ${m.alternatives.map((a) => `<a href="#" data-alt="${a.id}">${esc(a.title)}</a>`).join(" · ")}</p>` : ""}
      <ol class="as-steps">${m.steps.map((s) => `<li class="${s.status}"><span class="as-si" aria-hidden="true">${ICON[s.status] || "○"}</span><div>
        <b>${s.i}. ${esc(s.label)}</b> <small class="as-cost">${esc(s.cost)}${s.downloads ? " · may download" : ""}</small>
        <small>${esc(s.why || "")}</small>
        <small class="as-args">${esc(argText(s.resolved || s.args))}</small>
        ${s.status === "running" ? `<div class="as-pbar"><i style="width:${Math.round((s.progress || 0) * 100)}%"></i></div><small>${esc(s.message || "Running")}</small>` : ""}
        ${s.summary ? `<p class="as-sum">${esc(s.summary)}</p>` : ""}${s.error ? `<p class="as-err">${esc(s.error)}</p>` : ""}
        </div><div class="as-sbtn">${s.open && s.status === "done" ? `<button class="ghost small" data-open="${s.i}">Open</button>` : ""}${draft ? `<button class="as-drop" data-drop="${s.i}" aria-label="Remove step">×</button>` : ""}</div></li>`).join("")}</ol>
      ${m.report ? `<div class="as-report"><b>Report</b>${md(m.report)}</div>` : ""}
      <div class="as-acts">${btns.join("")}</div>
      <label class="as-check"><input type="checkbox" id="as-follow" ${A.follow ? "checked" : ""}> Follow along: open each result as it finishes</label>
    </div>`;
  }
  function argText(a) {
    return Object.entries(a || {}).filter(([k, v]) => v !== "" && v !== null && !(Array.isArray(v) && !v.length))
      .map(([k, v]) => `${k}: ${typeof v === "string" ? (v.length > 40 ? v.slice(0, 40).replace(/\n/g, " / ") + "…" : v) : JSON.stringify(v)}`).join(" · ");
  }
  function wireMission(host) {
    const m = A.mission;
    if (!m) return;
    host.querySelectorAll("[data-ma]").forEach((b) => (b.onclick = () => missionAct(b.dataset.ma)));
    host.querySelectorAll("[data-open]").forEach((b) => (b.onclick = () => openResult(m.steps.find((s) => s.i === +b.dataset.open).open)));
    host.querySelectorAll("[data-drop]").forEach((b) => (b.onclick = () => missionAct("drop", +b.dataset.drop)));
    host.querySelectorAll("[data-alt]").forEach((a) => (a.onclick = async (e) => { e.preventDefault(); await missionAct("delete"); plan({ playbook: a.dataset.alt }); }));
    const f = host.querySelector("#as-follow");
    if (f) f.onchange = (e) => { A.follow = e.target.checked; keep(); };
  }
  async function plan(body) {
    const host = $("#as-mission");
    if (host) host.innerHTML = `<div class="as-card"><div class="as-typing"><i></i><i></i><i></i></div><small class="as-goaltext">${A.sa && A.sa.has_key && body.goal ? "Claude is planning…" : "Drafting the plan…"}</small></div>`;
    try {
      const m = await api("/api/superagent/plan", { method: "POST", body: { ...body, view: where().view, approve: A.each ? "each" : "plan" } });
      A.mission = null;
      setMission(m);
      if (body.goal) $("#as-goal") && ($("#as-goal").value = "");
    } catch (e) { toast(e.message, true); }
    tab("do");
    if (!A.open) toggle(true);
    await poll();
  }
  async function missionAct(action, step) {
    const m = A.mission;
    if (!m) return;
    try {
      if (action === "delete") {
        await api(`/api/superagent/mission/${m.id}`, { method: "DELETE" });
        A.mission = null; A.missionId = null; keep();
      } else {
        if (action === "replan") $("#as-mission").insertAdjacentHTML("afterbegin", `<p class="as-goaltext">Claude is revising the plan…</p>`);
        setMission(await api(`/api/superagent/mission/${m.id}/${action}`, { method: "POST", body: step ? { step } : {} }));
      }
    } catch (e) { toast(e.message, true); }
    await poll();
    schedule();
  }
  async function openResult(o) {
    if (!o) return;
    if (o.type === "tune") {
      const nb = document.querySelector('nav button[data-view="tune"]');
      if (nb && nb.getAttribute("aria-selected") !== "true") nb.click();
      await wait(200);
      if (window.Tune) { await window.Tune.init(); window.Tune.adapter(o.adapter); }
      return;
    }
    const go = async (v, mode) => { await run({ type: "go", view: v, mode }, true); await wait(150); };
    try {
      if (o.type === "trace" && o.id && window.Trace) { await go("trace"); return window.Trace.open(o.id); }
      if (o.type === "wsp" && window.WsCore) { await go("wsp", o.mode); window.WsCore.history(); if (o.id) return window.WsCore.open(o.id); return; }
      if (o.type === "laws" && window.Laws) { await go("laws"); return window.Laws.loadAll(o.spec ? window.Laws.lawId(o.spec) : undefined); }
      if (o.type === "lab") { await go("lab"); if (o.run_id && window.openRun) window.openRun(o.run_id); return; }
    } catch (e) { toast(e.message, true); return; }
    await run(o, true);
    if (o.view === "wsp" && window.WsCore) window.WsCore.history();
    if (o.view === "laws" && window.Laws) window.Laws.loadAll();
    if (o.view === "atlas" && window.Atlas) window.Atlas.load();
  }

  // ------------------------------------------------------------ Agents: supervise everything that runs
  function agents() {
    const host = $("#as-agents"), S = A.sa;
    if (!S) { host.innerHTML = `<p class="as-note">Loading…</p>`; return; }
    const B = S.board, ap = B.autopilot || {};
    const running = B.jobs.filter((j) => ["queued", "running"].includes(j.state)), past = B.jobs.filter((j) => !["queued", "running"].includes(j.state));
    $("#as-tab-agents").textContent = B.alerts.length ? `Agents (${B.alerts.length})` : "Agents";
    host.innerHTML = `
      ${B.alerts.length ? `<div class="as-sec">Needs you</div>${B.alerts.map((a, i) => `<div class="as-alert ${a.level}"><span>${esc(a.text)}</span><div>${a.action ? `<button class="ghost small" data-al="${i}">${esc(a.action.label)}</button>` : ""}<button class="as-drop" data-dis="${esc(a.id)}" aria-label="Dismiss">×</button></div></div>`).join("")}` : ""}
      <div class="as-sec">Superagent</div>
      ${agentRow("✦", "Missions", B.missions.filter((m) => LIVE.includes(m.status)).map((m) => `${esc(m.title)}: ${m.done}/${m.of}${m.current ? `, ${esc(m.current)}` : ""}`).join("<br>") || "Idle.", "")}
      <div class="as-sec">Workers</div>
      ${agentRow("▤", "Job queue", running.length ? running.map((j) => `<div class="as-jr"><span>${esc(j.title)} <small>${esc(j.state)} · ${Math.round(j.progress * 100)}% · ${esc((j.message || "").slice(0, 80))}</small></span><button class="ghost small" data-stop="${j.id}">Stop</button></div>`).join("") : "Nothing running.",
        past.length ? `<details><summary>Recent</summary>${past.map((j) => `<div class="as-jr"><span class="${j.state}">${esc(j.title)} <small>${esc(j.state)}${j.error ? `: ${esc(j.error.slice(0, 120))}` : ""}</small></span></div>`).join("")}</details>` : "")}
      ${agentRow("⟳", "Autopilot", `<span class="as-st ${esc(ap.status || "idle")}">${esc(ap.status || "idle")}</span>${ap.queued ? ` ${ap.queued} queued, ${ap.done} done` : ""}${ap.current ? ` · now: ${esc(ap.current)}` : ""}${(ap.log || []).length ? `<small>${ap.log.map(esc).join("<br>")}</small>` : ""}`,
        `<div class="as-acts">${ap.status === "running" ? `<button class="ghost small" data-ap="pause">Pause</button>` : ""}${ap.status === "paused" ? `<button class="ghost small" data-ap="resume">Resume</button>` : ""}${["running", "paused"].includes(ap.status) ? `<button class="ghost small" data-ap="stop">Stop</button>` : `<button class="ghost small" data-pb="research-loop">Plan a campaign</button>`}</div>`)}
      ${agentRow("◇", "Claude in Weight space", B.claude.length ? B.claude.map((c) => `<div>${esc(c.kind)}${c.task ? ` · ${esc(c.task)}` : ""} · ${esc(String(c.spec || "").split("/").pop())}${c.line ? ` · <b>${esc(c.line)}</b>` : ""} <small>${c.created ? ago(c.created) : ""}</small></div>`).join("") : (S.has_key ? "No runs yet." : "Needs an Anthropic key (Engine)."),
        `<div class="as-acts"><button class="ghost small" data-go="wsp">Open Weight space</button>${S.has_key ? `<button class="ghost small" data-pb="claude-reads">Plan read, fix, write</button>` : ""}</div>`)}
      ${agentRow("≋", "Tune sessions", B.sessions.length ? B.sessions.map((s) => `<div class="as-jr"><span>${esc(s.name)} <small>${esc(s.source || "")} · ${s.step} steps · idle ${s.idle_min} min</small></span><span><button class="ghost small" data-watch="${s.id}">Watch</button><button class="ghost small" data-close="${s.id}">Close</button></span></div>`).join("") : "No live sessions.", "")}
      <p class="as-note">Controls here act at once (your click is the approval) and are logged as one-step missions.</p>`;
    host.querySelectorAll("[data-al]").forEach((b) => (b.onclick = () => alertAct(B.alerts[+b.dataset.al])));
    host.querySelectorAll("[data-dis]").forEach((b) => (b.onclick = async () => { await api("/api/superagent/dismiss", { method: "POST", body: { id: b.dataset.dis } }); poll(); }));
    host.querySelectorAll("[data-stop]").forEach((b) => (b.onclick = () => skill("job_stop", { id: b.dataset.stop })));
    host.querySelectorAll("[data-ap]").forEach((b) => (b.onclick = () => skill("autopilot_control", { action: b.dataset.ap })));
    host.querySelectorAll("[data-close]").forEach((b) => (b.onclick = () => skill("tune_close", { id: b.dataset.close })));
    host.querySelectorAll("[data-watch]").forEach((b) => (b.onclick = async () => { document.querySelector('nav button[data-view="tune"]').click(); await wait(200); window.Tune && window.Tune.watch(b.dataset.watch); }));
    host.querySelectorAll("[data-pb]").forEach((b) => (b.onclick = () => plan({ playbook: b.dataset.pb })));
    host.querySelectorAll("[data-go]").forEach((b) => (b.onclick = () => run({ type: "go", view: b.dataset.go }, true)));
  }
  function agentRow(icon, name, body, extra) {
    return `<div class="as-agent"><span class="as-ai" aria-hidden="true">${icon}</span><div><b>${esc(name)}</b><div class="as-ab">${body}</div>${extra || ""}</div></div>`;
  }
  async function skill(name, args) {
    try { await api("/api/superagent/skill", { method: "POST", body: { skill: name, args } }); toast("Done."); }
    catch (e) { toast(e.message, true); }
    setTimeout(poll, 600);
  }
  async function alertAct(a) {
    const x = a.action;
    if (x.skill) return skill(x.skill, x.args);
    if (x.mission) { setMission(await api(`/api/superagent/mission/${x.mission}`)); tab("do"); return; }
    if (x.open) return openResult(x.open);
    if (x.ask) { tab("ask"); ask(x.ask); }
  }

  // ------------------------------------------------------------ Guide: this page
  function page() {
    const p = A.data.page, host = $("#as-page");
    host.innerHTML = `<h4 class="as-h">${esc(p.title)}</h4><p class="as-intro">${esc(p.intro)}</p>
      ${p.actions.length ? `<div class="as-sec">Show me</div><div class="as-acts">${p.actions.map((a, i) => `<button class="ghost small" data-pa="${i}">${esc(a.label)}</button>`).join("")}</div>` : ""}
      ${p.steps && p.steps.length ? `<div class="as-sec">Steps on this page</div><ol class="as-list">${p.steps.map((s) => item(s, A.sa && A.sa.checks[s.id])).join("")}</ol>` : ""}
      <div class="as-sec">Ask about this page</div>
      <div class="as-qs">${p.prompts.map((q, i) => `<button data-pq="${i}">${esc(q)}</button>`).join("")}</div>
      <p class="as-note">Each question goes with the text of this page, so you can ask about the result in front of you.</p>`;
    host.querySelectorAll("[data-pa]").forEach((b) => (b.onclick = () => run(p.actions[+b.dataset.pa].action)));
    host.querySelectorAll("[data-pq]").forEach((b) => (b.onclick = () => { tab("ask"); ask(p.prompts[+b.dataset.pq]); }));
    host.querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => run(p.steps.find((s) => s.id === b.dataset.step).action)));
    host.querySelectorAll("[data-check]").forEach((b) => (b.onclick = () => plan({ check: b.dataset.check })));
  }
  function item(s, playbook) {
    const d = A.data;
    return `<li class="${s.done ? "done" : ""}${s.id === d.next ? " next" : ""}"><span class="as-mark" aria-hidden="true">${s.done ? "✓" : s.id === d.next ? "→" : ""}</span>
      <div><b>${esc(s.title)}</b><small>${esc(s.why)}</small></div><span class="as-2b">${playbook && !s.done ? `<button class="primary small" data-check="${s.id}" title="The superagent drafts the runs; you approve">Do it</button>` : ""}<button class="ghost small" data-step="${s.id}" title="Open the right tab and fill it in">${s.done ? "Again" : "Show me"}</button></span></li>`;
  }
  function chips() {
    const p = A.data.page, host = $("#as-chips");
    const qs = [...p.prompts.slice(0, 2), "What should I do next?"];
    host.innerHTML = qs.map((q, i) => `<button data-cq="${i}">${esc(q)}</button>`).join("") + `<button data-cq="screen">Explain my screen</button>`;
    host.querySelectorAll("[data-cq]").forEach((b) => (b.onclick = () => ask(b.dataset.cq === "screen" ? "Explain what is on my screen, and what it means." : qs[+b.dataset.cq])));
  }

  // ------------------------------------------------------------ page actions (open a tab, fill fields)
  const PRESS = new Set(["ws-plan-demo", "mo-pair", "ww-crafted", "ww-broken", "lw-all", "tu-sl-demo", "tu-rl-demo"]);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  async function until(sel, ms = 2500) {
    for (let t = 0; t < ms; t += 100) { const el = document.querySelector(sel); if (el && !el.disabled) return el; await wait(100); }
    return null;
  }
  async function run(a, quiet) {
    if (!a) return;
    if (a.type === "tour") { toggle(false); window.IsoGuide && window.IsoGuide.start(a.tour); return; }
    if (a.type === "mission") { await plan({ goal: a.goal }); return; }
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
    if (quiet) return;
    if (innerWidth < 1100) toggle(false);
    toast("Ready. Check the fields, then press the button to run it, or ask the superagent to do it.");
  }

  // ------------------------------------------------------------ Ask
  function msgs() {
    const host = $("#as-msgs");
    host.innerHTML = A.chat.length ? A.chat.map((m, i) => m.role === "page" ? `<div class="as-div"><span>${esc(m.text)}</span></div>`
      : `<div class="as-m ${m.role}">${md(m.text)}${(m.actions || []).length
        ? `<div class="as-acts">${m.actions.map((x, j) => `<button class="${x.action && x.action.type === "mission" ? "primary" : "ghost"} small" data-act="${i}:${j}">${x.action && x.action.type === "mission" ? "Plan: " : ""}${esc(x.label)}</button>`).join("")}</div>` : ""}</div>`).join("")
      : `<p class="as-note">Ask about ${esc(A.data ? A.data.page.title : "this page")}: how to start, what a number means, what to try next. Or say what you want done.</p>`;
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

  schedule();
  poll();
  window.IsoAssistant = { open: () => toggle(true), ask, run, where, plan, tab };
  window.Superagent = window.IsoAssistant;
})();
