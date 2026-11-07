"use strict";
// The docs layout: the job bar moves into the header, each tab's left sidebar gets the tab's name,
// "How it works" gets its sections on the left, and every page gets "On this page" on the right,
// built from the headings of whatever is on screen and following the scroll.
(function () {
  const header = document.querySelector("header.top");
  const root = document.documentElement;
  const job = document.getElementById("jobbar");
  if (job && header) header.appendChild(job);

  // header height, for the sticky sidebars
  const setH = () => root.style.setProperty("--hdr", `${header.getBoundingClientRect().height}px`);
  setH();
  if (window.ResizeObserver) new ResizeObserver(setH).observe(header);

  const views = [...document.querySelectorAll("main > .view")];
  const tabName = (v) => { const b = document.querySelector(`nav button[data-view="${v.id.replace("view-", "")}"]`); return b ? b.textContent.trim() : ""; };
  const slug = (t) => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "section";

  for (const v of views) {
    let rail = v.querySelector(":scope > .rail");
    const prose = v.querySelector(":scope > .prose");
    if (!rail && prose) {                       // a page of text: its sections become the left nav
      rail = document.createElement("aside");
      rail.className = "rail";
      rail.innerHTML = `<ul class="md-rail-nav"></ul>`;
      v.insertBefore(rail, prose);
    }
    if (rail && !rail.querySelector(".md-nav-title")) {
      const t = document.createElement("div");
      t.className = "md-nav-title";
      t.textContent = tabName(v);
      rail.prepend(t);
    }
    const toc = document.createElement("aside");
    toc.className = "md-toc";
    toc.setAttribute("aria-label", "On this page");
    v.appendChild(toc);
  }

  // ------------------------------------------------------------ "On this page"
  const active = () => views.find((v) => !v.hidden);
  function headings(v) {
    const host = v.querySelector(":scope > .stage") || v.querySelector(":scope > .prose");
    if (!host) return [];
    const sel = host.classList.contains("prose") ? "h1, h2" : ".head h1, .stage > h1, .stage > h2, .panel > h3, .intro .lede";
    return [...host.querySelectorAll(sel)].filter((h) => h.offsetParent !== null && h.textContent.trim());
  }
  const S = { built: null, items: [] };
  // ------------------------------------------------------------ Copy as Markdown / View as Markdown
  const ICO_COPY = '<svg viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg>';
  const ICO_MD = '<svg viewBox="0 0 24 24"><rect x="2.5" y="5" width="19" height="14" rx="2"/><path d="M6 15V9l2.5 3L11 9v6M15 9v6m0 0-2-2m2 2 2-2"/></svg>';
  function inline(n) {
    if (n.nodeType === 3) return n.textContent.replace(/\s+/g, " ");
    if (n.nodeType !== 1 || skip(n)) return "";
    const t = n.tagName, inner = [...n.childNodes].map(inline).join("");
    if (t === "B" || t === "STRONG") return inner.trim() ? `**${inner.trim()}**` : "";
    if (t === "I" || t === "EM") return inner.trim() ? `*${inner.trim()}*` : "";
    if (t === "CODE") return `\`${n.textContent}\``;
    if (t === "BR") return "\n";
    if (t === "A" && n.getAttribute("href") && /^https?:/.test(n.getAttribute("href"))) return `[${inner.trim()}](${n.getAttribute("href")})`;
    return inner;
  }
  function skip(n) {
    return ["BUTTON", "INPUT", "SELECT", "TEXTAREA", "SCRIPT", "STYLE", "CANVAS"].includes(n.tagName) || n.classList.contains("md-doc-tools") ||
      n.hidden || n.getAttribute("aria-hidden") === "true";
  }
  function md(node, out) {
    for (const n of node.childNodes) {
      if (n.nodeType === 3) { const t = n.textContent.trim(); if (t) out.push(t); continue; }
      if (n.nodeType !== 1 || skip(n)) continue;
      const t = n.tagName;
      if (/^H[1-6]$/.test(t)) out.push("", "#".repeat(Math.min(+t[1] === 3 ? 2 : +t[1], 6)) + " " + n.textContent.trim().replace(/\s+/g, " "), "");
      else if (t === "P" || n.classList.contains("finding") || n.classList.contains("sub") || n.classList.contains("meta")) { const x = inline(n).trim(); if (x) out.push(x, ""); }
      else if (t === "TABLE") {
        const rows = [...n.querySelectorAll("tr")].map((r) => [...r.children].map((c) => inline(c).trim().replace(/\|/g, "\\|").replace(/\n/g, " ")));
        if (!rows.length) continue;
        const w = Math.max(...rows.map((r) => r.length));
        const pad = (r) => [...r, ...Array(w - r.length).fill("")];
        out.push("", `| ${pad(rows[0]).join(" | ")} |`, `| ${Array(w).fill("---").join(" | ")} |`, ...rows.slice(1).map((r) => `| ${pad(r).join(" | ")} |`), "");
      } else if (t === "UL" || t === "OL") {
        [...n.children].forEach((li, i) => out.push(`${t === "OL" ? `${i + 1}.` : "-"} ${inline(li).trim()}`));
        out.push("");
      } else if (t === "PRE") out.push("", "```", n.textContent.replace(/\n$/, ""), "```", "");
      else if (t === "svg") { const lab = n.getAttribute("aria-label"); out.push(`*[chart${lab && lab !== "chart" ? `: ${lab}` : ""}]*`, ""); }
      else if (t === "DETAILS") { const s = n.querySelector("summary"); if (s) out.push(`**${s.textContent.trim()}**`, ""); md(n, out); }
      else if (t === "SUMMARY") continue;
      else md(n, out);
    }
    return out;
  }
  function toMarkdown(v) {
    const host = v.querySelector(":scope > .stage") || v.querySelector(":scope > .prose");
    const text = md(host, []).join("\n").replace(/\n{3,}/g, "\n\n").trim();
    return `<!-- Isomorph · ${tabName(v)} · ${new Date().toISOString().slice(0, 16).replace("T", " ")} -->\n\n${text}\n`;
  }
  const viewer = document.createElement("div");
  viewer.className = "md-view";
  viewer.hidden = true;
  viewer.innerHTML = `<div role="dialog" aria-label="Page as Markdown"><div class="md-vh"><span>Markdown</span><button class="ghost small" data-c>Copy</button><button class="ghost small" data-d>Download .md</button><button class="ghost small" data-x>Close</button></div><pre></pre></div>`;
  document.body.appendChild(viewer);
  viewer.onclick = (e) => { if (e.target === viewer || e.target.closest("[data-x]")) viewer.hidden = true; };
  viewer.querySelector("[data-c]").onclick = () => copy(viewer.querySelector("pre").textContent);
  viewer.querySelector("[data-d]").onclick = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([viewer.querySelector("pre").textContent], { type: "text/markdown" }));
    a.download = `${slug(viewer.dataset.title || "isomorph")}.md`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !viewer.hidden) viewer.hidden = true; });
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); }
    catch { const t = document.createElement("textarea"); t.value = text; document.body.appendChild(t); t.select(); document.execCommand("copy"); t.remove(); }
    toast("Copied as Markdown.");
  }
  function tools(v) {
    const host = v.querySelector(":scope > .stage") || v.querySelector(":scope > .prose");
    if (!host) return;
    let bar = host.querySelector(":scope > .md-doc-tools");
    const empty = !host.querySelector("h1, .lede, .panel, table, p");
    if (empty) { if (bar) bar.remove(); return; }   // never leave the area non-empty: tabs draw their intro into an empty one
    if (!bar) {
      bar = document.createElement("div");
      bar.className = "md-doc-tools";
      bar.innerHTML = `<button type="button" data-copy>${ICO_COPY}Copy as Markdown</button><button type="button" data-mdview>${ICO_MD}View as Markdown</button>`;
      bar.querySelector("[data-copy]").onclick = () => copy(toMarkdown(v));
      bar.querySelector("[data-mdview]").onclick = () => {
        viewer.querySelector("pre").textContent = toMarkdown(v);
        const h = host.querySelector("h1, .lede");
        viewer.dataset.title = h ? h.textContent.trim() : tabName(v);
        viewer.querySelector(".md-vh span").textContent = `${viewer.dataset.title} · Markdown`;
        viewer.hidden = false;
      };
      host.prepend(bar);
    }
  }

  function build() {
    const v = active();
    if (!v) return;
    tools(v);
    const toc = v.querySelector(":scope > .md-toc");
    const hs = headings(v);
    const used = new Set();
    S.items = hs.map((h) => {
      if (!h.id || used.has(h.id)) {
        let id = `${v.id.replace("view-", "")}-${slug(h.textContent)}`, k = 2;
        while (used.has(id) || (document.getElementById(id) && document.getElementById(id) !== h)) id = `${id.replace(/-\d+$/, "")}-${k++}`;
        h.id = id;
      }
      used.add(h.id);
      const sub = h.tagName === "H2" && v.id === "view-about";
      return { h, id: h.id, text: h.textContent.trim().replace(/\s+/g, " ").slice(0, 90), sub };
    });
    const sig = v.id + S.items.map((x) => x.id).join("|");
    if (sig !== S.built) {
      S.built = sig;
      toc.innerHTML = S.items.length > 1 ? `<p class="md-toc-t">On this page</p><ul>${S.items.map((x) => `<li class="${x.sub ? "sub" : ""}"><a href="#${x.id}" data-id="${x.id}">${esc(x.text)}</a></li>`).join("")}</ul>` : "";
      toc.querySelectorAll("a").forEach((a) => (a.onclick = (e) => { e.preventDefault(); jump(a.dataset.id); }));
      if (v.id === "view-about") {
        const nav = v.querySelector(".md-rail-nav");
        const tops = S.items.filter((x) => x.h.tagName === "H1");
        nav.innerHTML = tops.map((x) => `<li><a href="#${x.id}" data-id="${x.id}">${esc(x.text)}</a></li>`).join("");
        nav.querySelectorAll("a").forEach((a) => (a.onclick = (e) => { e.preventDefault(); jump(a.dataset.id); }));
      }
    }
    spy();
  }
  function jump(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }
  function spy() {
    const v = active();
    if (!v || !S.items.length) return;
    const line = parseFloat(getComputedStyle(root).getPropertyValue("--hdr")) + 40;
    let cur = S.items[0];
    for (const x of S.items) if (x.h.getBoundingClientRect().top <= line) cur = x;
    if (innerHeight + scrollY >= document.body.scrollHeight - 4) cur = S.items[S.items.length - 1];
    v.querySelectorAll(".md-toc a, .md-rail-nav a").forEach((a) => a.classList.toggle("on", a.dataset.id === cur.id));
  }
  let raf = 0;
  addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; spy(); }); }, { passive: true });

  let t = 0;
  const soon = () => { clearTimeout(t); t = setTimeout(build, 200); };
  const mo = new MutationObserver(soon);
  for (const v of views) {
    const host = v.querySelector(":scope > .stage") || v.querySelector(":scope > .prose");
    if (host) mo.observe(host, { childList: true, subtree: true });
    new MutationObserver(soon).observe(v, { attributes: true, attributeFilter: ["hidden"] });
  }
  document.addEventListener("click", (e) => { if (e.target.closest && e.target.closest("nav button[data-view]")) { scrollTo(0, 0); soon(); } }, true);
  build();
  window.IsoLayout = { rebuild: build, jump, markdown: () => toMarkdown(active()) };
})();
