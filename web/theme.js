"use strict";
// Light or dark. Follows the system until you choose; the choice is remembered on this browser,
// and shared with the Micro Opus page (same key), so both look the same.
(function () {
  const KEY = "mo-theme";
  const get = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  const set = (v) => { try { localStorage.setItem(KEY, v); } catch { /* private window */ } };
  const root = document.documentElement;
  const saved = get();
  if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  const isDark = () => (root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches);
  const SUN = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  const MOON = '<svg viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>';

  function paint(btn) {
    const dark = isDark();
    btn.innerHTML = dark ? SUN : MOON;
    btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
    btn.setAttribute("aria-label", btn.title);
  }

  function mount() {
    const env = document.getElementById("env");
    if (!env || document.getElementById("theme-btn")) return;
    const btn = document.createElement("button");
    btn.id = "theme-btn";
    btn.className = "ghost small theme-btn";
    env.parentNode.insertBefore(btn, env.nextSibling);
    paint(btn);
    btn.onclick = () => {
      root.dataset.theme = isDark() ? "light" : "dark";
      set(root.dataset.theme);
      paint(btn);
      document.dispatchEvent(new CustomEvent("isomorph-theme", { detail: root.dataset.theme }));
    };
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => paint(btn));
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
