(function () {
// Minimal SVG charts. No dependencies, so the app works offline.
const NS = "http://www.w3.org/2000/svg";
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function hexToRgb(h) {
  h = h.replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function mix(h1, h2, t) {
  const a = hexToRgb(h1), b = hexToRgb(h2);
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
}
function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 1e5)) return v.toExponential(1);
  return (+v).toFixed(d);
}
function niceTicks(lo, hi, n = 4) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function decimals(ticks) {
  if (ticks.length < 2) return 2;
  const step = Math.abs(ticks[1] - ticks[0]);
  let d = 0;
  while (d < 4 && Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) > 1e-6) d++;
  return d;
}
function lineChart(host, o) {
  o = { ...o, series: o.series.map((s) => ({ ...s, points: s.points.filter((p) => p && Number.isFinite(p[0]) && Number.isFinite(p[1])) })) };
  const W = o.width || 560, H = o.height || 230, m = { l: 44, r: 12, t: 10, b: 34 };
  const pts = o.series.flatMap((s) => s.points);
  if (!pts.length) { host.textContent = "No data."; return; }
  const xMin = o.xMin ?? Math.min(...pts.map((p) => p[0])), xMax = o.xMax ?? Math.max(...pts.map((p) => p[0]));
  const yMin = o.yMin ?? Math.min(...pts.map((p) => p[1]));
  let yMax = o.yMax ?? Math.max(...pts.map((p) => p[1]));
  if (o.yMax === undefined) { const t = niceTicks(yMin, yMax); if (t.length > 1 && t[t.length - 1] < yMax - 1e-12) yMax = t[t.length - 1] + (t[1] - t[0]); }
  const X = (v) => m.l + ((v - xMin) / (xMax - xMin || 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - yMin) / (yMax - yMin || 1)) * (H - m.t - m.b);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": o.label || "chart" });
  const yt = niceTicks(yMin, yMax), yd = decimals(yt);
  for (const t of yt) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }, svg);
    svgEl("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end" }, svg).textContent = o.yFmt ? o.yFmt(t) : (+t).toFixed(yd);
  }
  const intX = pts.every((p) => Number.isInteger(p[0]));
  let xt = niceTicks(xMin, xMax, 5);
  if (intX) xt = xt.filter((t) => Number.isInteger(t));
  const xd = intX ? 0 : decimals(xt);
  for (const t of xt) {
    svgEl("text", { x: X(t), y: H - m.b + 15, "text-anchor": "middle" }, svg).textContent = o.xFmt ? o.xFmt(t) : (+t).toFixed(xd);
  }
  svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
  if (o.xLabel) svgEl("text", { x: (W + m.l) / 2, y: H - 3, "text-anchor": "middle" }, svg).textContent = o.xLabel;
  if (o.yLabel) svgEl("text", { x: 10, y: m.t + (H - m.t - m.b) / 2, transform: `rotate(-90 10 ${m.t + (H - m.t - m.b) / 2})`, "text-anchor": "middle" }, svg).textContent = o.yLabel;
  for (const s of o.series) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round", opacity: s.opacity ?? 1 }, svg);
    if (s.dots) for (const p of s.points) svgEl("circle", { cx: X(p[0]), cy: Y(p[1]), r: 3, fill: s.color }, svg);
  }
  host.innerHTML = "";
  host.appendChild(svg);
  if (o.legend !== false) host.appendChild(legend(o.series.filter((s) => s.name)));
}

function legend(items) {
  const d = document.createElement("div");
  d.className = "legend";
  for (const s of items) {
    const sp = document.createElement("span");
    sp.style.setProperty("--c", s.color);
    sp.textContent = s.name;
    d.appendChild(sp);
  }
  return d;
}

function heatmap(host, o) {
  const rows = o.matrix.length, cols = o.matrix[0].length;
  const cell = o.cell || Math.max(10, Math.min(46, Math.floor(520 / Math.max(rows, cols))));
  const lw = o.rowLabelWidth ?? 110, th = o.colLabelHeight ?? 70;
  const W = lw + cols * cell + 8, H = th + rows * cell + 8;
  const lo = o.min ?? 0, hi = o.max ?? 1;
  const c0 = css("--soft") || "#e9edf2", c1 = o.color || css("--ab") || "#6b3fb3";
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", style: `max-width:${W}px`, role: "img", "aria-label": o.label || "heatmap" });
  o.matrix.forEach((r, i) => r.forEach((v, j) => {
    const t = Number.isFinite(v) ? Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1))) : 0;
    const rect = svgEl("rect", { x: lw + j * cell, y: th + i * cell, width: cell - 1.5, height: cell - 1.5, rx: 2, fill: mix(c0.startsWith("#") ? c0 : "#e9edf2", c1, t) }, svg);
    svgEl("title", {}, rect).textContent = `${o.rows[i]} × ${o.cols[j]}: ${fmt(v, 3)}`;
    if (o.values && cell >= 30) {
      svgEl("text", { x: lw + j * cell + cell / 2 - 1, y: th + i * cell + cell / 2 + 3.5, "text-anchor": "middle", style: `fill:${t > 0.55 ? "#fff" : "var(--ink)"};font-size:10.5px` }, svg).textContent = fmt(v, 2);
    }
  }));
  const step = (n) => Math.max(1, Math.ceil((n * 12) / (n * cell)));
  o.rows.forEach((l, i) => { if (i % step(rows) === 0) svgEl("text", { x: lw - 6, y: th + i * cell + cell / 2 + 3.5, "text-anchor": "end" }, svg).textContent = l; });
  o.cols.forEach((l, j) => {
    if (j % step(cols) === 0) {
      const x = lw + j * cell + cell / 2, y = th - 6;
      svgEl("text", { x, y, transform: `rotate(-45 ${x} ${y})`, "text-anchor": "start" }, svg).textContent = l;
    }
  });
  if (o.rowTitle) svgEl("text", { x: 0, y: 12, style: `fill:${css("--a")};font-weight:600` }, svg).textContent = o.rowTitle;
  if (o.colTitle) svgEl("text", { x: lw, y: 12, style: `fill:${css("--b")};font-weight:600` }, svg).textContent = o.colTitle;
  host.innerHTML = "";
  host.appendChild(svg);
}

function groupedBars(host, o) {
  const W = o.width || 560, H = o.height || 220, m = { l: 44, r: 10, t: 10, b: o.rotate ? 70 : 34 };
  const n = o.labels.length, k = o.series.length;
  const all = o.series.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined);
  const yMin = o.yMin ?? Math.min(0, ...all);
  let yMax = o.yMax ?? Math.max(...all);
  if (o.yMax === undefined) { const t = niceTicks(yMin, yMax); if (t.length > 1 && t[t.length - 1] < yMax - 1e-12) yMax = t[t.length - 1] + (t[1] - t[0]); }
  const Y = (v) => H - m.b - ((v - yMin) / (yMax - yMin || 1)) * (H - m.t - m.b);
  const gw = (W - m.l - m.r) / n, bw = Math.max(2, (gw * 0.78) / k);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": o.label || "bar chart" });
  const bt = niceTicks(yMin, yMax), bd = decimals(bt);
  for (const t of bt) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }, svg);
    svgEl("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end" }, svg).textContent = o.yFmt ? o.yFmt(t) : (+t).toFixed(bd);
  }
  o.labels.forEach((lab, i) => {
    o.series.forEach((s, j) => {
      const v = s.values[i];
      if (v === null || v === undefined) return;
      const x = m.l + i * gw + gw * 0.11 + j * bw;
      const y0 = Y(Math.max(yMin, 0)), y1 = Y(v);
      const r = svgEl("rect", { x, y: Math.min(y0, y1), width: bw - 1, height: Math.max(1, Math.abs(y1 - y0)), fill: s.color, rx: 1.5 }, svg);
      svgEl("title", {}, r).textContent = `${s.name || ""} ${lab}: ${fmt(v, 3)}`;
    });
    const every = Math.ceil(n / 12);
    if (i % every === 0) {
      const x = m.l + i * gw + gw / 2, y = H - m.b + 14;
      svgEl("text", o.rotate ? { x, y, transform: `rotate(40 ${x} ${y})`, "text-anchor": "start" } : { x, y, "text-anchor": "middle" }, svg).textContent = lab;
    }
  });
  svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(Math.max(yMin, 0)), y2: Y(Math.max(yMin, 0)), class: "axis" }, svg);
  if (o.yLabel) svgEl("text", { x: 10, y: m.t + (H - m.t - m.b) / 2, transform: `rotate(-90 10 ${m.t + (H - m.t - m.b) / 2})`, "text-anchor": "middle" }, svg).textContent = o.yLabel;
  host.innerHTML = "";
  host.appendChild(svg);
  if (o.legend !== false) host.appendChild(legend(o.series.filter((s) => s.name)));
}

function sparkPair(a, b, w = 120, h = 28) {
  const all = [...a, ...b].filter((v) => v > 0);
  const lo = Math.log10(Math.min(...all)), hi = Math.log10(Math.max(...all));
  const path = (v) => v.map((x, i) => `${i ? "L" : "M"}${((i / (v.length - 1 || 1)) * (w - 2) + 1).toFixed(1)},${(h - 2 - ((Math.log10(Math.max(x, 1e-12)) - lo) / (hi - lo || 1)) * (h - 4)).toFixed(1)}`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-label="singular values"><path d="${path(a)}" fill="none" stroke="var(--a)" stroke-width="1.5"/><path d="${path(b)}" fill="none" stroke="var(--b)" stroke-width="1.5" stroke-dasharray="3 2"/></svg>`;
}

window.Charts = { lineChart, heatmap, groupedBars, sparkPair, legend, fmt, css, mix };
})();
