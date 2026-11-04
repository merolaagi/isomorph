"use strict";
// Guide: an overlay that walks through Isomorph. A spotlight moves to each part of the
// screen, a pointer glides there and shows what to click or type, and a card says what it
// does. One tour of the whole app, and one for each tab. Nothing is run on your behalf:
// the only buttons the guide presses fill in examples.
(function () {
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const nav = (v) => `nav button[data-view="${v}"]`;
  const tab = (v) => () => { const b = document.querySelector(nav(v)); if (b && b.getAttribute("aria-selected") !== "true") b.click(); };
  const mode = (m) => () => { tab("wsp")(); const b = document.querySelector(`#ws-modes button[data-mode="${m}"]`); if (b) b.click(); };

  const TOURS = {
    tour: { title: "Tour of Isomorph", steps: [
      { title: "Welcome to Isomorph", text: "A bench for asking whether two neural networks are the same solution in different coordinates, and for reading what a network's weights do. This tour shows where everything is. Use the arrow keys or the buttons; Esc ends it." },
      { before: tab("trace"), sel: nav("trace"), act: "click", title: "Side by side", text: "Two models, one prompt. Follow the signal layer by layer through both: what each would predict at every depth, where attention goes, which neurons carry it, and which weights would move if you trained on the prompt." },
      { sel: nav("atlas"), act: "point", title: "Pattern library", text: "Profile models into a library: a fingerprint over fixed texts, attention circuits, concept probes and weight spectra, each judged against a random copy of the same architecture." },
      { sel: nav("laws"), act: "point", title: "Laws", text: "Physics and maths tests on raw weights: heavy-tailed spectra, random-matrix edges, level repulsion, Benford, conservation laws. Every law is checked against a Gaussian and a shuffled null." },
      { sel: nav("wsp"), act: "point", title: "Weight space", text: "Can the weights decide how to fine-tune or fix a model? Plan a fine-tune from the weights and the gradient (no key needed), or have Claude read and edit the weights directly, graded where the answer is known." },
      { sel: nav("engine"), act: "point", title: "Engine", text: "Mines rules from everything you have measured, tests the ones that promise a saving, builds a blueprint and writes a world view. Your Anthropic key is saved here, for the world view and for Weight space." },
      { sel: nav("lab"), act: "point", title: "Ground-truth lab", text: "Tiny transformers that learn (a + b) mod p, where the right algorithm is known. Check any tool here first: if it cannot find the Fourier circuits in these, do not trust it on a big model." },
      { sel: nav("hub"), act: "point", title: "Model statistics", text: "Compare two checkpoints from the Hub: weights directly (cosine, spectra, low-rank differences) or activations on the same texts (CKA, neuron matching, stitching)." },
      { sel: "#env", act: "point", title: "Server status", text: "Version, torch and the device in use (Apple GPU shows as mps). Long jobs show a progress bar under the header, with a Stop button; one job runs at a time." },
      { sel: "#iso-guide-btn", act: "click", title: "Guide, any time", text: "Open this for a walkthrough of the tab you are on. Start with Weight space or Side by side: Micro Opus models need no download." },
    ]},
    trace: { title: "Side by side, step by step", steps: [
      { before: tab("trace"), sel: "#tr-a", act: "type", title: "Model A", text: "A Hugging Face name or link, a local folder, a direct link to a .safetensors file, or microopus:words for the 7,440-weight glass-box model. Add @step3000 for a training checkpoint." },
      { sel: "#tr-b", act: "type", title: "Model B", text: "The model to compare against: another seed, an earlier checkpoint, a fine-tune." },
      { sel: "#tr-find", act: "point", title: "Find a model", text: "Search the Hub from inside the app, filter by size, and save favourites to My models." },
      { sel: "#tr-presets", act: "point", title: "Presets", text: "Ready pairs: seeds of one model, early against late checkpoints, a base model against its fine-tune. A preset only fills A and B." },
      { sel: "#tr-prompt", act: "type", title: "One prompt", text: "Both models read the same text. Short prompts trace faster." },
      { sel: "#tr-learn", act: "click", title: "The learning probe", text: "Adds one gradient step on the prompt: which weights would move, by how much, and how low-rank each change is." },
      { sel: "#tr-run", act: "click", title: "Trace prompt", text: "The first run downloads the models; later runs use the cache. Results open on the right, in tabs: Workflow, Architecture, Attention, Neurons, If trained on this, Weight files." },
      { sel: "#mo-pair", act: "point", title: "No download needed", text: "Micro Opus words against names: two fully transparent models. Good for a first look at every view." },
    ]},
    wsp: { title: "Weight space, step by step", steps: [
      { before: mode("plan"), sel: "#ws-modes", act: "point", title: "Three modes", text: "Plan a fine-tune needs no key. Read and Fix use Claude with the key saved under Engine." },
      { sel: "#ws-spec", act: "type", title: "The model", text: "Any causal language model Isomorph can load. Micro Opus (microopus:words) is instant and good for trying things." },
      { sel: "#ws-plan-demo", act: "click", press: true, title: "An example task", text: "This fills in a rule Micro Opus never saw: cats swim and fish fly. Your own examples go one per line: 'prompt => target', or plain text the model should model better." },
      { sel: "#ws-ex", act: "point", title: "Examples", text: "A quarter is set aside as held-out unless you give a separate held-out list. Nothing is judged on the examples it trained on." },
      { sel: "#ws-budget", act: "type", title: "Budget", text: "How many trainable values every plan gets, as a share of the tunable weights. All five plans get the same budget, so only the choice of where to spend it differs." },
      { sel: "#ws-plan button[type=submit]", act: "click", title: "Plan and check", text: "Reads each matrix's spectrum and the gradient, picks matrices two ways, then trains both against three controls (every matrix, last blocks, random matrices). The verdict says whether either pick beat the controls beyond the seeds' spread." },
      { before: mode("read"), sel: "#ws-rtask", act: "click", title: "Read: a quiz", text: "Claude reads the weights through fixed tools and answers. Three quizzes have known answers and are graded: what Micro Opus was trained on with its inputs hidden, a lab model's Fourier frequencies, a model's induction heads. Or ask your own question." },
      { sel: "#ws-rfb", act: "type", title: "Forward passes", text: "0 means from the weights alone. A few passes let Claude look at attention on texts it writes." },
      { before: mode("fix"), sel: "#ws-ftask", act: "click", title: "Fix: edit the weights", text: "'Find and remove a planted fault' plants a known fault and measures how much of the damage Claude repairs. 'Your goal' takes your own examples. Edits are kept only if practice improves, then judged on held-out examples against the same edits in random directions." },
      { sel: "#ws-key", act: "point", title: "The key", text: "Read and Fix call Claude with your Anthropic key, stored only on this Mac (Engine tab)." },
      { before: mode("write"), sel: "#ww-plot", act: "point", title: "Write: a network you can see whole", text: "From Weightspace Lab: a 2 → H → 1 network drawn as its decision landscape. White dots are grid points where it disagrees with the target. It starts broken: one output weight has the wrong sign." },
      { sel: "#ww-sliders", act: "point", title: "Edit the weights", text: "Drag a slider or type a number; the landscape and the agreement update at once. Apply crafted weights shows a hand-written solution; exports load straight into Weightspace Lab and back." },
      { sel: "#ws-write button[type=submit]", act: "click", title: "Ask Claude to write it", text: "Claude gets the rule in words and writes every weight, with no data and no training, or finds the fault in the broken network. It is graded on the same grid, against random weights and against the same network trained on 400 labelled points." },
      { before: mode("plan"), sel: "#ws-history", act: "point", title: "History", text: "Every plan, reading and fix is kept here with its full transcript. Saved checkpoints from Deep Network Designer arrive through its 'read' button and show up in the model list." },
    ]},
    atlas: { title: "Pattern library, step by step", steps: [
      { before: tab("atlas"), sel: "#at-spec", act: "type", title: "A model to profile", text: "Any Hugging Face model, link or local folder." },
      { sel: "#at-base", act: "click", title: "Random baseline", text: "Profiles a randomly initialised copy too. A pattern only counts if the trained model clearly beats it. Leave it on." },
      { sel: "#at-go", act: "click", title: "Profile and add", text: "Runs the fingerprint, circuit tests, concept probes and weight signature, then adds the model to the library." },
      { sel: "#at-list", act: "point", title: "The library", text: "With a few models in, it shows which patterns are shared and how close each model is to the consensus at every depth. Add the model you are curious about last." },
    ]},
    laws: { title: "Laws, step by step", steps: [
      { before: tab("laws"), sel: "#lw-cands", act: "click", title: "Pick models", text: "Micro Opus and lab models need no download. Tick a randomly initialised model as a control if you have one: emergent laws should fail on it." },
      { sel: "#lw-extra", act: "type", title: "More models", text: "One Hugging Face name per line." },
      { sel: "#lw-go", act: "click", title: "Run the laws", text: "Weights only, nothing is run on text, so any architecture works. Each law is repeated on a Gaussian and a shuffled null." },
      { sel: "#lw-stage", act: "point", title: "The table", text: "A tick means the law holds for that model; the grey number under each value is the null. Click a model for its spectra." },
    ]},
    engine: { title: "Engine, step by step", steps: [
      { before: tab("engine"), sel: "#en-mine", act: "click", title: "Mine rules", text: "Turns everything measured so far into rules, each with support, counterexamples and a confidence grade." },
      { sel: "#en-key", act: "type", title: "Your Anthropic key", text: "Optional for the world view, needed for Weight space's Read and Fix. It stays in data/settings.json on this Mac." },
      { sel: "#en-stage", act: "point", title: "Autopilot, rules, experiments", text: "Autopilot runs the whole loop unattended: profiles models, mines rules, tests the most informative ones, builds a blueprint and writes the world view." },
    ]},
    lab: { title: "Ground-truth lab, step by step", steps: [
      { before: tab("lab"), sel: "#train-form", act: "point", title: "A family", text: "Several tiny transformers on (a + b) mod p that differ only in their seed. The default (p = 53, four seeds) takes a few minutes." },
      { sel: "#train-form button[type=submit]", act: "click", title: "Train family", text: "Watch test accuracy jump: that is when a model stops memorising and finds the algorithm." },
      { sel: "#runs", act: "point", title: "Runs", text: "Open a run, then compare any two seeds: raw weights, after removing symmetries, and after relabelling by the task's own symmetry." },
    ]},
    hub: { title: "Model statistics, step by step", steps: [
      { before: tab("hub"), sel: "#hub-a", act: "type", title: "Two checkpoints", text: "Any Hugging Face checkpoints, written repo, repo::subfolder or repo@revision." },
      { sel: "#hub-weights", act: "click", title: "Compare weights", text: "Reads the files directly, so it works for custom architectures: cosine, singular-value shape and whether the difference is low-rank." },
      { sel: "#hub-acts", act: "click", title: "Compare activations", text: "Runs both on the same texts: layer-by-layer CKA, one-to-one neuron matching, next-token agreement and stitching." },
      { sel: "#presets", act: "point", title: "Presets", text: "Seeds, data order, initialisation and fine-tunes, ready to compare." },
    ]},
    about: { title: "How it works", steps: [
      { before: tab("about"), sel: "#view-about", act: "point", title: "How it works", text: "The method behind every view, in plain words: which symmetries are removed, what each number means, and what each control rules out." },
    ]},
  };

  // ------------------------------------------------------------ overlay
  let G = null;
  function build() {
    const root = document.createElement("div");
    root.className = "guide";
    root.innerHTML = `<div class="guide-spot"></div><div class="guide-ptr" aria-hidden="true"><svg viewBox="0 0 24 24" width="26" height="26"><path d="M4 2l15 11-7 1 4 8-3 1-4-8-5 5z"/></svg><span class="guide-ring"></span></div>
      <div class="guide-card" role="dialog" aria-live="polite"><div class="guide-top"><span class="guide-tt"></span><button class="guide-x" aria-label="End the guide">×</button></div>
        <h4></h4><p></p><div class="guide-act"></div>
        <div class="guide-bottom"><span class="guide-dots"></span><span class="guide-btns"><button class="ghost small guide-back">Back</button><button class="primary guide-next">Next</button></span></div></div>`;
    document.body.appendChild(root);
    root.querySelector(".guide-x").onclick = end;
    root.querySelector(".guide-back").onclick = () => go(G.i - 1);
    root.querySelector(".guide-next").onclick = () => (G.i + 1 < G.steps.length ? go(G.i + 1) : end());
    return root;
  }

  function start(name) {
    const t = TOURS[name] || TOURS.tour;
    if (G) end();
    G = { name, steps: t.steps, title: t.title, i: 0, root: build(), ptr: { x: innerWidth / 2, y: innerHeight / 2 } };
    document.addEventListener("keydown", keys);
    addEventListener("resize", place);
    addEventListener("scroll", place, true);
    requestAnimationFrame(() => G.root.classList.add("on"));
    go(0);
  }

  function end() {
    if (!G) return;
    document.removeEventListener("keydown", keys);
    removeEventListener("resize", place);
    removeEventListener("scroll", place, true);
    const r = G.root;
    r.classList.remove("on");
    setTimeout(() => r.remove(), reduce ? 0 : 250);
    G = null;
    try { localStorage.setItem("isomorph-guide-seen", "1"); } catch { /* private window */ }
  }

  function keys(e) {
    if (!G) return;
    if (e.key === "Escape") end();
    else if (e.key === "ArrowRight" || (e.key === "Enter" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName))) { e.preventDefault(); G.root.querySelector(".guide-next").click(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); if (G.i > 0) go(G.i - 1); }
  }

  function target() {
    const s = G.steps[G.i];
    if (!s.sel) return null;
    const el = document.querySelector(s.sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return r.width || r.height ? el : null;
  }

  function go(i) {
    if (!G || i < 0 || i >= G.steps.length) return;
    G.i = i;
    const s = G.steps[i];
    if (s.before) try { s.before(); } catch { /* the tab may not be ready */ }
    setTimeout(() => {
      if (!G) return;
      const el = target();
      if (el) el.scrollIntoView({ block: "center", inline: "nearest", behavior: reduce ? "auto" : "smooth" });
      setTimeout(() => { if (G) { place(); animate(el, s); } }, reduce ? 0 : 320);
    }, s.before ? 120 : 0);
    const c = G.root.querySelector(".guide-card");
    c.querySelector(".guide-tt").textContent = `${G.title} · ${i + 1} of ${G.steps.length}`;
    c.querySelector("h4").textContent = s.title;
    c.querySelector("p").textContent = s.text;
    c.querySelector(".guide-act").textContent = s.act === "click" ? (s.press ? "The guide presses this one for you." : "Click here when you are ready.")
      : s.act === "type" ? "Type here." : "";
    c.querySelector(".guide-dots").innerHTML = G.steps.map((_, k) => `<i class="${k === i ? "on" : k < i ? "done" : ""}"></i>`).join("");
    c.querySelector(".guide-back").disabled = i === 0;
    c.querySelector(".guide-next").textContent = i + 1 < G.steps.length ? "Next" : "Done";
    c.classList.remove("fresh"); void c.offsetWidth; c.classList.add("fresh");
  }

  function place() {
    if (!G) return;
    const spot = G.root.querySelector(".guide-spot"), card = G.root.querySelector(".guide-card");
    const el = target(), pad = 8;
    if (!el) {
      Object.assign(spot.style, { left: `${innerWidth / 2}px`, top: `${innerHeight / 2}px`, width: "0px", height: "0px" });
      const cw0 = Math.min(420, innerWidth - 32);
      Object.assign(card.style, { left: `${(innerWidth - cw0) / 2}px`, top: `${Math.max(16, innerHeight / 2 - 120)}px`, width: `${cw0}px` });
      return;
    }
    const r = el.getBoundingClientRect();
    const w = Math.min(r.width + 2 * pad, innerWidth - 8), h = Math.min(r.height + 2 * pad, innerHeight - 8);
    Object.assign(spot.style, { left: `${Math.max(4, r.left - pad)}px`, top: `${Math.max(4, r.top - pad)}px`, width: `${w}px`, height: `${h}px` });
    const cw = Math.min(380, innerWidth - 32), ch = card.offsetHeight || 200;
    let left, top;
    if (r.right + 16 + cw < innerWidth) { left = r.right + 16; top = r.top; }               // right of it
    else if (r.bottom + 16 + ch < innerHeight) { left = r.left; top = r.bottom + 16; }       // below
    else if (r.top - 16 - ch > 0) { left = r.left; top = r.top - 16 - ch; }                  // above
    else { left = r.left - 16 - cw; top = r.top; }                                           // left
    left = Math.min(Math.max(16, left), innerWidth - cw - 16);
    top = Math.min(Math.max(16, top), innerHeight - ch - 16);
    Object.assign(card.style, { left: `${left}px`, top: `${top}px`, width: `${cw}px` });
    const ptr = G.root.querySelector(".guide-ptr");
    if (ptr.classList.contains("show")) ptr.style.transform = `translate(${pointAt(r).join("px, ")}px)`;
  }

  function pointAt(r) {
    return [r.left + Math.min(r.width / 2, 60), r.top + Math.min(r.height / 2, 22)];
  }

  function animate(el, s) {
    const ptr = G.root.querySelector(".guide-ptr");
    if (!el || !s.act) { ptr.classList.remove("show"); return; }
    const [x, y] = pointAt(el.getBoundingClientRect());
    ptr.classList.remove("click", "type");
    ptr.classList.add("show");
    ptr.style.transform = `translate(${x}px, ${y}px)`;
    clearTimeout(G.t1); clearTimeout(G.t2);
    G.t1 = setTimeout(() => {
      if (!G) return;
      if (s.act === "click") {
        ptr.classList.add("click");
        el.classList.add("guide-pulse");
        G.t2 = setTimeout(() => el.classList.remove("guide-pulse"), 1200);
        if (s.press) el.click();
      } else if (s.act === "type") ptr.classList.add("type");
    }, reduce ? 0 : 650);
  }

  // ------------------------------------------------------------ launcher
  function launcher() {
    const env = document.getElementById("env");
    if (!env || document.getElementById("iso-guide-btn")) return;
    const wrap = document.createElement("div");
    wrap.className = "guide-launch";
    wrap.innerHTML = `<button class="ghost small" id="iso-guide-btn" aria-haspopup="menu" aria-expanded="false">Guide</button>
      <div class="guide-menu" role="menu" hidden>
        <button role="menuitem" data-tour="here">How to use this tab</button>
        <button role="menuitem" data-tour="tour">Tour of Isomorph</button>
        <button role="menuitem" data-tour="wsp">Weight space walkthrough</button>
      </div>`;
    env.parentNode.insertBefore(wrap, env.nextSibling);
    const btn = wrap.querySelector("#iso-guide-btn"), menu = wrap.querySelector(".guide-menu");
    btn.onclick = (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; btn.setAttribute("aria-expanded", String(!menu.hidden)); };
    document.addEventListener("click", () => { menu.hidden = true; btn.setAttribute("aria-expanded", "false"); });
    menu.querySelectorAll("[data-tour]").forEach((b) => (b.onclick = () => {
      menu.hidden = true;
      let t = b.dataset.tour;
      if (t === "here") {
        const cur = document.querySelector('nav button[aria-selected="true"]');
        t = cur ? cur.dataset.view : "tour";
      }
      start(t);
    }));
  }

  launcher();
  window.IsoGuide = { start, end, tours: TOURS };
  let seen = false;
  try { seen = !!localStorage.getItem("isomorph-guide-seen"); } catch { seen = true; }
  if (!seen && !location.hash) setTimeout(() => start("tour"), 900);
})();
