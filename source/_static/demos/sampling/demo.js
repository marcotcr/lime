// The LIME Stratified demo (demos/sampling/index.rst): why LIME Image's Monte Carlo sampling
// misses the neighbourhood of the explained photo, and what stratified sampling changes.
//
// - "Where samples land" is computed here, from the formulas of the paper: the distribution of
//   the number of preserved superpixels |x| under B(0.5) and under stratified sampling, with
//   masks drawn live, and the adjustment factor adj(i).
// - "The neighbourhood" and "The explanation" replay real runs, precomputed by ../compute
//   (explain.py logs what the library's unchanged explain_instance fed to its regressor, pack.py
//   writes data.js and one file per photo): the masks, the model's scores, the sample weights and
//   the coefficients, for Monte Carlo and stratified sampling and four segmentations.
//
// Without JavaScript, the <noscript> figure shows; when the data does not load, the script shows it.
(() => {
  const root = document.querySelector(".limestrat-demo");
  if (!root) return;
  const fallback = root.querySelector("noscript")?.textContent ?? "";
  const fail = (error) => {
    if (error) console.error("The LIME Stratified demo could not start:", error);
    root.classList.remove("limestrat-demo-ready");
    root.classList.add("limestrat-demo-failed");
    root.innerHTML = fallback;
  };

  const S = 224; // the photos' size
  const Z = 2; // canvas pixels per photo pixel
  const W = S * Z;
  const SVG = "http://www.w3.org/2000/svg";
  const POS = [214, 39, 40];
  const NEG = [31, 119, 180];
  const METHODS = {
    mc: { name: "Monte Carlo", title: "Monte Carlo sampling", cls: "limestrat-demo-mc" },
    st: { name: "Stratified", title: "Stratified sampling", cls: "limestrat-demo-st" },
  };
  const NEAR = 0.9; // "close to the photo": at least 90% of the superpixels kept

  const begin = () => {
    try {
      start(window.LIMESTRAT_DEMO);
    } catch (error) {
      fail(error);
    }
  };
  if (window.LIMESTRAT_DEMO) queueMicrotask(begin);
  else {
    const loader = document.createElement("script");
    loader.src = `${root.dataset.root ?? ""}data.js?v=${root.dataset.version ?? ""}`;
    loader.onload = begin;
    loader.onerror = () => fail();
    document.head.appendChild(loader);
  }

  // ---------------------------------------------------------------------------------------------
  // small helpers

  function el(tag, cls, ...children) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    for (const c of children) if (c != null) node.append(c);
    return node;
  }
  function svg(tag, attrs = {}, ...children) {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    for (const c of children) if (c != null) node.append(c);
    return node;
  }
  function html(node, markup) {
    node.innerHTML = markup;
    return node;
  }
  function bytes(s) {
    const b = atob(s);
    const a = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i += 1) a[i] = b.charCodeAt(i);
    return a.buffer;
  }
  const f32 = (s) => new Float32Array(bytes(s));
  const u16 = (s) => new Uint16Array(bytes(s));
  const u8 = (s) => new Uint8Array(bytes(s));
  const num = (x, d = 3) => (Math.abs(x) >= 1e4 || (x !== 0 && Math.abs(x) < 1e-3) ? x.toExponential(1) : x.toFixed(d));
  const pct = (x) => `${(100 * x).toFixed(x < 0.01 ? 2 : 1)}%`;
  const cvText = (cv) => (cv > 0 ? num(cv, 2) : "undefined (mean β ≤ 0)");
  function sci(p) {
    if (p === 0) return "0";
    if (p >= 1e-3) return p.toFixed(4);
    const e = Math.floor(Math.log10(p));
    return `${(p / 10 ** e).toFixed(1)}·10<sup>${e}</sup>`;
  }
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }
  function canvas(label) {
    const c = el("canvas", "limestrat-demo-canvas");
    c.width = W;
    c.height = W;
    c.setAttribute("role", "img");
    if (label) c.setAttribute("aria-label", label);
    return c;
  }
  function pixelOf(event, node) {
    const r = node.getBoundingClientRect();
    const x = Math.floor(((event.clientX - r.left) / r.width) * S);
    const y = Math.floor(((event.clientY - r.top) / r.height) * S);
    return x >= 0 && y >= 0 && x < S && y < S ? y * S + x : -1;
  }
  function toggle(options, value, onChange) {
    const box = el("div", "limestrat-demo-toggle");
    box.setAttribute("role", "group");
    const buttons = options.map(([v, label]) => {
      const b = el("button", null, label);
      b.type = "button";
      b.addEventListener("click", () => {
        set(v);
        onChange(v);
      });
      box.append(b);
      return [v, b];
    });
    const set = (v) => buttons.forEach(([w, b]) => b.setAttribute("aria-pressed", String(w === v)));
    set(value);
    box.set = set;
    return box;
  }
  function slider(label, min, max, value, onInput) {
    const box = el("label", "limestrat-demo-slider");
    const text = el("span", "limestrat-demo-slider-label");
    const input = el("input");
    Object.assign(input, { type: "range", min, max, step: 1, value });
    input.addEventListener("input", () => onInput(Number(input.value)));
    box.append(text, input);
    box.input = input;
    box.label = (t) => (text.textContent = t);
    box.label(label);
    return box;
  }
  function playButton(onClick) {
    const b = el("button", "limestrat-demo-play", "▶");
    b.type = "button";
    b.setAttribute("aria-label", "Play");
    b.addEventListener("click", onClick);
    b.playing = (on) => {
      b.textContent = on ? "❚❚" : "▶";
      b.setAttribute("aria-label", on ? "Pause" : "Play");
    };
    return b;
  }

  // log-factorials, for binomial coefficients of up to a few hundred superpixels
  const LF = new Float64Array(1025);
  for (let i = 1; i < LF.length; i += 1) LF[i] = LF[i - 1] + Math.log(i);
  const logBinom = (k, i) => LF[k] - LF[i] - LF[k - i];
  const pmfMC = (k, i) => Math.exp(logBinom(k, i) - k * Math.LN2); // B(k, i) for p = 0.5
  const pmfST = (k) => 1 / (k + 1);
  const adj = (k, i) => (k + 1) * pmfMC(k, i); // Eq. 12

  // a small seeded generator, so that "draw" is repeatable
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------------------------------------------------------------------------------------
  // a photo's runs, decoded once

  function loadPhoto(p) {
    const loaded = () => window.LIMESTRAT_DEMO_PHOTOS?.[p.id];
    if (loaded()) return Promise.resolve(loaded());
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `${root.dataset.root ?? ""}${p.file}?v=${root.dataset.version ?? ""}`;
      script.onload = () => (loaded() ? resolve(loaded()) : reject(new Error(p.file)));
      script.onerror = () => reject(new Error(`could not load ${p.file}`));
      document.head.appendChild(script);
    });
  }

  function decodeRun(r, k) {
    const masks = u8(r.masks);
    const rowBytes = Math.ceil(k / 8);
    const y = f32(r.y);
    const w = f32(r.w);
    const n = y.length;
    const size = new Uint16Array(n);
    for (let s = 0; s < n; s += 1) {
      let c = 0;
      for (let b = 0; b < rowBytes; b += 1) {
        let v = masks[s * rowBytes + b];
        while (v) {
          c += v & 1;
          v >>= 1;
        }
      }
      size[s] = c;
    }
    const beta = f32(r.beta);
    let wmax = 0;
    for (const v of w) wmax = Math.max(wmax, v);
    let bmax = 0;
    for (const v of beta) bmax = Math.max(bmax, Math.abs(v));
    const kept = (s, seg) => (masks[s * rowBytes + (seg >> 3)] >> (7 - (seg & 7))) & 1;
    return { ...r, n, y, w, wmax, beta, bmax: bmax || 1, size, kept };
  }

  async function prepare(p) {
    if (p._ready) return p._ready;
    p._ready = (async () => {
      const [data, img] = await Promise.all([loadPhoto(p), loadImage(root.dataset.root + p.photo)]);
      const c = document.createElement("canvas");
      c.width = S;
      c.height = S;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0, S, S);
      const rgb = ctx.getImageData(0, 0, S, S).data;
      const gray = new Float32Array(S * S);
      for (let i = 0; i < S * S; i += 1)
        gray[i] = (0.299 * rgb[4 * i] + 0.587 * rgb[4 * i + 1] + 0.114 * rgb[4 * i + 2]) / 255;
      const runs = {};
      for (const k of p.ks) {
        const r = data.runs[String(k)];
        const seg = new Int32Array(S * S);
        const pairs = u16(r.seg);
        for (let i = 0, at = 0; i < pairs.length; i += 2) {
          seg.fill(pairs[i], at, at + pairs[i + 1]);
          at += pairs[i + 1];
        }
        // the mean colour of every superpixel: what LIME paints over a removed one
        const sum = new Float64Array(3 * k);
        const area = new Float64Array(k);
        for (let i = 0; i < S * S; i += 1) {
          const s = seg[i];
          area[s] += 1;
          for (let ch = 0; ch < 3; ch += 1) sum[3 * s + ch] += rgb[4 * i + ch];
        }
        const mean = new Uint8ClampedArray(3 * k);
        for (let s = 0; s < k; s += 1) for (let ch = 0; ch < 3; ch += 1) mean[3 * s + ch] = sum[3 * s + ch] / area[s];
        const edge = new Uint8Array(S * S);
        for (let y = 0; y < S; y += 1)
          for (let x = 0; x < S; x += 1) {
            const i = y * S + x;
            if ((x + 1 < S && seg[i + 1] !== seg[i]) || (y + 1 < S && seg[i + S] !== seg[i])) edge[i] = 1;
          }
        runs[k] = { k, seg, mean, edge, maxDist: r.maxDist, mc: decodeRun(r.mc, k), st: decodeRun(r.st, k) };
      }
      return { rgb, gray, runs };
    })();
    p._ready.catch(() => (p._ready = null));
    return p._ready;
  }

  // paint S x S pixels onto a W x W canvas
  function paint(cnv, fill) {
    const off = document.createElement("canvas");
    off.width = S;
    off.height = S;
    const octx = off.getContext("2d");
    const im = octx.createImageData(S, S);
    fill(im.data);
    octx.putImageData(im, 0, 0);
    const ctx = cnv.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, W, W);
  }

  // a perturbed photo: superpixels not kept by the mask are painted in their mean colour
  function drawSample(cnv, photo, run, method, s) {
    const r = run[method];
    paint(cnv, (d) => {
      for (let i = 0; i < S * S; i += 1) {
        const seg = run.seg[i];
        const on = r.kept(s, seg);
        for (let ch = 0; ch < 3; ch += 1) d[4 * i + ch] = on ? photo.rgb[4 * i + ch] : run.mean[3 * seg + ch];
        if (run.edge[i]) for (let ch = 0; ch < 3; ch += 1) d[4 * i + ch] = 0.7 * d[4 * i + ch] + 0.3 * 255;
        d[4 * i + 3] = 255;
      }
    });
  }

  // the coefficients as in the paper's figures: matplotlib's "bwr" colour map, symmetric around 0
  // with its own scale ±max|β|. view "photo" blends it over the photo, "top" shows the 5 superpixels
  // that get_image_and_mask(positive_only=True, num_features=5) would select.
  function bwr(t) {
    return t >= 0 ? [255, 255 * (1 - t), 255 * (1 - t)] : [255 * (1 + t), 255 * (1 + t), 255];
  }
  function drawHeat(cnv, photo, run, method, view, focus) {
    const r = run[method];
    let top = null;
    if (view === "top") {
      const order = Array.from(r.beta.keys()).sort((a, b) => r.beta[b] - r.beta[a]);
      top = new Uint8Array(run.k);
      for (const s of order.slice(0, 5)) if (r.beta[s] > 0) top[s] = 1;
    }
    paint(cnv, (d) => {
      for (let i = 0; i < S * S; i += 1) {
        const seg = run.seg[i];
        let px;
        if (top) {
          const keep = top[seg];
          px = [0, 1, 2].map((ch) => (keep ? photo.rgb[4 * i + ch] : 0.25 * photo.rgb[4 * i + ch] + 0.75 * 128));
          if (keep && run.edge[i]) px = [255, 221, 0];
        } else {
          const c = bwr(Math.max(-1, Math.min(1, r.beta[seg] / r.bmax)));
          if (view === "photo") {
            const g = 255 * photo.gray[i];
            px = c.map((v) => 0.55 * v + 0.45 * g);
          } else px = c;
        }
        d[4 * i] = px[0];
        d[4 * i + 1] = px[1];
        d[4 * i + 2] = px[2];
        d[4 * i + 3] = 255;
      }
    });
    if (focus >= 0) {
      const ctx = cnv.getContext("2d");
      ctx.fillStyle = "#000";
      for (let i = 0; i < S * S; i += 1)
        if (run.seg[i] === focus) {
          const x = i % S;
          const y = (i / S) | 0;
          const out =
            x === 0 || y === 0 || x === S - 1 || y === S - 1 || run.seg[i - 1] !== focus || run.seg[i + 1] !== focus || run.seg[i - S] !== focus || run.seg[i + S] !== focus;
          if (out) ctx.fillRect(x * Z, y * Z, Z, Z);
        }
    }
  }

  // the dependent variable distribution, as in the paper's figures: f(ξx) against |x|/k, points
  // coloured from blue (low score) to orange (high score), f(ξ) as a dashed green line and the
  // explained photo as a magenta cross. `opts.small` drops the axes, for the Fig. 7 grid.
  const LOW = [0x6d, 0x9e, 0xeb];
  const HIGH = [0xf9, 0xcb, 0x9c];
  function distribution(run, method, opts = {}) {
    const r = run[method];
    const size = opts.small ? 160 : 360;
    const pad = opts.small ? 6 : 44;
    const node = svg("svg", { viewBox: `0 0 ${size} ${size}`, class: "limestrat-demo-chart limestrat-demo-dist", role: "img" });
    node.setAttribute("aria-label", `Scores of the ${METHODS[method].name} neighbourhood`);
    const x0 = pad;
    const x1 = size - (opts.small ? 6 : 10);
    const y0 = size - pad;
    const y1 = opts.small ? 6 : 10;
    const sx = (v) => x0 + ((v + 0.05) / 1.1) * (x1 - x0);
    const sy = (v) => y0 - ((v + 0.05) / 1.1) * (y0 - y1);
    const g = svg("g");
    g.append(svg("rect", { x: x0, y: y1, width: x1 - x0, height: y0 - y1, fill: "var(--ls-bg)", stroke: "var(--ls-border)" }));
    if (!opts.small) {
      for (const t of [0, 0.5, 1]) {
        const a = svg("text", { x: sx(t), y: y0 + 16, "text-anchor": "middle" });
        a.textContent = String(t);
        const b = svg("text", { x: x0 - 6, y: sy(t) + 5, "text-anchor": "end" });
        b.textContent = String(t);
        g.append(a, b);
      }
      const xl = svg("text", { x: (x0 + x1) / 2, y: size - 6, "text-anchor": "middle" });
      xl.textContent = "kept superpixels |x| / k";
      const yl = svg("text", { x: 12, y: (y0 + y1) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(y0 + y1) / 2})` });
      yl.textContent = "f(ξx)";
      g.append(xl, yl);
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of r.y) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    const dots = svg("g");
    const rad = opts.small ? 2.2 : 3.6;
    for (let i = 1; i < r.n; i += 1) {
      const t = hi > lo ? (r.y[i] - lo) / (hi - lo) : 0;
      const c = LOW.map((v, ch) => Math.round(v + t * (HIGH[ch] - v)));
      dots.append(svg("circle", { cx: sx(r.size[i] / run.k), cy: sy(r.y[i]), r: rad, fill: `rgb(${c})`, stroke: "#000", "stroke-width": opts.small ? 0.3 : 0.5 }));
    }
    g.append(dots);
    const fx = r.y[0];
    g.append(svg("line", { x1: x0, x2: x1, y1: sy(fx), y2: sy(fx), stroke: "#2ca02c", "stroke-width": opts.small ? 1.2 : 2, "stroke-dasharray": "6 4" }));
    const cx = sx(1);
    const cy = sy(fx);
    const m = opts.small ? 5 : 8;
    g.append(svg("path", { d: `M${cx - m},${cy - m}L${cx + m},${cy + m}M${cx - m},${cy + m}L${cx + m},${cy - m}`, stroke: "#c0c", "stroke-width": opts.small ? 1.5 : 2.5 }));
    node.append(g);
    node.scale = { sx, sy, x0, x1, y0, y1, size };
    return node;
  }

  // ---------------------------------------------------------------------------------------------
  // a simple chart frame: x over [0, k], y linear or logarithmic

  function chart(width, height, pad = { l: 64, r: 14, t: 14, b: 40 }) {
    const node = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "limestrat-demo-chart", role: "img" });
    const frame = {
      node,
      width,
      height,
      pad,
      x0: pad.l,
      x1: width - pad.r,
      y0: height - pad.b,
      y1: pad.t,
    };
    frame.sx = (v, lo, hi) => frame.x0 + ((v - lo) / (hi - lo || 1)) * (frame.x1 - frame.x0);
    return frame;
  }
  function axes(f, xlo, xhi, ylo, yhi, log, xlabel, ylabel, yfmt) {
    const g = svg("g");
    const sy = (v) => {
      const a = log ? Math.log10(Math.max(v, ylo)) : v;
      const lo = log ? Math.log10(ylo) : ylo;
      const hi = log ? Math.log10(yhi) : yhi;
      return f.y0 - ((a - lo) / (hi - lo)) * (f.y0 - f.y1);
    };
    const sx = (v) => f.sx(v, xlo, xhi);
    // y ticks
    const ticks = [];
    if (log) {
      const lo = Math.ceil(Math.log10(ylo));
      const hi = Math.floor(Math.log10(yhi));
      const step = Math.max(1, Math.ceil((hi - lo) / 5));
      for (let e = hi; e >= lo; e -= step) ticks.push(10 ** e);
    } else for (let i = 0; i <= 4; i += 1) ticks.push(ylo + ((yhi - ylo) * i) / 4);
    for (const t of ticks) {
      const y = sy(t);
      g.append(svg("line", { x1: f.x0, x2: f.x1, y1: y, y2: y, class: "ls-tick" }));
      const label = svg("text", { x: f.x0 - 6, y: y + 5, "text-anchor": "end" });
      label.textContent = yfmt ? yfmt(t) : log ? `1e${Math.round(Math.log10(t))}` : t.toFixed(2);
      g.append(label);
    }
    // x ticks
    const xs = [xlo, Math.round((xlo + xhi) / 4), Math.round((xlo + xhi) / 2), Math.round((3 * (xlo + xhi)) / 4), xhi];
    for (const t of new Set(xs)) {
      const label = svg("text", { x: sx(t), y: f.y0 + 17, "text-anchor": "middle" });
      label.textContent = String(t);
      g.append(label);
    }
    g.append(svg("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0, class: "ls-axis" }));
    g.append(svg("line", { x1: f.x0, x2: f.x0, y1: f.y0, y2: f.y1, class: "ls-axis" }));
    const xl = svg("text", { x: (f.x0 + f.x1) / 2, y: f.height - 4, "text-anchor": "middle" });
    xl.textContent = xlabel;
    const yl = svg("text", { x: 14, y: (f.y0 + f.y1) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(f.y0 + f.y1) / 2})` });
    yl.textContent = ylabel;
    g.append(xl, yl);
    return { g, sx, sy };
  }

  // ---------------------------------------------------------------------------------------------

  function start(data) {
    if (!data?.photos?.length) throw new Error("no photos in data.js");
    root.innerHTML = "";
    root.classList.add("limestrat-demo-ready");

    // the largest k by default, where Monte Carlo fails most clearly (as in the paper's Fig. 7, column B)
    const state = { photo: 0, tab: 0, kIndex: data.photos[0].ks.length - 1, sample: 0, view: "heat", focus: -1 };
    const TAB_IDS = ["sampling", "neighbourhood", "explanation", "figure"];

    // header
    const photos = el("div", "limestrat-demo-photos");
    const prediction = el("p", "limestrat-demo-prediction");
    const thumbs = data.photos.map((p, i) => {
      const b = el("button", "limestrat-demo-photo");
      b.type = "button";
      b.setAttribute("aria-label", p.title);
      b.title = p.title;
      const im = el("img");
      im.src = root.dataset.root + p.photo;
      im.alt = "";
      b.append(im);
      b.addEventListener("click", () => selectPhoto(i));
      photos.append(b);
      return b;
    });
    const tabs = el("div", "limestrat-demo-tabs");
    tabs.setAttribute("role", "tablist");
    const tabNames = ["1 · Where samples land", "2 · The neighbourhood", "3 · The explanation", "4 · All photos (Fig. 7)"];
    const tabButtons = tabNames.map((name, i) => {
      const b = el("button", "limestrat-demo-tab", name);
      b.type = "button";
      b.setAttribute("role", "tab");
      b.addEventListener("click", () => selectTab(i));
      tabs.append(b);
      return b;
    });
    const body = el("div", "limestrat-demo-body");
    root.append(el("div", "limestrat-demo-head", photos, prediction), tabs, body);

    let cleanup = () => {};
    const photo = () => data.photos[state.photo];

    function selectPhotoOnly(i) {
      state.photo = i;
      state.sample = 0;
      state.focus = -1;
      state.kIndex = Math.min(state.kIndex, photo().ks.length - 1);
      thumbs.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
      const p = photo();
      prediction.innerHTML = `${data.model} predicts <strong>${p.label}</strong> with probability ${pct(p.prob)}. Each LIME run uses n&nbsp;=&nbsp;${data.n} samples; explanations average ${data.repeats} runs, as in the paper.`;
    }
    function selectPhoto(i) {
      selectPhotoOnly(i);
      if (state.tab === 3) selectTab(2);
      else render();
    }
    function selectTab(i) {
      state.tab = i;
      tabButtons.forEach((b, j) => b.setAttribute("aria-selected", String(j === i)));
      render();
    }

    let token = 0;
    async function render() {
      cleanup();
      cleanup = () => {};
      const mine = (token += 1);
      if (state.tab === 0) {
        body.replaceChildren(samplingTab());
        return;
      }
      if (state.tab === 3) {
        body.replaceChildren(figureTab());
        return;
      }
      root.classList.add("limestrat-demo-loading");
      try {
        const ph = await prepare(photo());
        if (mine !== token) return;
        body.replaceChildren(state.tab === 1 ? neighbourhoodTab(ph) : explanationTab(ph));
      } catch (error) {
        if (mine === token) fail(error);
      } finally {
        if (mine === token) root.classList.remove("limestrat-demo-loading");
      }
    }

    function kButtons(onChange) {
      const ks = photo().ks;
      return toggle(
        ks.map((k, i) => [i, `k = ${k}`]),
        state.kIndex,
        (i) => {
          state.kIndex = i;
          state.sample = 0;
          state.focus = -1;
          onChange();
        },
      );
    }

    // -------------------------------------------------------------------------------------------
    // 1 · where samples land

    function samplingTab() {
      let k = 100;
      let log = false;
      let hover = -1;
      const draws = { mc: [], st: [] };
      const counts = { mc: null, st: null };
      const last = { mc: null, st: null };
      let timer = 0;
      let random = rng(data.seed);

      const f = chart(560, 280);
      const side = el("div", "limestrat-demo-side");
      const text = html(
        el("p", "limestrat-demo-text"),
        `LIME Image keeps each of the <em>k</em> superpixels with probability 0.5, so the number of kept superpixels |x| is binomial: almost every sample keeps about half of the photo. <span class="limestrat-demo-st">Stratified sampling</span> first draws q ~ U(0,1), then keeps each superpixel with probability q, so every |x| from 0 to k is equally likely.`,
      );
      const stats = el("p", "limestrat-demo-info");
      const point = el("p", "limestrat-demo-info");
      const maskBox = el("div", "limestrat-demo-masks");
      const legend = html(
        el("p", "limestrat-demo-legend"),
        `<span class="limestrat-demo-swatch limestrat-demo-swatch-mc"></span> Monte Carlo, B(0.5) &nbsp; <span class="limestrat-demo-swatch limestrat-demo-swatch-st"></span> Stratified &nbsp; <span class="limestrat-demo-swatch limestrat-demo-swatch-near"></span> close to the photo (|x| ≥ ${100 * NEAR}% of k). Lines: exact probabilities; bars: the masks drawn so far.`,
      );
      side.append(text, stats, point, maskBox);

      const play = playButton(() => (timer ? stop() : draw()));
      const kSlider = slider("", 2, 200, k, (v) => {
        k = v;
        reset();
      });
      const scale = toggle(
        [
          [false, "Linear"],
          [true, "Log"],
        ],
        log,
        (v) => {
          log = v;
          plot();
        },
      );
      const controls = el("div", "limestrat-demo-controls", play, kSlider, scale);
      const wrap = el("div", null, el("div", "limestrat-demo-stage limestrat-demo-stage-wide", el("figure", null, f.node, legend), side), controls);

      function reset() {
        stop();
        random = rng(data.seed + k);
        draws.mc = [];
        draws.st = [];
        counts.mc = new Uint32Array(k + 1);
        counts.st = new Uint32Array(k + 1);
        last.mc = last.st = null;
        hover = -1;
        kSlider.label(`k = ${k} superpixels`);
        plot();
      }
      function drawOne() {
        const mc = new Uint8Array(k);
        const st = new Uint8Array(k);
        const q = random();
        let a = 0;
        let b = 0;
        for (let j = 0; j < k; j += 1) {
          mc[j] = random() < 0.5 ? 1 : 0;
          st[j] = random() < q ? 1 : 0;
          a += mc[j];
          b += st[j];
        }
        draws.mc.push(a);
        draws.st.push(b);
        counts.mc[a] += 1;
        counts.st[b] += 1;
        last.mc = mc;
        last.st = st;
      }
      function draw() {
        if (draws.mc.length >= data.n) reset();
        play.playing(true);
        timer = setInterval(() => {
          for (let i = 0; i < 25 && draws.mc.length < data.n; i += 1) drawOne();
          plot();
          if (draws.mc.length >= data.n) stop();
        }, 40);
      }
      function stop() {
        if (timer) clearInterval(timer);
        timer = 0;
        play.playing(false);
      }

      function plot() {
        const peak = Math.max(pmfMC(k, k >> 1), pmfST(k));
        const ylo = log ? 1e-12 : 0;
        const yhi = log ? 1 : Math.min(1, peak * 1.15);
        const ax = axes(f, 0, k, ylo, yhi, log, "kept superpixels |x|", "probability", log ? null : (t) => t.toFixed(3));
        const g = svg("g");
        const near = Math.ceil(NEAR * k);
        g.append(svg("rect", { x: ax.sx(near - 0.5), y: f.y1, width: Math.max(0, ax.sx(k + 0.5) - ax.sx(near - 0.5)), height: f.y0 - f.y1, class: "ls-near" }));
        const bw = Math.max(1, ((f.x1 - f.x0) / (k + 1)) * 0.42);
        const n = Math.max(1, draws.mc.length);
        for (const [m, color, dx] of [
          ["mc", "var(--ls-mc)", -bw / 2],
          ["st", "var(--ls-st)", bw / 2],
        ]) {
          if (!counts[m]) continue;
          for (let i = 0; i <= k; i += 1) {
            if (!counts[m][i]) continue;
            const y = ax.sy(counts[m][i] / n);
            g.append(svg("rect", { x: ax.sx(i) + dx - bw / 2, y, width: bw, height: Math.max(0, f.y0 - y), fill: color, opacity: 0.35 }));
          }
        }
        const line = (fn, color) => {
          let d = "";
          for (let i = 0; i <= k; i += 1) d += `${i ? "L" : "M"}${ax.sx(i).toFixed(1)},${ax.sy(fn(i)).toFixed(1)}`;
          return svg("path", { d, fill: "none", stroke: color, "stroke-width": 2 });
        };
        g.append(line((i) => pmfMC(k, i), "var(--ls-mc)"), line(() => pmfST(k), "var(--ls-st)"));
        if (hover >= 0) g.append(svg("line", { x1: ax.sx(hover), x2: ax.sx(hover), y1: f.y0, y2: f.y1, class: "ls-cursor" }));
        const hit = svg("rect", { x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, fill: "transparent" });
        hit.addEventListener("pointermove", (e) => {
          const r = f.node.getBoundingClientRect();
          const x = ((e.clientX - r.left) / r.width) * f.width;
          const i = Math.round(((x - f.x0) / (f.x1 - f.x0)) * k);
          if (i !== hover && i >= 0 && i <= k) {
            hover = i;
            plot();
          }
        });
        hit.addEventListener("pointerleave", () => {
          hover = -1;
          plot();
        });
        f.node.replaceChildren(ax.g, g, hit);

        // the numbers
        let pNear = 0;
        for (let i = near; i <= k; i += 1) pNear += pmfMC(k, i);
        const sNear = (k - near + 1) / (k + 1);
        const seen = (m) => draws[m].filter((v) => v >= near).length;
        stats.innerHTML =
          `Of n = ${data.n} samples, how many keep at least ${near} of the ${k} superpixels (close to the photo)? ` +
          `<span class="limestrat-demo-mc">Monte Carlo</span> expects ${sci(data.n * pNear)} (probability ${sci(pNear)})` +
          `; <span class="limestrat-demo-st">stratified</span> expects ${num(data.n * sNear, 1)}.` +
          (draws.mc.length ? ` Drawn so far (${draws.mc.length}): ${seen("mc")} and ${seen("st")}.` : " Press ▶ to draw the masks.");
        if (hover >= 0) {
          const a = adj(k, hover);
          point.innerHTML =
            `At |x| = ${hover}: <span class="limestrat-demo-mc">P = ${sci(pmfMC(k, hover))}</span>, ` +
            `<span class="limestrat-demo-st">P = ${sci(pmfST(k))}</span>. ` +
            `Each stratified sample here gets the adjustment factor adj(${hover}) = (k+1)·C(k,${hover})/2<sup>k</sup> = ${sci(a)}, ` +
            `which multiplies the sample's distance to the photo before LIME's kernel: ` +
            (a < 1
              ? `it shrinks the distance, so these samples, which Monte Carlo almost never draws, keep a high weight in the regression.`
              : `it stretches the distance, so these mid-strata samples, which Monte Carlo over-represents, get a lower weight.`);
        } else point.innerHTML = `<span class="limestrat-demo-hint">Point at the chart to read the probabilities and the adjustment factor adj(i) of the stratum |x| = i.</span>`;
        maskBox.replaceChildren();
        for (const m of ["mc", "st"]) {
          if (!last[m]) continue;
          const strip = el("div", `limestrat-demo-mask limestrat-demo-mask-${m}`);
          for (const v of last[m]) strip.append(el("span", v ? "on" : null));
          const kept = last[m].reduce((a, b) => a + b, 0);
          maskBox.append(html(el("p", "limestrat-demo-legend"), `Last <span class="${METHODS[m].cls}">${METHODS[m].name}</span> mask: ${kept} of ${k} kept`), strip);
        }
      }

      reset();
      cleanup = stop;
      return wrap;
    }

    // -------------------------------------------------------------------------------------------
    // shared: how much the two explanations agree

    function ranks(beta) {
      const order = Array.from(beta.keys()).sort((a, b) => beta[a] - beta[b]);
      const r = new Float64Array(beta.length);
      for (let i = 0; i < order.length; ) {
        let j = i;
        while (j + 1 < order.length && beta[order[j + 1]] === beta[order[i]]) j += 1;
        for (let t = i; t <= j; t += 1) r[order[t]] = (i + j) / 2;
        i = j + 1;
      }
      return r;
    }
    // Spearman rank correlation of β and β̂, and the superpixels shared by their top-5 regions
    function agreement(run) {
      if (run._agree) return run._agree;
      const a = ranks(run.mc.beta);
      const b = ranks(run.st.beta);
      const n = a.length;
      const m = (n - 1) / 2;
      let sab = 0;
      let saa = 0;
      let sbb = 0;
      for (let i = 0; i < n; i += 1) {
        sab += (a[i] - m) * (b[i] - m);
        saa += (a[i] - m) ** 2;
        sbb += (b[i] - m) ** 2;
      }
      const top = (beta) => new Set(Array.from(beta.keys()).sort((x, y) => beta[y] - beta[x]).slice(0, 5));
      const ta = top(run.mc.beta);
      let shared = 0;
      for (const s of top(run.st.beta)) if (ta.has(s)) shared += 1;
      run._agree = { rho: sab / Math.sqrt(saa * sbb || 1), shared };
      return run._agree;
    }
    const cvShort = (x) => (x > 0 ? num(x, 3) : "—");
    const confused = (r) => !(r.cv >= 1);
    function header(m, r, compact = false) {
      const hat = m === "st" ? "̂" : "";
      const cv = `<span class="${confused(r) ? "limestrat-demo-bad" : ""}">${compact ? num(r.cv, 3) : cvShort(r.cv)}</span>`;
      const rc = `<span class="${r.rc < 0.5 ? "limestrat-demo-bad" : ""}">${num(r.rc, 3)}</span>`;
      return html(
        el("p", `limestrat-demo-head-metrics ${METHODS[m].cls}`),
        compact ? `${cv} | ${rc}` : `CV(β${hat}) = ${cv} | RC(Y${hat}) = ${rc}`,
      );
    }

    // -------------------------------------------------------------------------------------------
    // 2 · the neighbourhood, Monte Carlo and stratified side by side

    function neighbourhoodTab(ph) {
      const p = photo();
      const cols = {};
      const pick = { mc: 0, st: 0 };
      const ks = kButtons(() => update());
      const legend = html(
        el("p", "limestrat-demo-legend"),
        `Each dot is one of the ${data.n} perturbed photos of the neighbourhood: its share of kept superpixels |x|/k and the model's probability f(ξ<sub>x</sub>) for <strong>${p.label}</strong>, coloured from blue (low) to orange (high), as in the paper. Dashed green: f(ξ), the photo itself (magenta ×). RC(Y) is the share of [0, f(ξ)] covered by the 1–99% range of the scores (Eq. 7), averaged over ${data.repeats} runs. Point at a dot to see that perturbed photo: removed superpixels are filled with their mean colour.`,
      );
      const pair = el("div", "limestrat-demo-pair");
      for (const m of ["mc", "st"]) {
        const cnv = canvas(`A perturbed photo from the ${METHODS[m].name} neighbourhood`);
        cnv.classList.add("limestrat-demo-sample");
        const cap = el("figcaption");
        const metrics = el("div");
        const plot = el("div");
        const col = el("figure", "limestrat-demo-col", html(el("h4", METHODS[m].cls), METHODS[m].title), metrics, plot, el("div", "limestrat-demo-sample-row", cnv, cap));
        cols[m] = { cnv, cap, metrics, plot };
        pair.append(col);
      }
      const info = el("p", "limestrat-demo-info");
      const wrap = el("div", null, el("div", "limestrat-demo-controls limestrat-demo-controls-top", ks), pair, info, legend);

      function showSample(m) {
        const run = ph.runs[p.ks[state.kIndex]];
        const r = run[m];
        const s = pick[m];
        drawSample(cols[m].cnv, ph, run, m, s);
        cols[m].cap.innerHTML =
          `Sample #${s}${s === 0 ? " (the photo)" : ""}: keeps ${r.size[s]} of ${run.k} superpixels (${pct(r.size[s] / run.k)}), ` +
          `f(ξ<sub>x</sub>) = ${num(r.y[s])}, weight ${num(r.w[s])}` +
          (m === "st" && s > 0 ? ` (the kernel of its distance × adj(${r.size[s]}) = ${sci(adj(run.k, r.size[s]))})` : "") +
          ".";
        const node = cols[m].plot.firstChild;
        node?.querySelector(".ls-pick")?.remove();
        if (node) node.append(svg("circle", { class: "ls-pick", cx: node.scale.sx(r.size[s] / run.k), cy: node.scale.sy(r.y[s]), r: 8, fill: "none", stroke: "var(--ls-fg)", "stroke-width": 2.5 }));
      }

      function update() {
        const run = ph.runs[p.ks[state.kIndex]];
        for (const m of ["mc", "st"]) {
          const r = run[m];
          const node = distribution(run, m);
          node.addEventListener("pointermove", (e) => {
            const b = node.getBoundingClientRect();
            const x = ((e.clientX - b.left) / b.width) * node.scale.size;
            const y = ((e.clientY - b.top) / b.height) * node.scale.size;
            let best = -1;
            let bd = 16 * 16;
            for (let i = 0; i < r.n; i += 1) {
              const dx = node.scale.sx(r.size[i] / run.k) - x;
              const dy = node.scale.sy(r.y[i]) - y;
              if (dx * dx + dy * dy < bd) {
                bd = dx * dx + dy * dy;
                best = i;
              }
            }
            if (best >= 0 && best !== pick[m]) {
              pick[m] = best;
              showSample(m);
            }
          });
          cols[m].plot.replaceChildren(node);
          cols[m].metrics.replaceChildren(header(m, r));
          pick[m] = 0;
          showSample(m);
        }
        const near = Math.ceil(NEAR * run.k);
        const count = (r) => {
          let c = 0;
          for (let i = 1; i < r.n; i += 1) if (r.size[i] >= near) c += 1;
          return c;
        };
        info.innerHTML =
          `With k = ${run.k}, <span class="limestrat-demo-mc">Monte Carlo</span> puts ${count(run.mc)} of its ${data.n - 1} random samples close to the photo (|x| ≥ ${near}), ` +
          `<span class="limestrat-demo-st">stratified sampling</span> ${count(run.st)}. ` +
          (run.mc.rc < 0.5
            ? `The Monte Carlo scores are squashed near ${num(run.mc.y.slice(1).reduce((a, b) => a + b, 0) / (run.mc.n - 1), 2)}: RC(Y) = ${num(run.mc.rc, 3)}, against RC(Ŷ) = ${num(run.st.rc, 3)} for stratified sampling. A linear regressor fit on Monte Carlo's flat scores has almost nothing to explain.`
            : `RC(Y) = ${num(run.mc.rc, 3)} for Monte Carlo, RC(Ŷ) = ${num(run.st.rc, 3)} for stratified sampling.`);
      }

      update();
      return wrap;
    }

    // -------------------------------------------------------------------------------------------
    // 3 · the explanation, side by side, as one row of the paper's Fig. 7

    function explanationTab(ph) {
      const p = photo();
      const cols = {};
      const pair = el("div", "limestrat-demo-pair");
      for (const m of ["mc", "st"]) {
        const cnv = canvas(`${METHODS[m].name} explanation`);
        const cap = el("figcaption");
        const metrics = el("div");
        const plot = el("div");
        pair.append(el("figure", "limestrat-demo-col", html(el("h4", METHODS[m].cls), METHODS[m].title), metrics, el("div", "limestrat-demo-duo", cnv, plot), cap));
        cols[m] = { cnv, cap, metrics, plot };
        cnv.addEventListener("pointermove", (e) => {
          const i = pixelOf(e, cnv);
          const s = i >= 0 ? ph.runs[p.ks[state.kIndex]].seg[i] : -1;
          if (s !== state.focus) {
            state.focus = s;
            paintBoth();
            focusInfo();
          }
        });
        cnv.addEventListener("pointerleave", () => {
          state.focus = -1;
          paintBoth();
          focusInfo();
        });
      }
      const ks = kButtons(() => update());
      const viewToggle = toggle(
        [
          ["heat", "β (paper colours)"],
          ["photo", "β over the photo"],
          ["top", "Top 5 superpixels"],
        ],
        state.view,
        (v) => {
          state.view = v;
          paintBoth();
        },
      );
      const verdict = el("p", "limestrat-demo-verdict");
      const info = el("p", "limestrat-demo-info");
      const legend = html(
        el("p", "limestrat-demo-legend"),
        `<span class="limestrat-demo-swatch limestrat-demo-swatch-pos"></span> raises the probability of <strong>${p.label}</strong> &nbsp; <span class="limestrat-demo-swatch limestrat-demo-swatch-neg"></span> lowers it; white: no effect. As in the paper's Fig. 7, each heatmap is the average of ${data.repeats} runs, in its own symmetric scale ±max|β|. A <em>confused</em> explanation (CV(β) < 1, in red) is almost uniform: every superpixel gets about the same importance, so no region of the photo stands out. <em>Top 5</em> shows what <code>get_image_and_mask(positive_only=True, num_features=5)</code> would select.`,
      );
      const table = el("table");
      const wrap = el("div", null, el("div", "limestrat-demo-controls limestrat-demo-controls-top", ks, viewToggle), pair, verdict, info, legend, el("div", "limestrat-demo-table-wrap", table));

      function paintBoth() {
        const run = ph.runs[p.ks[state.kIndex]];
        for (const m of ["mc", "st"]) drawHeat(cols[m].cnv, ph, run, m, state.view, state.focus);
      }
      function rank(beta, s) {
        let r = 1;
        for (const v of beta) if (v > beta[s]) r += 1;
        return r;
      }
      function focusInfo() {
        const run = ph.runs[p.ks[state.kIndex]];
        if (state.focus < 0) {
          info.innerHTML = `<span class="limestrat-demo-hint">Point at a superpixel to compare its importance in the two explanations.</span>`;
          return;
        }
        const s = state.focus;
        info.innerHTML =
          `Superpixel #${s}: <span class="limestrat-demo-mc">β = ${num(run.mc.beta[s])}</span> (rank ${rank(run.mc.beta, s)} of ${run.k}), ` +
          `<span class="limestrat-demo-st">β̂ = ${num(run.st.beta[s])}</span> (rank ${rank(run.st.beta, s)} of ${run.k}).`;
      }

      function update() {
        const run = ph.runs[p.ks[state.kIndex]];
        paintBoth();
        for (const m of ["mc", "st"]) {
          const r = run[m];
          cols[m].metrics.replaceChildren(header(m, r));
          cols[m].plot.replaceChildren(distribution(run, m, { small: true }));
          cols[m].cap.innerHTML = `R² = ${num(r.r2, 3)} (mean of ${data.repeats} runs) · scale ±${num(r.bmax)}`;
        }
        const { rho, shared } = agreement(run);
        const mcBad = confused(run.mc);
        verdict.innerHTML =
          (mcBad
            ? `<strong>Monte Carlo's explanation is confused</strong> (CV(β) = ${cvShort(run.mc.cv)} < 1): it gives every superpixel about the same importance and points at no region. `
            : `Monte Carlo's explanation is not confused at this k (CV(β) = ${cvShort(run.mc.cv)}). `) +
          `<strong>Stratified sampling</strong> ${confused(run.st) ? "is also confused here" : `identifies a region (CV(β̂) = ${cvShort(run.st.cv)}, ${num(run.st.cv / Math.max(run.mc.cv, 1e-9), 1)}× higher)`}, ` +
          `with RC(Ŷ) = ${num(run.st.rc, 3)} against RC(Y) = ${num(run.mc.rc, 3)}. ` +
          `The two explanations ${rho > 0.7 ? "broadly agree" : rho > 0.4 ? "partly disagree" : "disagree"} (rank correlation ${num(rho, 2)}, top-5 regions share ${shared} of 5 superpixels).`;
        focusInfo();

        const head = `<thead><tr><th>k</th><th class="limestrat-demo-mc">CV(β)</th><th class="limestrat-demo-mc">RC(Y)</th><th class="limestrat-demo-mc">R²</th><th class="limestrat-demo-st">CV(β̂)</th><th class="limestrat-demo-st">RC(Ŷ)</th><th class="limestrat-demo-st">R²</th><th>rank corr.</th><th>top-5 shared</th></tr></thead>`;
        const rows = p.ks
          .map((k, i) => {
            const a = ph.runs[k].mc;
            const b = ph.runs[k].st;
            const ag = agreement(ph.runs[k]);
            const bad = (x, txt) => `<td${x ? ' class="limestrat-demo-bad"' : ""}>${txt}</td>`;
            const good = (y, x, txt) => `<td${y > x ? ' class="limestrat-demo-better"' : ""}>${txt}</td>`;
            return (
              `<tr data-i="${i}" aria-selected="${i === state.kIndex}"><td>${k}</td>` +
              bad(confused(a), cvShort(a.cv)) +
              bad(a.rc < 0.5, num(a.rc, 3)) +
              `<td>${num(a.r2, 3)}</td>` +
              good(b.cv, a.cv > 0 ? a.cv : -Infinity, cvShort(b.cv)) +
              good(b.rc, a.rc, num(b.rc, 3)) +
              good(b.r2, a.r2, num(b.r2, 3)) +
              `<td>${num(ag.rho, 2)}</td><td>${ag.shared}/5</td></tr>`
            );
          })
          .join("");
        table.innerHTML = `${head}<tbody>${rows}</tbody><caption class="limestrat-demo-legend">All four segmentations of this photo. Red: Monte Carlo is confused (CV < 1) or covers less than half of the score range (RC < 0.5). Green: stratified sampling is higher. The last two columns compare the two explanations. Click a row to show it.</caption>`;
        table.querySelectorAll("tbody tr").forEach((tr) =>
          tr.addEventListener("click", () => {
            state.kIndex = Number(tr.dataset.i);
            state.focus = -1;
            ks.set(state.kIndex);
            update();
          }),
        );
      }

      update();
      return wrap;
    }

    // -------------------------------------------------------------------------------------------
    // 4 · the paper's Fig. 7: every photo, k = 50 and 200, both samplers

    function figureTab() {
      const grid = el("div", "limestrat-demo-fig7");
      const status = el("p", "limestrat-demo-hint", "Loading the photos…");
      const legend = html(
        el("p", "limestrat-demo-legend"),
        `The low-CV rows of the paper's Fig. 7, recomputed with the library: (A, B) Monte Carlo and (C, D) stratified sampling, with k ≈ 50 and k ≈ 200 superpixels. Each cell shows CV | RC, the explanation (average of ${data.repeats} runs) and the scores of the neighbourhood. Red numbers: confused (CV < 1) or poor range coverage (RC < 0.5). Click a row to explore that photo.`,
      );
      const wrap = el("div", null, grid, status, legend);
      const head = el("div", "limestrat-demo-fig7-row limestrat-demo-fig7-head");
      head.append(el("div", null, "Image"));
      for (const [m, label] of [
        ["mc", "(A) Monte Carlo<br>k ≈ 50"],
        ["mc", "(B) Monte Carlo<br>k ≈ 200"],
        ["st", "(C) Stratified<br>k ≈ 50"],
        ["st", "(D) Stratified<br>k ≈ 200"],
      ])
        head.append(html(el("div", METHODS[m].cls), `${label}<br><small>CV | RC</small>`));
      grid.append(head);

      (async () => {
        let done = 0;
        for (const [i, p] of data.photos.entries()) {
          const row = el("div", "limestrat-demo-fig7-row");
          row.tabIndex = 0;
          row.setAttribute("role", "button");
          row.setAttribute("aria-label", `Explore ${p.title}`);
          const thumb = el("img");
          thumb.src = root.dataset.root + p.photo;
          thumb.alt = p.title;
          row.append(html(el("div", "limestrat-demo-fig7-image"), `<strong>${p.label}</strong><br>${pct(p.prob)}`));
          row.firstChild.prepend(thumb);
          grid.append(row);
          const open = () => {
            selectPhotoOnly(i);
            state.kIndex = 0;
            selectTab(2);
          };
          row.addEventListener("click", open);
          row.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && open());
          const ph = await prepare(p);
          const kk = [p.ks[0], p.ks[p.ks.length - 1]];
          for (const m of ["mc", "st"])
            for (const k of kk) {
              const run = ph.runs[k];
              const cell = el("div", "limestrat-demo-fig7-cell");
              cell.append(header(m, run[m], true));
              const cnv = canvas(`${METHODS[m].name}, k = ${k}`);
              drawHeat(cnv, ph, run, m, "heat", -1);
              cell.append(el("div", "limestrat-demo-duo", cnv, distribution(run, m, { small: true })), el("figcaption", null, `k = ${k}`));
              row.append(cell);
            }
          done += 1;
          status.textContent = done < data.photos.length ? `Loading the photos… ${done} of ${data.photos.length}` : "";
        }
      })().catch(fail);
      return wrap;
    }

    // a link to #<anchor>-<tab> (e.g. #sampling-demo-explanation) opens that tab
    const fromHash = () => {
      const i = TAB_IDS.findIndex((t) => location.hash === `#${root.id}-${t}`);
      return i >= 0 ? i : null;
    };
    window.addEventListener("hashchange", () => {
      const i = fromHash();
      if (i != null) selectTab(i);
    });
    const first = fromHash();
    state.tab = first ?? 0;
    tabButtons.forEach((b, j) => b.setAttribute("aria-selected", String(j === state.tab)));
    selectPhotoOnly(0);
    render();
    if (first != null) root.scrollIntoView();
  }
})();
