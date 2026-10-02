/* ══════════════════════════════════════════════════════════════
 * FIGURE · FUNCTION-SPACE REWARD UPDATES VS GRADIENT DESCENT ON THE DUAL
 * (theory.html → after Eq. (1); <figure id="fig-update-race">)
 *
 * A small version of the comparison in Arenz et al. (2016), on the two-state problem and its
 * features instead of their linear-quadratic systems. Layout after the hero figure of
 * distill.pub/2017/momentum: optimisation paths over the loss contours, sliders below.
 *
 * Both methods start at θ⁽⁰⁾ (drag it in Θ-space) and run N_ITER iterations:
 *   gradient descent        θ ← θ − α ∇G(θ),       ∇G = Ψᵀρ_π − Ψᵀρ̂,  ρ̂ ∝ ρ_E exp(−r_θ/β)
 *   function-space update   θ ← θ − ε δ(θ),        δ = θ − β(φ_E − φ_ρ(θ))   (Eq. 1 in parameters)
 * φ_ρ(θ) is the moment-matching (M-)projection of ρ_π onto the Boltzmann family: the φ with
 * E_{ρ_φ}[ψ] = E_{ρ_π}[ψ] (Newton on log Z(φ) − φᵀm). With it δ = 0 exactly where ∇G = 0, so both
 * methods share the fixed point θ⋆ (a least-squares fit of log ρ_π would not: the agent's
 * occupancies are only approximately Boltzmann in these two features).
 * Left: Θ-space with the contours of G (recomputed per β), θ⋆ and both paths.
 * Right: the suboptimality G(θ⁽ⁱ⁾) − G(θ⋆) per iteration, log scale.
 *
 * Edit the CONFIG block only. Styles: style.css (.ur-*, and the .tp-* boxes).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-update-race');
  if (!fig || !window.TwoStateMDP || !window.TwoStateIRL || !window.PlotBox) return;
  const { EDGES, MU0, FEATURES, softVI, occupancy, el } = window.TwoStateMDP;
  const { GAMMA, VIEW, isoPaths } = window.TwoStateIRL;
  const L = window.PlotBox.label;

  // ── CONFIG ──────────────────────────────────────────────────
  const START  = [-0.4, 0.8];                          // initial θ⁽⁰⁾
  const BETA   = { min: 1, max: 30, step: 0.5, value: 10 };
  const ALPHA    = { min: 0.1, max: 20, value: 4 };       // gradient-descent step (log slider)
  const EPS    = { min: 0.002, max: 1, value: 0.06 };   // function-space step (log slider)
  const N_ITER = 40;                                    // iterations per method
  const VI     = 200;                                   // soft VI sweeps per iterate
  const GRID   = 41, GRID_VI = 80, N_LEVELS = 14;       // contour grid of G (per β, cached)
  const LOG_Y  = [-8, 1];                               // suboptimality range, log10
  // ─────────────────────────────────────────────────────────────

  // ── Model: dual, gradient, projection (β is a variable here) ──
  const lse = xs => { const m = Math.max(...xs); return m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0)); };
  const reward = th => FEATURES.map(f => f[0] * th[0] + f[1] * th[1]);
  const moments = p => [0, 1].map(k => FEATURES.reduce((s, f, i) => s + p[i] * f[k], 0));
  const rhoE = occupancy(softVI(EDGES.map(e => e.r), GAMMA, VI).pi, GAMMA);
  const logRhoE = rhoE.map(Math.log);

  function evalAt(th, beta, sweeps = VI) {             // G(θ), ∇G(θ) and ρ_π
    const r = reward(th), { V, pi } = softVI(r, GAMMA, sweeps);
    const z = logRhoE.map((l, i) => l - r[i] / beta), lz = lse(z);
    const G = (1 - GAMMA) * (MU0[0] * V[0] + MU0[1] * V[1]) + beta * lz;
    if (sweeps !== VI) return { G };
    const rho = occupancy(pi, GAMMA), hat = z.map(v => Math.exp(v - lz));
    return { G, rho, grad: moments(rho.map((p, i) => p - hat[i])) };
  }

  // M-projection: argmin_φ log Σ exp(φᵀψ) − φᵀm, damped Newton (convex, 2-D)
  function project(m, phi = [0, 0]) {
    const obj = ph => lse(FEATURES.map(f => ph[0] * f[0] + ph[1] * f[1])) - ph[0] * m[0] - ph[1] * m[1];
    for (let it = 0; it < 60; it++) {
      const z = FEATURES.map(f => phi[0] * f[0] + phi[1] * f[1]), lz = lse(z), p = z.map(v => Math.exp(v - lz));
      const mu = moments(p);
      const C = [0, 1].map(a => [0, 1].map(b => FEATURES.reduce((s, f, i) => s + p[i] * (f[a] - mu[a]) * (f[b] - mu[b]), 0)));
      const g = [mu[0] - m[0], mu[1] - m[1]], det = C[0][0] * C[1][1] - C[0][1] * C[1][0];
      const step = [(C[1][1] * g[0] - C[0][1] * g[1]) / det, (C[0][0] * g[1] - C[1][0] * g[0]) / det];
      let t = 1; const f0 = obj(phi);
      while (t > 1e-8 && obj([phi[0] - t * step[0], phi[1] - t * step[1]]) > f0) t /= 2;
      phi = [phi[0] - t * step[0], phi[1] - t * step[1]];
      if (Math.abs(t * step[0]) + Math.abs(t * step[1]) < 1e-12) break;
    }
    return phi;
  }
  const phiE = project(moments(rhoE));

  // θ⋆ = argmin G: Newton with a finite-difference Hessian of ∇G, backtracking on G
  let thGuess = [1.5, -0.5];
  function thetaStar(beta) {
    let th = thGuess.slice();
    for (let it = 0; it < 30; it++) {
      const { G, grad } = evalAt(th, beta), h = 1e-4;
      const col = k => { const u = th.slice(), d = th.slice(); u[k] += h; d[k] -= h;
        const gu = evalAt(u, beta).grad, gd = evalAt(d, beta).grad; return [(gu[0] - gd[0]) / (2 * h), (gu[1] - gd[1]) / (2 * h)]; };
      const c0 = col(0), c1 = col(1), H = [[c0[0], (c0[1] + c1[0]) / 2], [(c0[1] + c1[0]) / 2, c1[1]]];
      const det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
      const step = [(H[1][1] * grad[0] - H[0][1] * grad[1]) / det, (H[0][0] * grad[1] - H[1][0] * grad[0]) / det];
      let t = 1;
      while (t > 1e-6 && evalAt([th[0] - t * step[0], th[1] - t * step[1]], beta).G > G) t /= 2;
      th = [th[0] - t * step[0], th[1] - t * step[1]];
      if (Math.hypot(grad[0], grad[1]) < 1e-11) break;
    }
    thGuess = th;
    return th;
  }

  function run(th0, beta, Gstar, method, step) {
    let th = th0.slice(), phi = [0, 0];
    const path = [], gap = [];
    for (let k = 0; k <= N_ITER; k++) {
      const e = evalAt(th, beta);
      path.push(th); gap.push(e.G - Gstar);
      if (k === N_ITER) break;
      let d;
      if (method === 'gd') d = e.grad;
      else { phi = project(moments(e.rho), phi); d = [0, 1].map(j => th[j] - beta * (phiE[j] - phi[j])); }
      th = [th[0] - step * d[0], th[1] - step * d[1]];
      if (!isFinite(th[0]) || !isFinite(th[1]) || Math.hypot(th[0], th[1]) > 1e3) break;   // diverged
    }
    return { path, gap };
  }

  // ── Markup (typeset by MathJax after load) ──────────────────
  fig.classList.add('ifig', 'ur-fig');
  const caption = fig.querySelector('figcaption');
  fig.innerHTML = String.raw`
    <div class="ifig__canvas">
      <div class="ur-body">
        <div class="ur-panel"><svg class="tp-plot tp-plot--theta ur-theta" viewBox="0 0 300 318" role="img"
             aria-label="Reward parameter space with the contours of the dual, the IRL solution, and the paths of gradient descent and of the function-space update from a draggable start"></svg></div>
        <div class="ur-panel"><div class="oc-title ur-title">𝒢(θ<sup>(i)</sup>) − 𝒢(θ<sup>⋆</sup>)</div>
          <svg class="ur-conv" viewBox="0 0 300 334" role="img"
             aria-label="Suboptimality of both methods per iteration, log scale"></svg></div>
      </div>
      <div class="ur-controls">
        <label class="ur-ctl"><span class="ur-name">Gradient descent $\alpha$ = <span class="ur-val" data-v="alpha"></span></span>
          <input type="range" class="ur-range ur-range--gd" data-s="alpha" min="0" max="1000"></label>
        <label class="ur-ctl"><span class="ur-name">Function-space $\epsilon$ = <span class="ur-val" data-v="eps"></span></span>
          <input type="range" class="ur-range ur-range--fu" data-s="eps" min="0" max="1000"></label>
        <label class="ur-ctl"><span class="ur-name">$\beta$ = <span class="ur-val" data-v="beta"></span></span>
          <input type="range" class="ur-range" data-s="beta" min="${BETA.min}" max="${BETA.max}" step="${BETA.step}"></label>
      </div>
    </div>`;
  if (caption) fig.appendChild(caption);

  // ── Θ-space box ─────────────────────────────────────────────
  const thetaSvg = fig.querySelector('.ur-theta');
  const tBox = window.PlotBox(thetaSvg, { id: 'ur-theta', title: 'Θ-space', sym: 'θ', view: VIEW });
  const gLabel = document.createElement('div');                 // 𝒢(θ) in the box's bottom-left corner (as theta-space.js)
  gLabel.className = 'tp-glabel';
  gLabel.textContent = '$\\mathcal{G}(\\theta)$';
  thetaSvg.parentNode.appendChild(gLabel);
  const contours = el('g', { class: 'ur-contours' }, tBox.under);
  const paths = {};
  ['gd', 'fu'].forEach(m => {
    const g = el('g', { class: 'ur-run ur-run--' + m }, tBox.marks);
    paths[m] = { line: el('polyline', { class: 'ur-line' }, g), dots: el('g', {}, g) };
  });
  const starPt  = tBox.point([0, 0], { label: L('θ', { sup: '⋆' }) });
  const startPt = tBox.point(START, { label: L('θ', { sup: '(0)' }), cls: 'ur-start' });

  // ── Convergence box ─────────────────────────────────────────
  const conv = fig.querySelector('.ur-conv');
  const C = { x0: 44, y0: 30, w: 246, h: 256 };
  const CX = k => C.x0 + k / N_ITER * C.w;
  const CY = g => { const l = Math.log10(Math.max(g, 10 ** LOG_Y[0])); return C.y0 + (LOG_Y[1] - l) / (LOG_Y[1] - LOG_Y[0]) * C.h; };
  const cclip = el('clipPath', { id: 'ur-conv-clip' }, el('defs', {}, conv));
  el('rect', { x: C.x0, y: C.y0, width: C.w, height: C.h }, cclip);
  el('rect', { x: C.x0, y: C.y0, width: C.w, height: C.h, class: 'tp-frame' }, conv);
  [[0, '1'], [-4, '10⁻⁴'], [-8, '10⁻⁸']].forEach(([l, s]) =>
    el('text', { x: C.x0 - 6, y: CY(10 ** l) + 4, class: 'ur-tick', 'text-anchor': 'end' }, conv).textContent = s);
  [0, N_ITER / 2, N_ITER].forEach(k =>
    el('text', { x: CX(k), y: C.y0 + C.h + 15, class: 'ur-tick', 'text-anchor': 'middle' }, conv).textContent = k);
  el('text', { x: C.x0 + C.w / 2, y: C.y0 + C.h + 40, class: 'tp-axis', 'text-anchor': 'middle' }, conv)
    .innerHTML = 'iteration <tspan font-style="italic">i</tspan>';
  const curves = {};
  const cg = el('g', { 'clip-path': 'url(#ur-conv-clip)' }, conv);
  ['gd', 'fu'].forEach(m => { curves[m] = el('polyline', { class: 'ur-curve ur-run--' + m }, cg); });

  // ── State, sliders ──────────────────────────────────────────
  const logMap = ({ min, max }) => ({
    toVal: t => min * Math.pow(max / min, t / 1000),
    toPos: v => Math.round(1000 * Math.log(v / min) / Math.log(max / min)),
  });
  const S = { alpha: ALPHA.value, eps: EPS.value, beta: BETA.value, start: START.slice() };
  const maps = { alpha: logMap(ALPHA), eps: logMap(EPS) };
  const fmt = { alpha: v => v.toFixed(2), eps: v => v.toFixed(3), beta: v => v.toFixed(1) };
  fig.querySelectorAll('.ur-range').forEach(inp => {
    const k = inp.dataset.s;
    const fill = () => inp.style.setProperty('--fill', (100 * (inp.value - inp.min) / (inp.max - inp.min)) + '%');   // filled part of the track
    inp.value = maps[k] ? maps[k].toPos(S[k]) : S[k];
    fill();
    inp.addEventListener('input', () => { S[k] = maps[k] ? maps[k].toVal(+inp.value) : +inp.value; fill(); schedule(); });
  });
  const vals = Object.fromEntries([...fig.querySelectorAll('.ur-val')].map(n => [n.dataset.v, n]));

  // Per-β cache: θ⋆, G(θ⋆) and the contour paths
  const cache = new Map();
  function forBeta(beta) {
    if (cache.has(beta)) return cache.get(beta);
    const ts = thetaStar(beta), Gs = evalAt(ts, beta).G;
    const t1 = Array.from({ length: GRID }, (_, i) => VIEW.x[0] + i * (VIEW.x[1] - VIEW.x[0]) / (GRID - 1));
    const t2 = Array.from({ length: GRID }, (_, i) => VIEW.y[0] + i * (VIEW.y[1] - VIEW.y[0]) / (GRID - 1));
    const Z = t2.map(b => t1.map(a => evalAt([a, b], beta, GRID_VI).G));
    const sorted = Z.flat().sort((a, b) => a - b);
    const levels = Array.from({ length: N_LEVELS - 1 }, (_, k) => sorted[Math.floor((k + 1) / N_LEVELS * sorted.length)]);
    const entry = { ts, Gs, d: isoPaths(Z, t1, t2, levels, tBox.X, tBox.Y) };
    cache.set(beta, entry);
    return entry;
  }

  let shownBeta = null;
  function draw() {
    Object.entries(vals).forEach(([k, n]) => { n.textContent = fmt[k](S[k]); });
    const B = forBeta(S.beta);
    if (shownBeta !== S.beta) {
      contours.replaceChildren(...B.d.map(d => el('path', { d, class: 'tp-contour' })));
      starPt.move(B.ts);
      shownBeta = S.beta;
    }
    startPt.move(S.start);
    ['gd', 'fu'].forEach(m => {
      const { path, gap } = run(S.start, S.beta, B.Gs, m, m === 'gd' ? S.alpha : S.eps);
      paths[m].line.setAttribute('points', path.map(p => tBox.X(p[0]).toFixed(1) + ',' + tBox.Y(p[1]).toFixed(1)).join(' '));
      paths[m].dots.replaceChildren(...path.map(p => el('circle', { cx: tBox.X(p[0]).toFixed(1), cy: tBox.Y(p[1]).toFixed(1), r: 2.6 })));
      curves[m].setAttribute('points', gap.map((g, k) => CX(k).toFixed(1) + ',' + (isFinite(g) ? CY(g) : C.y0 - 10).toFixed(1)).join(' '));
    });
  }
  let queued = false;
  const schedule = () => { if (!queued) { queued = true; setTimeout(() => { queued = false; draw(); }, 0); } };

  // Drag (or click) in Θ-space to move θ⁽⁰⁾
  let dragging = false;
  const moveStart = ev => { const p = tBox.toData(ev); if (p) { S.start = p; schedule(); } };
  thetaSvg.addEventListener('pointerdown', ev => { ev.preventDefault(); dragging = true; thetaSvg.setPointerCapture(ev.pointerId); moveStart(ev); });
  thetaSvg.addEventListener('pointermove', ev => { if (dragging) moveStart(ev); });
  ['pointerup', 'pointercancel'].forEach(t => thetaSvg.addEventListener(t, () => { dragging = false; }));

  draw();
})();
