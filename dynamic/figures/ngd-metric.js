/* ══════════════════════════════════════════════════════════════
 * FIGURE · THE DUAL UNDER THE EUCLIDEAN AND THE FISHER METRIC   (theory.html →
 * Function-Space Reward Updates, Natural Gradient Descent; <figure id="fig-ngd-metric">)
 *
 * Static. The objective is the dual G(θ) of the two-state problem (two-state-irl.js).
 *   left   Θ with the Euclidean metric: contours of G, the steepest step −∇G at θ⁽ⁱ⁾ on the
 *          circle of steps of equal Euclidean length
 *   right  the same contours in coordinates η = F^{1/2}(θ − θ⁽ⁱ⁾), in which the Fisher metric
 *          at θ⁽ⁱ⁾ is Euclidean; the steepest step there is −F^{-1/2}∇G, i.e. the natural
 *          gradient −F^{-1}∇G in θ
 * F = F(θ⁽ⁱ⁾) = Σ_{s,a} ρ ∇_θ log ρ ∇_θ log ρᵀ is the Fisher information of the agent's occupancy
 * ρ = ρ_{π_θ} with respect to θ (π_θ the soft-optimal policy of r_θ; ∇_θ log ρ by central
 * differences). Both panels use the same contour levels.
 *
 * Edit the CONFIG block only. Styles: style.css (.nm-*, and the .tp-* boxes and arrows).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-ngd-metric');
  if (!fig || !window.TwoStateMDP || !window.TwoStateIRL || !window.PlotBox) return;
  const { el } = window.TwoStateMDP;
  const { VIEW, thStar, dual, dualGrad, rhoOf, isoPaths, dualGrid } = window.TwoStateIRL;
  const L = window.PlotBox.label;

  // ── CONFIG ──────────────────────────────────────────────────
  const THETA   = [0, 0];            // θ⁽ⁱ⁾ where both steps are drawn
  const RADIUS  = 0.4;               // step length ε, as a fraction of the distance θ⁽ⁱ⁾ → θ⋆ in each panel
  const PAD     = 1.35;              // right-panel view: half-width relative to the extent of {0, η⋆}
  const GRID = 41, GRID_VI = 80;     // right-panel contour grid (the left panel uses the shared grid of G)
  // ─────────────────────────────────────────────────────────────

  // Fisher information of ρ_{π_θ} w.r.t. θ at θ⁽ⁱ⁾, and its symmetric square roots
  const h = 1e-4, rho = rhoOf(THETA);
  const dlog = [0, 1].map(k => {
    const u = THETA.slice(), d = THETA.slice(); u[k] += h; d[k] -= h;
    const ru = rhoOf(u), rd = rhoOf(d);
    return ru.map((v, i) => (Math.log(v) - Math.log(rd[i])) / (2 * h));
  });
  const F = [0, 1].map(a => [0, 1].map(b => rho.reduce((s, p, i) => s + p * dlog[a][i] * dlog[b][i], 0)));
  const sym = (M, f) => {                                          // f(M) for a symmetric 2×2 M, via its eigenvectors
    const tr = M[0][0] + M[1][1], det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)), l = [tr / 2 + disc, tr / 2 - disc];
    const v1 = Math.abs(M[0][1]) > 1e-14 ? [l[0] - M[1][1], M[0][1]] : (M[0][0] >= M[1][1] ? [1, 0] : [0, 1]);
    const n = Math.hypot(v1[0], v1[1]), a = [v1[0] / n, v1[1] / n], b = [-a[1], a[0]];
    return [0, 1].map(i => [0, 1].map(j => f(l[0]) * a[i] * a[j] + f(l[1]) * b[i] * b[j]));
  };
  const Fh = sym(F, Math.sqrt), Fih = sym(F, x => 1 / Math.sqrt(x));
  const mul = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
  const toTheta = eta => { const t = mul(Fih, eta); return [THETA[0] + t[0], THETA[1] + t[1]]; };
  const toEta = th => mul(Fh, [th[0] - THETA[0], th[1] - THETA[1]]);

  const g = dualGrad(THETA);
  const etaStar = toEta(thStar);
  const half = PAD * Math.max(Math.abs(etaStar[0]), Math.abs(etaStar[1]), 1e-3);
  const c = [etaStar[0] / 2, etaStar[1] / 2];
  const VIEW_ETA = { x: [c[0] - half, c[0] + half], y: [c[1] - half, c[1] + half] };

  // Contour grids with shared levels (the Euclidean panel uses the shared grid of G and its levels)
  const axis = (v, n) => Array.from({ length: n }, (_, i) => v[0] + i * (v[1] - v[0]) / (n - 1));
  const { ts1: tx, ts2: ty, Z: Zt, levels } = dualGrid();
  const ex = axis(VIEW_ETA.x, GRID), ey = axis(VIEW_ETA.y, GRID);
  const Ze = ey.map(b => ex.map(a => dual(toTheta([a, b]), GRID_VI)));

  function panel(svgSel, { id, title, sym: s, view, Z, xs, ys, at, star, dir, cls }) {
    const svg = fig.querySelector(svgSel);
    const box = window.PlotBox(svg, { id, title, sym: s, view });
    isoPaths(Z, xs, ys, levels, box.X, box.Y).forEach(d => el('path', { d, class: 'tp-contour' }, box.under));
    const dist = Math.hypot(star[0] - at[0], star[1] - at[1]), r = RADIUS * dist;
    const px = box.X(at[0]), py = box.Y(at[1]), rr = box.X(at[0] + r) - px;
    el('circle', { cx: px, cy: py, r: rr, class: 'nm-ball' }, box.under);
    el('line', { x1: px, y1: py, x2: box.X(star[0]), y2: box.Y(star[1]), class: 'nm-tostar' }, box.under);
    const defs = el('defs', {}, svg);
    const m = el('marker', { id: id + '-m', class: 'nm-m--' + cls, viewBox: '0 0 10 10', refX: 7, refY: 5,
      markerWidth: 9, markerHeight: 9, markerUnits: 'userSpaceOnUse', orient: 'auto' }, defs);
    el('path', { d: 'M 0 1 L 10 5 L 0 9 z' }, m);
    const n = Math.hypot(dir[0], dir[1]) || 1;
    el('line', { x1: px, y1: py, x2: box.X(at[0] + r * dir[0] / n), y2: box.Y(at[1] + r * dir[1] / n),
      class: 'tp-g nm-g--' + cls, 'marker-end': 'url(#' + id + '-m)' }, box.under);
    box.point(star, { label: L('θ', { sup: '⋆' }) });
    box.point(at, { label: L('θ', { sup: '(i)' }), side: 'left' });
  }

  // Left: Euclidean. Steepest step −∇G.
  panel('.nm-euc', { id: 'nm-euc', title: 'Euclidean metric', sym: 'θ', view: VIEW, Z: Zt, xs: tx, ys: ty,
    at: THETA, star: thStar, dir: [-g[0], -g[1]], cls: 'grad' });
  // Right: Fisher. In η the steepest step is −F^{-1/2}∇G (the natural gradient mapped to η).
  const gEta = mul(Fih, g);
  panel('.nm-fis', { id: 'nm-fis', title: 'Fisher metric', sym: 'η', view: VIEW_ETA, Z: Ze, xs: ex, ys: ey,
    at: [0, 0], star: etaStar, dir: [-gEta[0], -gEta[1]], cls: 'nat' });
})();
