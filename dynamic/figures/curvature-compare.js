/* ══════════════════════════════════════════════════════════════
 * FIGURE · CURVATURE SEEN BY GRADIENT DESCENT AND BY THE FUNCTIONAL UPDATE
 * (theory.html → First-Order Contraction, after Corollary C.3; <figure id="fig-curvature">)
 *
 * From a draggable θ⁽⁰⁾, N_ITER steps of
 *   gradient descent     θ ← θ − α ∇G(θ)
 *   functional update    θ ← θ − εβ (H⁽ⁱ⁾)⁻¹ ∇G(θ)      (integral-FIM form, Lemma 3.8)
 * on the dual G of the two-state problem. At every iterate:
 *   H̄⁽ⁱ⁾ = integral Hessian of G from θ⋆ to θ⁽ⁱ⁾ (TwoStateIRL.intHess)
 *   H⁽ⁱ⁾ = integral FIM along the Φ-segment (TwoStateIRL.intFIM)
 * Cell ellipses: the level set ½ dᵀ S d = CELL_LEVEL of the step-scaled curvature at each method's best
 * step size, S = 2M / (λ_min(M) + λ_max(M)) (= α⋆ H̄ for GD, ε⋆β H^{-1/2} H̄ H^{-1/2} for the functional
 * update). Its eigenvalues lie in [2/(κ+1), 2κ/(κ+1)] around 1, so sizes are comparable across rows and
 * the shape shows κ. (An eigenvalue 1 means one best step lands on the minimum in that direction.)
 *   gradient descent     M = H̄⁽ⁱ⁾ (Θ coordinates)
 *   functional update    M = (H⁽ⁱ⁾)^{-1/2} H̄⁽ⁱ⁾ (H⁽ⁱ⁾)^{-1/2}, the curvature in the update's own
 *                        (preconditioned) coordinates; it has the eigenvalues of (H⁽ⁱ⁾)⁻¹H̄⁽ⁱ⁾
 * Each ellipse is labelled with κ and the best contraction coefficient (κ − 1)/(κ + 1); for the functional
 * update also the norm-change factor c = √λ_max((H⁽ⁱ⁾)⁻¹H⁽ⁱ⁺¹⁾) and c × coeff. (Corollary C.3).
 * The 3D panel draws the quadratic approximation ½ dᵀ H̄⁽⁰⁾ d (isometric) with the same level set.
 *
 * Edit the CONFIG block only. Styles: style.css (.cc-*, and the .tp-* boxes).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-curvature');
  if (!fig || !window.TwoStateMDP || !window.TwoStateIRL || !window.PlotBox) return;
  const { el } = window.TwoStateMDP;
  const { BETA, VIEW, thStar, dualGrad, intHess, intFIM, contourPaths } = window.TwoStateIRL;
  const L = window.PlotBox.label;

  // ── CONFIG ──────────────────────────────────────────────────
  const START  = [-0.4, 0.8];            // θ⁽⁰⁾
  const ALPHA  = 4;                      // gradient-descent step size
  const EPS    = 0.06;                   // functional-update step size
  const N_ITER = 3;                      // iterations shown (ellipses for i = 0 … N_ITER − 1)
  const LEVEL  = 0.02;                   // level set ½ dᵀ M d = LEVEL in the 3D panel
  const CELL_LEVEL = 0.15;               // level set ½ dᵀ S d = CELL_LEVEL of the step-scaled curvature S in the cells
  const GLYPH  = 1.0;                    // half-width of each ellipse cell, in d units (grown if an ellipse would not fit)
  // ─────────────────────────────────────────────────────────────

  // 2×2 helpers
  const symEig = M => {                                           // eigenvalues (desc) and eigenvectors of a symmetric 2×2
    const a = M[0][0], b = (M[0][1] + M[1][0]) / 2, c = M[1][1];
    const tr = a + c, disc = Math.sqrt(Math.max(0, (a - c) * (a - c) / 4 + b * b));
    const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    let v = Math.abs(b) > 1e-14 ? [l1 - c, b] : (a >= c ? [1, 0] : [0, 1]);
    const n = Math.hypot(v[0], v[1]); v = [v[0] / n, v[1] / n];
    return { l: [l1, l2], v: [v, [-v[1], v[0]]] };
  };
  const symFn = (M, f) => { const { l, v } = symEig(M);
    return [0, 1].map(i => [0, 1].map(j => f(l[0]) * v[0][i] * v[0][j] + f(l[1]) * v[1][i] * v[1][j])); };
  const mm = (A, B) => [0, 1].map(i => [0, 1].map(j => A[i][0] * B[0][j] + A[i][1] * B[1][j]));
  const solve = (M, g) => { const d = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    return [(M[1][1] * g[0] - M[0][1] * g[1]) / d, (M[0][0] * g[1] - M[1][0] * g[0]) / d]; };

  function run(th0, method) {
    let th = th0.slice();
    const its = [];
    for (let k = 0; k <= N_ITER; k++) {
      if (k === N_ITER) { its.push({ th, H: method === 'fu' ? intFIM(th) : null }); break; }
      const Hb = intHess(th), g = dualGrad(th);
      let M, d, H = null;
      if (method === 'gd') { M = Hb; d = [ALPHA * g[0], ALPHA * g[1]]; }
      else {
        H = intFIM(th); const Hih = symFn(H, x => 1 / Math.sqrt(x));
        M = mm(mm(Hih, Hb), Hih);                                   // same eigenvalues as H⁻¹H̄
        const s = solve(H, g); d = [EPS * BETA * s[0], EPS * BETA * s[1]];
      }
      const { l } = symEig(M), kappa = l[0] / l[1];
      its.push({ th, M, H, kappa, coeff: (kappa - 1) / (kappa + 1) });
      th = [th[0] - d[0], th[1] - d[1]];
    }
    // Functional update: the H⁽ⁱ⁾-norm changes between iterations, so scale the coefficient by
    // c = √λ_max((H⁽ⁱ⁾)⁻¹H⁽ⁱ⁺¹⁾) (Corollary C.3); λ_max via (H⁽ⁱ⁾)^{-1/2} H⁽ⁱ⁺¹⁾ (H⁽ⁱ⁾)^{-1/2}
    if (method === 'fu') its.slice(0, N_ITER).forEach((o, k) => {
      const Hih = symFn(o.H, x => 1 / Math.sqrt(x));
      o.c = Math.sqrt(symEig(mm(mm(Hih, its[k + 1].H), Hih)).l[0]);
    });
    return its;
  }

  // ── Θ-space box ─────────────────────────────────────────────
  const thetaSvg = fig.querySelector('.cc-theta');
  const tBox = window.PlotBox(thetaSvg, { id: 'cc-theta', title: 'Θ-space', sym: 'θ', view: VIEW });
  contourPaths(tBox.X, tBox.Y).forEach(d => el('path', { d, class: 'tp-contour' }, tBox.under));
  const gLabel = document.createElement('div');                 // 𝒢(θ) in the box's bottom-left corner (as theta-space.js)
  gLabel.className = 'tp-glabel';
  gLabel.textContent = '$\\mathcal{G}(\\theta)$';
  thetaSvg.parentNode.appendChild(gLabel);
  const runs = {};
  ['gd', 'fu'].forEach(m => {
    const g = el('g', { class: 'cc-run cc-run--' + m }, tBox.marks);
    runs[m] = { line: el('polyline', { class: 'cc-line' }, g), dots: el('g', {}, g) };
  });
  tBox.point(thStar, { label: L('θ', { sup: '⋆' }) });
  const startPt = tBox.point(START, { label: L('θ', { sup: '(0)' }), cls: 'cc-start' });

  // ── 3D quadratic approximation (isometric) ──────────────────────────
  const q3 = fig.querySelector('.cc-3d');
  const C3 = { cx: 150, cy: 150, s: 92, zs: 260 };               // centre, xy scale, z scale (SVG units; set per M)
  const proj = (x, y, z) => [C3.cx + (x - y) * 0.866 * C3.s, C3.cy + (x + y) * 0.5 * C3.s - z * C3.zs];
  const g3 = { grid: el('g', { class: 'cc-grid' }, q3), floor: el('path', { class: 'cc-floor' }, q3),
               lvl: el('path', { class: 'cc-level' }, q3), shadow: el('path', { class: 'cc-shadow' }, q3),
               long: el('line', { class: 'cc-axis cc-axis--long' }, q3), short: el('line', { class: 'cc-axis cc-axis--short' }, q3) };
  const labLong = el('text', { class: 'cc-axlab' }, q3), labShort = el('text', { class: 'cc-axlab' }, q3);
  labLong.textContent = 'long axis: flat'; labShort.textContent = 'short axis: steep';
  function draw3d(M) {
    const q = (x, y) => 0.5 * (M[0][0] * x * x + 2 * M[0][1] * x * y + M[1][1] * y * y);
    const { l, v } = symEig(M);
    const R = 1.35 * Math.sqrt(2 * LEVEL / l[1]);                 // grid just wider than the long axis
    const zmax = 4 * LEVEL, n = 12;                                // bowl drawn up to 4 × the level
    C3.s = 92 / R; C3.zs = 120 / zmax;                             // fit the panel whatever M is
    const zc = z => Math.min(z, zmax), pts = [];
    for (let k = 0; k <= n; k++) {
      const u = -R + 2 * R * k / n;
      let a = '', b = '';
      for (let j = 0; j <= 40; j++) {
        const w = -R + 2 * R * j / 40;
        const p1 = proj(u, w, zc(q(u, w))), p2 = proj(w, u, zc(q(w, u)));
        a += (j ? 'L' : 'M') + p1[0].toFixed(1) + ' ' + p1[1].toFixed(1);
        b += (j ? 'L' : 'M') + p2[0].toFixed(1) + ' ' + p2[1].toFixed(1);
      }
      pts.push(a, b);
    }
    g3.grid.replaceChildren(...pts.map(d => el('path', { d })));
    const sq = [[-R, -R], [R, -R], [R, R], [-R, R]].map(([x, y]) => proj(x, y, 0));
    g3.floor.setAttribute('d', 'M' + sq.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('L') + 'Z');
    const ring = [], shadow = [];
    for (let j = 0; j <= 80; j++) {
      const t = 2 * Math.PI * j / 80, a0 = Math.sqrt(2 * LEVEL / l[1]) * Math.cos(t), a1 = Math.sqrt(2 * LEVEL / l[0]) * Math.sin(t);
      const x = a0 * v[1][0] + a1 * v[0][0], y = a0 * v[1][1] + a1 * v[0][1];
      ring.push(proj(x, y, LEVEL)); shadow.push(proj(x, y, 0));
    }
    const path = P => 'M' + P.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('L');
    g3.lvl.setAttribute('d', path(ring)); g3.shadow.setAttribute('d', path(shadow));
    // Half-axes on the floor, labels just beyond their ends, pointing away from the centre
    const o = proj(0, 0, 0);
    const ax = (line, lab, lam, dir) => {
      const r = Math.sqrt(2 * LEVEL / lam), p = proj(r * dir[0], r * dir[1], 0);
      [['x1', o[0]], ['y1', o[1]], ['x2', p[0]], ['y2', p[1]]].forEach(([k, val]) => line.setAttribute(k, val));
      const dx = p[0] - o[0], dy = p[1] - o[1], nn = Math.hypot(dx, dy) || 1;
      lab.setAttribute('x', p[0] + 10 * dx / nn); lab.setAttribute('y', p[1] + 10 * dy / nn + 4);
      lab.setAttribute('text-anchor', dx >= 0 ? 'start' : 'end');
      try {                                                        // keep the label inside the 300-wide panel
        const bb = lab.getBBox();
        if (bb.x + bb.width > 298) { lab.setAttribute('text-anchor', 'end'); lab.setAttribute('x', 298); }
        if (bb.x < 2) { lab.setAttribute('text-anchor', 'start'); lab.setAttribute('x', 2); }
      } catch (e) { /* not rendered yet */ }
    };
    ax(g3.long, labLong, l[1], v[1][0] * 0.866 - v[1][1] * 0.866 >= 0 ? v[1] : [-v[1][0], -v[1][1]]);
    ax(g3.short, labShort, l[0], v[0][0] * 0.866 - v[0][1] * 0.866 < 0 ? v[0] : [-v[0][0], -v[0][1]]);
  }

  // ── Ellipse grid (rows: methods; columns: iterations) ───────
  const grid = fig.querySelector('.cc-cells');
  const cells = { gd: [], fu: [] };
  ['gd', 'fu'].forEach(m => {
    for (let k = 0; k < N_ITER; k++) {
      const cell = grid.querySelector('[data-m="' + m + '"][data-i="' + k + '"]');
      const svg = el('svg', { viewBox: '-60 -60 120 120', class: 'cc-cell-svg' }, cell.querySelector('.cc-cell-plot'));
      el('rect', { x: -56, y: -56, width: 112, height: 112, class: 'tp-frame' }, svg);
      el('line', { x1: -56, y1: 0, x2: 56, y2: 0, class: 'cc-cross' }, svg);
      el('line', { x1: 0, y1: -56, x2: 0, y2: 56, class: 'cc-cross' }, svg);
      cells[m].push({ path: el('path', { class: 'cc-ell cc-ell--' + m }, svg), val: cell.querySelector('.cc-val') });
    }
  });
  const stepScaled = M => { const { l } = symEig(M), k = 2 / (l[0] + l[1]); return [[k * M[0][0], k * M[0][1]], [k * M[1][0], k * M[1][1]]]; };
  const ellipse = (M, half) => {
    const { l, v } = symEig(stepScaled(M)), sc = 56 / half, P = [];
    for (let j = 0; j <= 64; j++) {
      const t = 2 * Math.PI * j / 64, a0 = Math.sqrt(2 * CELL_LEVEL / l[1]) * Math.cos(t), a1 = Math.sqrt(2 * CELL_LEVEL / l[0]) * Math.sin(t);
      const x = a0 * v[1][0] + a1 * v[0][0], y = a0 * v[1][1] + a1 * v[0][1];
      P.push((x * sc).toFixed(1) + ' ' + (-y * sc).toFixed(1));
    }
    return 'M' + P.join('L') + 'Z';
  };

  let start = START.slice();
  function draw() {
    startPt.move(start);
    const res = { gd: run(start, 'gd'), fu: run(start, 'fu') };
    // One scale for every cell, so the ellipses stay comparable: wide enough for the longest half-axis
    const longest = Math.max(...['gd', 'fu'].flatMap(m => res[m].slice(0, N_ITER).map(o => Math.sqrt(2 * CELL_LEVEL / symEig(stepScaled(o.M)).l[1]))));
    const half = Math.max(GLYPH, longest / 0.9);
    ['gd', 'fu'].forEach(m => {
      const pts = res[m].map(o => o.th);
      runs[m].line.setAttribute('points', pts.map(p => tBox.X(p[0]).toFixed(1) + ',' + tBox.Y(p[1]).toFixed(1)).join(' '));
      runs[m].dots.replaceChildren(...pts.map(p => el('circle', { cx: tBox.X(p[0]).toFixed(1), cy: tBox.Y(p[1]).toFixed(1), r: 3 })));
      res[m].slice(0, N_ITER).forEach((o, k) => {
        cells[m][k].path.setAttribute('d', ellipse(o.M, half));
        cells[m][k].val.innerHTML = m === 'gd'
          ? 'κ = ' + o.kappa.toFixed(1) + ', coeff. ' + o.coeff.toFixed(2)
          : 'κ = ' + o.kappa.toFixed(1) + ', c = ' + o.c.toFixed(2) + '<br>c × coeff. ' + (o.c * o.coeff).toFixed(2);
      });
    });
    draw3d(res.gd[0].M);
  }

  // Drag θ⁽⁰⁾ in Θ
  let dragging = false, queued = false;
  const schedule = () => { if (!queued) { queued = true; setTimeout(() => { queued = false; draw(); }, 0); } };
  thetaSvg.addEventListener('pointerdown', ev => { const p = tBox.toData(ev); if (!p) return;
    ev.preventDefault(); dragging = true; thetaSvg.setPointerCapture(ev.pointerId); start = p; schedule(); });
  thetaSvg.addEventListener('pointermove', ev => { if (!dragging) return; const p = tBox.toData(ev); if (p) { start = p; schedule(); } });
  ['pointerup', 'pointercancel'].forEach(t => thetaSvg.addEventListener(t, () => { dragging = false; }));

  draw();
})();
