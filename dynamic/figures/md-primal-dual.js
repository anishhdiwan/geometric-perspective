/* ══════════════════════════════════════════════════════════════
 * FIGURE · ONE MIRROR DESCENT STEP IN THE PRIMAL AND THE DUAL SPACE   (theory.html → Mirror
 * Descent, above "An intuitive picture…"; <figure id="fig-md-primal-dual">)
 * After the primal-dual picture in anishhdiwan.github.io/notes/mirror-descent, with X = Θ.
 *
 * Objective: the dual G(θ) of the two-state problem (two-state-irl.js).
 * Mirror map: the quadratic map of Prop. 3.7, Ω(θ) = ⟨θ, Hθ⟩ / 2β, with H = H⋆ the integral
 * Fisher information at θ⋆ (TwoStateIRL.intFIM), fixed, so the dual space and y⋆ do not move. Then
 *   ∇Ω(θ) = Hθ / β          (primal Θ → dual space)
 *   (∇Ω)⁻¹(y) = β H⁻¹ y     (dual → primal)
 * One step: y⁽ⁱ⁾ = ∇Ω(θ⁽ⁱ⁾), y⁽ⁱ⁺¹⁾ = y⁽ⁱ⁾ − ε∇G(θ⁽ⁱ⁾), θ⁽ⁱ⁺¹⁾ = (∇Ω)⁻¹(y⁽ⁱ⁺¹⁾).
 * Left: Θ with the contours of G, θ⁽ⁱ⁾ (draggable), θ⁽ⁱ⁺¹⁾ and θ⋆. Right: the dual space in
 * coordinates y, with the contours of G((∇Ω)⁻¹(y)), y⁽ⁱ⁾, the step to y⁽ⁱ⁺¹⁾ and y⋆ = ∇Ω(θ⋆).
 * The maps between the spaces are two fixed short arrows in the gap (labels: HTML overlays, MathJax);
 * grey links that follow the points join θ⁽ⁱ⁾ → y⁽ⁱ⁾ and y⁽ⁱ⁺¹⁾ → θ⁽ⁱ⁺¹⁾.
 * The dual view is the square around the image of the primal box, so y⁽ⁱ⁾ is always inside.
 *
 * Edit the CONFIG block only. Styles: style.css (.mdp-*, and the .tp-* boxes).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-md-primal-dual');
  if (!fig || !window.TwoStateMDP || !window.TwoStateIRL || !window.PlotBox) return;
  const { el } = window.TwoStateMDP;
  const { BETA, VIEW, thStar, dual, dualGrad, intFIM, isoPaths, dualGrid } = window.TwoStateIRL;
  const L = window.PlotBox.label;

  // ── CONFIG ──────────────────────────────────────────────────
  const START = [-0.4, 0.8];             // initial θ⁽ⁱ⁾
  const EPS   = 0.018;                   // step size ε (≈ half-way to θ⋆ from START with H = H⋆)
  const PAD   = 0.04;                    // dual view: padding around the image of the primal box (fraction)
  const GRID = 41, GRID_VI = 80;         // dual-panel contour grid (the primal panel uses the shared grid of G)
  // ─────────────────────────────────────────────────────────────

  // Layout (viewBox 780 × 350): two 256-unit boxes with a gap between them that holds the two
  // fixed maps, ∇Ω (→, upper) and (∇Ω)⁻¹ (←, lower); their labels are HTML overlays (style.css)
  const S = 256, B = [{ x0: 34, y0: 48 }, { x0: 490, y0: 48 }];
  const gx0 = B[0].x0 + S + 16, gx1 = B[1].x0 - 34, ARR_UP = B[0].y0 + 0.40 * S, ARR_DN = B[0].y0 + 0.60 * S;
  const svg = fig.querySelector('.mdp-plot');
  const defs = el('defs', {}, svg);
  B.forEach((b, k) => el('rect', { x: b.x0, y: b.y0, width: S, height: S }, el('clipPath', { id: 'mdp-clip-' + k }, defs)));
  const marker = (id, cls, size) => {
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: size, markerHeight: size,
      markerUnits: 'userSpaceOnUse', orient: 'auto' }, defs);
    el('path', { d: 'M 0 1 L 10 5 L 0 9 z', class: cls }, m);
  };
  marker('mdp-arrow', 'mdp-arrowhead', 10); marker('mdp-step', 'mdp-stephead', 9); marker('mdp-link-arrow', 'mdp-linkhead', 7);

  // Fixed maps between the spaces, and grey links that follow the points: θ⁽ⁱ⁾ → y⁽ⁱ⁾ and y⁽ⁱ⁺¹⁾ → θ⁽ⁱ⁺¹⁾
  el('line', { class: 'mdp-map-arrow', 'marker-end': 'url(#mdp-arrow)', x1: gx0, y1: ARR_UP, x2: gx1, y2: ARR_UP }, svg);
  el('line', { class: 'mdp-map-arrow', 'marker-end': 'url(#mdp-arrow)', x1: gx1, y1: ARR_DN, x2: gx0, y2: ARR_DN }, svg);
  const links = el('g', { class: 'mdp-links' }, svg);
  const linkFwd = el('path', { class: 'mdp-link', 'marker-end': 'url(#mdp-link-arrow)' }, links);
  const linkBack = el('path', { class: 'mdp-link', 'marker-end': 'url(#mdp-link-arrow)' }, links);

  const view = [VIEW, null];
  const toX = (k, v) => B[k].x0 + (v - view[k].x[0]) / (view[k].x[1] - view[k].x[0]) * S;
  const toY = (k, v) => B[k].y0 + (view[k].y[1] - v) / (view[k].y[1] - view[k].y[0]) * S;
  const P = (k, p) => [toX(k, p[0]), toY(k, p[1])];

  const layers = B.map((b, k) => {
    el('text', { x: b.x0 + S / 2, y: b.y0 - 12, class: 'tp-title', 'text-anchor': 'middle' }, svg)
      .textContent = k ? 'Dual space' : 'Primal space';
    const under = el('g', { 'clip-path': 'url(#mdp-clip-' + k + ')' }, svg);
    const contours = el('g', {}, under);
    el('rect', { x: b.x0, y: b.y0, width: S, height: S, class: 'tp-frame' + (k ? '' : ' mdp-frame--primal') }, svg);
    const sym = k ? 'y' : 'θ';
    el('text', { x: b.x0 + S / 2, y: b.y0 + S + 25, class: 'tp-axis', 'text-anchor': 'middle' }, svg)
      .innerHTML = '<tspan font-style="italic">' + sym + '</tspan><tspan dy="4" class="tp-sub">1</tspan>';
    el('text', { x: b.x0 - 16, y: b.y0 + S / 2, class: 'tp-axis', 'text-anchor': 'middle',
      transform: 'rotate(-90 ' + (b.x0 - 16) + ' ' + (b.y0 + S / 2) + ')' }, svg)
      .innerHTML = '<tspan font-style="italic">' + sym + '</tspan><tspan dy="4" class="tp-sub">2</tspan>';
    return { contours, marks: el('g', { 'clip-path': 'url(#mdp-clip-' + k + ')' }, svg) };
  });
  // The update in the dual space, y⁽ⁱ⁾ → y⁽ⁱ⁺¹⁾
  const stepD = el('line', { class: 'mdp-step', 'marker-end': 'url(#mdp-step)' }, layers[1].marks);

  const pt = (k, lab, cls) => {
    const g = el('g', { class: 'mdp-pt ' + cls }, layers[k].marks);
    return { c: el('circle', { r: 5 }, g), t: Object.assign(el('text', { class: 'tp-label' }, g), { innerHTML: lab }) };
  };
  const mk = {
    th0: pt(0, L('θ', { sup: '(i)' }), 'mdp-pt--start'), th1: pt(0, L('θ', { sup: '(i+1)' }), 'mdp-pt--next'),
    ths: pt(0, L('θ', { sup: '⋆' }), 'mdp-pt--star'),
    y0: pt(1, L('y', { sup: '(i)' }), 'mdp-pt--start'), y1: pt(1, L('y', { sup: '(i+1)' }), 'mdp-pt--next'),
    ys: pt(1, L('y', { sup: '⋆' }), 'mdp-pt--star'),
  };
  const put = (m, [x, y], dx, dy, anchor) => {
    m.c.setAttribute('cx', x); m.c.setAttribute('cy', y);
    m.t.setAttribute('x', x + dx); m.t.setAttribute('y', y + dy); m.t.setAttribute('text-anchor', anchor);
  };
  const line = (ln, a, b) => [['x1', a[0]], ['y1', a[1]], ['x2', b[0]], ['y2', b[1]]].forEach(([k, v]) => ln.setAttribute(k, v));

  // Primal contours (shared grid of G) and levels shared with the dual panel
  const axis = (v, n) => Array.from({ length: n }, (_, i) => v[0] + i * (v[1] - v[0]) / (n - 1));
  const { ts1: px, ts2: py, Z: Zp, levels } = dualGrid();
  isoPaths(Zp, px, py, levels, v => toX(0, v), v => toY(0, v))
    .forEach(d => el('path', { d, class: 'tp-contour' }, layers[0].contours));

  const mul = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
  const inv = M => { const d = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]; };

  // Fixed mirror map (H = H⋆), fixed dual view (square around the image of the primal box) and contours
  const H = intFIM(thStar), Hi = inv(H);
  const toDual = th => mul(H, th).map(v => v / BETA);                // ∇Ω(θ) = Hθ/β
  const toPrimal = y => mul(Hi, y).map(v => v * BETA);               // (∇Ω)⁻¹(y) = βH⁻¹y
  const ys = toDual(thStar);
  {
    const corners = [[VIEW.x[0], VIEW.y[0]], [VIEW.x[0], VIEW.y[1]], [VIEW.x[1], VIEW.y[0]], [VIEW.x[1], VIEW.y[1]]].map(toDual);
    const xs = corners.map(p => p[0]), yv = corners.map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...yv) + Math.max(...yv)) / 2;
    const half = (1 + PAD) * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...yv) - Math.min(...yv)) / 2;
    view[1] = { x: [cx - half, cx + half], y: [cy - half, cy + half] };
    const qx = axis(view[1].x, GRID), qy = axis(view[1].y, GRID);
    const Zd = qy.map(b => qx.map(a => dual(toPrimal([a, b]), GRID_VI)));
    isoPaths(Zd, qx, qy, levels, v => toX(1, v), v => toY(1, v)).forEach(d => el('path', { d, class: 'tp-contour' }, layers[1].contours));
  }

  let theta = START.slice();
  function draw() {
    const g = dualGrad(theta), y0 = toDual(theta);
    const y1 = [y0[0] - EPS * g[0], y0[1] - EPS * g[1]], th1 = toPrimal(y1);
    line(stepD, P(1, y0), P(1, y1));
    // Links: curved, bowing up for ∇Ω and down for (∇Ω)⁻¹, stopping short of the target point
    const link = (path, a, b, bow) => {
      const dx = b[0] - a[0], dy = b[1] - a[1], n = Math.hypot(dx, dy) || 1, e = [b[0] - 8 * dx / n, b[1] - 8 * dy / n];
      const mx = (a[0] + e[0]) / 2, my = (a[1] + e[1]) / 2 + bow;
      path.setAttribute('d', 'M ' + a[0] + ' ' + a[1] + ' Q ' + mx + ' ' + my + ' ' + e[0] + ' ' + e[1]);
    };
    link(linkFwd, P(0, theta), P(1, y0), -40); link(linkBack, P(1, y1), P(0, th1), 40);
    put(mk.th0, P(0, theta), -8, -9, 'end'); put(mk.th1, P(0, th1), 8, 16, 'start'); put(mk.ths, P(0, thStar), 8, 18, 'start');
    put(mk.y0, P(1, y0), -8, -9, 'end');     put(mk.y1, P(1, y1), 8, 16, 'start');   put(mk.ys, P(1, ys), 8, 18, 'start');
  }

  // Drag θ⁽ⁱ⁾ in the primal box
  let dragging = false, queued = false;
  const schedule = () => { if (!queued) { queued = true; setTimeout(() => { queued = false; draw(); }, 0); } };
  const toData = ev => {
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse());
    if (p.x < B[0].x0 || p.x > B[0].x0 + S || p.y < B[0].y0 || p.y > B[0].y0 + S) return null;
    return [view[0].x[0] + (p.x - B[0].x0) / S * (view[0].x[1] - view[0].x[0]),
            view[0].y[1] - (p.y - B[0].y0) / S * (view[0].y[1] - view[0].y[0])];
  };
  svg.addEventListener('pointerdown', ev => {
    const p = toData(ev); if (!p) return;
    ev.preventDefault(); dragging = true; svg.setPointerCapture(ev.pointerId); theta = p; schedule();
  });
  // The primal box is the interactive one: its frame thickens on hover (as the Θ boxes elsewhere)
  svg.addEventListener('pointermove', ev => {
    const p = toData(ev);
    svg.classList.toggle('is-primal-hover', !!p || dragging);
    svg.style.cursor = p ? 'crosshair' : '';
    if (dragging && p) { theta = p; schedule(); }
  });
  svg.addEventListener('pointerleave', () => { if (!dragging) svg.classList.remove('is-primal-hover'); });
  ['pointerup', 'pointercancel'].forEach(t => svg.addEventListener(t, () => {
    dragging = false;
  }));

  draw();
})();
