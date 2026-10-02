/* ══════════════════════════════════════════════════════════════
 * FIGURE · THE BREGMAN DIVERGENCE IN ONE DIMENSION   (theory.html → Mirror Descent,
 * below the definition of D_Ω; <figure id="fig-bregman">)
 * After the illustration by Constantine Caramanis (as in anishhdiwan.github.io/notes/mirror-descent).
 *
 * The mirror map Ω(x) (curve), its tangent at x₂ (grey line), and the Bregman divergence
 *   D_Ω(x₁, x₂) = Ω(x₁) − [Ω(x₂) + Ω'(x₂)(x₁ − x₂)]
 * as the red vertical gap at x₁ between the curve and the tangent. Drag x₁ and x₂ (the
 * nearer one follows the pointer, clamped to the map's drag range); arrow keys move x₁.
 * The chips choose between the paper's two mirror maps, in one dimension:
 *   quad    Ω(θ) = ⟨θ, Hθ⟩ / 2β (Prop. 3.7) with H/β = 1, i.e. ½x²; D_Ω = ½(x₁ − x₂)²
 *   negent  Ω(ρ) = Σ ρ log ρ (entropic MD), i.e. x log x;      D_Ω = x₁ log(x₁/x₂) − x₁ + x₂
 *
 * Edit the CONFIG block only. Styles: style.css (.bg-*).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-bregman');
  if (!fig || !window.TwoStateMDP) return;
  const { el } = window.TwoStateMDP;

  // ── CONFIG ──────────────────────────────────────────────────
  // Each mirror map: Ω, Ω', the x and y ranges shown, the range x₁, x₂ can be dragged in,
  // and their start values
  const MAPS = {
    quad:   { f: x => 0.5 * x * x, df: x => x,
              x: [-2.4, 2.4], y: [-1.0, 3.2], drag: [-2.1, 2.1], start: [1.6, 0.3] },
    negent: { f: x => (x > 0 ? x * Math.log(x) : 0), df: x => Math.log(x) + 1,
              x: [0, 3], y: [-1.0, 3.2], drag: [0.08, 2.8], start: [2.5, 1.1] },
  };
  const N = 240;                                   // curve resolution
  // ─────────────────────────────────────────────────────────────

  const P = { x0: 26, x1: 326, y0: 18, y1: 296 };                 // plot area in the 340 × 340 viewBox
  const svg = fig.querySelector('.bg-plot');
  const chips = [...fig.querySelectorAll('.chip[data-map]')];
  const readout = fig.querySelector('.bg-val');
  let key = 'quad', M = MAPS[key], pts = M.start.slice();

  const X = x => P.x0 + (x - M.x[0]) / (M.x[1] - M.x[0]) * (P.x1 - P.x0);
  const Y = y => P.y1 - (y - M.y[0]) / (M.y[1] - M.y[0]) * (P.y1 - P.y0);
  const invX = px => M.x[0] + (px - P.x0) / (P.x1 - P.x0) * (M.x[1] - M.x[0]);
  const clampX = x => Math.min(M.drag[1], Math.max(M.drag[0], x));   // dragging past the range keeps the point at its edge

  const clip = el('clipPath', { id: 'bg-clip' }, el('defs', {}, svg));
  el('rect', { x: P.x0, y: P.y0, width: P.x1 - P.x0, height: P.y1 - P.y0 }, clip);
  const arrow = el('marker', { id: 'bg-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 8, markerHeight: 8,
    markerUnits: 'userSpaceOnUse', orient: 'auto' }, svg.querySelector('defs'));
  el('path', { d: 'M 0 1 L 10 5 L 0 9 z', class: 'bg-arrowhead' }, arrow);

  const xAxis = el('line', { class: 'bg-axis' }, svg), yAxis = el('line', { class: 'bg-axis' }, svg);
  const plot = el('g', { 'clip-path': 'url(#bg-clip)' }, svg);
  const tangent = el('line', { class: 'bg-tangent' }, plot);
  const curve = el('path', { class: 'bg-curve' }, plot);
  const gap = el('line', { class: 'bg-gap' }, svg);
  const ticks = [0, 1].map(() => el('line', { class: 'bg-tick' }, svg));
  const tickLab = [0, 1].map(i => {
    const t = el('text', { class: 'bg-xlab', 'text-anchor': 'middle' }, svg);
    t.innerHTML = '<tspan font-style="italic">x</tspan><tspan dy="4" class="bg-sub">' + (i + 1) + '</tspan>';
    return t;
  });
  const dotTan = el('circle', { r: 5, class: 'bg-dot bg-dot--tan' }, svg);   // (x₂, Ω(x₂))
  const dotF   = el('circle', { r: 5, class: 'bg-dot' }, svg);               // (x₁, Ω(x₁))
  const dotLin = el('circle', { r: 5, class: 'bg-dot bg-dot--lin' }, svg);   // (x₁, tangent at x₁)
  const labF = el('text', { class: 'bg-lab' }, svg);
  labF.innerHTML = 'Ω(<tspan font-style="italic">x</tspan><tspan dy="4" class="bg-sub">1</tspan><tspan dy="-4">)</tspan>';
  const yLab = el('text', { class: 'bg-ylab', 'text-anchor': 'end' }, svg);   // Ω on the y axis
  yLab.textContent = 'Ω';
  const redLab = el('text', { class: 'bg-red' }, svg);
  redLab.textContent = 'Bregman divergence';
  const leader = el('path', { class: 'bg-leader', 'marker-end': 'url(#bg-arrow)' }, svg);

  function draw() {
    const [x1, x2] = pts, f = M.f, df = M.df;
    const lin = x => f(x2) + df(x2) * (x - x2);
    // Axes: x at y = 0 (or the bottom), y at x = 0 (or the left edge)
    const ay = Math.min(P.y1, Math.max(P.y0, Y(0))), ax = Math.min(P.x1, Math.max(P.x0, X(0)));
    [['x1', P.x0], ['y1', ay], ['x2', P.x1], ['y2', ay]].forEach(([a, v]) => xAxis.setAttribute(a, v));
    [['x1', ax], ['y1', P.y0], ['x2', ax], ['y2', P.y1]].forEach(([a, v]) => yAxis.setAttribute(a, v));
    yLab.setAttribute('x', ax - 7); yLab.setAttribute('y', P.y0 + 14);
    const lo = M.x[0] > 0 || key === 'negent' ? Math.max(1e-3, M.x[0]) : M.x[0];   // x log x is only defined for x > 0
    const xs = Array.from({ length: N + 1 }, (_, i) => lo + i * (M.x[1] - lo) / N);
    curve.setAttribute('d', 'M' + xs.map(x => X(x).toFixed(1) + ' ' + Y(f(x)).toFixed(1)).join('L'));
    [['x1', X(M.x[0])], ['y1', Y(lin(M.x[0]))], ['x2', X(M.x[1])], ['y2', Y(lin(M.x[1]))]].forEach(([a, v]) => tangent.setAttribute(a, v));

    // x₁, x₂ as ticks on the x axis
    pts.forEach((x, i) => {
      [['x1', X(x)], ['y1', ay - 5], ['x2', X(x)], ['y2', ay + 5]].forEach(([a, v]) => ticks[i].setAttribute(a, v));
      tickLab[i].setAttribute('x', X(x)); tickLab[i].setAttribute('y', ay + 22);
    });
    dotTan.setAttribute('cx', X(x2)); dotTan.setAttribute('cy', Y(f(x2)));
    dotF.setAttribute('cx', X(x1));   dotF.setAttribute('cy', Y(f(x1)));
    dotLin.setAttribute('cx', X(x1)); dotLin.setAttribute('cy', Y(lin(x1)));
    [['x1', X(x1)], ['y1', Y(f(x1))], ['x2', X(x1)], ['y2', Y(lin(x1))]].forEach(([a, v]) => gap.setAttribute(a, v));

    // Labels: each tries candidate positions in order and takes the first whose box is inside
    // the plot and clear of the curve, the tangent, the x axis and the labels placed before it.
    const placed = [];
    const collisions = b => {                                     // 0 = free; counts what the box hits
      let n = 0;
      if (b.x < P.x0 || b.x + b.width > P.x1 || b.y < P.y0 - 4 || b.y + b.height > P.y1 + 30) n += 10;
      if (ay > b.y - 3 && ay < b.y + b.height + 3) n += 1;
      for (let q = b.x; q <= b.x + b.width; q += 3) {
        const u = invX(q), cy = Y(f(u)), ty = Y(lin(u));
        if ((cy > b.y - 3 && cy < b.y + b.height + 3) || (ty > b.y - 3 && ty < b.y + b.height + 3)) n += 1;
      }
      return n + 5 * placed.filter(o => b.x < o.x + o.width + 4 && o.x < b.x + b.width + 4 && b.y < o.y + o.height + 2 && o.y < b.y + b.height + 2).length;
    };
    const place = (node, cands) => {                              // first free candidate, else the least bad
      let best = null;
      for (const [x, y, anchor] of cands) {
        node.setAttribute('x', x); node.setAttribute('y', y); node.setAttribute('text-anchor', anchor);
        const b = node.getBBox(), n = collisions(b);
        if (n === 0) { placed.push(b); return b; }
        if (!best || n < best.n) best = { n, x, y, anchor };
      }
      node.setAttribute('x', best.x); node.setAttribute('y', best.y); node.setAttribute('text-anchor', best.anchor);
      const b = node.getBBox(); placed.push(b); return b;
    };
    const qx = X(x1), fy = Y(f(x1)), ly = Y(lin(x1)), my = (fy + ly) / 2;
    [...ticks, ...tickLab, yLab].forEach(n => { try { placed.push(n.getBBox()); } catch (e) { /* hidden */ } });
    place(labF, [[qx + 9, fy - 4, 'start'], [qx - 9, fy - 4, 'end'], [qx, fy - 12, 'middle']]);
    const rb = place(redLab, [[P.x1 - 2, P.y0 + 14, 'end'], [P.x0 + 2, P.y0 + 14, 'start'],
                              [qx + 40, my - 30, 'start'], [qx - 40, my - 30, 'end'],
                              [P.x1 - 2, P.y0 + 60, 'end'], [P.x0 + 2, P.y0 + 60, 'start'],
                              [P.x1 - 2, ay - 10, 'end'], [P.x0 + 2, ay - 10, 'start']]);

    // Red leader: from the bottom of the label, down and then sideways into the middle of the gap,
    // arriving from the side the label is on
    const cx = rb.x + rb.width / 2, fromRight = cx > qx, sy = rb.y + rb.height + 3;
    const sx = Math.max(rb.x + 12, Math.min(rb.x + rb.width - 12, qx + (fromRight ? 40 : -40)));
    const tx = qx + (fromRight ? 6 : -6);
    leader.setAttribute('d', 'M ' + sx + ' ' + sy + ' C ' + sx + ' ' + (sy + 0.7 * (my - sy)) + ', ' +
      (tx + (fromRight ? 24 : -24)) + ' ' + my + ', ' + tx + ' ' + my);

    readout.textContent = (f(x1) - lin(x1)).toFixed(3);
  }

  // Drag: the nearer of x₁, x₂ follows the pointer
  let active = -1;
  const px = ev => new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse()).x;
  svg.addEventListener('pointerdown', ev => {
    ev.preventDefault();
    const x = px(ev);
    active = Math.abs(X(pts[0]) - x) <= Math.abs(X(pts[1]) - x) ? 0 : 1;
    pts[active] = clampX(invX(x)); draw();
    svg.setPointerCapture(ev.pointerId);
  });
  svg.addEventListener('pointermove', ev => { if (active >= 0) { pts[active] = clampX(invX(px(ev))); draw(); } });
  ['pointerup', 'pointercancel'].forEach(t => svg.addEventListener(t, () => { active = -1; }));
  svg.addEventListener('keydown', ev => {
    const d = { ArrowLeft: -1, ArrowRight: 1 }[ev.key];
    if (!d) return;
    ev.preventDefault();
    pts[0] = clampX(pts[0] + d * 0.01 * (M.x[1] - M.x[0])); draw();
  });

  chips.forEach(c => c.addEventListener('click', () => {
    key = c.dataset.map; M = MAPS[key]; pts = M.start.slice();
    chips.forEach(o => o.classList.toggle('is-active', o === c));
    draw();
  }));
  draw();
})();
