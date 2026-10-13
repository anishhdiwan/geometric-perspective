/* ══════════════════════════════════════════════════════════════
 * FIGURE · QUADRATIC FORMS AND DEFINITENESS   (theory.html → Appendix, Linear Algebra,
 * below the list; <figure id="fig-quad-form">)
 *
 * A symmetric A = Q Λ Qᵀ with Λ = diag(λ₁, λ₂) and Q the rotation by an angle. Shown:
 *   left    contours of f(x) = xᵀ A x, the level set f(x) = 1 (or = −1 when A has no positive eigenvalue), the
 *           principal axes q₁, q₂ (dashed) and, when A ≻ 0, the semi-axes of length 1/√λᵢ
 *   right   the surface f(x) in 3D (isometric, as the curvature figure), its height scaled to fit
 *           the panel, with the same level set when A ≻ 0
 *   label   positive definite / semi-definite, negative (semi-)definite or indefinite, and κ
 *           when A ≻ 0; below it, Q, Λ and A with their current entries
 * Sliders set λ₁, λ₂ and the angle of Q.
 *
 * Edit the CONFIG block only. Styles: style.css (.qf-*, the .tp-* boxes, the .cc-* 3D panel and
 * the .ur-* sliders).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-quad-form');
  if (!fig || !window.TwoStateMDP || !window.PlotBox || !window.TwoStateIRL) return;
  const { el } = window.TwoStateMDP;
  const { isoPaths } = window.TwoStateIRL;

  // ── CONFIG ──────────────────────────────────────────────────
  const VIEW    = { x: [-2, 2], y: [-2, 2] };   // contour panel range
  const START   = { l1: 1.5, l2: 0.5, deg: 30 };
  const GRID    = 61, N_LEVELS = 4;             // contour grid and number of levels on each side of 0
  const ZERO    = 1e-9;                          // |λ| below this counts as zero
  const R3      = 1.5;                           // 3D panel: half-width of the drawn square
  // ─────────────────────────────────────────────────────────────

  const st = { ...START };
  const box = window.PlotBox(fig.querySelector('.qf-plot'), { id: 'qf', title: 'f(x) contours', sym: 'x', view: VIEW });
  const { X, Y } = box;
  const contours = el('g', { class: 'qf-contours' }, box.under);
  const axes = [0, 1].map(i => el('line', { class: 'qf-axis qf-axis--' + (i + 1) }, box.under));
  const level = el('path', { class: 'qf-level' }, box.under);
  const semi = [0, 1].map(i => el('line', { class: 'qf-semi qf-semi--' + (i + 1) }, box.under));
  el('circle', { cx: X(0), cy: Y(0), r: 2.5, class: 'qf-origin' }, box.under);
  const svg = fig.querySelector('.qf-plot');
  const lab = (cls, html) => { const n = el('text', { class: 'qf-lab ' + cls }, svg); n.innerHTML = html; return n; };
  const qLab = [1, 2].map(i => lab('qf-lab--' + i, '<tspan font-weight="700">q</tspan><tspan dy="4" class="qf-sub">' + i + '</tspan>'));
  const sLab = [1, 2].map(i => lab('qf-lab--' + i, '1/√<tspan font-style="italic">λ</tspan><tspan dy="4" class="qf-sub">' + i + '</tspan>'));

  // 3D panel (isometric, as in the curvature figure)
  const s3 = fig.querySelector('.qf-3d');
  const C3 = { cx: 150, cy: 132, s: 64 / R3, zs: 28 };              // zs is set per A so the surface fits
  const proj = (x, y, z) => [C3.cx + (x - y) * 0.866 * C3.s, C3.cy + (x + y) * 0.5 * C3.s - z * C3.zs];
  const floor = el('path', { class: 'cc-floor' }, s3), grid3 = el('g', { class: 'cc-grid' }, s3);
  const lvl3 = el('path', { class: 'cc-level' }, s3), shadow3 = el('path', { class: 'cc-shadow' }, s3);

  const typeEl = fig.querySelector('.qf-type');
  const P = pts => 'M' + pts.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('L');

  function draw() {
    const th = st.deg * Math.PI / 180, c = Math.cos(th), s = Math.sin(th);
    const q1 = [c, s], q2 = [-s, c], l = [st.l1, st.l2];
    // A = Q Λ Qᵀ
    const A = [[l[0] * c * c + l[1] * s * s, (l[0] - l[1]) * c * s], [(l[0] - l[1]) * c * s, l[0] * s * s + l[1] * c * c]];
    const q = (x, y) => A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y;

    // Principal axes and their labels at the box edge
    [q1, q2].forEach((v, i) => {
      const r = 3;
      [['x1', X(-r * v[0])], ['y1', Y(-r * v[1])], ['x2', X(r * v[0])], ['y2', Y(r * v[1])]].forEach(([k, val]) => axes[i].setAttribute(k, val));
      const t = 1.85 / Math.max(Math.abs(v[0]), Math.abs(v[1]));        // where the axis leaves the box
      const sgn = v[0] * 0.3 + v[1] >= 0 ? 1 : -1;                     // label on the upper end
      qLab[i].setAttribute('x', X(sgn * t * v[0]) + 6); qLab[i].setAttribute('y', Y(sgn * t * v[1]) + 4);
      qLab[i].setAttribute('text-anchor', sgn * v[0] < -0.2 ? 'end' : 'start');
      if (sgn * v[0] < -0.2) qLab[i].setAttribute('x', X(sgn * t * v[0]) - 6);
    });

    // Contours: levels spaced so rings near the origin are visible, on both sides of 0
    const ts = Array.from({ length: GRID }, (_, i) => VIEW.x[0] + i * (VIEW.x[1] - VIEW.x[0]) / (GRID - 1));
    const Z = ts.map(y => ts.map(x => q(x, y)));
    const L = Math.max(...Z.flat().map(Math.abs)) || 1;
    const levels = [];
    for (let k = 1; k <= N_LEVELS; k++) { const v = L * (k / (N_LEVELS + 1)) ** 2; levels.push(v, -v); }
    contours.replaceChildren(...isoPaths(Z, ts, ts, levels, X, Y).filter(Boolean).map(d => el('path', { d, class: 'tp-contour' })));

    // Level set f = 1 (an ellipse with its semi-axes when A ≻ 0); f = −1 if A has no positive eigenvalue
    const pd = l[0] > ZERO && l[1] > ZERO;
    const lev = l.some(v => v > ZERO) ? 1 : l.some(v => v < -ZERO) ? -1 : null;
    [lvl3, shadow3, ...semi, ...sLab].forEach(n => { n.style.display = pd ? '' : 'none'; });
    level.style.display = lev === null ? 'none' : '';
    if (!pd && lev !== null) level.setAttribute('d', isoPaths(Z, ts, ts, [lev], X, Y)[0]);
    if (pd) {
      const a = [1 / Math.sqrt(l[0]), 1 / Math.sqrt(l[1])], ring = [], ring3 = [], sh3 = [];
      for (let j = 0; j <= 96; j++) {
        const u = 2 * Math.PI * j / 96, x = a[0] * Math.cos(u) * q1[0] + a[1] * Math.sin(u) * q2[0],
              y = a[0] * Math.cos(u) * q1[1] + a[1] * Math.sin(u) * q2[1];
        ring.push([X(x), Y(y)]);
        if (Math.abs(x) <= R3 && Math.abs(y) <= R3) { ring3.push(proj(x, y, 1)); sh3.push(proj(x, y, 0)); }
      }
      level.setAttribute('d', P(ring) + 'Z');
      const inside = ring3.length === 97;                               // whole ring inside the 3D square
      lvl3.setAttribute('d', inside ? P(ring3) : ''); shadow3.setAttribute('d', inside ? P(sh3) : '');
      [q1, q2].forEach((v, i) => {
        const e = [a[i] * v[0], a[i] * v[1]];
        [['x1', X(0)], ['y1', Y(0)], ['x2', X(e[0])], ['y2', Y(e[1])]].forEach(([k, val]) => semi[i].setAttribute(k, val));
        const n = [-v[1], v[0]];                                        // label beside the middle of the semi-axis
        sLab[i].setAttribute('x', X(e[0] / 2 + 0.16 * n[0])); sLab[i].setAttribute('y', Y(e[1] / 2 + 0.16 * n[1]) + 4);
        sLab[i].setAttribute('text-anchor', 'middle');
      });
    }

    // 3D surface: grid lines in x and y, height scaled so the largest |q| on the square fits
    let zAbs = 0;
    for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) zAbs = Math.max(zAbs, Math.abs(q(-R3 + R3 * i / 20, -R3 + R3 * j / 20)));
    C3.zs = 56 / Math.max(zAbs, 0.25);
    const zc = z => z, n = 12, paths = [];
    for (let k = 0; k <= n; k++) {
      const u = -R3 + 2 * R3 * k / n;
      const a = [], b = [];
      for (let j = 0; j <= 40; j++) {
        const w = -R3 + 2 * R3 * j / 40;
        a.push(proj(u, w, zc(q(u, w)))); b.push(proj(w, u, zc(q(w, u))));
      }
      paths.push(P(a), P(b));
    }
    grid3.replaceChildren(...paths.map(d => el('path', { d })));
    floor.setAttribute('d', P([[-R3, -R3], [R3, -R3], [R3, R3], [-R3, R3]].map(([x, y]) => proj(x, y, 0))) + 'Z');

    // Type and condition number
    const pos = l.filter(v => v > ZERO).length, neg = l.filter(v => v < -ZERO).length;
    const type = pos === 2 ? 'Positive definite' : neg === 2 ? 'Negative definite'
               : pos && neg ? 'Indefinite' : pos ? 'Positive semi-definite' : neg ? 'Negative semi-definite' : 'Zero matrix';
    typeEl.innerHTML = '<strong>' + type + '</strong>' + (pd ? ' &middot; κ = λ<sub>max</sub> / λ<sub>min</sub> = ' +
      (Math.max(...l) / Math.min(...l)).toFixed(2) : '');

    // Q, Λ and A
    const f2 = v => (Math.abs(v) < 5e-3 ? 0 : v).toFixed(2).replace('-', '−');
    [['q11', c], ['q12', -s], ['q21', s], ['q22', c], ['l11', l[0]], ['l12', 0], ['l21', 0], ['l22', l[1]],
     ['a11', A[0][0]], ['a12', A[0][1]], ['a21', A[1][0]], ['a22', A[1][1]]]
      .forEach(([k, v]) => { fig.querySelector('[data-m="' + k + '"]').textContent = f2(v); });

    // Slider readouts and fills
    fig.querySelector('[data-v="l1"]').textContent = st.l1.toFixed(2).replace('-', '−');
    fig.querySelector('[data-v="l2"]').textContent = st.l2.toFixed(2).replace('-', '−');
    fig.querySelector('[data-v="deg"]').textContent = st.deg + '°';
    fig.querySelectorAll('.ur-range').forEach(inp => {
      inp.style.setProperty('--fill', (100 * (inp.value - inp.min) / (inp.max - inp.min)) + '%');
    });
  }

  fig.querySelectorAll('.ur-range[data-s]').forEach(inp => {
    inp.value = st[inp.dataset.s];
    inp.addEventListener('input', () => { st[inp.dataset.s] = +inp.value; draw(); });
  });
  draw();
})();
