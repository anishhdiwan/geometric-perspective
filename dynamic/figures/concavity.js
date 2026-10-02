/* ══════════════════════════════════════════════════════════════
 * FIGURE · CONCAVITY OF THE ENTROPY AND THE NEGATIVE KL IN THE OCCUPANCY
 * (theory.html → Entropy-Regularized Inverse RL, below fact (ii); <figure id="fig-concavity">)
 *
 * Both terms of Eq. (2) as functions of the state-action occupancy ρ_π(s,a):
 *   left   E_{ρ_π(s)}[H(π(a|s))] = −Σ_{s,a} ρ_π(s,a) log(ρ_π(s,a) / ρ_π(s)),  ρ_π(s) = Σ_a ρ_π(s,a)
 *   right  −KL(ρ_π ‖ ρ_E)       = −Σ_{s,a} ρ_π(s,a) log(ρ_π(s,a) / ρ_E(s,a))
 * The occupancy is 4-D, so we plot along a segment of occupancy space:
 *   ρ_t = (1−t) ρ_{π_A} + t ρ_{π_B},  t ∈ [0, 1],
 * between the occupancies of two policies, each set with a policy pad. Valid occupancies form a
 * convex set (the Bellman flow constraints are linear), so every ρ_t is the occupancy of
 * π_t(a|s) = ρ_t(s,a) / Σ_a' ρ_t(s,a'); and a function is concave on a convex set iff it is
 * concave along every such segment. Each panel shows a chord between two draggable points
 * (shared by both panels) and shades the region between curve and chord.
 *
 * Edit the CONFIG block only. Styles: style.css (.cv-*, and the .tp-* titles/axes).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-concavity');
  if (!fig || !window.TwoStateMDP) return;
  const { EDGES, softVI, occupancy, el } = window.TwoStateMDP;

  // ── CONFIG ──────────────────────────────────────────────────
  const GAMMA  = window.TwoStateIRL ? window.TwoStateIRL.GAMMA : 0.9;   // same γ as the other figures
  const POLICY = { A: [0.1, 0.2], B: [0.9, 0.8] };   // initial π(a=1|s=0), π(a=1|s=1) of the two end policies
  const START  = [0.15, 0.75];                       // initial chord endpoints (values of t)
  const N      = 200;                                // curve resolution
  // ─────────────────────────────────────────────────────────────

  const rhoE  = occupancy(softVI(EDGES.map(e => e.r), GAMMA, 200).pi, GAMMA);   // expert occupancy
  const xlogy = (x, y) => (x > 0 ? x * Math.log(x / y) : 0);                   // 0 log 0 = 0
  const stateSum = rho => [0, 1].map(s => EDGES.reduce((acc, e, i) => acc + (e.s === s ? rho[i] : 0), 0));
  const F = {
    entropy: rho => { const d = stateSum(rho); return -EDGES.reduce((acc, e, i) => acc + xlogy(rho[i], d[e.s]), 0); },
    negkl:   rho => -rho.reduce((acc, r, i) => acc + xlogy(r, rhoE[i]), 0),
  };
  const occOf = p => occupancy(EDGES.map(e => (e.a === 1 ? p[e.s] : 1 - p[e.s])), GAMMA);   // p[s] = π(a=1|s)

  // ── Controls: the two end policies (two policy pads) ────────
  const controls = fig.querySelector('.cv-controls');
  const pads = {};
  ['A', 'B'].forEach(k => {                                     // one policy pad each (policy-pad.js)
    pads[k] = window.PolicyPad(controls, { value: POLICY[k], title: 'π<sub>' + k + '</sub>(a|s)', onChange: () => update() });
  });

  // ── Panels ──────────────────────────────────────────────────
  const B = { x0: 44, y0: 30, w: 246, h: 240 };                // plot box in a 300 × 318 viewBox
  const PANELS = [
    // Titles in the site's figure-title style (DM Mono, as the occupancy figure's .oc-title), HTML subscripts
    { title: 'E<sub>ρ<sub>π</sub>(s)</sub>[H(π(a|s))]', label: 'Expected entropy', f: F.entropy },
    { title: '−KL(ρ<sub>π</sub>&thinsp;‖&thinsp;ρ<sub>E</sub>)', label: 'Negative KL', f: F.negkl },
  ];
  const body = fig.querySelector('.cv-body');
  const pts = START.slice();                                    // shared chord endpoints (t)
  const ts = Array.from({ length: N + 1 }, (_, i) => i / N);
  let rhoA, rhoB;
  const rhoAt = t => rhoA.map((a, i) => (1 - t) * a + t * rhoB[i]);
  const draws = [];
  const drawAll = () => draws.forEach(d => d());

  PANELS.forEach(P => {
    const wrap = document.createElement('div');
    wrap.className = 'cv-panel';
    body.appendChild(wrap);
    const svg = el('svg', { viewBox: '0 0 300 318', class: 'cv-plot', role: 'img', 'aria-label': P.label + ' along a segment of occupancy space, with a chord' }, wrap);
    const title = document.createElement('div');                // title (HTML, so subscripts render cleanly)
    title.className = 'oc-title cv-title'; title.innerHTML = P.title; wrap.appendChild(title);
    const xlab = document.createElement('div');                 // MathJax x label (HTML overlay)
    xlab.className = 'cv-xlab'; xlab.textContent = '$t$'; wrap.appendChild(xlab);

    const X = t => B.x0 + t * B.w;
    const invX = x => Math.min(1, Math.max(0, (x - B.x0) / B.w));
    el('rect', { x: B.x0, y: B.y0, width: B.w, height: B.h, class: 'tp-frame' }, svg);
    [[0, '0'], [1, '1']].forEach(([t, s]) =>
      el('text', { x: X(t), y: B.y0 + B.h + 14, class: 'cv-tick', 'text-anchor': 'middle' }, svg).textContent = s);

    const shade = el('path', { class: 'cv-shade' }, svg);
    const curve = el('path', { class: 'cv-curve' }, svg);
    const chord = el('line', { class: 'cv-chord' }, svg);
    const handles = START.map(() => el('circle', { r: 6, class: 'cv-handle' }, svg));

    draws.push(() => {
      const vals = ts.map(t => P.f(rhoAt(t)));
      let lo = Math.min(...vals), hi = Math.max(...vals);
      const pad = 0.08 * Math.max(hi - lo, 1e-3); lo -= pad; hi += pad;
      const Y = v => B.y0 + (hi - v) / (hi - lo) * B.h;
      const pt = t => X(t).toFixed(1) + ' ' + Y(P.f(rhoAt(t))).toFixed(1);
      curve.setAttribute('d', 'M' + ts.map((t, i) => X(t).toFixed(1) + ' ' + Y(vals[i]).toFixed(1)).join('L'));
      const [a, b] = pts[0] <= pts[1] ? pts : [pts[1], pts[0]];
      chord.setAttribute('x1', X(a)); chord.setAttribute('y1', Y(P.f(rhoAt(a))));
      chord.setAttribute('x2', X(b)); chord.setAttribute('y2', Y(P.f(rhoAt(b))));
      shade.setAttribute('d', 'M' + [a, ...ts.filter(t => t > a && t < b), b].map(pt).join('L') + 'Z');
      handles.forEach((h, i) => { h.setAttribute('cx', X(pts[i])); h.setAttribute('cy', Y(P.f(rhoAt(pts[i])))); });
    });

    // Drag: grab the nearer endpoint and move it along t; both panels follow
    let active = -1;
    const px = ev => new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse()).x;
    svg.addEventListener('pointerdown', ev => {
      ev.preventDefault();                                          // no text selection while dragging
      const x = px(ev);
      active = Math.abs(X(pts[0]) - x) <= Math.abs(X(pts[1]) - x) ? 0 : 1;
      pts[active] = invX(x); drawAll();
      svg.setPointerCapture(ev.pointerId);
    });
    svg.addEventListener('pointermove', ev => { if (active >= 0) { pts[active] = invX(px(ev)); drawAll(); } });
    ['pointerup', 'pointercancel'].forEach(t => svg.addEventListener(t, () => { active = -1; }));
  });

  function update() {
    rhoA = occOf(pads.A.get());
    rhoB = occOf(pads.B.get());
    drawAll();
  }
  update();
})();
