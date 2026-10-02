/* ══════════════════════════════════════════════════════════════
 * FIGURE · OCCUPANCY MEASURE OF THE TWO-STATE PROBLEM   (theory.html →
 * Preliminaries, <figure id="fig-occupancy">)
 *
 * One row, left to right:
 *   Policy    a policy pad (policy-pad.js): one point, x = π(a=1|s=0), y = π(a=1|s=1).
 *   Rollouts  N_ROLLOUTS independent rollouts of π for T steps from s₀ ~ μ₀,
 *             drawn as states (circles) and actions (labelled arrows), with "…"
 *             for the infinite continuation. Each step uses a fixed random number,
 *             so moving a slider changes the rollouts smoothly.
 *   Grid      the normalised discounted occupancy ρ_π(s,a) = d(s) π(a|s),
 *             d = (1−γ)(I − γ P_πᵀ)⁻¹ μ₀ (two-state-mdp.js → occupancy), on a
 *             fixed [0, 1] colour scale.
 *
 * Edit the CONFIG block only. Colours: style.css (--oc-lo, --oc-hi).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-occupancy');
  if (!fig || !window.TwoStateMDP) return;
  const { EDGES, MU0, occupancy, el } = window.TwoStateMDP;

  // ── CONFIG ──────────────────────────────────────────────────
  const GAMMA      = window.TwoStateIRL ? window.TwoStateIRL.GAMMA : 0.9;   // same γ as the other figures
  const START      = [0.5, 0.5];          // initial π(a=1|s=0), π(a=1|s=1)
  const N_ROLLOUTS = 4;                   // rollouts shown
  const T          = 5;                   // states per rollout (s₀ … s₄), then "…"
  const DIGITS     = 2;                   // decimals shown
  const STRONG     = 0.65;                // cells above this value get light text
  // ─────────────────────────────────────────────────────────────

  const body = fig.querySelector('.oc-body');
  const num  = v => v.toFixed(DIGITS);
  const add  = (cls, row, col, html, tag = 'div') => {
    const n = document.createElement(tag);
    n.className = cls;
    n.style.gridRow = row;
    n.style.gridColumn = col;
    if (html !== undefined) n.innerHTML = html;
    body.appendChild(n);
    return n;
  };

  // Grid columns: 1 policy pad · 2 rollouts · 3 axis label s · 4 row ticks · 5–6 cells · 7 colour bar · 8 its ticks
  // Grid rows:    1 titles · 2–3 states · 4 column ticks · 5 axis label a
  add('oc-title oc-title--roll', 1, 2, 'Rollouts of π');
  add('oc-title', 1, '5 / span 2', 'ρ<sub>π</sub>(s, a)');                    // regular weight: bold s, a blur in DM Mono

  // Policy pad (policy-pad.js): x = π(a=1|s=0), y = π(a=1|s=1); its own title sits in the titles row
  const padTitle = add('oc-pad', 1, 1);
  const padCell  = add('oc-pad', '2 / span 2', 1);                             // centred on the rollouts / grid rows
  const pad = window.PolicyPad(padCell, { value: START, titleIn: padTitle, onChange: () => update() });

  add('oc-axis oc-axis--y', '2 / span 2', 3, '<span class="oc-sym">s</span>');
  [0, 1].forEach(k => {
    add('oc-tick oc-tick--row', k + 2, 4, k);
    add('oc-tick oc-tick--col', 4, 5 + k, k);
  });
  add('oc-axis oc-axis--x', 5, '5 / span 2', '<span class="oc-sym">a</span>');

  const cells = EDGES.map(e => ({ e, n: add('oc-cell', e.s + 2, 5 + e.a) }));

  add('oc-bar', '2 / span 2', 7);
  add('oc-bartick', '2 / span 2', 8, '<span>1</span><span>0</span>');


  // ── Rollouts panel (SVG) ────────────────────────────────────
  const NS = 'http://www.w3.org/2000/svg';
  const roll = document.createElementNS(NS, 'svg');
  roll.setAttribute('class', 'oc-roll');
  roll.setAttribute('viewBox', '0 0 350 200');
  roll.setAttribute('role', 'img');
  roll.setAttribute('aria-label', N_ROLLOUTS + ' rollouts of the policy, ' + T + ' steps each');
  roll.style.gridRow = '2 / span 2';
  roll.style.gridColumn = 2;
  body.appendChild(roll);

  const R = 13, DX = 66, X0 = 20, Y0 = 46, DY = 44;             // circle radius, spacing, first centre
  const cx = t => X0 + t * DX, cy = r => Y0 + r * DY;
  const defs = el('defs', {}, roll);
  const mk = el('marker', { id: 'oc-arrow', viewBox: '0 0 10 10', refX: 10, refY: 5, markerWidth: 6, markerHeight: 6,
                            markerUnits: 'userSpaceOnUse', orient: 'auto' }, defs);
  el('path', { d: 'M 0 1 L 10 5 L 0 9 z', class: 'oc-roll__head' }, mk);
  for (let t = 0; t < T; t++) {                                  // time labels
    el('text', { x: cx(t), y: 14, class: 'oc-roll__t', 'text-anchor': 'middle' }, roll)
      .innerHTML = '<tspan font-style="italic">t</tspan> = ' + t;
  }
  const rows = Array.from({ length: N_ROLLOUTS }, (_, r) => {
    const g = el('g', {}, roll);
    const states = Array.from({ length: T }, (_, t) => {
      el('circle', { cx: cx(t), cy: cy(r), r: R, class: 'oc-roll__state' }, g);
      return el('text', { x: cx(t), y: cy(r), class: 'oc-roll__s' }, g);
    });
    const actions = Array.from({ length: T }, (_, t) => {       // a_t: arrow after s_t (the last one leads to "…")
      const x1 = cx(t) + R + 2, x2 = t < T - 1 ? cx(t + 1) - R - 2 : cx(t) + R + 30;
      el('line', { x1, y1: cy(r), x2, y2: cy(r), class: 'oc-roll__arrow', 'marker-end': 'url(#oc-arrow)' }, g);
      return el('text', { x: (x1 + x2) / 2, y: cy(r) - 8, class: 'oc-roll__a', 'text-anchor': 'middle' }, g);
    });
    el('text', { x: cx(T - 1) + R + 34, y: cy(r), class: 'oc-roll__dots' }, g).textContent = '…';
    return { states, actions };
  });

  // Fixed random numbers per rollout and step (seeded): the same rollouts on every load
  const seed = 1;
  const rng = s => () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let x = Math.imul(s ^ (s >>> 15), 1 | s);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x; return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };   // mulberry32
  let U;
  const draw = () => { const r = rng(seed); U = Array.from({ length: N_ROLLOUTS }, () => Array.from({ length: T + 1 }, r)); };
  draw();

  function drawRollouts(p) {                                     // p[s] = π(a=1|s)
    rows.forEach(({ states, actions }, r) => {
      let s = U[r][T] < MU0[0] ? 0 : 1;                          // s₀ ~ μ₀
      for (let t = 0; t < T; t++) {
        const a = U[r][t] < p[s] ? 1 : 0;                        // a_t ~ π(·|s_t)
        states[t].textContent = s;
        actions[t].innerHTML = '<tspan font-weight="700">a</tspan> = ' + a;
        s = EDGES.find(e => e.s === s && e.a === a).next;        // deterministic transition
      }
    });
  }


  // ── Live update ─────────────────────────────────────────────
  function update() {
    const p   = pad.get();                                                      // π(a=1|s)
    const pi  = EDGES.map(e => (e.a === 1 ? p[e.s] : 1 - p[e.s]));              // π(a|s) per edge
    const rho = occupancy(pi, GAMMA);
    const d   = [0, 1].map(s => EDGES.reduce((acc, e, i) => acc + (e.s === s ? rho[i] : 0), 0));

    cells.forEach(({ e, n }, i) => {
      const v = rho[i];
      n.style.background = 'color-mix(in oklab, var(--oc-hi) ' + (100 * v).toFixed(1) + '%, var(--oc-lo))';
      n.textContent = num(v);
      n.classList.toggle('is-strong', v >= STRONG);
      n.title = 'ρ(s=' + e.s + ', a=' + e.a + ') = d(s) π(a|s) = ' + num(d[e.s]) + ' × ' + num(pi[i]) + ' = ' + num(v);
    });
    drawRollouts(p);
  }

  update();
})();
