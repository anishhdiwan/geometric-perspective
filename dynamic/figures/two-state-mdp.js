/* ══════════════════════════════════════════════════════════════
 * SHARED · TWO-STATE MDP   (used by two-state.js and theta-space.js)
 *
 * One place for the MDP (states, actions, rewards, features), how it is drawn,
 * and the solvers ported from the paper code (irl.py / geometries.py):
 *   hardVI      — value iteration, greedy policy
 *   softVI      — soft value iteration (α = 1): V(s) = log Σ_a exp Q(s,a),
 *                 Q(s,a) = r(s,a) + γ V(s'), π(a|s) = exp(Q(s,a) − V(s))
 *   occupancy   — normalised discounted occupancy ρ(s,a) = d(s) π(a|s),
 *                 d = (1−γ)(I − γ P_πᵀ)⁻¹ μ₀   (geometries._rho_from_policy)
 *
 * Reward vectors are arrays over EDGES, in EDGES order.
 * Edit the CONFIG block only.
 * ══════════════════════════════════════════════════════════════ */
window.TwoStateMDP = (function () {

  // ── CONFIG ──────────────────────────────────────────────────
  // States. `loop` is the side ('left' | 'right') of the state's self-loop.
  const NODES = [
    { s: 0, x: 170, y: 120, loop: 'left'  },
    { s: 1, x: 350, y: 120, loop: 'right' },
  ];
  const MU0 = [1, 0];                          // initial state distribution μ₀

  // Actions: one edge per (state s, action a), deterministic → next state,
  // true reward r. Keep the order (0,0), (0,1), (1,0), (1,1): feature
  // matrices and reward vectors follow it.
  // Edges between states need `bend` ('up' | 'down'): which side they curve to.
  const EDGES = [
    { s: 0, a: 0, next: 0, r: 1 },
    { s: 0, a: 1, next: 1, r: 0, bend: 'down' },
    { s: 1, a: 0, next: 0, r: 0, bend: 'up'   },
    { s: 1, a: 1, next: 1, r: 2 },
  ];

  // Features ψ(s,a) ∈ R², rows in EDGES order. The expert occupancy is exactly
  // Boltzmann in them (Assumption 3.1). Regenerate with
  // dynamic/figures/tools/two_state_features.py after changing the MDP or γ.
  const FEATURES = [
    [-0.154902, -0.851687],   // (s=0, a=0)
    [-0.009393,  0.270927],   // (s=0, a=1)
    [-0.611546,  0.418024],   // (s=1, a=0)
    [ 0.775841,  0.162737],   // (s=1, a=1)
  ];

  // Drawing sizes (SVG units)
  const R_NODE     = 20;                              // state circle radius
  const LOOP_SIZE  = 87;                              // how far a self-loop reaches out
  const ARC_HEIGHT = 60;                              // how far an arc bulges
  const ARROW      = { base: 7, feature: 10, reward: 10, policy: 10, expert: 10 };  // arrowhead size per highlight
  // ─────────────────────────────────────────────────────────────


  const out = s => EDGES.map((e, i) => i).filter(i => EDGES[i].s === s);

  // ── Solvers ─────────────────────────────────────────────────
  // Returns, per edge, whether its action is optimal. Ties (e.g. r = 0) mark every tied action.
  function hardVI(r, gamma, sweeps) {
    const V = NODES.map(() => 0);
    const q = i => r[i] + gamma * V[EDGES[i].next];
    for (let k = 0; k < sweeps; k++) {
      NODES.forEach(({ s }) => { V[s] = Math.max(...out(s).map(q)); });
    }
    return EDGES.map((e, i) => q(i) >= V[e.s] - 1e-9 * (1 + Math.abs(V[e.s])));
  }

  // Plain loops, no allocation per sweep: this runs once per contour grid point.
  const OUT = NODES.map(({ s }) => out(s));
  function softVI(r, gamma, sweeps) {
    const V = new Float64Array(NODES.length), Q = new Float64Array(EDGES.length);
    for (let k = 0; k < sweeps; k++) {
      for (let i = 0; i < EDGES.length; i++) Q[i] = r[i] + gamma * V[EDGES[i].next];
      for (let s = 0; s < NODES.length; s++) {                 // V(s) = log Σ_a exp Q(s,a)
        let m = -Infinity, z = 0;
        for (const i of OUT[s]) m = Math.max(m, Q[i]);
        for (const i of OUT[s]) z += Math.exp(Q[i] - m);
        V[s] = m + Math.log(z);
      }
    }
    return { V, pi: EDGES.map((e, i) => Math.exp(Q[i] - V[e.s])) };   // π(a|s) per edge
  }

  // Two states: solve the 2×2 system (I − γ P_πᵀ) d = (1−γ) μ₀ in closed form.
  function occupancy(pi, gamma) {
    const P = [[0, 0], [0, 0]];                       // P_π(s → s')
    EDGES.forEach((e, i) => { P[e.s][e.next] += pi[i]; });
    const a = 1 - gamma * P[0][0], b = -gamma * P[1][0];
    const c = -gamma * P[0][1],    d = 1 - gamma * P[1][1];
    const det = a * d - b * c, u = (1 - gamma) * MU0[0], v = (1 - gamma) * MU0[1];
    const ds = [(d * u - b * v) / det, (a * v - c * u) / det];
    return EDGES.map((e, i) => ds[e.s] * pi[i]);
  }


  // ── Drawing ─────────────────────────────────────────────────
  const NS  = 'http://www.w3.org/2000/svg';
  const el  = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => n.setAttribute(k, v));
    if (parent) parent.appendChild(n);
    return n;
  };
  const node = s => NODES.find(n => n.s === s);
  const deg  = d => d * Math.PI / 180;
  const pt   = (x, y) => x.toFixed(1) + ' ' + y.toFixed(1);

  // Self-loop on the node's `loop` side: leaves from the top, returns at the bottom.
  function loopPath(n) {
    const dx = n.loop === 'left' ? -1 : 1;
    const ex = n.x + dx * R_NODE * Math.cos(deg(35)), ey = R_NODE * Math.sin(deg(35));
    const cx = n.x + dx * LOOP_SIZE * Math.cos(deg(37)), cy = LOOP_SIZE * Math.sin(deg(37));
    return {
      d: 'M ' + pt(ex, n.y - ey) + ' C ' + pt(cx, n.y - cy) + ', ' + pt(cx, n.y + cy) + ', ' + pt(ex, n.y + ey),
      lx: n.x + dx * 42, ly: n.y + 48,
    };
  }
  // Arc between two states on the same row, curving up or down.
  function arcPath(from, to, bend) {
    const dx = Math.sign(to.x - from.x), dy = bend === 'up' ? -1 : 1;
    const ox = R_NODE * Math.cos(deg(50)), oy = R_NODE * Math.sin(deg(50));
    const x0 = from.x + dx * ox, x1 = to.x - dx * ox, y = from.y + dy * oy;
    return {
      d: 'M ' + pt(x0, y) + ' C ' + pt(x0 + dx * 39, y + dy * ARC_HEIGHT) + ', ' +
         pt(x1 - dx * 39, y + dy * ARC_HEIGHT) + ', ' + pt(x1, y),
      lx: (from.x + to.x) / 2, ly: from.y + dy * 80 + 4,
    };
  }

  // Draw the MDP into `svg`. `id` makes the arrowhead markers unique per figure;
  // `labels: false` leaves out the "a = …, r = …" edge labels.
  // Returns the edge and node elements (each edge with its label position `geo`
  // and the reward-value tspan `rVal`, to update), and setArrow(path, kind).
  function draw(svg, { id, labels = true }) {
    const defs = el('defs', {}, svg);
    Object.entries(ARROW).forEach(([kind, size]) => {
      const m = el('marker', {
        id: id + '-arrow-' + kind, class: 'ts-arrow ts-arrow--' + kind,
        viewBox: '0 0 10 10', refX: 10, refY: 5,
        markerWidth: size, markerHeight: size, markerUnits: 'userSpaceOnUse', orient: 'auto',
      }, defs);
      el('path', { d: 'M 0 1 L 10 5 L 0 9 z' }, m);
    });
    const setArrow = (path, kind) => path.setAttribute('marker-end', 'url(#' + id + '-arrow-' + kind + ')');

    const edges = EDGES.map(e => {
      const from = node(e.s);
      const geo  = e.next === e.s ? loopPath(from) : arcPath(from, node(e.next), e.bend);
      const g    = el('g', { class: 'ts-edge' }, svg);
      const path = el('path', { d: geo.d }, g);
      // Stop the line a little before the tip, so thick strokes don't poke out of the arrowhead
      const len = path.getTotalLength();
      path.setAttribute('stroke-dasharray', (len - 4) + ' ' + len);
      setArrow(path, 'base');
      let rVal = null;
      if (labels) {
        const t = el('text', { x: geo.lx, y: geo.ly, class: 'ts-lbl', 'text-anchor': 'middle' }, g);
        t.innerHTML =
          '<tspan class="ts-a"><tspan font-weight="700">a</tspan> = ' + e.a + '</tspan>, ' +
          '<tspan class="ts-r"><tspan font-style="italic">r</tspan> = <tspan class="ts-rval">' + e.r + '</tspan></tspan>';
        rVal = t.querySelector('.ts-rval');
      }
      return { e, g, path, geo, rVal };
    });

    const nodes = NODES.map(n => {
      const g = el('g', { class: 'ts-node' }, svg);
      el('circle', { cx: n.x, cy: n.y, r: R_NODE }, g);
      el('text', { x: n.x, y: n.y, class: 'ts-node__label' }, g).textContent = n.s;
      return { n, g };
    });

    return { edges, nodes, setArrow, el };
  }

  return { NODES, EDGES, MU0, FEATURES, hardVI, softVI, occupancy, draw, el };
})();
