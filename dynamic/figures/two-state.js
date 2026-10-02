/* ══════════════════════════════════════════════════════════════
 * FIGURE · TWO-STATE PROBLEM   (theory.html → "Reward & Occupancy
 * Parameter Spaces", <figure id="fig-two-state">)
 *
 * Draws the two-state MDP (as on Peters' Robot Learning slides) and,
 * for the selected state(s), highlights its features, its rewards or
 * its optimal action. The optimal policy is computed below by value
 * iteration, so it stays correct if you edit the rewards or GAMMA.
 *
 * The MDP itself (states, actions, rewards, drawing sizes) lives in
 * two-state-mdp.js; this file's CONFIG holds only what is specific to
 * this figure. Colours and line widths live in style.css (Section 1
 * palette + Section 15.6).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-two-state');
  if (!fig || !window.TwoStateMDP) return;
  const { NODES, EDGES, FEATURES, hardVI, draw } = window.TwoStateMDP;   // MDP, ψ, drawing: two-state-mdp.js

  // ── CONFIG ──────────────────────────────────────────────────
  const GAMMA    = 0.9;                       // discount for the optimal policy
  const VI_STEPS = 500;                       // value-iteration sweeps
  const DIGITS   = 2;                         // decimals shown for ψ(s,a)
  // ─────────────────────────────────────────────────────────────


  // ── Optimal policy: value iteration, then greedy action(s) ──
  const OPT = hardVI(EDGES.map(e => e.r), GAMMA, VI_STEPS);       // per edge: is its action optimal?
  const optActions = s => EDGES.filter((e, i) => e.s === s && OPT[i]).map(e => e.a);


  // ── Drawing ─────────────────────────────────────────────────
  const vec = v => '(' + v.map(x => x.toFixed(DIGITS).replace('-', '−')).join(', ') + ')';
  const psi = (s, a) => FEATURES[EDGES.findIndex(e => e.s === s && e.a === a)];
  const svg = fig.querySelector('svg');
  const { edges: edgeEls, nodes: nodeEls, setArrow, el } = draw(svg, { id: 'ts' });

  // ψ(s,a) label on each edge, in place of the "a = …, r = …" label while Features is on
  edgeEls.forEach(({ e, g, geo }) => {
    el('text', { x: geo.lx, y: geo.ly, class: 'ts-feat', 'text-anchor': 'middle' }, g)
      .innerHTML = '<tspan font-style="italic">ψ</tspan> = ' + vec(psi(e.s, e.a));
  });
  nodeEls.forEach(({ n, g }) => {
    g.setAttribute('tabindex', 0);
    g.setAttribute('role', 'button');
    g.setAttribute('aria-label', 'Select state ' + n.s);
  });


  // ── Readout table (one row per state) ───────────────────────
  const table = fig.querySelector('.ts-table');
  table.innerHTML =
    '<thead><tr><th>State</th>' +
    '<th data-aspect="feature">$\\psi(\\mathbf{s}, \\mathbf{a}{=}0)$</th>' +
    '<th data-aspect="feature">$\\psi(\\mathbf{s}, \\mathbf{a}{=}1)$</th>' +
    '<th data-aspect="reward">$r(\\mathbf{s}, \\mathbf{a}{=}0)$</th>' +
    '<th data-aspect="reward">$r(\\mathbf{s}, \\mathbf{a}{=}1)$</th>' +
    '<th data-aspect="policy">$\\pi^\\star(\\mathbf{s})$</th></tr></thead><tbody>' +
    NODES.map(({ s }) => {
      const r = a => EDGES.find(e => e.s === s && e.a === a).r;
      return '<tr data-s="' + s + '"><td>' + s + '</td>' +
        '<td data-aspect="feature">' + vec(psi(s, 0)) + '</td>' +
        '<td data-aspect="feature">' + vec(psi(s, 1)) + '</td>' +
        '<td data-aspect="reward">' + r(0) + '</td>' +
        '<td data-aspect="reward">' + r(1) + '</td>' +
        '<td data-aspect="policy">$\\mathbf{a} ' + (optActions(s).length > 1
          ? '\\in \\{' + optActions(s).join(', ') + '\\}' : '= ' + optActions(s)[0]) + '$</td></tr>';
    }).join('') + '</tbody>';
  const rows = [...table.querySelectorAll('tbody tr')];

  const gammaEl = fig.querySelector('.ts-gamma');
  if (gammaEl) gammaEl.textContent = GAMMA;


  // ── State & rendering ───────────────────────────────────────
  // aspect: '' | 'feature' | 'reward' | 'policy';  state: 'both' | '0' | '1'
  const sel  = { aspect: '', state: 'both' };
  let   view = { ...sel };        // what is drawn (differs from sel while hovering the caption)

  function render() {
    const { aspect, state } = view;
    const on = s => state === 'both' || String(s) === state;

    edgeEls.forEach(({ e, g, path }, i) => {
      const feature = aspect === 'feature' && on(e.s);
      const reward  = aspect === 'reward'  && on(e.s);
      const policy  = aspect === 'policy'  && on(e.s) && OPT[i];
      g.classList.toggle('is-feature', feature);
      g.classList.toggle('is-reward', reward);
      g.classList.toggle('is-policy', policy);
      g.classList.toggle('is-dim', !!aspect && !feature && !reward && !policy);
      setArrow(path, feature ? 'feature' : reward ? 'reward' : policy ? 'policy' : 'base');
    });
    nodeEls.forEach(({ n, g }) => g.classList.toggle('is-dim', !!aspect && !on(n.s)));
    rows.forEach(tr => tr.classList.toggle('is-dim', !on(tr.dataset.s)));
    table.querySelectorAll('[data-aspect]').forEach(c =>
      c.classList.toggle('is-on', c.dataset.aspect === aspect));

    fig.querySelectorAll('.chip[data-aspect]').forEach(b =>
      b.classList.toggle('is-active', b.dataset.aspect === sel.aspect));
    fig.querySelectorAll('.chip[data-state]').forEach(b =>
      b.classList.toggle('is-active', b.dataset.state === sel.state));
  }

  const commit = () => { view = { ...sel }; render(); };
  const toggleAspect = a => { sel.aspect = sel.aspect === a ? '' : a; commit(); };
  const toggleState  = s => { sel.state = sel.state === s ? 'both' : s; commit(); };

  // Chips
  fig.querySelectorAll('.chip[data-aspect]').forEach(b =>
    b.addEventListener('click', () => toggleAspect(b.dataset.aspect)));
  fig.querySelectorAll('.chip[data-state]').forEach(b =>
    b.addEventListener('click', () => { sel.state = b.dataset.state; commit(); }));

  // Click a state (circle or table row) to focus it; click again for both
  nodeEls.forEach(({ n, g }) => {
    g.addEventListener('click', () => toggleState(String(n.s)));
    g.addEventListener('keydown', ev => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); toggleState(String(n.s)); }
    });
  });
  rows.forEach(tr => tr.addEventListener('click', () => toggleState(tr.dataset.s)));

  // Coloured words in the caption: hover previews, click selects
  fig.querySelectorAll('figcaption .hl[data-aspect]').forEach(w => {
    w.addEventListener('mouseenter', () => { view = { ...sel, aspect: w.dataset.aspect }; render(); });
    w.addEventListener('mouseleave', commit);
    w.addEventListener('click', () => toggleAspect(w.dataset.aspect));
  });

  render();
})();
