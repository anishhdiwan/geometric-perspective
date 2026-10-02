/* ══════════════════════════════════════════════════════════════
 * FIGURE · MATCHING THE EXPERT'S OCCUPANCY   (theory.html →
 * Entropy-Regularized Inverse RL, <figure id="fig-occupancy-match">)
 *
 * The same policy pad as the occupancy figure (policy-pad.js: x = π(a=1|s=0),
 * y = π(a=1|s=1)). Four horizontal bars, one per state-action pair:
 *   light, full height : the expert's occupancy ρ_E(s,a), fixed — soft value
 *                        iteration (α = 1) on the true rewards, as in the Θ/Φ figure
 *   dark, thinner      : the agent's occupancy ρ_π(s,a) for the pad's policy
 * and the reverse KL divergence KL(ρ_π ‖ ρ_E) = Σ ρ_π log(ρ_π / ρ_E), live.
 * Bars use a fixed [0, 1] scale.
 *
 * Edit the CONFIG block only. Colours: style.css (--oc-lo, --oc-hi and .om-*).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-occupancy-match');
  if (!fig || !window.TwoStateMDP) return;
  const { EDGES, softVI, occupancy } = window.TwoStateMDP;

  // ── CONFIG ──────────────────────────────────────────────────
  const GAMMA   = window.TwoStateIRL ? window.TwoStateIRL.GAMMA : 0.9;   // same γ as the other figures
  const VI      = 200;                    // soft VI sweeps for the expert
  const START   = [0.5, 0.5];             // initial π(a=1|s=0), π(a=1|s=1)
  const DIGITS  = 2;                      // decimals for the occupancies
  const KL_DIGITS = 3;                    // decimals for the KL divergence
  // ─────────────────────────────────────────────────────────────

  const rhoE = occupancy(softVI(EDGES.map(e => e.r), GAMMA, VI).pi, GAMMA);   // expert, fixed

  const body = fig.querySelector('.om-body');
  const num  = v => v.toFixed(DIGITS);
  const sym  = (s, k) => '<span><span class="oc-sym">' + s + '</span> = ' + k + '</span>';
  const add  = (cls, row, col, html) => {
    const n = document.createElement('div');
    n.className = cls;
    n.style.gridRow = row;
    n.style.gridColumn = col;
    if (html !== undefined) n.innerHTML = html;
    body.appendChild(n);
    return n;
  };

  // Grid columns: 1 policy pad · 2 row labels · 3 bars · 4 values
  // Grid rows:    1 title · 2–3 state 0 (a = 0, 1) · 4–5 state 1 · 6 axis ticks · 7 legend · 8 KL
  add('oc-title', 1, '2 / span 3', 'Occupancy measures');

  // Policy pad (policy-pad.js): x = π(a=1|s=0), y = π(a=1|s=1)
  const padTitle = add('oc-pad', 1, 1);
  const padCell  = add('oc-pad om-pad', '2 / span 4', 1);                      // centred on the four bars
  const pad = window.PolicyPad(padCell, { value: START, titleIn: padTitle, onChange: () => update() });

  const bars = EDGES.map((e, i) => {
    const row = 2 + 2 * e.s + e.a;
    add('om-label', row, 2, sym('s', e.s) + ', ' + sym('a', e.a));
    const track = add('om-track', row, 3, '<div class="om-bar om-bar--expert"></div><div class="om-bar om-bar--agent"></div>');
    const [ex, ag] = track.children;
    ex.style.width = (100 * rhoE[i]) + '%';
    const val = add('om-val', row, 4, '<span class="om-v"></span><span class="om-ve">' + num(rhoE[i]) + '</span>');
    return { e, ag, v: val.firstChild, track };
  });

  add('om-ticks', 6, 3, '<span>0</span><span>1</span>');
  add('om-legend', 7, '2 / span 3',
    '<span><i class="om-sw om-sw--expert"></i>$\\rho_E$ (expert)</span>' +
    '<span><i class="om-sw om-sw--agent"></i>$\\rho_\\pi$ (agent)</span>');
  const kl = add('om-kl', 8, '2 / span 3',
    '$\\mathrm{KL}(\\rho_\\pi \\,\\|\\, \\rho_E) =$ <span class="om-kl__v"></span>').querySelector('.om-kl__v');


  // ── Live update ─────────────────────────────────────────────
  function update() {
    const p   = pad.get();                                                      // π(a=1|s)
    const pi  = EDGES.map(e => (e.a === 1 ? p[e.s] : 1 - p[e.s]));              // π(a|s) per edge
    const rho = occupancy(pi, GAMMA);

    bars.forEach(({ ag, v, track }, i) => {
      ag.style.width = (100 * rho[i]) + '%';
      v.textContent = num(rho[i]);
      track.title = 'ρ_π = ' + num(rho[i]) + ',  ρ_E = ' + num(rhoE[i]);
    });
    // Reverse KL; terms with ρ_π = 0 contribute 0 (ρ_E > 0 everywhere for the soft expert)
    const d = rho.reduce((acc, r, i) => acc + (r > 0 ? r * Math.log(r / rhoE[i]) : 0), 0);
    kl.textContent = Math.max(0, d).toFixed(KL_DIGITS);
  }

  update();
})();
