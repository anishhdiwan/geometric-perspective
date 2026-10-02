/* ══════════════════════════════════════════════════════════════
 * FIGURE · THE EXPERT'S OCCUPANCY OVER EACH FEATURE   (theory.html →
 * Reward & Occupancy Parameter Spaces, below Assumption 3.1; <figure id="fig-expert-density">)
 *
 * Static. Two x-y plots, one per feature k = 1, 2: the marginal of the expert occupancy
 * ρ_E(s,a) over ψ_k, i.e. one stem per state-action pair at x = ψ_k(s,a) with height
 * ρ_E(s,a) (soft value iteration, α = 1, on the true rewards).
 *
 * Edit the CONFIG block only. Styles: style.css (.ed-*, and the .tp-* frame/axes).
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-expert-density');
  if (!fig || !window.TwoStateMDP || !window.TwoStateIRL) return;
  const { EDGES, FEATURES, softVI, occupancy, el } = window.TwoStateMDP;
  const { GAMMA } = window.TwoStateIRL;

  // ── CONFIG ──────────────────────────────────────────────────
  const XR     = [-1.1, 1.1];              // feature range (both panels)
  const YR     = [0, 1];                   // probability range
  const XTICKS = [-1, 0, 1];
  const YTICKS = [0, 0.5, 1];
  // ─────────────────────────────────────────────────────────────

  const rhoE = occupancy(softVI(EDGES.map(e => e.r), GAMMA, 200).pi, GAMMA);

  const W = 300, H = 280, B = { x0: 44, y0: 10, w: 240, h: 220 };   // viewBox and plot box
  const X = v => B.x0 + (v - XR[0]) / (XR[1] - XR[0]) * B.w;
  const Y = v => B.y0 + (YR[1] - v) / (YR[1] - YR[0]) * B.h;
  const body = fig.querySelector('.ed-body');

  [0, 1].forEach(k => {
    const panel = document.createElement('div');
    panel.className = 'ed-panel';
    body.appendChild(panel);
    const title = document.createElement('div');                  // HTML title (DM Mono, clean subscripts)
    title.className = 'oc-title ed-title';
    title.innerHTML = 'ρ<sub>E</sub> over ψ<sub>' + (k + 1) + '</sub>';
    title.style.marginLeft = (100 * B.x0 / W) + '%';
    title.style.width = (100 * B.w / W) + '%';
    panel.appendChild(title);
    const svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, class: 'ed-plot', role: 'img',
      'aria-label': 'Marginal of the expert occupancy over feature ' + (k + 1) + ': one stem per state-action pair' }, panel);

    el('rect', { x: B.x0, y: B.y0, width: B.w, height: B.h, class: 'tp-frame' }, svg);
    XTICKS.forEach(v => el('text', { x: X(v), y: B.y0 + B.h + 15, class: 'ed-tick', 'text-anchor': 'middle' }, svg).textContent = v);
    YTICKS.forEach(v => el('text', { x: B.x0 - 6, y: Y(v) + 4, class: 'ed-tick', 'text-anchor': 'end' }, svg).textContent = v);
    el('text', { x: B.x0 + B.w / 2, y: B.y0 + B.h + 40, class: 'tp-axis', 'text-anchor': 'middle' }, svg)
      .innerHTML = '<tspan font-style="italic">ψ</tspan><tspan dy="4" class="tp-sub">' + (k + 1) + '</tspan>';

    FEATURES.forEach((f, i) => {
      el('line', { x1: X(f[k]), y1: Y(0), x2: X(f[k]), y2: Y(rhoE[i]), class: 'ed-stem' }, svg);
      el('circle', { cx: X(f[k]), cy: Y(rhoE[i]), r: 4.5, class: 'ed-dot' }, svg);
    });
  });
})();
