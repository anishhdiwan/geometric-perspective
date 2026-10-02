/* ══════════════════════════════════════════════════════════════
 * FIGURE · Θ-SPACE, Φ-SPACE AND THE TWO-STATE PROBLEM   (theory.html →
 * "Reward & Occupancy Parameter Spaces", <figure id="fig-theta-space">)
 *
 * Left:   Θ-space with the IRL solution θ⋆, the hovered (or pinned)
 *         θ⁽ⁱ⁾, and (toggles) contours of the dual G(θ) and the three
 *         update directions at θ⁽ⁱ⁾ from the paper's Fig. 1.
 * Middle: Φ-space (occupancy natural parameters), squares as in the
 *         paper's Fig. 1:
 *           φ_E      expert occupancy
 *           φ⋆       occupancy of the IRL solution, π_θ⋆ (≠ φ_E: the
 *                    entropy regularisation keeps it away from φ_E)
 *           φ_ρ⁽ⁱ⁾   occupancy of the soft-optimal policy for θ⁽ⁱ⁾ (moves)
 *           φ̂        target φ_E − θ⁽ⁱ⁾/β, Eq. (3) (moves)
 *         and the e-geodesic, the straight line φ̂ → φ_ρ⁽ⁱ⁾ (dashed).
 * Right:  the two-state MDP with the rewards r_θ and their optimal policy
 *         (value iteration) at θ⁽ⁱ⁾, or (toggle) the expert: value
 *         iteration on the true rewards.
 * Third panel, chosen by a menu that doubles as its title: the MDP (above), or
 *         Θ-space Geometry and Φ-space Geometry, contours of the quadratics
 *         ½(θ − θ⋆)ᵀH̄⁽ⁱ⁾(θ − θ⋆) (integral Hessian, Prop. 3.4) and
 *         ½(φ − φ̂)ᵀH⁽ⁱ⁾(φ − φ̂) (integral FIM, Lemma 3.8), with a thicker ring
 *         through θ⁽ⁱ⁾ / φ_ρ⁽ⁱ⁾. Updated live.
 * Only hovering over Θ moves things; left-click fixes θ⁽ⁱ⁾, right-click or
 * the Reset chip unfixes it.
 *
 * This file also holds the figure's markup and caption, so theory.html and
 * index.html (which only have an empty <figure id="fig-theta-space">) always
 * show the same figure.
 *
 * Model (G, θ⋆, φ fits, contour grid): two-state-irl.js. MDP, ψ and
 * solvers: two-state-mdp.js. Plot box: plot-box.js. Edit the CONFIG block
 * (and the markup below it) only.
 * ══════════════════════════════════════════════════════════════ */
(function () {
  const fig = document.getElementById('fig-theta-space');
  if (!fig || !window.TwoStateIRL) return;
  const { EDGES, hardVI, draw, el } = window.TwoStateMDP;
  const { GAMMA, BETA, THETA0, VIEW, reward, dualGrad, delta, thStar,
          phiE, phiStar, phiOf, phiHat, intFIM, intHess, contourPaths, quadPaths } = window.TwoStateIRL;
  const L = window.PlotBox.label;

  // ── CONFIG ──────────────────────────────────────────────────
  const VIEW_PHI      = { x: [-0.6, 3.0], y: [-1.8, 1.8] };   // plotted Φ range: φ₁ (x), φ₂ (y)
  const HARD_VI_STEPS = 200;             // value iteration for the displayed policy
  const DIGITS        = 2;               // decimals shown for r_θ
  const GRAD_LEN      = 40;              // drawn length of every gradient arrow (SVG units; directions only)
  // ─────────────────────────────────────────────────────────────


  // ── Markup & caption: one source for theory.html and index.html ──
  // Each page only has <figure class="ap" id="fig-theta-space"></figure>; edit the figure here.
  // A page may keep its own caption: put a <figcaption> inside the <figure> and it replaces
  // the caption below (index.html does this).
  // (String.raw keeps the LaTeX backslashes; MathJax typesets it after the page loads.)
  fig.classList.add('ifig', 'ts-fig', 'tp-fig');
  const pageCaption = fig.querySelector('figcaption');
  fig.innerHTML = String.raw`
    <div class="ifig__controls">
      <span class="ifig__label">Show</span>
      <button class="chip" data-toggle="dual"><span class="chip__dot"></span>Dual contours $\mathcal{G}(\theta)$</button>
      <button class="chip" data-toggle="grads"><span class="chip__dot"></span>Gradients</button>
      <button class="chip" data-toggle="expert"><span class="chip__dot"></span>Expert reward and policy</button>
      <button class="chip chip--reset" data-action="reset" hidden>Reset $\theta^{(i)}$</button>
    </div>
    <div class="ifig__canvas tp-body">
      <div class="tp-left">
        <svg class="tp-plot tp-plot--theta" viewBox="0 0 300 318" role="img"
             aria-label="Reward parameter space with the initial reward, the IRL solution and a movable reward parameter"></svg>
      </div>
      <div class="tp-left">
        <svg class="tp-plot tp-plot--phi" viewBox="0 0 300 318" role="img"
             aria-label="Occupancy parameter space with the expert, optimal, initial, current and target occupancy parameters"></svg>
      </div>
      <div class="tp-left tp-panel">
        <label class="tp-pick">
          <select aria-label="Choose what the third panel shows">
            <option value="mdp">Learnt Reward &amp; Policy</option>
            <option value="tgeo">Θ-space Geometry</option>
            <option value="pgeo">Φ-space Geometry</option>
          </select>
        </label>
        <svg class="tp-mdp" data-panel="mdp" viewBox="60 -91 400 424" role="img"
             aria-label="Two-state MDP with the rewards and optimal policy for the selected reward parameters"></svg>
        <svg class="tp-plot tp-plot--tgeo" data-panel="tgeo" viewBox="0 0 300 318" role="img" hidden
             aria-label="Reward parameter space geometry: contours of the integral Hessian around the IRL solution"></svg>
        <svg class="tp-plot tp-plot--pgeo" data-panel="pgeo" viewBox="0 0 300 318" role="img" hidden
             aria-label="Occupancy parameter space geometry: contours of the integral Fisher information around the target"></svg>
      </div>
    </div>
    <figcaption>
      <strong>A geometric picture of inverse reinforcement learning.</strong>
      The IRL solution $\theta^\star$ minimises the Lagrangian dual $\mathcal{G}(\theta)$.
      $\phi_E$ is the expert and $\phi^\star$ belongs to $\theta^\star$.
      Hover over the $\Theta$-space to move $\theta^{(i)}$ (left-click to fix it; right-click or Reset to follow the cursor again).
      The right-side panel computes the optimal policy for the rewards defined by $\theta^{(i)}$ using soft value iteration.
      We also compute the <span class="hl-grad">vanilla gradient</span> $-\nabla_\theta \mathcal{G}$,
      our <span class="hl-func">functional update</span> $-\delta^{(i)} = -\beta\,(\phi_{\rho^{(i)}} - \hat{\phi})$ (Lemma 3.8)
      and the <span class="hl-direct">direct optimisation direction</span> $\theta^\star - \theta^{(i)}$ (Prop. 3.4).
    </figcaption>
  `;
  if (pageCaption) fig.querySelector('figcaption').replaceWith(pageCaption);


  // ── Θ plot ──────────────────────────────────────────────────
  const thetaSvg = fig.querySelector('.tp-plot--theta');
  const tBox = window.PlotBox(thetaSvg, { id: 'tp-t', title: 'Θ-space', sym: 'θ', view: VIEW });
  const contours = el('g', { class: 'tp-contours' }, tBox.under);

  // 𝒢(θ) in the top-left corner of the box, as in the paper's Fig. 1 — only when the page
  // asks for it with <figure … data-g-label> (index.html). HTML overlay so MathJax typesets it.
  if (fig.hasAttribute('data-g-label')) {
    const lbl = document.createElement('div');
    lbl.className = 'tp-glabel';
    lbl.textContent = '$\\mathcal{G}(\\theta)$';
    thetaSvg.parentNode.appendChild(lbl);
  }

  // Gradients at θ⁽ⁱ⁾ (toggle), as in the paper's Fig. 1 and geometries.py (negated, fixed length):
  //   grad    vanilla gradient step direction      −∇_θ G = Ψᵀ(ρ̂ − ρ_π)
  //   func    functional update direction (ours)   −δ⁽ⁱ⁾ = −(θ − β(φ_E − φ_ρ))
  //   direct  direct optimisation direction         θ⋆ − θ⁽ⁱ⁾   (+ dashed line to θ⋆)
  const grads = el('g', { class: 'tp-grads' }, tBox.under);
  const gDefs = el('defs', {}, thetaSvg);
  const toStar = el('line', { class: 'tp-gline' }, grads);
  const arrows = {};
  ['grad', 'func', 'direct'].forEach(k => {
    const m = el('marker', { id: 'tp-g-' + k, class: 'tp-gm--' + k, viewBox: '0 0 10 10', refX: 7, refY: 5,
                             markerWidth: 9, markerHeight: 9, markerUnits: 'userSpaceOnUse', orient: 'auto' }, gDefs);
    el('path', { d: 'M 0 1 L 10 5 L 0 9 z' }, m);
    arrows[k] = el('line', { class: 'tp-g tp-g--' + k, 'marker-end': 'url(#tp-g-' + k + ')' }, grads);
  });
  const setLine = (ln, x1, y1, x2, y2) => [['x1', x1], ['y1', y1], ['x2', x2], ['y2', y2]]
    .forEach(([a, v]) => ln.setAttribute(a, v));

  function drawGrads(th) {
    const x0 = tBox.X(th[0]), y0 = tBox.Y(th[1]);
    const dirs = {
      grad:   dualGrad(th).map(v => -v),
      func:   delta(th).map(v => -v),
      direct: [thStar[0] - th[0], thStar[1] - th[1]],
    };
    Object.entries(dirs).forEach(([k, v]) => {
      const px = tBox.X(th[0] + v[0]) - x0, py = tBox.Y(th[1] + v[1]) - y0, n = Math.hypot(px, py) || 1;
      setLine(arrows[k], x0, y0, x0 + GRAD_LEN * px / n, y0 + GRAD_LEN * py / n);
    });
    setLine(toStar, x0, y0, tBox.X(thStar[0]), tBox.Y(thStar[1]));
  }

  tBox.point(thStar, { label: L('θ', { sup: '⋆' }) });
  const probe = tBox.point(THETA0, { label: L('θ', { sup: '(i)' }), cls: 'tp-probe' });


  // ── Φ plot ──────────────────────────────────────────────────
  const pBox = window.PlotBox(fig.querySelector('.tp-plot--phi'), { id: 'tp-p', title: 'Φ-space', sym: 'ϕ', view: VIEW_PHI });
  const geodesic = el('line', { class: 'tp-geodesic' }, pBox.marks);        // φ̂ → φ_ρ⁽ⁱ⁾

  // φ_E, φ⋆ and φ̂ lie close together: labels below, left and right keep them apart
  pBox.point(phiE,    { shape: 'square', label: L('ϕ', { sub: 'E' }), side: 'below' });
  pBox.point(phiStar, { shape: 'square', label: L('ϕ', { sup: '⋆' }), side: 'left' });
  const phiRho = pBox.point(phiE, { shape: 'square', label: L('ϕ', { sub: 'ρ', subsup: '(i)' }), cls: 'tp-probe' });
  const phiT   = pBox.point(phiE, { shape: 'square', label: L('ϕ', { hat: true }), cls: 'tp-probe' });


  // ── Third panel: a menu (styled as the panel title) picks the MDP or a geometry view ──
  // Only the visible view is computed on hover. A page can pick the start view with
  // <figure … data-panel="mdp|tgeo|pgeo">.
  const picker = fig.querySelector('.tp-pick select');
  picker.value = fig.dataset.panel || 'mdp';
  let panel = picker.value;
  // (SVG elements have no .hidden property, so toggle the attribute)
  const showPanel = () => fig.querySelectorAll('[data-panel]').forEach(n => n.toggleAttribute('hidden', n.dataset.panel !== panel));
  showPanel();

  // MDP: optimal policy and rewards for r_θ (or for the expert). Its viewBox has the plots'
  // aspect ratio and puts the diagram at the height of their box centre.
  const mdp = draw(fig.querySelector('.tp-mdp'), { id: 'tp' });
  const R_TRUE   = EDGES.map(e => e.r);
  const OPT_TRUE = hardVI(R_TRUE, GAMMA, HARD_VI_STEPS);          // expert: VI on the true rewards


  // ── Geometry views: contours of the integral Hessian (Θ) and integral FIM (Φ) at θ⁽ⁱ⁾ ──
  // q̄(θ) = ½(θ − θ⋆)ᵀ H̄⁽ⁱ⁾ (θ − θ⋆)  and  q(φ) = ½(φ − φ̂)ᵀ H⁽ⁱ⁾ (φ − φ̂): centred on the targets,
  // with a thicker ring through θ⁽ⁱ⁾ / φ_ρ⁽ⁱ⁾. Recomputed on every hover (32 soft VIs for H̄).
  const tgBox = window.PlotBox(fig.querySelector('.tp-plot--tgeo'), { id: 'tp-tg', title: '', sym: 'θ', view: VIEW });
  const pgBox = window.PlotBox(fig.querySelector('.tp-plot--pgeo'), { id: 'tp-pg', title: '', sym: 'ϕ', view: VIEW_PHI });
  const tgRings = el('g', { class: 'tp-rings tp-rings--hess' }, tgBox.under);
  const pgRings = el('g', { class: 'tp-rings tp-rings--fim' }, pgBox.under);
  tgBox.point(thStar, { label: L('θ', { sup: '⋆' }) });
  const tgProbe = tgBox.point(THETA0, { label: L('θ', { sup: '(i)' }), cls: 'tp-probe' });
  const pgHat   = pgBox.point(phiE,   { shape: 'square', label: L('ϕ', { hat: true }), side: 'left', cls: 'tp-probe' });
  const pgRho   = pgBox.point(phiE,   { shape: 'square', label: L('ϕ', { sub: 'ρ', subsup: '(i)' }), cls: 'tp-probe' });

  function drawRings(g, box, view, M, c, p) {
    const { rings, through } = quadPaths(M, c, p, view, box.X, box.Y);
    g.textContent = '';
    rings.forEach(d => el('path', { d, class: 'tp-ring' }, g));
    el('path', { d: through, class: 'tp-ring is-through' }, g);
  }
  function drawGeometry(th, pr, ph) {
    if (panel === 'tgeo') { drawRings(tgRings, tgBox, VIEW, intHess(th), thStar, th); tgProbe.move(th); }
    if (panel === 'pgeo') { drawRings(pgRings, pgBox, VIEW_PHI, intFIM(th), phiHat(th), phiOf(th)); pgRho.move(pr); pgHat.move(ph); }
  }


  // ── Hover ───────────────────────────────────────────────────
  // Clamp a φ to the Φ box, inset by half a marker (5 SVG units of the 256-unit box)
  const clampPhi = v => ['x', 'y'].map((a, k) => {
    const [lo, hi] = VIEW_PHI[a], m = 5 / 256 * (hi - lo);
    return Math.min(hi - m, Math.max(lo + m, v[k]));
  });
  const num = v => v.toFixed(DIGITS).replace('-', '−');
  // A page can start with θ⁽ⁱ⁾ pinned somewhere: <figure … data-theta="-0.4,0.9"> (index.html).
  // Otherwise θ⁽ⁱ⁾ starts at θ⁽⁰⁾ and its markers stay hidden until you hover or pin.
  const START = fig.dataset.theta ? fig.dataset.theta.split(',').map(Number) : THETA0.slice();
  let pinned = START, expert = false, showGrads = false, current = pinned;

  function show(th) {
    current = th;
    const moved = th[0] !== THETA0[0] || th[1] !== THETA0[1];      // θ⁽ⁱ⁾ markers hidden until you hover or pin

    // Θ and Φ (moving φ's that leave the Φ box are held at its edge)
    const pr = clampPhi(phiOf(th)), ph = clampPhi(phiHat(th));
    probe.move(th);  phiRho.move(pr);  phiT.move(ph);
    [probe, phiRho, phiT].forEach(p => p.show(moved));
    geodesic.setAttribute('x1', pBox.X(ph[0])); geodesic.setAttribute('y1', pBox.Y(ph[1]));
    geodesic.setAttribute('x2', pBox.X(pr[0])); geodesic.setAttribute('y2', pBox.Y(pr[1]));
    geodesic.classList.toggle('is-hidden', !moved);
    if (showGrads) drawGrads(th);
    drawGeometry(th, pr, ph);
    [tgProbe, pgRho, pgHat].forEach(p => p.show(moved));

    // MDP
    if (panel !== 'mdp') return;
    const r   = expert ? R_TRUE : reward(th);
    const OPT = expert ? OPT_TRUE : hardVI(r, GAMMA, HARD_VI_STEPS);           // per edge: optimal action?
    mdp.edges.forEach(({ g, path, rVal }, i) => {
      g.classList.toggle('is-policy', OPT[i] && !expert);
      g.classList.toggle('is-expert', OPT[i] && expert);
      g.classList.toggle('is-dim', !OPT[i]);
      mdp.setArrow(path, !OPT[i] ? 'base' : expert ? 'expert' : 'policy');
      rVal.textContent = expert ? r[i] : num(r[i]);
    });
  }

  // At most one update per animation frame, however fast the pointer moves
  let queued = null;
  const schedule = th => {
    if (!queued) requestAnimationFrame(() => { const t = queued; queued = null; if (t && !locked) show(t); });
    queued = th;
  };
  // Hover moves θ⁽ⁱ⁾. Left-click fixes it there (hover then does nothing; another left-click moves
  // the fixed point). Right-click or the Reset chip unfixes it and returns it to the start.
  let locked = false;
  const resetChip = fig.querySelector('.chip[data-action="reset"]');
  const unlock = () => { locked = false; resetChip.hidden = true; pinned = START; };
  thetaSvg.addEventListener('pointermove', ev => { if (!locked) schedule(tBox.toData(ev) || pinned); });
  thetaSvg.addEventListener('pointerleave', () => { if (!locked) schedule(pinned); });
  thetaSvg.addEventListener('click', ev => {
    const th = tBox.toData(ev);
    if (th) { pinned = th; locked = true; resetChip.hidden = false; show(pinned); }
  });
  thetaSvg.addEventListener('contextmenu', ev => {
    ev.preventDefault();
    unlock();
    show(tBox.toData(ev) || pinned);                                             // back to following the cursor
  });
  resetChip.addEventListener('click', () => { unlock(); show(pinned); });
  picker.addEventListener('change', () => { panel = picker.value; showPanel(); show(current); });


  // ── Controls & caption values ───────────────────────────────
  let drawn = false;
  const dualChip = fig.querySelector('.chip[data-toggle="dual"]');
  dualChip.addEventListener('click', () => {
    if (!drawn) { contourPaths(tBox.X, tBox.Y).forEach(d => el('path', { d, class: 'tp-contour' }, contours)); drawn = true; }
    contours.classList.toggle('is-on', dualChip.classList.toggle('is-active'));
  });
  const gradChip = fig.querySelector('.chip[data-toggle="grads"]');
  gradChip.addEventListener('click', () => {
    showGrads = gradChip.classList.toggle('is-active');
    grads.classList.toggle('is-on', showGrads);
    show(current);
  });
  const expertChip = fig.querySelector('.chip[data-toggle="expert"]');
  expertChip.addEventListener('click', () => {
    expert = expertChip.classList.toggle('is-active');
    show(current);
  });
  // Coloured caption words do the same as their chip
  fig.querySelectorAll('figcaption .hl[data-toggle]').forEach(w => w.addEventListener('click', () =>
    fig.querySelector('.chip[data-toggle="' + w.dataset.toggle + '"]')?.click()));   // no-op if data-always removed the chip
  // A page can make toggles permanent: <figure … data-always="dual grads"> switches those on
  // and removes their chips (index.html: dual and grads; theory.html: dual only).
  (fig.dataset.always || '').split(/\s+/).filter(Boolean).forEach(k => {
    const chip = fig.querySelector('.chip[data-toggle="' + k + '"]');
    if (chip) { chip.click(); chip.remove(); }
  });
  fig.querySelectorAll('.tp-gamma').forEach(n => { n.textContent = GAMMA; });
  fig.querySelectorAll('.tp-beta').forEach(n => { n.textContent = BETA; });

  show(pinned);
})();
