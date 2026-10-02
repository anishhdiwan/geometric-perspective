/* ══════════════════════════════════════════════════════════════
 * SHARED · IRL ON THE TWO-STATE MDP   (used by the Θ-space figures;
 * needs two-state-mdp.js)
 *
 * Θ (reward parameters):  r_θ = Ψθ.
 *   Dual   G(θ) = (1−γ) μ₀ᵀ V_θ + β log Σ_{s,a} ρ_E(s,a) exp(−r_θ(s,a)/β),
 *          V_θ the soft value function of r_θ (α = 1).
 *   ∇G     = Ψᵀρ_π − Ψᵀρ̂,  ρ̂ ∝ ρ_E exp(−r_θ/β)   (Prop. 3.5).
 *   θ⋆     = argmin G, by gradient descent from θ⁽⁰⁾ (as in
 *          irl.mceirl_vanilla_gradient).
 * Φ (occupancy natural parameters, Assumption 3.1):
 *   φ(ρ)   = least-squares fit of log ρ onto [Ψ, 1], first two
 *            coefficients (geometries._fit_phi_from_log_rho).
 *   φ_E = φ(ρ_E),  φ_ρ(θ) = φ(ρ_{π_θ}) with π_θ the soft-optimal policy
 *   of r_θ,  φ⋆ = φ_ρ(θ⋆),  φ̂(θ) = φ_E − θ/β  (Eq. 3).
 *   δ(θ)   = θ − β(φ_E − φ_ρ(θ)), the functional-update direction (Eq. 1).
 * Geometry at θ⁽ⁱ⁾ (τ-integrals by 8-point Gauss–Legendre):
 *   H⁽ⁱ⁾  = ∫₀¹ F(φ̂ + τ(φ_ρ − φ̂)) dτ, integral FIM (Lemma 3.8), F(φ) = Cov_ρφ[ψ]
 *   H̄⁽ⁱ⁾  = ∫₀¹ ∇²G(θ⋆ + τ(θ − θ⋆)) dτ, integral Hessian (Prop. 3.4), ∇²G by
 *           central differences of ∇G (geometries._hess_at_theta)
 * The expert π_E is soft VI on the true rewards (computed on load).
 * The contour grid of G is computed on first use and cached, so every
 * figure that shows the contours shares it.
 *
 * Edit the CONFIG block only.
 * ══════════════════════════════════════════════════════════════ */
window.TwoStateIRL = (function () {
  const { EDGES, MU0, FEATURES, softVI, occupancy } = window.TwoStateMDP;

  // ── CONFIG ──────────────────────────────────────────────────
  const GAMMA  = 0.9;                            // discount
  const BETA   = 10;                             // β in Eq. (2)
  const THETA0 = [0, 0];                         // θ⁽⁰⁾
  const VIEW   = { x: [-0.8, 2.4], y: [-1.9, 1.3] }; // plotted Θ range: θ₁ (x), θ₂ (y)

  const SOFT_VI_STEPS = 200;                     // soft VI for π_E, θ⋆, φ
  const GRID_VI_STEPS = 80;                      // cheap soft VI per contour grid point
  const GRID     = 41;                           // contour grid is GRID × GRID
  const N_LEVELS = 14;                           // number of contour lines
  const GD       = { step: 4, maxIter: 200, tol: 1e-7 };  // gradient descent for θ⋆
  // ─────────────────────────────────────────────────────────────


  // ── Rewards, expert, dual ───────────────────────────────────
  const reward = th => FEATURES.map(f => f[0] * th[0] + f[1] * th[1]);        // r_θ = Ψθ
  const lse    = xs => { const m = Math.max(...xs); return m + Math.log(xs.reduce((s, x) => s + Math.exp(x - m), 0)); };
  const rhoOf  = th => occupancy(softVI(reward(th), GAMMA, SOFT_VI_STEPS).pi, GAMMA);

  const logRhoE = occupancy(softVI(EDGES.map(e => e.r), GAMMA, SOFT_VI_STEPS).pi, GAMMA).map(Math.log);

  function dual(th, steps) {
    const r = reward(th), { V } = softVI(r, GAMMA, steps);
    return (1 - GAMMA) * (MU0[0] * V[0] + MU0[1] * V[1]) + BETA * lse(logRhoE.map((l, i) => l - r[i] / BETA));
  }
  function dualGrad(th) {
    const r = reward(th), rho = rhoOf(th);
    const z = logRhoE.map((l, i) => l - r[i] / BETA), Z = lse(z);
    const diff = rho.map((p, i) => p - Math.exp(z[i] - Z));                  // ρ_π − ρ̂
    return [0, 1].map(k => FEATURES.reduce((s, f, i) => s + f[k] * diff[i], 0));
  }

  let thStar = THETA0.slice();                                                // θ⋆ = argmin G
  for (let k = 0; k < GD.maxIter; k++) {
    const g = dualGrad(thStar), next = [thStar[0] - GD.step * g[0], thStar[1] - GD.step * g[1]];
    const moved = Math.hypot(next[0] - thStar[0], next[1] - thStar[1]);
    thStar = next;
    if (moved < GD.tol) break;
  }


  // ── Φ: least-squares fit of log ρ onto [Ψ, 1] ───────────────
  // Normal equations (AᵀA) c = Aᵀ log ρ with A = [Ψ, 1] (4 × 3), solved by Cramer's rule.
  const A   = FEATURES.map(f => [f[0], f[1], 1]);
  const AtA = [0, 1, 2].map(i => [0, 1, 2].map(j => A.reduce((s, a) => s + a[i] * a[j], 0)));
  const det3 = M => M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                  - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                  + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  function fitPhi(logRho) {
    const b = [0, 1, 2].map(i => A.reduce((s, a, k) => s + a[i] * logRho[k], 0));
    const D = det3(AtA);
    return [0, 1].map(c => det3(AtA.map((row, i) => row.map((v, j) => (j === c ? b[i] : v)))) / D);
  }
  const phiOf  = th => fitPhi(rhoOf(th).map(Math.log));                      // φ_ρ(θ)
  const phiHat = th => [0, 1].map(k => phiE[k] - th[k] / BETA);              // φ̂(θ), Eq. (3)
  const phiE    = fitPhi(logRhoE);
  const phi0    = phiOf(THETA0);
  const phiStar = phiOf(thStar);

  // Functional-update direction δ = θ − β(φ_E − φ_ρ) = β(φ_ρ − φ̂)  (Eq. 1, Lemma 3.8);
  // the same as θ − lstsq(Ψ, β log(ρ_E/ρ_π)) in irl.functional_reward_update.
  const delta = th => { const pr = phiOf(th); return [0, 1].map(k => th[k] - BETA * (phiE[k] - pr[k])); };


  // ── Integral geometry matrices at θ⁽ⁱ⁾ ─────────────────────
  // τ-integrals by 8-point Gauss–Legendre on [0, 1]: the integrands are smooth, and this
  // matches 16 points to 8 digits (the code's 200-point uniform mean is ~1e-3 off).
  const GL8 = [[0.1834346424956498, 0.3626837833783620], [0.5255324099163290, 0.3137066458778873],
               [0.7966664774136267, 0.2223810344533745], [0.9602898564975363, 0.1012285362903763]]
    .flatMap(([x, w]) => [[(1 - x) / 2, w / 2], [(1 + x) / 2, w / 2]]);            // [τ, weight]
  const integrate = f => GL8.reduce((M, [t, w]) => {
    const A = f(t);
    return M.map((row, i) => row.map((v, j) => v + w * A[i][j]));
  }, [[0, 0], [0, 0]]);
  const sym = M => [[M[0][0], (M[0][1] + M[1][0]) / 2], [(M[0][1] + M[1][0]) / 2, M[1][1]]];
  const lerp = (a, b, t) => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];

  // FIM of the Boltzmann occupancy ρ_φ ∝ exp(Ψφ): F(φ) = Cov_ρφ[ψ]  (Prop. 3.5; geometries._F_from_phi)
  function fim(phi) {
    const z = FEATURES.map(f => f[0] * phi[0] + f[1] * phi[1]), Z = lse(z);
    const p = z.map(v => Math.exp(v - Z));
    const mu = [0, 1].map(k => FEATURES.reduce((s, f, i) => s + p[i] * f[k], 0));
    return [0, 1].map(a => [0, 1].map(b =>
      FEATURES.reduce((s, f, i) => s + p[i] * (f[a] - mu[a]) * (f[b] - mu[b]), 0)));
  }
  // Hessian ∇²_θ G by central differences of the exact gradient (geometries._hess_at_theta)
  function hessG(th, eps = 1e-3) {
    const col = k => {
      const up = th.slice(), dn = th.slice();
      up[k] += eps; dn[k] -= eps;
      const gu = dualGrad(up), gd = dualGrad(dn);
      return [0, 1].map(i => (gu[i] - gd[i]) / (2 * eps));
    };
    const c0 = col(0), c1 = col(1);
    return sym([[c0[0], c1[0]], [c0[1], c1[1]]]);
  }
  // Integral FIM along the e-geodesic, H⁽ⁱ⁾ = ∫₀¹ F(φ̂ + τ(φ_ρ − φ̂)) dτ  (Lemma 3.8).
  // Exact identity: H⁽ⁱ⁾(φ_ρ − φ̂) = μ(φ_ρ) − μ(φ̂).
  const intFIM = th => { const pr = phiOf(th), ph = phiHat(th); return sym(integrate(t => fim(lerp(ph, pr, t)))); };
  // Integral Hessian from θ⋆ to θ, H̄⁽ⁱ⁾ = ∫₀¹ ∇²G(θ⋆ + τ(θ − θ⋆)) dτ  (Prop. 3.4; θ⋆ is known here).
  // Exact identity: H̄⁽ⁱ⁾(θ − θ⋆) = ∇G(θ). Costs 32 soft VIs.
  const intHess = th => sym(integrate(t => hessG(lerp(thStar, th, t))));


  // ── Contours: marching squares on a grid of values Z[j][i] at (ts1[i], ts2[j]) ──
  // One SVG path string per level, in the plot's coordinates X, Y.
  function isoPaths(Z, ts1, ts2, levels, X, Y) {
    return levels.map(level => {
      let d = '';
      for (let j = 0; j < ts2.length - 1; j++) for (let i = 0; i < ts1.length - 1; i++) {
        const c = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];            // cell corners
        const pts = [];
        for (let e = 0; e < 4; e++) {                                           // crossings on the 4 sides
          const [ia, ja] = c[e], [ib, jb] = c[(e + 1) % 4];
          const va = Z[ja][ia] - level, vb = Z[jb][ib] - level;
          if ((va < 0) !== (vb < 0)) {
            const t = va / (va - vb);
            pts.push([X(ts1[ia] + t * (ts1[ib] - ts1[ia])), Y(ts2[ja] + t * (ts2[jb] - ts2[ja]))]);
          }
        }
        for (let p = 0; p + 1 < pts.length; p += 2) {
          d += 'M' + pts[p][0].toFixed(1) + ' ' + pts[p][1].toFixed(1) +
               'L' + pts[p + 1][0].toFixed(1) + ' ' + pts[p + 1][1].toFixed(1);
        }
      }
      return d;
    });
  }
  const axisGrid = (view, n) => ['x', 'y'].map(a =>
    Array.from({ length: n }, (_, i) => view[a][0] + i * (view[a][1] - view[a][0]) / (n - 1)));

  // Dual G: grid computed once, on first use, and shared by every figure
  let grid = null;
  function dualGrid() {                                                       // { ts1, ts2, Z, levels }; read-only
    if (!grid) {
      const [ts1, ts2] = axisGrid(VIEW, GRID);
      const Z = ts2.map(t2 => ts1.map(t1 => dual([t1, t2], GRID_VI_STEPS)));
      const sorted = Z.flat().sort((a, b) => a - b);
      const levels = Array.from({ length: N_LEVELS - 1 },                     // quantiles: lines spread evenly
        (_, k) => sorted[Math.floor((k + 1) / N_LEVELS * sorted.length)]);
      grid = { ts1, ts2, Z, levels };
    }
    return grid;
  }
  function contourPaths(X, Y) {
    const g = dualGrid();
    return isoPaths(g.Z, g.ts1, g.ts2, g.levels, X, Y);
  }

  // Quadratic q(x) = ½ (x − c)ᵀ M (x − c) over a plot's view, as contour paths: N_RINGS − 1 rings
  // evenly spaced in √q across the box (so the density doesn't depend on p), plus the ring
  // through p, returned separately (drawn thicker): { rings, through }.
  const N_RINGS = 6;
  function quadPaths(M, c, p, view, X, Y) {
    const q = x => { const a = x[0] - c[0], b = x[1] - c[1]; return 0.5 * (M[0][0] * a * a + 2 * M[0][1] * a * b + M[1][1] * b * b); };
    const [ts1, ts2] = axisGrid(view, GRID);
    const Z = ts2.map(t2 => ts1.map(t1 => q([t1, t2])));
    const qMax = Math.max(...Z.flat());
    const levels = Array.from({ length: N_RINGS - 1 }, (_, k) => qMax * ((k + 1) / N_RINGS) ** 2);
    const [through] = isoPaths(Z, ts1, ts2, [q(p)], X, Y);
    return { rings: isoPaths(Z, ts1, ts2, levels, X, Y), through };
  }

  return { GAMMA, BETA, THETA0, VIEW, reward, dualGrad, delta, thStar,
           phiE, phi0, phiStar, phiOf, phiHat, intFIM, intHess, contourPaths, dualGrid, quadPaths, isoPaths, dual, rhoOf };
})();
