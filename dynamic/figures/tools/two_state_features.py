"""Features psi(s,a) in R^2 for the two-state problem (cached in theta-space.js).

Assumption 3.1 asks for occupancies that are Boltzmann in psi:
    rho(s,a) = exp(phi^T psi(s,a)) / Z.
With 4 state-action pairs this means log rho lies in span{psi_1, psi_2, 1}.

  psi_1 : centred log rho_E              -> the expert is *exactly* Boltzmann
  psi_2 : top principal direction of the centred log-occupancies of a grid of
          policies, orthogonal to psi_1  -> agent occupancies are as Boltzmann
          as a 2D feature space allows (the rest is out of span, cf. App. A.2)

Run:  python dynamic/figures/tools/two_state_features.py   (numpy only)
Keep GAMMA, the MDP and the expert in sync with two-state-mdp.js / theta-space.js.
"""
import numpy as np

GAMMA = 0.9
NEXT = np.array([[0, 1], [0, 1]])         # next state for (s, a)
MU0 = np.array([1.0, 0.0])
R_TRUE = np.array([1.0, 0.0, 0.0, 2.0])   # order (0,0), (0,1), (1,0), (1,1)


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def soft_vi(r, sweeps=200):               # alpha = 1, as in irl.soft_value_iteration
    r, V = r.reshape(2, 2), np.zeros(2)
    for _ in range(sweeps):
        Q = r + GAMMA * V[NEXT]
        V = np.array([lse(q) for q in Q])
    return np.exp(Q - V[:, None])


def occupancy(pi):                         # normalised discounted rho(s,a)
    P = np.zeros((2, 2))
    for s in range(2):
        for a in range(2):
            P[s, NEXT[s, a]] += pi[s, a]
    d = np.linalg.solve(np.eye(2) - GAMMA * P.T, (1 - GAMMA) * MU0)
    return (d[:, None] * pi).ravel()


def centre(v):
    return v - v.mean()


log_rho_E = np.log(occupancy(soft_vi(R_TRUE)))
psi1 = centre(log_rho_E)
psi1 /= np.linalg.norm(psi1)

grid = np.linspace(0.02, 0.98, 49)        # policies pi(a=1|s=0), pi(a=1|s=1)
L = np.array([centre(np.log(occupancy(np.array([[1 - p, p], [1 - q, q]]))))
              for p in grid for q in grid])
L_res = L - np.outer(L @ psi1, psi1)
psi2 = np.linalg.svd(L_res - L_res.mean(0), full_matrices=False)[2][0]
psi2 = centre(psi2)
psi2 -= (psi2 @ psi1) * psi1
psi2 /= np.linalg.norm(psi2)

Psi = np.stack([psi1, psi2], axis=1)
explained = 1 - ((L - L @ Psi @ Psi.T) ** 2).sum() / (L ** 2).sum()
print(f"agent log-occupancy variance in span: {explained:.3f} (expert: exact)")
print("const FEATURES = [   // psi(s,a) = [psi_1, psi_2], rows in EDGES order")
for (s, a), row in zip([(0, 0), (0, 1), (1, 0), (1, 1)], Psi):
    print(f"  [{row[0]: .6f}, {row[1]: .6f}],   // (s={s}, a={a})")
print("];")
