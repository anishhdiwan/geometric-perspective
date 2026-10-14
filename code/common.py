import numpy as np
from omegaconf import DictConfig, OmegaConf

import gridworld as W
import irl as IRL
import trajectory as T
import optimizer as O

DAMP = 1e-8

IRL_ALGOS = {
    "trirl":                      IRL.trirl_discrete,
    "vanilla_grad":               IRL.mceirl_vanilla_gradient,
    "natural_grad":               IRL.mceirl_natural_gradient,
    "path_integral_natural_grad": IRL.mceirl_path_natural_gradient,
}


def build(cfg: DictConfig):
    """Builds the gridworld and expert, learns the features, and runs the IRL algorithm
    `cfg.irl.algo`. Returns the world, the features, the IRL logs, and the IRL config."""
    env_cfg    = OmegaConf.to_container(cfg.env, resolve=True)
    expert_cfg = OmegaConf.to_container(cfg.expert, resolve=True)
    irl_cfg    = OmegaConf.to_container(cfg.irl, resolve=True)
    seed       = irl_cfg["seed"]
    np.random.seed(seed)

    # GridWorld
    absorbing = np.zeros(int(env_cfg["world_size"]**2), dtype=bool)
    absorbing[env_cfg["absorbing_idx"]] = True
    terminal = np.zeros(int(env_cfg["world_size"]**2), dtype=bool)
    terminal[env_cfg["terminal_idx"]] = True
    initial_states = env_cfg["initial_idx"]

    world = W.IcyGridWorld(size=env_cfg["world_size"], p_slip=env_cfg["p_slip"],
                           absorbing=absorbing, terminal=terminal, initial=initial_states)

    r_default, r_abs, r_goal = env_cfg["r_default"], env_cfg["r_abs"], env_cfg["r_goal"]
    penultimate_coords, rewarding_actions_idx = env_cfg["penultimate_coords"], env_cfg["rewarding_actions_idx"]
    reward = np.full((world.n_states, world.n_actions), r_default)
    reward[absorbing.nonzero(), :] = r_abs
    for coords, action_idx in zip(penultimate_coords, rewarding_actions_idx):
        row, col, n = coords[0], coords[1], world.size
        reward[int(row*n + col), action_idx] = r_goal

    # Expert trajectories
    _, expert_policy, _ = IRL.soft_value_iteration(world.p_transition, reward, irl_cfg["discount"], eps=1e-5)
    expert_policy_exec  = T.stochastic_policy_adapter(expert_policy)
    expert_trajectories = list(T.generate_trajectories(expert_cfg["n_trajectories"], world, expert_policy_exec))

    # Features / IRL
    features = W.boltzmann_features(world, expert_trajectories,
                                    base_features=W.rbf_features(world), plot=False, seed=seed)
    init     = O.Constant(irl_cfg["init_theta"])

    _, _, logs = IRL_ALGOS[irl_cfg["algo"]](world, features, expert_policy, init, irl_cfg)

    return world, features, logs, irl_cfg


class DualGeometry:
    """Occupancy-space and reward-space geometry of the IRL dual for a fixed world,
    feature map, discount, beta, and expert occupancy rho_E."""

    def __init__(self, world, features, discount, beta, rho_E, eps_vi=1e-5):
        self.world    = world
        self.features = features
        self.gamma    = discount
        self.beta     = beta
        self.eps_vi   = eps_vi

        self.n_states  = world.n_states
        self.p_initial = np.zeros(self.n_states)
        self.p_initial[world.initial] = 1.0 / len(world.initial)

        self.rho_E    = np.asarray(rho_E, dtype=float).reshape(-1)
        self.rho_E   /= self.rho_E.sum()
        self.log_rhoE = np.log(self.rho_E.clip(1e-10, 1.0))
        self.phi_E    = self.fit_phi_from_log_rho(self.log_rhoE)

    @staticmethod
    def logsumexp(x):
        m = np.max(x)
        return m + np.log(np.sum(np.exp(x - m)))

    def rho_from_policy(self, pi):
        P_pi = np.einsum("sna,sa->sn", self.world.p_transition, pi)
        rho  = np.linalg.solve(np.eye(self.n_states) - self.gamma * P_pi.T, self.p_initial)
        rho  = np.clip(rho, 0.0, None)
        rho /= rho.sum()
        return rho

    def fit_phi_from_log_rho(self, log_rho):
        A = np.column_stack([self.features, np.ones(self.features.shape[0])])
        sol, _, _, _ = np.linalg.lstsq(A, log_rho, rcond=None)
        return sol[:2]

    def rho_exp_family(self, phi):
        logits  = self.features @ phi
        logits -= self.logsumexp(logits)
        rho     = np.exp(logits)
        rho    /= rho.sum()
        return rho

    def F_from_rho(self, rho):
        rho = np.asarray(rho, dtype=float).reshape(-1)
        rho = rho / rho.sum()
        mu  = self.features.T @ rho
        F   = (self.features * rho[:, None]).T @ self.features - np.outer(mu, mu)
        return 0.5 * (F + F.T)

    def F_from_phi(self, phi):
        return self.F_from_rho(self.rho_exp_family(phi))

    def integral_fim(self, phi_rho, phi_hat, n_taus=200):
        """H = ∫₀¹ F(phi_hat + tau (phi_rho - phi_hat)) dtau on a uniform tau-grid."""
        H = np.zeros((2, 2))
        for tau in np.linspace(0.0, 1.0, n_taus):
            H += self.F_from_phi(phi_hat + tau * (phi_rho - phi_hat))
        return 0.5 * (H / n_taus + (H / n_taus).T)

    def grad_at_theta(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(2)
        r     = self.features @ theta
        _, pi, _ = IRL.soft_value_iteration(
            self.world.p_transition, np.repeat(r[:, None], self.world.n_actions, axis=1),
            self.gamma, eps=self.eps_vi)
        rho         = self.rho_from_policy(pi)
        log_rho_hat = self.log_rhoE - r / self.beta
        log_rho_hat = log_rho_hat - self.logsumexp(log_rho_hat)
        rho_hat     = np.exp(log_rho_hat)
        return self.features.T @ rho - self.features.T @ rho_hat

    def hess_at_theta(self, theta, fd_eps=1e-3):
        theta = np.asarray(theta, dtype=float).reshape(2)
        I2    = np.eye(2)
        H     = np.zeros((2, 2))
        for k in range(2):
            H[:, k] = (self.grad_at_theta(theta + fd_eps * I2[:, k])
                      - self.grad_at_theta(theta - fd_eps * I2[:, k])) / (2.0 * fd_eps)
        return 0.5 * (H + H.T)
