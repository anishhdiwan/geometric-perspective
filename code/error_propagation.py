import hydra
from omegaconf import DictConfig
import numpy as np

import common as C

# Sweep design
RATIO_VALUES = np.arange(0.0, 2.01, 0.5)
DAMP         = 1e-12
RNG_SEED     = 0


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def _rho_from_phi(features, phi):
    logits  = features @ phi
    logits -= _logsumexp(logits)
    rho     = np.exp(logits)
    return rho / rho.sum()


def _F_from_rho(features, rho):
    mu = features.T @ rho
    F  = (features * rho[:, None]).T @ features - np.outer(mu, mu)
    return 0.5 * (F + F.T)


def fit_phi_mle(features, rho, n_iter=10, damp=DAMP):
    """Moment-matching MLE fit: solves mu(phi) = features.T @ rho by Newton's method."""
    d      = features.shape[1]
    phi    = np.zeros(d)
    target = features.T @ rho
    for _ in range(n_iter):
        rho_phi = _rho_from_phi(features, phi)
        F       = _F_from_rho(features, rho_phi)
        res     = features.T @ rho_phi - target
        phi    -= np.linalg.solve(F + damp * np.eye(d), res)
    return phi


def make_residual_g(features, rho_E, rng):
    """Random state function g with E[g] = 0, E[g^2] = 1 and Cov[psi, g] = 0 under rho_E."""
    n_states = features.shape[0]
    g0   = rng.standard_normal(n_states)
    A    = np.column_stack([features, np.ones(n_states)])
    AtWA = (A * rho_E[:, None]).T @ A
    AtWg = (A * rho_E[:, None]).T @ g0
    b    = np.linalg.solve(AtWA + DAMP * np.eye(A.shape[1]), AtWg)
    g    = g0 - A @ b
    g   /= np.sqrt(np.sum(rho_E * g**2))
    return g


def measure_gradient_error(features, theta_i, rho_E_mis, phi_E, beta):
    """Target gap: E_rho_hat[psi] - mu(phi_hat). Returns (phi_hat, grad_error)."""
    r_i     = features @ theta_i
    phi_hat = phi_E - theta_i / beta

    rho_bhat = _rho_from_phi(features, phi_hat)
    mu_hat   = features.T @ rho_bhat

    log_rho_hat  = np.log(rho_E_mis) - r_i / beta
    log_rho_hat -= _logsumexp(log_rho_hat)
    rho_hat      = np.exp(log_rho_hat)
    E_hat        = features.T @ rho_hat

    return phi_hat, E_hat - mu_hat


def _angle_deg(u, v):
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    cos_angle = np.dot(u, v) / denom if denom > 1e-12 else 1.0
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def sweep(features, rho_E, iterates, beta, ratio_values=RATIO_VALUES, seed=RNG_SEED):
    """||target gap|| and angle(delta, delta_tilde) per (ratio, iterate). Returns (records, g)."""
    rng = np.random.default_rng(seed)
    g   = make_residual_g(features, rho_E, rng)

    # c-independent agent fits and clean (c = 0) updates
    phi_rho_list = [fit_phi_mle(features, rho_i) for _, rho_i in iterates]
    phi_E_clean  = fit_phi_mle(features, rho_E)
    delta_clean  = [beta * (phi_rho_list[k] - (phi_E_clean - theta_i / beta))
                    for k, (theta_i, _) in enumerate(iterates)]

    # in-basis scale std_rho_E(phi_E . psi)
    r_E_clean   = features @ phi_E_clean
    phi_E_scale = np.sqrt(np.sum(rho_E * r_E_clean**2) - (np.sum(rho_E * r_E_clean))**2)

    records = np.zeros((len(ratio_values), len(iterates), 2))
    for j, ratio in enumerate(ratio_values):
        c          = ratio * phi_E_scale
        rho_E_mis  = rho_E * np.exp(c * g)
        rho_E_mis /= rho_E_mis.sum()
        phi_E      = fit_phi_mle(features, rho_E_mis)

        for k, (theta_i, _) in enumerate(iterates):
            phi_hat, grad_error = measure_gradient_error(features, theta_i, rho_E_mis, phi_E, beta)
            delta_tilde = beta * (phi_rho_list[k] - phi_hat)
            records[j, k] = (np.linalg.norm(grad_error), _angle_deg(delta_clean[k], delta_tilde))
    return records, g


def print_table(records):
    """Rows = ratio, cols = the two errors (mean over iterates)."""
    mean = records.mean(axis=1)
    print("\n| ratio | grad. error | misalign [deg] |")
    print("|---|---|---|")
    for ratio, (grad_error, misalign) in zip(RATIO_VALUES, mean):
        print(f"| {ratio:.1f} | {grad_error:.4e} | {misalign:.3f} |")


@hydra.main(version_base=None, config_path=".", config_name="config")
def experiment(cfg : DictConfig) -> None:
    world, features, logs, irl_cfg = C.build(cfg)

    # Error propagation sweep
    geom_logs = logs["dual_geometry"]
    beta      = irl_cfg["beta"]

    rho_E  = np.asarray(geom_logs[0]["rho_E"], dtype=float).reshape(-1)
    rho_E /= rho_E.sum()

    iterates = []
    for item in geom_logs:
        theta_i = np.asarray(item["theta"], dtype=float).reshape(-1)
        rho_i   = np.asarray(item["rho"],   dtype=float).reshape(-1)
        iterates.append((theta_i, rho_i / rho_i.sum()))

    records, g = sweep(features, rho_E, iterates, beta)
    print_table(records)


if __name__ == "__main__":
    experiment()
