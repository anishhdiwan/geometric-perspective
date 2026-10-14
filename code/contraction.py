"""Contraction rates of the vanilla gradient and the functional update along the IRL iterates."""
import hydra
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm

import common as C


@hydra.main(version_base=None, config_path=".", config_name="config")
def experiment(cfg : DictConfig) -> None:
    world, features, logs, irl_cfg = C.build(cfg)
    geom_logs = logs["dual_geometry"]
    geo       = C.DualGeometry(world, features, irl_cfg["discount"], irl_cfg["beta"], geom_logs[0]["rho_E"])
    beta      = geo.beta
    damp      = C.DAMP

    # Integral FIM H_i per iterate
    results = []
    for tqdm_i, item in enumerate(tqdm(geom_logs, desc="Processing logged thetas")):
        theta_i     = np.asarray(item["theta"],       dtype=float).reshape(2)
        rho_i       = np.asarray(item["rho"],         dtype=float).reshape(-1)
        rho_i      /= rho_i.sum()
        tqdm.write(f"theta[{tqdm_i}] = {theta_i}")

        phi_rho_i    = geo.fit_phi_from_log_rho(np.log(rho_i.clip(1e-10, 1.0)))
        phi_hat_i    = geo.phi_E - theta_i / beta

        results.append({
            "theta":      theta_i,
            "H":          geo.integral_fim(phi_rho_i, phi_hat_i),
        })

    # Contraction coefficients; H_bar_i is the dual Hessian averaged on the line theta* -> theta_i
    theta_star = np.array([0.07079738, 0.90061526], dtype=float) # converged theta from a 100-iteration run
    taus_H     = np.linspace(0.0, 1.0, 10)
    print("\nkappa(H_bar_i) along the TRIRL theta trajectory")
    print(f"theta* (optimum) = {theta_star}")
    print()
    print("| i | κ(H̄⁽ⁱ⁾) | coeff. VG | κ((H⁽ⁱ⁾)⁻¹H̄⁽ⁱ⁾) | coeff. functional | c = √λmax((H⁽ⁱ⁾)⁻¹H⁽ⁱ⁺¹⁾) | c × coeff. functional |")
    print("|---|---|---|---|---|---|---|")
    for i, r in enumerate(results):
        theta_i = np.asarray(r["theta"], dtype=float).reshape(2)
        H_bar   = np.zeros((2, 2))
        for tau in taus_H:
            H_bar += geo.hess_at_theta(theta_star + tau * (theta_i - theta_star))
        H_bar /= len(taus_H)
        H_bar  = 0.5 * (H_bar + H_bar.T)
        eig    = np.linalg.eigvalsh(H_bar)
        kappa  = np.abs(eig).max() / max(np.abs(eig).min(), 1e-12)

        H_i = r["H"]

        # kappa((H_i)^-1 H_bar_i)
        M_i       = np.linalg.solve(H_i + damp * np.eye(2), H_bar)
        eig_nat   = np.linalg.eigvals(M_i).real
        kappa_nat = np.abs(eig_nat).max() / max(np.abs(eig_nat).min(), 1e-12)
        rate_nat  = (kappa_nat - 1.0) / (kappa_nat + 1.0)

        # change of the H-norm between consecutive iterates
        if i + 1 < len(results):
            H_next      = results[i + 1]["H"]
            A_i         = np.linalg.solve(H_i + damp * np.eye(2), H_next)
            lambda_max  = np.sqrt(np.abs(np.linalg.eigvals(A_i).real).max())

            extra_cols = f"{lambda_max:.4f} | {lambda_max * rate_nat:.4f}"
        else:
            extra_cols = "n/a | n/a"

        print(f"| {i} | {kappa:.4f} | {(kappa - 1.0) / (kappa + 1.0):.4f} | "
              f"{kappa_nat:.4f} | {rate_nat:.4f} | "
              f"{extra_cols} |")


if __name__ == "__main__":
    experiment()
