"""Dual-geometry figure and, optionally, the angle-to-direct-optimisation table."""
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.lines import Line2D

import common as C


def compute_results(geo, geom_logs, n_taus=200, damp=C.DAMP):
    """Per-iterate update directions, reward-space Hessian, and FIMs F and H."""
    results = []
    features, beta = geo.features, geo.beta

    for tqdm_i, item in enumerate(tqdm(geom_logs, desc="Processing logged thetas")):
        theta_i     = np.asarray(item["theta"],       dtype=float).reshape(2)
        rho_i       = np.asarray(item["rho"],         dtype=float).reshape(-1)
        rho_i      /= rho_i.sum()
        tqdm.write(f"theta[{tqdm_i}] = {theta_i}")

        r_i           = features @ theta_i
        log_rho_hat_i = geo.log_rhoE - r_i / beta
        log_rho_hat_i = log_rho_hat_i - geo.logsumexp(log_rho_hat_i)
        rho_hat_i     = np.exp(log_rho_hat_i)

        phi_rho_i    = geo.fit_phi_from_log_rho(np.log(rho_i.clip(1e-10, 1.0)))
        phi_hat_i    = geo.phi_E - theta_i / beta
        grad_theta_i = features.T @ rho_i - features.T @ rho_hat_i

        F_i = geo.F_from_rho(rho_i)
        H_i = geo.integral_fim(phi_rho_i, phi_hat_i, n_taus)

        nat_F_i      = np.linalg.solve(F_i + damp * np.eye(2), grad_theta_i)
        nat_H_i      = np.linalg.solve(H_i + damp * np.eye(2), grad_theta_i)
        Hess_theta_i = geo.hess_at_theta(theta_i)

        results.append({
            "theta":      theta_i,
            "phi_rho":    phi_rho_i,
            "phi_hat":    phi_hat_i,
            "grad":      -grad_theta_i,
            "nat_F":     -nat_F_i,
            "nat_H":     -nat_H_i,
            "Hess_theta": Hess_theta_i,
            "F":          F_i,
            "H":          H_i,
        })
    return results


def print_angle_table(geo, results, damp=C.DAMP):
    theta_star  = np.array([0.07079738, 0.90061526], dtype=float)  # converged theta from a longer run
    taus_Hbar   = np.linspace(0.0, 1.0, 10)

    def _angle_deg(u, v):
        cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    print("\nAngle to the direct-optimization direction")
    print("| i | angle(vanilla, optimal) | angle(natF, optimal) | angle(natH, optimal) |")
    print("|---|---|---|---|")
    for i, r in enumerate(results):
        theta_i = r["theta"]
        H_bar_i = np.zeros((2, 2))
        for tau in taus_Hbar:
            H_bar_i += geo.hess_at_theta(theta_star + tau * (theta_i - theta_star))
        H_bar_i /= len(taus_Hbar)
        H_bar_i  = 0.5 * (H_bar_i + H_bar_i.T)

        nat_Hbar_i = np.linalg.solve(H_bar_i + damp * np.eye(2), r["grad"])

        ang_vanilla = _angle_deg(r["grad"],  nat_Hbar_i)
        ang_natF    = _angle_deg(r["nat_F"], nat_Hbar_i)
        ang_natH    = _angle_deg(r["nat_H"], nat_Hbar_i)
        print(f"| {i} | {ang_vanilla:.4f} | {ang_natF:.4f} | {ang_natH:.4f} |")


def plot_dual_geometries(results, PC, damp=C.DAMP):
    """Theta-space directions, Phi-space paths, and their metric ellipses."""
    plt.rcParams.update({
        "font.family":     PC["font"],
        "axes.titlesize":  PC["titlesize"],
        "axes.labelsize":  PC["labelsize"],
        "legend.fontsize": PC["legendsize"],
        "xtick.labelsize": PC["ticksize"],
        "ytick.labelsize": PC["ticksize"],
    })
    total_figsize = (PC["figsize"][0] * PC["fig_scale_w"],
                     PC["figsize"][1] * PC["fig_scale_h"])
    fig, axes = plt.subplots(1, 4, figsize=total_figsize,
                             sharey=False, sharex=False, constrained_layout=False)
    ax_theta, ax_geom, ax_theta_metric, ax_phi_metric = axes
    if PC["axes_aspect"] is not None:
        for ax in axes:
            ax.set_box_aspect(PC["axes_aspect"])

    def _ellipse_points(M, center, level=1.0, normalize=False):
        w, V = np.linalg.eigh(0.5 * (M + M.T))
        w    = np.clip(w, 1e-12, None)
        if normalize:
            w = w / np.max(w)
        t      = np.linspace(0.0, 2.0 * np.pi, PC["n_ellipse_pts"])
        circle = np.stack([np.cos(t), np.sin(t)], axis=0)
        pts    = (V @ ((level / np.sqrt(w))[:, None] * circle)).T + center[None, :]
        return pts

    def _scaled(v, span, frac=0.1):
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.zeros_like(v)
        return frac * span * v / n

    def _swap(x):
        x = np.asarray(x, dtype=float)
        return x[[1, 0]] if x.ndim == 1 else x[:, [1, 0]]

    def _set_limits(ax, pts, pad_frac=PC["pad_frac_limits"]):
        pts = np.asarray(pts, dtype=float)
        mn  = pts.min(axis=0)
        mx  = pts.max(axis=0)
        pad = pad_frac * np.maximum(mx - mn, 1.0)
        ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
        ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])

    # colors
    grad_color          = PC["color_grad"]
    natF_color          = PC["color_natF"]
    natH_color          = PC["color_natH"]
    F_curve_color       = PC["color_F_curve"]
    H_curve_color       = PC["color_H_curve"]
    theta_hess_color    = PC["color_theta_hess"]
    theta_path_color    = PC["color_theta_path"]
    phi_curr_path_color = PC["color_phi_curr"]
    phi_tgt_path_color  = PC["color_phi_tgt"]

    theta_path      = np.array([r["theta"]   for r in results])
    phi_rho_path    = np.array([r["phi_rho"] for r in results])
    phi_hat_path    = np.array([r["phi_hat"] for r in results])
    span_draw_theta = max(np.max(theta_path.max(axis=0) - theta_path.min(axis=0)), 1.0)

    theta_pts        = [_swap(theta_path)]
    phi_pts          = [_swap(phi_rho_path), _swap(phi_hat_path)]
    theta_metric_pts = []
    phi_metric_pts   = []

    for r in results:
        theta_i        = r["theta"]
        g_theta        = _scaled(r["grad"],  span_draw_theta)
        f_theta        = _scaled(r["nat_F"], span_draw_theta)
        h_theta        = _scaled(r["nat_H"], span_draw_theta)
        pts_theta_hess = _ellipse_points(r["Hess_theta"] + damp * np.eye(2), theta_i,      level=1.0)
        pts_F          = _ellipse_points(r["F"]          + damp * np.eye(2), r["phi_rho"], level=1.0)
        pts_H          = _ellipse_points(r["H"]          + damp * np.eye(2), r["phi_rho"], level=1.0)

        theta_pts.extend([_swap(theta_i)[None, :],
                           (_swap(theta_i) + _swap(g_theta))[None, :],
                           (_swap(theta_i) + _swap(f_theta))[None, :],
                           (_swap(theta_i) + _swap(h_theta))[None, :]])
        phi_pts.extend([_swap(r["phi_rho"])[None, :], _swap(r["phi_hat"])[None, :]])
        theta_metric_pts.extend([_swap(theta_i)[None, :], _swap(pts_theta_hess)])
        phi_metric_pts.extend([_swap(r["phi_rho"])[None, :], _swap(pts_F), _swap(pts_H)])

    theta_all        = np.vstack(theta_pts)
    phi_all          = np.vstack(phi_pts)
    theta_metric_all = np.vstack(theta_metric_pts)
    phi_metric_all   = np.vstack(phi_metric_pts)

    theta_path_plot   = _swap(theta_path)
    phi_rho_path_plot = _swap(phi_rho_path)
    phi_hat_path_plot = _swap(phi_hat_path)

    # ax_theta
    ax_theta.plot(theta_path_plot[:, 0], theta_path_plot[:, 1],
                  "-", lw=PC["path_lw"], color=theta_path_color, zorder=1)
    ax_theta.plot(theta_path_plot[0, 0],  theta_path_plot[0, 1],
                  "o", ms=PC["path_ms"], mfc="white", mec=theta_path_color, mew=1.3, zorder=3)
    ax_theta.plot(theta_path_plot[-1, 0], theta_path_plot[-1, 1],
                  "o", ms=PC["path_ms"], color=theta_path_color, zorder=3)

    for r in results:
        theta_i = _swap(r["theta"])
        g_theta = _swap(_scaled(r["grad"],  span_draw_theta))
        f_theta = _swap(_scaled(r["nat_F"], span_draw_theta))
        h_theta = _swap(_scaled(r["nat_H"], span_draw_theta))
        ax_theta.plot(theta_i[0], theta_i[1],
                      "o", ms=PC["step_ms"], color=theta_path_color, zorder=2)
        for vec, col in [(g_theta, grad_color), (f_theta, natF_color), (h_theta, natH_color)]:
            ax_theta.arrow(theta_i[0], theta_i[1], vec[0], vec[1],
                           width=PC["arrow_width"], head_width=PC["arrow_head_w"],
                           length_includes_head=True, color=col,
                           alpha=PC["arrow_alpha"], zorder=2)

    _set_limits(ax_theta, theta_all)
    ax_theta.set_aspect("equal", adjustable="datalim")
    ax_theta.set_xlabel(PC["theta_xlabel"], labelpad=1)
    ax_theta.set_ylabel(PC["theta_ylabel"], labelpad=1)
    ax_theta.set_title(PC["title_theta"], pad=PC["titlepad"])

    # ax_geom
    ax_geom.plot(phi_rho_path_plot[:, 0], phi_rho_path_plot[:, 1],
                 "-", lw=PC["path_lw"], color=phi_curr_path_color, zorder=1)
    ax_geom.plot(phi_hat_path_plot[:, 0], phi_hat_path_plot[:, 1],
                 "-", lw=PC["path_lw"], color=phi_tgt_path_color,  zorder=1)
    ax_geom.plot(phi_rho_path_plot[0, 0],  phi_rho_path_plot[0, 1],
                 "o", ms=PC["path_ms"], mfc="white", mec=phi_curr_path_color, mew=1.3, zorder=3)
    ax_geom.plot(phi_rho_path_plot[-1, 0], phi_rho_path_plot[-1, 1],
                 "o", ms=PC["path_ms"], color=phi_curr_path_color, zorder=3)
    ax_geom.plot(phi_hat_path_plot[0, 0],  phi_hat_path_plot[0, 1],
                 "o", ms=PC["path_ms"], mfc="white", mec=phi_tgt_path_color, mew=1.3, zorder=3)
    ax_geom.plot(phi_hat_path_plot[-1, 0], phi_hat_path_plot[-1, 1],
                 "o", ms=PC["path_ms"], color=phi_tgt_path_color, zorder=3)

    for r in results:
        phi_rho_i = _swap(r["phi_rho"])
        phi_hat_i = _swap(r["phi_hat"])
        ax_geom.plot([phi_hat_i[0], phi_rho_i[0]], [phi_hat_i[1], phi_rho_i[1]],
                     "--", lw=PC["connector_lw"], color=PC["color_connector"],
                     alpha=PC["connector_alpha"], zorder=1)

    _set_limits(ax_geom, phi_all)
    ax_geom.set_aspect("equal", adjustable="datalim")
    ax_geom.set_xlabel(PC["phi_xlabel"], labelpad=1)
    ax_geom.set_ylabel(PC["phi_ylabel"], labelpad=1)
    ax_geom.set_title(PC["title_phi"], pad=PC["titlepad"])

    label_angles = np.linspace(0.0, 2.0 * np.pi, max(len(results), 2), endpoint=False)

    # ax_theta_metric
    for i, r in enumerate(results):
        theta_i        = _swap(r["theta"])
        pts_theta_hess = _swap(_ellipse_points(
            r["Hess_theta"] + damp * np.eye(2), r["theta"], level=1.0))
        ax_theta_metric.plot(pts_theta_hess[:, 0], pts_theta_hess[:, 1],
                             lw=PC["ellipse_lw"], color=theta_hess_color,
                             alpha=PC["ellipse_alpha"], zorder=1)
        ax_theta_metric.plot(theta_i[0], theta_i[1],
                             "o", ms=PC["metric_dot_ms"], color=theta_hess_color, zorder=2)
        if PC["show_iter_labels"]:
            label_idx = int((label_angles[i] / (2.0 * np.pi)) * (pts_theta_hess.shape[0] - 1))
            label_pt  = pts_theta_hess[label_idx]
            label_dir = label_pt - theta_i
            label_dir = label_dir / (np.linalg.norm(label_dir) + 1e-12)
            label_pt  = label_pt + (PC["label_offset_base"] + PC["label_offset_step"] * (i % 4)) * label_dir
            ax_theta_metric.text(label_pt[0], label_pt[1], rf"$i={i}$",
                                 fontsize=PC["label_idx_fontsize"],
                                 color="black", ha="center", va="center", clip_on=False)

    _set_limits(ax_theta_metric, theta_metric_all, pad_frac=PC["pad_frac_metric"])
    ax_theta_metric.set_aspect("equal", adjustable="datalim")
    ax_theta_metric.set_xlabel(PC["theta_xlabel"], labelpad=1)
    ax_theta_metric.set_ylabel(PC["theta_ylabel"], labelpad=1)
    ax_theta_metric.set_title(PC["title_theta_g"], pad=PC["titlepad"])

    # ax_phi_metric
    for i, r in enumerate(results):
        phi_rho_i = _swap(r["phi_rho"])
        pts_F     = _swap(_ellipse_points(r["F"] + damp * np.eye(2), r["phi_rho"], level=1.0))
        pts_H     = _swap(_ellipse_points(r["H"] + damp * np.eye(2), r["phi_rho"], level=1.0))
        ax_phi_metric.plot(pts_F[:, 0], pts_F[:, 1],
                           lw=PC["ellipse_lw"], color=F_curve_color,
                           alpha=PC["ellipse_alpha"], zorder=1)
        ax_phi_metric.plot(pts_H[:, 0], pts_H[:, 1],
                           lw=PC["ellipse_lw"], ls="--", color=H_curve_color,
                           alpha=PC["ellipse_alpha"], zorder=1)
        ax_phi_metric.plot(phi_rho_i[0], phi_rho_i[1],
                           "o", ms=PC["metric_dot_ms"], color=F_curve_color, zorder=2)
        if PC["show_iter_labels"]:
            label_idx_F = int((label_angles[i] / (2.0 * np.pi)) * (pts_F.shape[0] - 1))
            label_F     = pts_F[label_idx_F]
            dir_F       = label_F - phi_rho_i
            dir_F       = dir_F / (np.linalg.norm(dir_F) + 1e-12)
            label_F     = label_F + (PC["label_offset_base"] + PC["label_offset_step"] * (i % 4)) * dir_F
            ax_phi_metric.text(label_F[0], label_F[1], rf"$i={i}$",
                               fontsize=PC["label_idx_fontsize"],
                               color="black", ha="center", va="center", clip_on=False)

    _set_limits(ax_phi_metric, phi_metric_all, pad_frac=PC["pad_frac_metric"])
    ax_phi_metric.set_aspect("equal", adjustable="datalim")
    ax_phi_metric.set_xlabel(PC["phi_xlabel"], labelpad=1)
    ax_phi_metric.set_ylabel(PC["phi_ylabel"], labelpad=1)
    ax_phi_metric.set_title(PC["title_phi_g"], pad=PC["titlepad"])

    for ax in [ax_theta, ax_geom, ax_theta_metric, ax_phi_metric]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both",
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
        ax.margins(0.0)

    # Legend (arrowhead at the end only)
    leg = PC["legend"]
    lw  = PC["legend_lw"]
    ms  = PC["legend_marker_ms"]
    handles = [
        Line2D([0], [0], lw=lw, color=theta_path_color,    label=leg["theta_path"]),
        Line2D([0], [0], lw=lw, color=phi_curr_path_color, label=leg["phi_rho"]),
        Line2D([0], [0], lw=lw, color=phi_tgt_path_color,  label=leg["phi_hat"]),
        Line2D([0], [0], lw=0.8, ls="--", color=PC["color_connector"], label=leg["connector"]),
        Line2D([0, 1], [0, 0], lw=lw, color=grad_color,
               marker=">", markersize=ms, markevery=[-1], label=leg["grad"]),
        Line2D([0, 1], [0, 0], lw=lw, color=natF_color,
               marker=">", markersize=ms, markevery=[-1], label=leg["natF"]),
        Line2D([0, 1], [0, 0], lw=lw, color=natH_color,
               marker=">", markersize=ms, markevery=[-1], label=leg["natH"]),
        Line2D([0], [0], lw=PC["ellipse_lw"], color=theta_hess_color, label=leg["theta_hess"]),
        Line2D([0], [0], lw=PC["ellipse_lw"], color=F_curve_color,    label=leg["F"]),
        Line2D([0], [0], lw=PC["ellipse_lw"], ls="--", color=H_curve_color, label=leg["H"]),
    ]

    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, PC["legend_y"]),
               ncol=PC["legend_ncol"], frameon=False,
               columnspacing=PC["legend_col_spacing"],
               handlelength=PC["legend_handle_len"],
               handletextpad=PC["legend_handle_pad"],
               borderaxespad=0.0)
    fig.subplots_adjust(**PC["subadj"])
    return fig


@hydra.main(version_base=None, config_path=".", config_name="config")
def experiment(cfg : DictConfig) -> None:
    world, features, logs, irl_cfg = C.build(cfg)
    geom_logs = logs["dual_geometry"]
    geo       = C.DualGeometry(world, features, irl_cfg["discount"], irl_cfg["beta"], geom_logs[0]["rho_E"])

    results = compute_results(geo, geom_logs)
    if cfg.angle_table:
        print_angle_table(geo, results)

    PC  = OmegaConf.to_container(cfg.plotting, resolve=True)
    fig = plot_dual_geometries(results, PC)
    fig.savefig(os.path.join(HydraConfig.get().runtime.output_dir, "dual_geometry.pdf"),
                dpi=PC["dpi"], bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    experiment()
