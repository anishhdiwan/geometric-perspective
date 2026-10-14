import numpy as np
from tqdm import tqdm
import copy
import optimizer as O


def no_op(lr0=0.2):
    def _lr(k):
        return lr0
    return _lr

def logsumexp(x, axis=0):
    x_max = np.max(x, axis=axis, keepdims=True)
    return (x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))).squeeze()


def expected_svf_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps=1e-5):
    """Expected state visitation frequency of the policy `p_action`."""
    n_states, _, n_actions = p_transition.shape
    final = list(terminal.nonzero()[0]) + list(absorbing.nonzero()[0])
    p_transition = np.copy(p_transition)
    p_transition[final, :, :] = 0.0

    # set-up transition matrices for each action
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # actual forward-computation of state expectations
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)

        delta, d = np.max(np.abs(d_ - d)), d_

    rho = d / d.sum()
    return rho.flatten()



def soft_policy_evaluation(p_transition, pi, reward, discount, terminal=None, absorbing=None, alpha=1.0, eps=1e-5):
    """Soft (entropy-regularised) policy evaluation with temperature `alpha`."""
    n_states, _, n_actions = p_transition.shape
    pi = pi.reshape(n_states, n_actions)
    reward = reward.reshape(n_states, n_actions)

    if absorbing is None:
        absorbing = np.zeros(n_states, dtype=bool)
    if terminal is None:
        terminal = np.zeros(n_states, dtype=bool)
    terminal_set = terminal | absorbing

    # transitions probabilities
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
    r = [reward[:, a] for a in range(n_actions)]

    # entropy term: -alpha log pi(a|s)
    log_pi = np.log(np.clip(pi, 1e-10, 1.0))
    ent = -alpha * log_pi  # (n_states,n_actions)

    v = np.zeros(n_states)
    delta = np.inf

    while delta > eps:
        v_old = v.copy()

        # clamp bootstrap on terminals
        v_boot = v_old.copy()
        v_boot[terminal_set] = 0.0

        # Q(s,a) = E_{s'}[ R + gamma V(s') ]
        q = np.array([
            r[a] + discount * (p[a] @ v_boot)
            for a in range(n_actions)
        ]).T


        # V(s) = E_{a~pi}[ Q(s,a) - alpha log pi(a|s) ]
        v = (pi * (q + ent)).sum(axis=1)

        # hard clamp terminals
        v[terminal_set] = 0.0

        delta = np.max(np.abs(v - v_old))

    return v



def soft_value_iteration(p_transition, reward, discount, alpha=1.0, pi_old=None, eta=None, eps=1e-5):
    """Soft value iteration with temperature `alpha`; with `pi_old`, uses the TRIRL trust-region reward."""
    n_states, _, n_actions = p_transition.shape
    reward = reward.reshape(n_states, n_actions)

    if pi_old is not None:
        reward = (1/(1 + eta))*(reward + eta * np.log(pi_old))

    # set up transition probability matrices
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
    reward = [reward[:, a] for a in range(n_actions)]

    # compute state log partition V and state-action log partition Q
    v = -1e200 * np.ones(n_states)  # np.dot doesn't behave with -np.inf

    delta = np.inf
    while delta > eps:
        v_old = v.copy()

        q = np.array([
            p[a] @ (reward[a] + discount * v_old)
            for a in range(n_actions)
        ]).T

        v = alpha * logsumexp(q / alpha, axis=1)

        delta = np.max(np.abs(v_old - v))

    pi = np.exp((q - v[:, None]) / alpha)

    logging_kl = None
    if pi_old is not None:
        # KL(pi || pi_old), for logging only (the KL penalty is already in the reward)
        logging_kl = np.mean((pi.clip(1e-10, 1.0) * (np.log(pi.clip(1e-10, 1.0)) - np.log(pi_old.clip(1e-10, 1.0)))).sum(axis=-1))

    return v, pi, logging_kl


def reinforcement_learning(p_transition, p_initial, terminal, absorbing, reward, discount, alpha=1.0, pi_old=None, eta=None,
                           eps_vi=1e-5, eps_rho=1e-5):

    # Soft Value Iteration
    v, p_action, logging_kl = soft_value_iteration(p_transition, reward, discount, alpha=alpha, pi_old=pi_old, eta=eta, eps=eps_vi)
    rho = expected_svf_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps_rho)

    return rho, p_action.clip(1e-10, 1.0), v, logging_kl

def functional_reward_update(theta, rho_expert, rho_agent, features, epsilon=0.1, beta=0.5):
    reward_tilde = beta * (
        np.log(np.clip(rho_expert, 1e-6, 1.0)) - np.log(np.clip(rho_agent, 1e-6, 1.0))
    )
    theta_tilde, _, _, _ = np.linalg.lstsq(features, reward_tilde, rcond=None)
    theta_new = (1 - epsilon) * theta + epsilon * theta_tilde
    return theta_new, theta_tilde


def trirl_discrete(world, features, expert_policy, init, irl_cfg, eps_vi=1e-5, eps_rho=1e-5):
    """Trust Region IRL (TRIRL) with functional reward updates."""

    def generator():
        max_iter = irl_cfg["max_iter"]
        count = 0
        while count < max_iter:
            yield count
            count += 1

    p_transition = world.p_transition
    absorbing = world.absorbing.astype(bool)
    terminal = world.terminal.astype(bool)
    discount = irl_cfg["discount"]
    n_states, _, n_actions = p_transition.shape

    # compute static properties
    p_initial = np.zeros((n_states))
    p_initial[world.initial] = 1.0 / len(world.initial)
    e_rho = expected_svf_from_policy(p_transition, p_initial, terminal, expert_policy, absorbing, eps_rho)

    # parameters
    theta = init((features.shape[1],))
    delta = np.inf
    eta = lambda x: x

    # logging
    kl_pi_piold = []
    kl_rho_pi_rho_es = []
    value_fns = []
    log_rhoE_minus_r_betas = []
    reward_vecs = []
    rho_pis = []
    H_pis = []
    thetas = [theta.copy()]
    geom_logs = []

    # initialize iterants
    eta = no_op(lr0=irl_cfg["init_eta"])
    reward = features @ theta
    p_action = np.full((n_states, n_actions), 1/n_actions)
    rho = expected_svf_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps_rho)
    rho_agent = rho.copy()

    geom_logs.append({
        "theta": theta.copy(),
        "theta_tilde": theta.copy(),
        "theta_projected": theta.copy(),
        "rho": rho_agent.copy(),
        "p_action": p_action.copy(),
        "rho_E": e_rho.copy(),
    })


    with tqdm(generator()) as pbar:
        for count in pbar:
            pbar.set_description(f"Delta: {delta}")

            # reset iterants
            theta_old = theta.copy()
            rho_agent_old = rho_agent.copy()
            pi_old = p_action.copy()

            # intermediate, large-stepped reward
            theta_intermediate, _ = functional_reward_update(theta_old, e_rho, rho_agent_old, features, epsilon=irl_cfg["epsilon"], beta=irl_cfg["beta"])
            reward_vec = features @ theta_intermediate  # r(s) or r(s,a) vector
            reward = reward_vec
            reward = np.repeat(reward[:, None], n_actions, axis=1)
            thetas.append(theta_intermediate.copy())

            # tr policy update
            rho, p_action, value_fn, kl = reinforcement_learning(p_transition, p_initial, terminal, absorbing, reward, discount, pi_old=pi_old, eta=eta(count),
                                                         eps_vi=eps_vi, eps_rho=eps_rho)

            # reward projection
            theta, theta_tilde = functional_reward_update(theta_old, e_rho, rho_agent_old, features, epsilon=irl_cfg["epsilon"]/(1 + eta(count)), beta=irl_cfg["beta"])
            rho_agent = rho.copy()
            geom_logs.append({
                "theta": theta_intermediate.copy(),
                "theta_projected": theta.copy(),
                "theta_tilde": theta_tilde.copy(),
                "rho": rho.copy(),
                "rho_old": rho_agent_old.copy(),
                "p_action": p_action.copy(),
                "pi_old": pi_old.copy(),
                "rho_E": e_rho.copy(),
                "eta": eta(count),
                "kl": kl,
            })

            # logging
            kl_rho_pi_rho_e = np.sum(rho_agent * (np.log(rho_agent.clip(1e-10, 1.0)) - np.log(e_rho.clip(1e-10, 1.0))))
            kl_rho_pi_rho_es.append(kl_rho_pi_rho_e)
            kl_pi_piold.append(kl)
            H_pi = -np.sum(rho_agent * np.sum(p_action * np.log(p_action.clip(1e-10, 1.0)), axis=-1))
            H_pis.append(H_pi)

            # Eval Dual
            if irl_cfg["eval_dual"]:
                reward_projected = features @ theta
                reward_projected = np.repeat(reward_projected[:, None], n_actions, axis=1)
                v_mce = soft_policy_evaluation(p_transition, copy.deepcopy(p_action), reward_projected, discount, terminal=None, absorbing=None, alpha=1/irl_cfg["beta"])
                value_fns.append(copy.deepcopy(v_mce))
                log_rhoE = np.log(e_rho.clip(eps_rho, 1.0))
                log_rhoE_minus_r_betas.append(log_rhoE - ((features @ theta) / irl_cfg["beta"]))
                reward_vecs.append((features @ theta).copy())
                rho_pis.append(rho.copy())

            # stopping criterion
            delta = np.max(np.abs(theta_old - theta))


    logs = {}
    logs["KL(pi || pi_old)"] = kl_pi_piold
    logs["KL(ρ_π || ρ_E)"] = kl_rho_pi_rho_es
    logs["value_fns"] = value_fns
    logs["log(ρ_E) - r/beta"] = log_rhoE_minus_r_betas
    logs["reward_vecs"] = reward_vecs
    logs["rho_pi"] = rho_pis
    logs["rho_E"] = e_rho
    logs["H_pi"] = H_pis
    logs["thetas"] = thetas
    logs["dual_geometry"] = geom_logs

    return features @ theta, p_action, logs


def mceirl_vanilla_gradient(world, features, expert_policy, init, irl_cfg, learning_rate=0.01,
               eps_vi=1e-5, eps_rho=1e-5):
    """MCE-IRL with vanilla gradient dual updates."""

    def generator():
        max_iter = irl_cfg["max_iter"]
        count = 0
        while count < max_iter:
            yield count
            count += 1

    p_transition = world.p_transition
    absorbing = world.absorbing.astype(bool)
    terminal = world.terminal.astype(bool)
    discount = irl_cfg["discount"]
    n_states, _, n_actions = p_transition.shape

    # compute static properties
    p_initial = np.zeros((n_states))
    p_initial[world.initial] = 1.0 / len(world.initial)
    e_rho = expected_svf_from_policy(p_transition, p_initial, terminal, expert_policy, absorbing, eps_rho)

    # parameters
    theta = init((features.shape[1],))
    delta = np.inf
    optim = O.Sga(lr=learning_rate)
    optim.reset(theta)

    # logging
    kl_rho_pi_rho_es = []
    value_fns = []
    log_rhoE_minus_r_betas = []
    reward_vecs = []
    rho_pis = []
    H_pis = []
    thetas = [theta.copy()]
    geom_logs = []

    # initialize iterants
    reward = features @ theta
    p_action = np.full((n_states, n_actions), 1/n_actions)
    rho = expected_svf_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps_rho)
    rho_agent = rho.copy()

    with tqdm(generator()) as pbar:
        for count in pbar:
            pbar.set_description(f"Delta: {delta}")

            # reset iterants
            theta_old = theta.copy()
            rho_old = rho.copy()

            # compute reward
            reward_vec = features @ theta
            reward = np.repeat(reward_vec[:, None], n_actions, axis=1)

            # compute the policy and occupancy measure
            rho, p_action, value_fn, kl = reinforcement_learning(p_transition, p_initial, terminal, absorbing, reward, discount, alpha=1/irl_cfg["beta"],
                                                         eps_vi=eps_vi, eps_rho=eps_rho)
            rho_closed_form = rho.copy()

            # compute vanilla gradient and update dual
            reward_old = features @ theta_old
            log_rhoE = np.log(e_rho.clip(1e-8, 1.0))
            rho_pi_hat = np.exp(log_rhoE - (reward_old / irl_cfg["beta"]))
            rho_pi_hat /= rho_pi_hat.sum()
            rho_closed_form /= rho_closed_form.sum()
            grad = features.T @ (rho_closed_form - rho_pi_hat)

            # log (theta, rho) for the dual geometry
            geom_logs.append({
                "theta": theta_old.copy(),
                "rho": rho_closed_form.copy(),
                "rho_E": e_rho.copy(),
            })


            optim.step(-irl_cfg["beta"]*grad)
            rho_agent = rho_closed_form

            # logging
            kl_rho_pi_rho_e = np.sum(rho_agent * (np.log(rho_agent.clip(1e-10, 1.0)) - np.log(e_rho.clip(1e-10, 1.0))))
            kl_rho_pi_rho_es.append(kl_rho_pi_rho_e)
            H_pi = -np.sum(rho_agent * np.sum(p_action * np.log(p_action.clip(1e-10, 1.0)), axis=-1))
            H_pis.append(H_pi)
            thetas.append(theta.copy())

            # Eval Dual
            if irl_cfg["eval_dual"]:
                reward_eval = features @ theta
                reward_eval = np.repeat(reward_eval[:, None], n_actions, axis=1)
                v_mce = soft_policy_evaluation(p_transition, copy.deepcopy(p_action), reward_eval, discount, terminal=None, absorbing=None, alpha=1/irl_cfg["beta"])
                value_fns.append(copy.deepcopy(v_mce))
                log_rhoE = np.log(e_rho.clip(eps_rho, 1.0))
                log_rhoE_minus_r_betas.append(log_rhoE - ((features @ theta) / irl_cfg["beta"]))
                reward_vecs.append((features @ theta).copy())
                rho_pis.append(rho.copy())

            # stopping criterion
            delta = np.max(np.abs(theta_old - theta))


    logs = {}
    logs["KL(ρ_π || ρ_E)"] = kl_rho_pi_rho_es
    logs["value_fns"] = value_fns
    logs["log(ρ_E) - r/beta"] = log_rhoE_minus_r_betas
    logs["reward_vecs"] = reward_vecs
    logs["rho_pi"] = rho_pis
    logs["rho_E"] = e_rho
    logs["H_pi"] = H_pis
    logs["thetas"] = thetas
    logs["dual_geometry"] = geom_logs

    return features @ theta, p_action, logs


def mceirl_natural_gradient(world, features, expert_policy, init, irl_cfg, learning_rate=0.01,
               eps_vi=1e-5, eps_rho=1e-5):
    """MCE-IRL with natural gradient dual updates (FIM at rho)."""

    def generator():
        max_iter = irl_cfg["max_iter"]
        count = 0
        while count < max_iter:
            yield count
            count += 1

    p_transition = world.p_transition
    absorbing = world.absorbing.astype(bool)
    terminal = world.terminal.astype(bool)
    discount = irl_cfg["discount"]
    n_states, _, n_actions = p_transition.shape

    # compute static properties
    p_initial = np.zeros((n_states))
    p_initial[world.initial] = 1.0 / len(world.initial)
    e_rho = expected_svf_from_policy(p_transition, p_initial, terminal, expert_policy, absorbing, eps_rho)

    # parameters
    theta = init((features.shape[1],))
    delta = np.inf
    optim = O.Sga(lr=learning_rate)
    optim.reset(theta)

    # logging
    kl_rho_pi_rho_es = []
    value_fns = []
    log_rhoE_minus_r_betas = []
    reward_vecs = []
    rho_pis = []
    H_pis = []
    thetas = [theta.copy()]
    geom_logs = []

    # initialize iterants
    reward = features @ theta
    p_action = np.full((n_states, n_actions), 1/n_actions)
    rho = expected_svf_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps_rho)
    rho_agent = rho.copy()

    with tqdm(generator()) as pbar:
        for count in pbar:
            pbar.set_description(f"Delta: {delta}")

            # reset iterants
            theta_old = theta.copy()
            rho_old = rho.copy()

            # compute reward
            reward_vec = features @ theta
            reward = np.repeat(reward_vec[:, None], n_actions, axis=1)

            # compute the policy and occupancy measure
            rho, p_action, value_fn, kl = reinforcement_learning(p_transition, p_initial, terminal, absorbing, reward, discount, alpha=1/irl_cfg["beta"],
                                                         eps_vi=eps_vi, eps_rho=eps_rho)
            rho_closed_form = rho.copy()

            # compute natural gradient
            reward_old = features @ theta_old
            log_rhoE = np.log(e_rho.clip(1e-8, 1.0))
            rho_pi_hat = np.exp(log_rhoE - (reward_old / irl_cfg["beta"]))
            rho_pi_hat /= rho_pi_hat.sum()
            rho_closed_form /= rho_closed_form.sum()
            grad = features.T @ (rho_closed_form - rho_pi_hat)

            # log (theta, rho) for the dual geometry
            geom_logs.append({
                "theta": theta_old.copy(),
                "rho": rho_closed_form.copy(),
                "rho_E": e_rho.copy(),
            })

            # FIM and update dual
            mu = features.T @ rho_closed_form
            fisher = (features * rho_closed_form[:, None]).T @ features - np.outer(mu, mu)
            fisher += irl_cfg.get("fisher_damping", 1e-8) * np.eye(fisher.shape[0])
            nat_grad = np.linalg.solve(fisher, grad)
            optim.step(-irl_cfg["beta"]*nat_grad)
            rho_agent = rho_closed_form

            # logging
            kl_rho_pi_rho_e = np.sum(rho_agent * (np.log(rho_agent.clip(1e-10, 1.0)) - np.log(e_rho.clip(1e-10, 1.0))))
            kl_rho_pi_rho_es.append(kl_rho_pi_rho_e)
            H_pi = -np.sum(rho_agent * np.sum(p_action * np.log(p_action.clip(1e-10, 1.0)), axis=-1))
            H_pis.append(H_pi)
            thetas.append(theta.copy())

            # Eval Dual
            if irl_cfg["eval_dual"]:
                reward_eval = features @ theta_old
                reward_eval = np.repeat(reward_eval[:, None], n_actions, axis=1)
                v_mce = soft_policy_evaluation(p_transition, copy.deepcopy(p_action), reward_eval, discount, terminal=None, absorbing=None)
                value_fns.append(copy.deepcopy(v_mce))
                log_rhoE = np.log(e_rho.clip(eps_rho, 1.0))
                log_rhoE_minus_r_betas.append(log_rhoE - ((features @ theta_old) / irl_cfg["beta"]))
                reward_vecs.append((features @ theta_old).copy())
                rho_pis.append(rho.copy())

            # stopping criterion
            delta = np.max(np.abs(theta_old - theta))


    logs = {}
    logs["KL(ρ_π || ρ_E)"] = kl_rho_pi_rho_es
    logs["value_fns"] = value_fns
    logs["log(ρ_E) - r/beta"] = log_rhoE_minus_r_betas
    logs["reward_vecs"] = reward_vecs
    logs["rho_pi"] = rho_pis
    logs["rho_E"] = e_rho
    logs["H_pi"] = H_pis
    logs["thetas"] = thetas
    logs["dual_geometry"] = geom_logs

    return features @ theta, p_action, logs


def mceirl_path_natural_gradient(world, features, expert_policy, init, irl_cfg, learning_rate=0.01,
               eps_vi=1e-5, eps_rho=1e-5):
    """MCE-IRL with path-integral natural gradient dual updates."""

    def generator():
        max_iter = irl_cfg["max_iter"]
        count = 0
        while count < max_iter:
            yield count
            count += 1

    p_transition = world.p_transition
    absorbing = world.absorbing.astype(bool)
    terminal = world.terminal.astype(bool)
    discount = irl_cfg["discount"]
    n_states, _, n_actions = p_transition.shape

    # compute static properties
    p_initial = np.zeros((n_states))
    p_initial[world.initial] = 1.0 / len(world.initial)
    e_rho = expected_svf_from_policy(p_transition, p_initial, terminal, expert_policy, absorbing, eps_rho)

    # parameters
    theta = init((features.shape[1],))
    delta = np.inf
    optim = O.Sga(lr=learning_rate)
    optim.reset(theta)

    # logging
    kl_rho_pi_rho_es = []
    value_fns = []
    log_rhoE_minus_r_betas = []
    reward_vecs = []
    rho_pis = []
    H_pis = []
    thetas = [theta.copy()]
    geom_logs = []

    # initialize iterants
    reward = features @ theta
    p_action = np.full((n_states, n_actions), 1/n_actions)
    rho = expected_svf_from_policy(p_transition, p_initial, terminal, p_action, absorbing, eps_rho)
    rho_agent = rho.copy()

    with tqdm(generator()) as pbar:
        for count in pbar:
            pbar.set_description(f"Delta: {delta}")

            # reset iterants
            theta_old = theta.copy()
            rho_old = rho.copy()

            # compute reward
            reward_vec = features @ theta
            reward = np.repeat(reward_vec[:, None], n_actions, axis=1)

            # compute the policy and occupancy measure
            rho, p_action, value_fn, kl = reinforcement_learning(p_transition, p_initial, terminal, absorbing, reward, discount,
                                                         eps_vi=eps_vi, eps_rho=eps_rho)
            rho_closed_form = rho.copy()

            # compute gradient
            reward_old = features @ theta_old
            log_rhoE = np.log(e_rho.clip(1e-8, 1.0))
            rho_pi_hat = np.exp(log_rhoE - (reward_old / irl_cfg["beta"]))
            rho_pi_hat /= rho_pi_hat.sum()
            rho_closed_form /= rho_closed_form.sum()
            grad = features.T @ (rho_closed_form - rho_pi_hat)

            # log (theta, rho) for the dual geometry
            geom_logs.append({
                "theta": theta_old.copy(),
                "rho": rho_closed_form.copy(),
                "rho_E": e_rho.copy(),
            })

            # Integral FIM and update dual
            n_tau = irl_cfg.get("fim_path_n", 16)
            fisher = np.zeros((features.shape[1], features.shape[1]))
            for tau in np.linspace(0.0, 1.0, n_tau):
                rho_tau = tau * rho_closed_form + (1.0 - tau) * rho_pi_hat
                rho_tau /= rho_tau.sum()
                mu = features.T @ rho_tau
                fisher += (features * rho_tau[:, None]).T @ features - np.outer(mu, mu)
            fisher /= n_tau
            fisher += irl_cfg.get("fisher_damping", 1e-8) * np.eye(fisher.shape[0])
            path_nat_grad = np.linalg.solve(fisher, grad)
            optim.step(-irl_cfg["beta"]*path_nat_grad)
            rho_agent = rho_closed_form


            # logging
            kl_rho_pi_rho_e = np.sum(rho_agent * (np.log(rho_agent.clip(1e-10, 1.0)) - np.log(e_rho.clip(1e-10, 1.0))))
            kl_rho_pi_rho_es.append(kl_rho_pi_rho_e)
            H_pi = -np.sum(rho_agent * np.sum(p_action * np.log(p_action.clip(1e-10, 1.0)), axis=-1))
            H_pis.append(H_pi)
            thetas.append(theta.copy())

            # Eval Dual
            if irl_cfg["eval_dual"]:
                reward_eval = features @ theta
                reward_eval = np.repeat(reward_eval[:, None], n_actions, axis=1)
                v_mce = soft_policy_evaluation(p_transition, copy.deepcopy(p_action), reward_eval, discount, terminal=None, absorbing=None, alpha=1/irl_cfg["beta"])
                value_fns.append(copy.deepcopy(v_mce))
                log_rhoE = np.log(e_rho.clip(eps_rho, 1.0))
                log_rhoE_minus_r_betas.append(log_rhoE - ((features @ theta) / irl_cfg["beta"]))
                reward_vecs.append((features @ theta).copy())
                rho_pis.append(rho.copy())

            # stopping criterion
            delta = np.max(np.abs(theta_old - theta))


    logs = {}
    logs["KL(ρ_π || ρ_E)"] = kl_rho_pi_rho_es
    logs["value_fns"] = value_fns
    logs["log(ρ_E) - r/beta"] = log_rhoE_minus_r_betas
    logs["reward_vecs"] = reward_vecs
    logs["rho_pi"] = rho_pis
    logs["rho_E"] = e_rho
    logs["H_pi"] = H_pis
    logs["thetas"] = thetas
    logs["dual_geometry"] = geom_logs

    return features @ theta, p_action, logs