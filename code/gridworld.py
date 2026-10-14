import numpy as np
from itertools import product
from tqdm import tqdm


class GridWorld:
    def __init__(self, size, absorbing=None, terminal=None, initial=None):
        self.size = size

        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.n_states = size**2
        self.n_actions = len(self.actions)
        self.absorbing = absorbing
        self.terminal = terminal
        self.initial = initial
        self.p_transition = self._add_absorbing_transitions(self._transition_prob_table(), self.absorbing)

    def state_index_to_point(self, state):
        return state % self.size, state // self.size

    def state_point_to_index(self, state):
        return state[1] * self.size + state[0]

    def state_point_to_index_clipped(self, state):
        s = (max(0, min(self.size - 1, state[0])), max(0, min(self.size - 1, state[1])))
        return self.state_point_to_index(s)

    def state_index_transition(self, s, a):
        s = self.state_index_to_point(s)
        s = s[0] + self.actions[a][0], s[1] + self.actions[a][1]
        return self.state_point_to_index_clipped(s)

    def _add_absorbing_transitions(self, transition_prob_table, absorbing_states):
        if absorbing_states is None:
            return transition_prob_table
        else:
            absorbing_idxs = absorbing_states.nonzero()
            for absorbing_idx in absorbing_idxs:
                transition_prob_table[absorbing_idx, :, :] = 0.0
                transition_prob_table[absorbing_idx, absorbing_idx, :] = 1.0
            return transition_prob_table

    def _transition_prob_table(self):
        """Builds the transition probability table."""
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

        s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)
        for s_from, s_to, a in product(s1, s2, a):
            table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a)

        return table

    def _transition_prob(self, s_from, s_to, a):
        """Transition probability for a single transition."""
        fx, fy = self.state_index_to_point(s_from)
        tx, ty = self.state_index_to_point(s_to)
        ax, ay = self.actions[a]

        # deterministic transition defined by action
        if fx + ax == tx and fy + ay == ty:
            return 1.0

        # we can stay at the same state if we would move over an edge
        if fx == tx and fy == ty:
            if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
                return 1.0

        # otherwise this transition is impossible
        return 0.0

    def __repr__(self):
        return "GridWorld(size={})".format(self.size)


class IcyGridWorld(GridWorld):
    """Grid world with slippery transitions (Frozen Lake without holes)."""

    def __init__(self, size, absorbing=None, p_slip=0.0, terminal=None, initial=None):
        self.p_slip = p_slip

        super().__init__(size, absorbing=absorbing, terminal=terminal, initial=initial)

    def _transition_prob(self, s_from, s_to, a):
        """Transition probability for a single transition."""
        fx, fy = self.state_index_to_point(s_from)
        tx, ty = self.state_index_to_point(s_to)
        ax, ay = self.actions[a]

        # intended transition defined by action
        if fx + ax == tx and fy + ay == ty:
            return 1.0 - self.p_slip + self.p_slip / self.n_actions

        # we can slip to all neighboring states
        if abs(fx - tx) + abs(fy - ty) == 1:
            return self.p_slip / self.n_actions

        # we can stay at the same state if we would move over an edge
        if fx == tx and fy == ty:
            # intended move over an edge
            if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
                # double slip chance at corners
                if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
                    return 1.0 - self.p_slip + 2.0 * self.p_slip / self.n_actions

                # regular probability at normal edges
                return 1.0 - self.p_slip + self.p_slip / self.n_actions

            # double slip chance at corners
            if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
                return 2.0 * self.p_slip / self.n_actions

            # single slip chance at edge
            if not 0 < fx < self.size - 1 or not 0 < fy < self.size - 1:
                return self.p_slip / self.n_actions

            # otherwise we cannot stay at the same state
            return 0.0

        # otherwise this transition is impossible
        return 0.0

    def __repr__(self):
        return "IcyGridWorld(size={}, p_slip={})".format(self.size, self.p_slip)


def console_curve(values, title="", width=64, height=10):
    """Text loss curve: column means drawn as eighth-block bars (log scale if the range exceeds 100x)."""
    values = np.asarray(values, dtype=float)
    width  = min(width, len(values))
    cols   = np.array([c.mean() for c in np.array_split(values, width)])

    log_scale = bool(np.all(cols > 0) and cols.max() / cols.min() > 100)
    y         = np.log10(cols) if log_scale else cols
    lo, hi    = y.min(), y.max()
    levels    = np.round((y - lo) / (hi - lo + 1e-12) * (height * 8 - 1)).astype(int) + 1

    blocks = " ▁▂▃▄▅▆▇█"
    fmt    = lambda v: f"{10**v if log_scale else v:.3g}"
    label_w = max(len(fmt(hi)), len(fmt(lo)))

    lines = [f"  {title}" + ("  (log scale)" if log_scale else "")]
    for row in range(height):
        base  = (height - 1 - row) * 8
        bar   = "".join(blocks[min(max(l - base, 0), 8)] for l in levels)
        label = fmt(hi) if row == 0 else fmt(lo) if row == height - 1 else ""
        lines.append(f"  {label:>{label_w}} │{bar}")
    lines.append(f"  {'':>{label_w}} └{'─' * width}")
    steps = f"1{str(len(values)).rjust(width - 1)}"
    lines.append(f"  {'':>{label_w}}  {steps}  step")
    return "\n".join(lines)


def rbf_features(world):
    """2D RBF features centred at the goal and the top-left corner."""
    features = np.zeros((world.n_states, 2))

    goal_idx = world.absorbing.nonzero()[0][0]
    goal_x, goal_y = world.state_index_to_point(goal_idx)

    anchor_x, anchor_y = 0, world.size - 1
    sigma_goal = 4.0
    sigma_anchor = 4.0

    for s in range(world.n_states):
        x, y = world.state_index_to_point(s)

        d_goal_sq = (x - goal_x)**2 + (y - goal_y)**2
        d_anchor_sq = (x - anchor_x)**2 + (y - anchor_y)**2

        features[s, 0] = np.exp(-d_goal_sq / (2.0 * sigma_goal**2))
        features[s, 1] = np.exp(-d_anchor_sq / (2.0 * sigma_anchor**2))

    return features


def boltzmann_features(world, trajectories, base_features=None, plot=True, seed=0):
    """Learn Boltzmann features from expert states with the DSM encoder loss.
    Returns features of shape (n_states, latent_dim)."""
    import os
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax
    import matplotlib.pyplot as plt
    from flax.linen.initializers import constant, orthogonal

    seed = seed
    hidden1 = 8
    hidden2 = 4
    latent_dim = 2
    energy_hidden = 16

    n_steps = 5000
    batch_size = 16
    learning_rate = 1e-4
    max_grad_norm = 0.1

    dsm_sigma = 0.5
    dsm_alpha = 1.0
    feature_std_target = 0.6
    feature_var_weight = 1.0

    logging_frequency = 100
    plot_dir = "."
    hist_file = "boltzmann_feat_hist.pdf"
    loss_file = "boltzmann_feat_loss.pdf"

    if base_features is None:
        base_features = rbf_features(world)

    base_features = np.asarray(base_features, dtype=np.float32)

    def _trajectory_states(trajectories):
        states = []
        for trajectory in trajectories:
            if hasattr(trajectory, "states"):
                traj_states = trajectory.states() if callable(trajectory.states) else trajectory.states
                states.extend([int(s) for s in traj_states])
            elif hasattr(trajectory, "transitions"):
                traj_transitions = trajectory.transitions() if callable(trajectory.transitions) else trajectory.transitions
                for transition in traj_transitions:
                    if hasattr(transition, "state"):
                        states.append(int(transition.state))
                    else:
                        states.append(int(transition[0]))
            else:
                for transition in trajectory:
                    if hasattr(transition, "state"):
                        states.append(int(transition.state))
                    else:
                        states.append(int(transition[0]))
        return np.asarray(states, dtype=int)

    expert_states = _trajectory_states(trajectories)
    expert_states = expert_states[(0 <= expert_states) & (expert_states < world.n_states)]

    if expert_states.size == 0:
        raise ValueError("No expert states found in trajectories.")

    x_mean = base_features.mean(axis=0, keepdims=True)
    x_std = base_features.std(axis=0, keepdims=True) + 1e-8
    x = (base_features - x_mean) / x_std

    expert_x = jnp.asarray(x[expert_states], dtype=jnp.float32)
    all_x = jnp.asarray(x, dtype=jnp.float32)

    class EncoderNetwork(nn.Module):
        def setup(self):
            self.feat_dense1 = nn.Dense(hidden1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
            self.feat_dense2 = nn.Dense(hidden2, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
            self.feat_dense3 = nn.Dense(latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
            self.energy_dense1 = nn.Dense(energy_hidden, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))
            self.energy_dense2 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        def encode_state(self, x):
            z = self.feat_dense1(x)
            z = nn.relu(z)
            z = self.feat_dense2(z)
            z = nn.relu(z)
            z = self.feat_dense3(z)
            return z

        def energy_from_z(self, z):
            e = self.energy_dense1(z)
            e = nn.gelu(e)
            e = self.energy_dense2(e)
            return jnp.squeeze(e, axis=-1)

        def __call__(self, x):
            z = self.encode_state(x)
            return self.energy_from_z(z)

    encoder = EncoderNetwork()
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = encoder.init(init_key, expert_x[:1])
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
    opt_state = optimizer.init(params)

    def _encode(params, x):
        return encoder.apply(params, x, method=EncoderNetwork.encode_state)

    def _energy_from_z(params, z):
        return encoder.apply(params, z, method=EncoderNetwork.energy_from_z)

    def _var_floor_penalty(z, target_std=0.25, eps=1e-6):
        std = jnp.sqrt(jnp.var(z, axis=0) + eps)
        return jnp.mean(jnp.square(jnp.maximum(0.0, target_std - std))), jnp.mean(std)

    def _loss_fn(params, expert_batch, key):
        expert_z = _encode(params, expert_batch)
        noise = jax.random.normal(key, expert_z.shape)
        noisy_z = expert_z + dsm_sigma * noise

        def energy_sum(z):
            return jnp.sum(_energy_from_z(params, z))

        score = -jax.grad(energy_sum)(noisy_z)
        target = -(noisy_z - expert_z) / (dsm_sigma ** 2)

        dsm_loss = 0.5 * jnp.mean(jnp.sum(jnp.square(score - target), axis=-1))
        var_loss, feature_std = _var_floor_penalty(expert_z, target_std=feature_std_target)
        total_loss = dsm_alpha * dsm_loss + feature_var_weight * var_loss

        metrics = {
            "loss": total_loss,
            "dsm_loss": dsm_loss,
            "var_loss": var_loss,
            "feature_std": feature_std,
        }
        return total_loss, metrics

    @jax.jit
    def _train_step(params, opt_state, expert_x, key):
        key, sample_key, loss_key = jax.random.split(key, 3)
        idx = jax.random.choice(sample_key, expert_x.shape[0], shape=(batch_size,), replace=expert_x.shape[0] < batch_size)
        expert_batch = expert_x[idx]

        (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params, expert_batch, loss_key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics["grad_norm"] = optax.global_norm(grads)
        return params, opt_state, key, metrics

    loss_hist = []
    dsm_loss_hist = []
    var_loss_hist = []
    feature_std_hist = []
    grad_norm_hist = []

    pbar = tqdm(range(1, n_steps + 1), desc="Feature learning (DSM)", leave=False)
    for step in pbar:
        params, opt_state, key, metrics = _train_step(params, opt_state, expert_x, key)

        loss_hist.append(float(metrics["loss"]))
        dsm_loss_hist.append(float(metrics["dsm_loss"]))
        var_loss_hist.append(float(metrics["var_loss"]))
        feature_std_hist.append(float(metrics["feature_std"]))
        grad_norm_hist.append(float(metrics["grad_norm"]))

        if step == 1 or step % logging_frequency == 0:
            pbar.set_postfix(loss=f"{loss_hist[-1]:.4f}", dsm=f"{dsm_loss_hist[-1]:.4f}", var=f"{var_loss_hist[-1]:.4f}")
    pbar.close()

    print(console_curve(loss_hist, title="Feature learning loss (DSM + feature variance)"))
    print(f"  final: loss={loss_hist[-1]:.6f} dsm={dsm_loss_hist[-1]:.6f} var={var_loss_hist[-1]:.6f} "
          f"std={feature_std_hist[-1]:.6f} grad_norm={grad_norm_hist[-1]:.6f}\n")

    features = np.asarray(_encode(params, all_x))

    if plot:
        os.makedirs(plot_dir, exist_ok=True)
        expert_features = np.asarray(_encode(params, expert_x))

        fig, axes = plt.subplots(1, latent_dim, figsize=(3.0 * latent_dim, 2.6), squeeze=False)
        axes = axes.flatten()
        for k in range(latent_dim):
            axes[k].hist(expert_features[:, k], bins=20, alpha=0.85)
            axes[k].set_title(rf"$z_{k + 1}$ on expert states")
            axes[k].set_xlabel(rf"$z_{k + 1}$")
            axes[k].set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, hist_file), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2))
        ax.plot(loss_hist, label="total")
        ax.plot(dsm_loss_hist, label="DSM")
        ax.plot(var_loss_hist, label="feature variance")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Boltzmann Encoder Loss")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
        fig.subplots_adjust(bottom=0.30)
        fig.savefig(os.path.join(plot_dir, loss_file), bbox_inches="tight")
        plt.close(fig)

    return features

