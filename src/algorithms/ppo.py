from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax import core
import flax.linen as nn

from src.configs import PPOConfig
from src.networks.ppo_resnet import ActorResnetV2, CriticResnetV2
from src.algorithms.agent import Agent


@struct.dataclass
class AgentState:
    actor_train_state: TrainState
    critic_train_state: TrainState
    agent_config: PPOConfig = struct.field(pytree_node=False)


class Transition(NamedTuple):
    obs: jnp.ndarray  # o_t
    action: jnp.ndarray  # a_t
    reward: jnp.ndarray  # r[t+1]
    done: jnp.ndarray  # done[t+1] (always False in ContinualGo)
    value: jnp.ndarray  # v(o_t)
    log_prob: jnp.ndarray  # log_prob Pi(a_t|o_t)
    legal_mask: jnp.ndarray  # legal actions at o_t


METRIC_NAMES = (
    "policy_loss",
    "value_loss",
    "entropy",
    "actor_dormant",
    "critic_dormant",
    "approx_kl",
    "actor_kernels_l2",
    "critic_kernels_l2",
)


def _capture_conv(mdl: nn.Module, method_name: str) -> bool:
    return method_name == "__call__" and isinstance(mdl, nn.Conv)


def dormant_fraction(
    mut,
    tau: float = 0.025,
    eps: float = 1e-8,
) -> jnp.ndarray:
    intermediates = mut.get("intermediates", {})
    dormant = jnp.array(0.0, dtype=jnp.float32)
    total = jnp.array(0.0, dtype=jnp.float32)

    for leaf in jax.tree_util.tree_leaves(intermediates):
        act = jax.nn.relu(leaf)
        # last axis is channels
        mean_act = act.mean(axis=tuple(range(act.ndim - 1)))
        scores = mean_act / (mean_act.mean() + eps)
        dormant = dormant + jnp.sum(scores <= tau)
        total = total + scores.size

    return jnp.where(total > 0, dormant / total, 0.0)


def kernel_l2_norm(params):
    kernels = optax.tree_utils.tree_get_all_with_path(params, "kernel")
    return optax.tree_utils.tree_norm([v for _, v in kernels], ord=2)


def masked_logits(logits, legal_mask):
    return jnp.where(legal_mask, logits, jnp.finfo(logits.dtype).min)


def init_agent_state(
    agent_config: PPOConfig, action_dim: int, obs_shape: tuple, rng: jax.random.PRNGKey
):
    actor_network = ActorResnetV2(
        action_dim=action_dim,
        num_channels=agent_config.num_channels,
        num_blocks=agent_config.num_blocks,
        layer_norm=agent_config.layer_norm,
    )
    critic_network = CriticResnetV2(
        num_channels=agent_config.num_channels,
        num_blocks=agent_config.num_blocks,
        layer_norm=agent_config.layer_norm,
    )

    init_x = jnp.zeros(obs_shape, dtype=jnp.float32)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)
    actor_network_params = actor_network.init(actor_rng, init_x)
    critic_network_params = critic_network.init(critic_rng, init_x)

    actor_chain, critic_chain = [], []

    # Masks to prevent decaying LayerNorm's params
    actor_wd_mask = jax.tree_util.tree_map(
        lambda p: p.ndim > 1,
        actor_network_params,
    )
    critic_wd_mask = jax.tree_util.tree_map(
        lambda p: p.ndim > 1,
        critic_network_params,
    )

    actor_chain.append(
        optax.add_decayed_weights(agent_config.weight_decay, mask=actor_wd_mask)
    )
    critic_chain.append(
        optax.add_decayed_weights(agent_config.weight_decay, mask=critic_wd_mask)
    )

    if agent_config.gradient_clipping:
        actor_chain.append(optax.clip_by_global_norm(agent_config.max_grad_norm))
        critic_chain.append(optax.clip_by_global_norm(agent_config.max_grad_norm))

    actor_chain.append(optax.adam(agent_config.actor_lr, eps=1e-5))
    critic_chain.append(optax.adam(agent_config.critic_lr, eps=1e-5))

    tx_actor = optax.chain(*actor_chain)
    tx_critic = optax.chain(*critic_chain)

    actor_train_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network_params,
        tx=tx_actor,
    )
    critic_train_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_network_params,
        tx=tx_critic,
    )

    return (
        AgentState(
            actor_train_state=actor_train_state,
            critic_train_state=critic_train_state,
            agent_config=agent_config,
        ),
        rng,
    )


@jax.jit
def agent_step(
    agent_state: AgentState,
    obs: jnp.ndarray,
    legal_mask: jnp.ndarray,
    rng: jax.random.PRNGKey,
):
    logits = agent_state.actor_train_state.apply_fn(
        agent_state.actor_train_state.params, obs
    )
    logits = masked_logits(logits, legal_mask)
    value = agent_state.critic_train_state.apply_fn(
        agent_state.critic_train_state.params, obs
    )
    rng, rng_a = jax.random.split(rng)
    action = jax.random.categorical(rng_a, logits=logits)
    log_prob = jax.nn.log_softmax(logits)[action]
    return action, value, log_prob, rng


@partial(jax.jit, static_argnums=(1,))
def critic_loss(
    critic_params, critic_fn, traj_batch, targets, clip_eps, vf_coef, redo_tau
):
    # Re-run the network through the batch
    value, mutables = critic_fn(
        critic_params,
        traj_batch.obs,
        mutable=["intermediates"],
        capture_intermediates=_capture_conv,
    )
    # Calculate value loss
    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
        -clip_eps, clip_eps
    )
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    value_loss = vf_coef * value_loss

    aux = {}
    aux["dormant"] = dormant_fraction(mutables, tau=redo_tau)

    return value_loss, aux


@partial(jax.jit, static_argnums=(1,))
def policy_loss(actor_params, actor_fn, traj_batch, gae, clip_eps, ent_coef, redo_tau):
    # Re-run the network through the batch
    logits, mutables = actor_fn(
        actor_params,
        traj_batch.obs,
        mutable=["intermediates"],
        capture_intermediates=_capture_conv,
    )
    logits = masked_logits(logits, traj_batch.legal_mask)
    all_log_probs = jax.nn.log_softmax(logits)
    log_prob = jnp.take_along_axis(
        all_log_probs,
        jnp.expand_dims(traj_batch.action, axis=-1),
        axis=-1,
    ).squeeze(axis=-1)
    # Calculate actor loss
    log_ratio = log_prob - traj_batch.log_prob
    ratio = jnp.exp(log_ratio)
    approx_kl = ((ratio - 1) - log_ratio).mean()
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - clip_eps,
            1.0 + clip_eps,
        )
        * gae
    )

    # Calculate entropy (illegal actions have zero probability and contribute zero)
    probs = jnp.exp(all_log_probs)
    entropy = (
        -jnp.where(traj_batch.legal_mask, probs * all_log_probs, 0.0)
        .sum(axis=-1)
        .mean()
    )

    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    total_loss = loss_actor - ent_coef * entropy

    dormant = dormant_fraction(mutables, tau=redo_tau)

    aux_info = {
        "approx_kl": approx_kl,
        "policy_loss": loss_actor,
        "entropy": entropy,
        "actor_dormant": dormant,
    }

    return total_loss, aux_info


@jax.jit
def _calculate_gae(traj_batch, last_val, gamma, gae_lambda):
    """Calculate advantages and value targets
    GAE_t = delta_t + gamma * lambda * (1 - done_{t+1}) * GAE_{t+1}
    GAE_{traj_len+1} = 0
    delta_t = reward_t + gamma * value_{t+1} * (1 - done_{t+1}) - value_t
    Args:
        traj_batch (Transition): Transitions with shape (num_steps, _)
        last_val (float): Value of the last observation. This is the observation that follows the last observation in traj_batch.
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda
    Returns:
        advantages (jnp.ndarray): Advantages with shape (num_steps, _)
        targets (jnp.ndarray): Value targets with shape (num_steps, _)
    """

    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


def _create_minibatches(config, batch, rng):
    """Shuffle the rollout steps and split them into minibatches.
    The output will have the shape of (num_minibatches, minibatch_size, _)
    """
    minibatch_size = config.rollout_steps // config.num_mini_batch
    permutation = jax.random.permutation(rng, config.rollout_steps)
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    return jax.tree_util.tree_map(
        lambda x: x.reshape(
            (
                config.num_mini_batch,
                minibatch_size,
            )
            + x.shape[1:]
        ),
        shuffled_batch,
    )


@jax.jit
def update_step(agent_state: AgentState, traj_batch: Transition, last_val, rng):
    cfg = agent_state.agent_config
    # calculate advantages and targets
    advantages, targets = _calculate_gae(
        traj_batch, last_val, cfg.gamma, cfg.gae_lambda
    )

    def _update_minibatch(carry_in, batch_info):
        actor_train_state, critic_train_state = carry_in
        mb_traj_batch, mb_advantages, mb_targets = batch_info
        actor_grad_fn = jax.value_and_grad(policy_loss, has_aux=True)

        (_, aux_info), actor_grads = actor_grad_fn(
            actor_train_state.params,
            actor_train_state.apply_fn,
            mb_traj_batch,
            mb_advantages,
            cfg.clip_eps,
            cfg.entropy_coef,
            cfg.redo_tau,
        )

        actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)

        critic_grad_fn = jax.value_and_grad(critic_loss, has_aux=True)
        (value_loss, critic_aux), critic_grads = critic_grad_fn(
            critic_train_state.params,
            critic_train_state.apply_fn,
            mb_traj_batch,
            mb_targets,
            cfg.clip_eps,
            cfg.vf_coef,
            cfg.redo_tau,
        )

        critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
        aux_info["value_loss"] = value_loss
        aux_info["actor_kernels_l2"] = kernel_l2_norm(actor_train_state.params)
        aux_info["critic_kernels_l2"] = kernel_l2_norm(critic_train_state.params)
        aux_info["critic_dormant"] = critic_aux["dormant"]
        return (actor_train_state, critic_train_state), aux_info

    def _batch_update(update_state, unused):
        actor_train_state, critic_train_state, rng = update_state
        rng, rng_perm = jax.random.split(rng)
        minibatches_info = _create_minibatches(
            cfg, (traj_batch, advantages, targets), rng_perm
        )
        # Loop through minibatches
        carry_in = (actor_train_state, critic_train_state)
        carry_out, aux_info = jax.lax.scan(
            _update_minibatch, carry_in, minibatches_info
        )
        actor_train_state, critic_train_state = carry_out
        return (actor_train_state, critic_train_state, rng), aux_info

    # Learning
    update_state = (agent_state.actor_train_state, agent_state.critic_train_state, rng)
    update_state, aux_info = jax.lax.scan(_batch_update, update_state, None, cfg.epochs)
    actor_train_state, critic_train_state, rng = update_state

    # ReDo
    updates_per_step = cfg.epochs * cfg.num_mini_batch
    num_update_steps = actor_train_state.step // updates_per_step
    do_redo = cfg.redo_freq is not None and (num_update_steps % cfg.redo_freq) == 0
    actor_train_state, rng = _maybe_redo(
        actor_train_state, traj_batch.obs, rng, do_redo, cfg.num_blocks, cfg.redo_tau
    )
    critic_train_state, rng = _maybe_redo(
        critic_train_state, traj_batch.obs, rng, do_redo, cfg.num_blocks, cfg.redo_tau
    )

    new_agent_state = AgentState(
        actor_train_state=actor_train_state,
        critic_train_state=critic_train_state,
        agent_config=cfg,
    )

    metrics = jnp.array(
        [
            aux_info["policy_loss"].mean(),
            aux_info["value_loss"].mean(),
            aux_info["entropy"].mean(),
            aux_info["actor_dormant"].mean(),
            aux_info["critic_dormant"].mean(),
            aux_info["approx_kl"].mean(),
            aux_info["actor_kernels_l2"].mean(),
            aux_info["critic_kernels_l2"].mean(),
        ]
    )
    return new_agent_state, metrics, rng


def _maybe_redo(
    train_state: TrainState,
    obs: jnp.ndarray,
    rng: jax.random.PRNGKey,
    do_redo: bool,
    num_res_blocks: int,
    tau: float,
):
    rng, key = jax.random.split(rng)
    new_train_state = jax.lax.cond(
        do_redo,
        lambda: apply_redo(
            train_state,
            train_state.apply_fn,
            obs,
            key,
            num_res_blocks=num_res_blocks,
            tau=tau,
        ),
        lambda: train_state,
    )
    return new_train_state, rng


def _dormant_mask(
    act: jnp.ndarray, tau: float = 0.025, eps: float = 1e-8
) -> jnp.ndarray:
    """Boolean mask of shape (C,) for conv channels."""
    act = jax.nn.relu(act)
    mean_act = act.mean(axis=tuple(range(act.ndim - 1)))
    scores = mean_act / (mean_act.mean() + eps)
    return scores <= tau


def _get_activation(intermediates, module_path):
    node = intermediates
    for name in module_path:
        node = node[name]
    return node["__call__"][-1]


def _zero_moment_slice(moment_tree, path, mask, axis):
    node = moment_tree
    for key in path[:-1]:
        node = node[key]
    moment = node[path[-1]]

    shape = [1] * moment.ndim
    shape[axis] = mask.shape[0]

    node[path[-1]] = jnp.where(
        mask.reshape(shape),
        jnp.zeros_like(moment),
        moment,
    )


def apply_redo(
    train_state: TrainState,
    apply_fn,
    obs: jnp.ndarray,
    rng: jax.random.PRNGKey,
    num_res_blocks: int,
    tau: float = 0.025,
    # NOTE kernel_init has to match the original ResNet initializer
    kernel_init=nn.initializers.lecun_normal(),
) -> TrainState:
    params = train_state.params

    _, mut = apply_fn(
        params,
        obs,
        mutable=["intermediates"],
        capture_intermediates=_capture_conv,
    )
    activations = mut.get("intermediates", {})

    # Make params mutable
    params_frozen = isinstance(params, core.FrozenDict)
    params = core.unfreeze(params)
    p = params["params"]

    # Get Adam's parameters and make them mutable
    opt_state = train_state.opt_state
    adam_chain_state = opt_state[-1]  # ty: ignore
    adam_state = adam_chain_state[0]

    if not isinstance(adam_state, optax.ScaleByAdamState):
        raise TypeError("Expected Optax Adam state.")

    mu_frozen = isinstance(adam_state.mu, core.FrozenDict)
    nu_frozen = isinstance(adam_state.nu, core.FrozenDict)

    mu = core.unfreeze(adam_state.mu)  # ty: ignore
    nu = core.unfreeze(adam_state.nu)  # ty: ignore

    def reset_moments(path, mask, axis):
        _zero_moment_slice(mu, path, mask, axis)
        _zero_moment_slice(nu, path, mask, axis)

    # One key per conv layer
    keys = jax.random.split(rng, num_res_blocks + 1)

    # Recycle conv1 in every residual block
    for i in range(num_res_blocks):
        block = p[f"res{i}"]

        # Get activations and dormancy mask
        activation = activations[f"res{i}"]["conv1"]["__call__"][-1]
        mask = _dormant_mask(activation, tau)

        # Reinitialize incoming weights
        kernel = block["conv1"]["kernel"]

        fresh_kernel = kernel_init(
            keys[i],
            kernel.shape,
            dtype=kernel.dtype,
        )

        block["conv1"]["kernel"] = jnp.where(
            mask[None, None, None, :],
            fresh_kernel,
            kernel,
        )

        if "bias" in block["conv1"]:
            block["conv1"]["bias"] = jnp.where(
                mask,
                0.0,
                block["conv1"]["bias"],
            )
            reset_moments(
                ("params", f"res{i}", "conv1", "bias"),
                mask,
                axis=0,
            )

        # Reset Adam's parameters of incoming weights
        reset_moments(
            ("params", f"res{i}", "conv1", "kernel"),
            mask,
            axis=-1,
        )

        # Zero outgoing weights
        kernel = block["conv2"]["kernel"]

        block["conv2"]["kernel"] = jnp.where(
            mask[None, None, :, None],
            0.0,
            kernel,
        )

        # Reset Adam's parameters of outgoing weights
        reset_moments(
            ("params", f"res{i}", "conv2", "kernel"),
            mask,
            axis=-2,
        )

    # Recycle head_conv
    activation = activations["head_conv"]["__call__"][-1]
    mask = _dormant_mask(activation, tau)

    # Incoming weights
    kernel = p["head_conv"]["kernel"]
    fresh_kernel = kernel_init(
        keys[-1],
        kernel.shape,
        dtype=kernel.dtype,
    )
    p["head_conv"]["kernel"] = jnp.where(
        mask[None, None, None, :],
        fresh_kernel,
        kernel,
    )
    reset_moments(
        ("params", "head_conv", "kernel"),
        mask,
        axis=-1,
    )
    if "bias" in p["head_conv"]:
        p["head_conv"]["bias"] = jnp.where(
            mask,
            0.0,
            p["head_conv"]["bias"],
        )
        reset_moments(
            ("params", "head_conv", "bias"),
            mask,
            axis=0,
        )

    # Outgoing weights (head_conv -> final_layer)
    # activations have NHWC shape
    height, width = activation.shape[-3:-1]

    dense_mask = jnp.tile(mask, height * width)

    dense_kernel = p["final_layer"]["kernel"]

    p["final_layer"]["kernel"] = jnp.where(
        dense_mask[:, None],
        0.0,
        dense_kernel,
    )
    reset_moments(
        ("params", "final_layer", "kernel"),
        dense_mask,
        axis=0,
    )

    new_params = core.freeze(params) if params_frozen else params

    # Re-construct Adam's state
    new_mu = core.freeze(mu) if mu_frozen else mu
    new_nu = core.freeze(nu) if nu_frozen else nu

    new_adam_state = optax.ScaleByAdamState(
        count=adam_state.count,  # maintain counter
        mu=new_mu,
        nu=new_nu,
    )
    new_adam_chain_state = (new_adam_state,) + adam_chain_state[1:]  # ty: ignore
    new_opt_state = opt_state[:-1] + (new_adam_chain_state,)  # ty: ignore

    # Re-construct TrainState
    return train_state.replace(
        params=new_params,
        opt_state=new_opt_state,
    )


PPOAgent = Agent(init_agent_state, agent_step, update_step, METRIC_NAMES)
