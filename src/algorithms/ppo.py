from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from src.configs import PPOConfig
from src.networks.ppo_resnet import ActorResnetV2, CriticResnetV2
from src.algorithms.agent import Agent


@struct.dataclass
class AgentState:
    actor_train_state: TrainState
    critic_train_state: TrainState
    agent_config: PPOConfig = struct.field(pytree_node=False)


class Transition(NamedTuple):
    obs: jnp.ndarray        # o_t
    action: jnp.ndarray     # a_t
    reward: jnp.ndarray     # r[t+1]
    done: jnp.ndarray       # done[t+1] (always False in ContinualGo)
    value: jnp.ndarray      # v(o_t)
    log_prob: jnp.ndarray   # log_prob Pi(a_t|o_t)
    legal_mask: jnp.ndarray # legal actions at o_t


METRIC_NAMES = ("policy_loss", "value_loss", "entropy", "approx_kl")


def masked_logits(logits, legal_mask):
    return jnp.where(legal_mask, logits, jnp.finfo(logits.dtype).min)


def init_agent_state(agent_config: PPOConfig, action_dim: int, obs_shape: tuple, rng: jax.random.PRNGKey):
    actor_network = ActorResnetV2(
        action_dim=action_dim,
        num_channels=agent_config.num_channels,
        num_blocks=agent_config.num_blocks,
    )
    critic_network = CriticResnetV2(
        num_channels=agent_config.num_channels,
        num_blocks=agent_config.num_blocks,
    )

    init_x = jnp.zeros(obs_shape, dtype=jnp.float32)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)
    actor_network_params = actor_network.init(actor_rng, init_x)
    critic_network_params = critic_network.init(critic_rng, init_x)

    if agent_config.gradient_clipping:
        tx_actor = optax.chain(
            optax.clip_by_global_norm(agent_config.max_grad_norm),
            optax.adam(agent_config.actor_lr, eps=1e-5),
        )
        tx_critic = optax.chain(
            optax.clip_by_global_norm(agent_config.max_grad_norm),
            optax.adam(agent_config.critic_lr, eps=1e-5),
        )
    else:
        tx_actor = optax.adam(agent_config.actor_lr, eps=1e-5)
        tx_critic = optax.adam(agent_config.critic_lr, eps=1e-5)

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
def agent_step(agent_state: AgentState, obs: jnp.ndarray, legal_mask: jnp.ndarray, rng: jax.random.PRNGKey):
    logits = agent_state.actor_train_state.apply_fn(agent_state.actor_train_state.params, obs)
    logits = masked_logits(logits, legal_mask)
    value = agent_state.critic_train_state.apply_fn(agent_state.critic_train_state.params, obs)
    rng, rng_a = jax.random.split(rng)
    action = jax.random.categorical(rng_a, logits=logits)
    log_prob = jax.nn.log_softmax(logits)[action]
    return action, value, log_prob, rng


@partial(jax.jit, static_argnums=(1,))
def critic_loss(critic_params, critic_fn, traj_batch, targets, clip_eps, vf_coef):
    # Re-run the network through the batch
    value = critic_fn(critic_params, traj_batch.obs)
    # Calculate value loss
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )
    value_loss = vf_coef * value_loss
    return value_loss


@partial(jax.jit, static_argnums=(1,))
def policy_loss(actor_params, actor_fn, traj_batch, gae, clip_eps, ent_coef):
    # Re-run the network through the batch
    logits = actor_fn(actor_params, traj_batch.obs)
    logits = masked_logits(logits, traj_batch.legal_mask)
    all_log_probs = jax.nn.log_softmax(logits)
    log_prob = jnp.take_along_axis(
        all_log_probs, jnp.expand_dims(traj_batch.action, axis=-1), axis=-1,
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
    entropy = -jnp.where(
        traj_batch.legal_mask, probs * all_log_probs, 0.0
    ).sum(axis=-1).mean()

    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    total_loss = loss_actor - ent_coef * entropy

    aux_info = {'approx_kl': approx_kl,
                'policy_loss': loss_actor,
                'entropy': entropy}

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
        gae = (
            delta
            + gamma * gae_lambda * (1 - done) * gae
        )
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
        lambda x: jnp.take(x, permutation, axis=0), batch)
    return jax.tree_util.tree_map(
        lambda x: x.reshape((config.num_mini_batch, minibatch_size,) + x.shape[1:]),
        shuffled_batch)


@jax.jit
def update_step(agent_state: AgentState, traj_batch: Transition, last_val, rng):
    cfg = agent_state.agent_config
    # calculate advantages and targets
    advantages, targets = _calculate_gae(traj_batch, last_val, cfg.gamma, cfg.gae_lambda)

    def _update_minibatch(carry_in, batch_info):
        actor_train_state, critic_train_state = carry_in
        mb_traj_batch, mb_advantages, mb_targets = batch_info
        actor_grad_fn = jax.value_and_grad(policy_loss, has_aux=True)

        (_, aux_info), actor_grads = actor_grad_fn(
            actor_train_state.params, actor_train_state.apply_fn, mb_traj_batch, mb_advantages,
            cfg.clip_eps, cfg.entropy_coef)

        actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)

        critic_grad_fn = jax.value_and_grad(critic_loss)
        value_loss, critic_grads = critic_grad_fn(
            critic_train_state.params, critic_train_state.apply_fn, mb_traj_batch, mb_targets,
            cfg.clip_eps, cfg.vf_coef)

        critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
        aux_info['value_loss'] = value_loss
        return (actor_train_state, critic_train_state), aux_info

    def _batch_update(update_state, unused):
        actor_train_state, critic_train_state, rng = update_state
        rng, rng_perm = jax.random.split(rng)
        minibatches_info = _create_minibatches(cfg, (traj_batch, advantages, targets), rng_perm)
        # Loop through minibatches
        carry_in = (actor_train_state, critic_train_state)
        carry_out, aux_info = jax.lax.scan(_update_minibatch, carry_in, minibatches_info)
        actor_train_state, critic_train_state = carry_out
        return (actor_train_state, critic_train_state, rng), aux_info

    # Learning
    update_state = (agent_state.actor_train_state, agent_state.critic_train_state, rng)
    update_state, aux_info = jax.lax.scan(_batch_update, update_state, None, cfg.epochs)
    actor_train_state, critic_train_state, rng = update_state

    new_agent_state = AgentState(
        actor_train_state=actor_train_state,
        critic_train_state=critic_train_state,
        agent_config=cfg,
    )
    metrics = jnp.array([
        aux_info['policy_loss'].mean(),
        aux_info['value_loss'].mean(),
        aux_info['entropy'].mean(),
        aux_info['approx_kl'].mean(),
    ])
    return new_agent_state, metrics, rng


PPOAgent = Agent(init_agent_state, agent_step, update_step, METRIC_NAMES)
