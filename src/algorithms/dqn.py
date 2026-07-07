from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from jax.tree_util import tree_map, tree_leaves, tree_map_with_path
from trac_optimizer.experimental.jax.trac import start_trac # Import for using trac

from src.configs import DQNConfig
from src.networks.dqn_resnet import DQNResnetV2
from src.utils.replay_buffer import ReplayBuffer
from src.algorithms.agent import Agent


def default_float():
    return jax.dtypes.canonicalize_dtype(jnp.float32)


class TrainStateWithTargetNet(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int


@struct.dataclass
class AgentState:
    q_train_state: TrainStateWithTargetNet
    buffer_state: Any                                  # (location, full, buffers)
    # agent_config and buffer are Python objects, not jax pytrees — mark static.
    init_params : Any # For L2 Init and Wasserstein Regularization
    agent_config: DQNConfig = struct.field(pytree_node=False)
    buffer: ReplayBuffer = struct.field(pytree_node=False)


METRIC_NAMES = ("loss",)

def linear_epsilon(cfg: DQNConfig, t):
    anneal = cfg.explore_frac * cfg.total_steps
    eps = ((cfg.end_epsilon - cfg.start_epsilon) / anneal) * t + cfg.start_epsilon
    return jnp.clip(eps, cfg.end_epsilon)

def init_agent_state(agent_config: DQNConfig,action_dim: int,obs_shape: tuple,rng: jax.random.PRNGKey):
    # Buffer transition: (obs, next_obs, action, reward, done).
    dummy_items = (
        jnp.zeros(obs_shape, dtype=default_float()),
        jnp.zeros(obs_shape, dtype=default_float()),
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros(()),
        jnp.zeros((), dtype=bool),
    )
    buf = ReplayBuffer(buffer_size=agent_config.buffer_size, dummy_items=dummy_items)
    buffer_state = buf.initialize()

    network = DQNResnetV2(action_dim=action_dim,num_channels=agent_config.num_channels,num_blocks=agent_config.num_blocks)
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jnp.zeros(obs_shape, dtype=default_float()))
    params = tree_map(lambda p: p.astype(default_float()), params)

    tx = optax.adam(learning_rate=agent_config.q_lr)

    # Use the TRAC optimizer if enabled
    if agent_config.trac : 
        tx = start_trac(tx)

    base = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    q_train_state = TrainStateWithTargetNet(
        step=base.step,
        apply_fn=base.apply_fn,
        params=base.params,
        tx=base.tx,
        opt_state=base.opt_state,
        target_network_params=tree_map(jnp.copy, base.params),
        timesteps=0,
    )

    return (
        AgentState(
            q_train_state=q_train_state,
            buffer_state=buffer_state,
            agent_config=agent_config,
            buffer=buf,
            init_params = params
        ),
        rng,
    )


@partial(jax.jit)
def agent_step(agent_state: AgentState, obs: jnp.ndarray,legal_mask: jnp.ndarray,t,rng: jax.random.PRNGKey):
    cfg = agent_state.agent_config
    epsilon = linear_epsilon(cfg, t)

    params = agent_state.q_train_state.params
    q = agent_state.q_train_state.apply_fn(params, obs)
    q = jnp.where(legal_mask, q, jnp.finfo(q.dtype).min)
    argmax = jnp.argmax(q)

    rng, rng_e = jax.random.split(rng)

    def random_action_fn(rng_in):
        rng_out, rng_a = jax.random.split(rng_in)
        # sample uniformly from legal actions only.
        probs = legal_mask.astype(default_float())
        probs = probs / probs.sum()
        logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
        action = jax.random.categorical(rng_a, logits=logits)
        return action, rng_out

    def greedy_action_fn(rng_in):
        return argmax, rng_in

    action, rng_out = jax.lax.cond(
        jax.random.uniform(rng_e) < epsilon,
        random_action_fn,
        greedy_action_fn,
        rng,
    )
    is_nongreedy = (action != argmax)
    return action.squeeze(), rng_out, {"epsilon": epsilon, "is_nongreedy": is_nongreedy}


@jax.jit
def update_step(agent_state: AgentState, transition, rng):
    obs, action, next_obs, reward, terminated, truncated, _aux = transition
    init_params = agent_state.init_params 
    cfg = agent_state.agent_config
    train_state = agent_state.q_train_state
    buf = agent_state.buffer

    done = jnp.logical_or(terminated, truncated)
    buffer_state = buf.add(
        agent_state.buffer_state, obs, next_obs, action, reward, done
    )
    train_state = train_state.replace(timesteps=train_state.timesteps + 1)

    def _learn(args):
        train_state, rng = args
        rng, rng_sample = jax.random.split(rng)
        obs_b, next_obs_b, action_b, reward_b, done_b = buf.sample(
            buffer_state, cfg.buffer_batch_size, rng_sample
        )
        obs_b = obs_b.astype(default_float())
        next_obs_b = next_obs_b.astype(default_float())

        q_next_target = train_state.apply_fn(train_state.target_network_params, next_obs_b)
        q_next_target = jnp.max(q_next_target, axis=-1)
        target = reward_b + (1.0 - done_b) * cfg.gamma * q_next_target

        def _loss_fn(params):

            def _wasserstein2(path,p,q):
                if path[-1].key == 'kernel' : # Can check the naming by using tree_leaves_with_path
                    return jnp.sum((jnp.sort(p.ravel()) - jnp.sort(q.ravel()))**2) # Preserve order statistics 
                return jnp.float32(0.0)

            q_vals = train_state.apply_fn(params, obs_b)
            chosen_q = jnp.take_along_axis(
                q_vals, jnp.expand_dims(action_b, axis=-1), axis=-1,
            ).squeeze(axis=-1)

            loss = jnp.mean((chosen_q - target) ** 2)

            #[TODO] : Can keep w2,l2init as a boolean. Currently, I am using 0 weight to disable both methods. 
            if cfg.l2_init_weight != 0 : 
                loss += cfg.l2_init_weight * sum(tree_leaves(tree_map(lambda p,q : jnp.sum((p - q) ** 2), params, init_params)))
            if cfg.w2_weight != 0 : 
                loss += cfg.w2_weight * sum(tree_leaves(tree_map_with_path(_wasserstein2, params, init_params)))

            return loss

        loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
        updates, new_opt_state = train_state.tx.update(
            grads, train_state.opt_state, train_state.params,
        )
        new_params = optax.apply_updates(train_state.params, updates)
        new_train_state = train_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=train_state.step + 1,
        )
        return new_train_state, rng, loss

    def _no_learn(args):
        train_state, rng = args
        return train_state, rng, jnp.array(jnp.nan)

    is_learn_time = (
        buf.can_sample(buffer_state, cfg.buffer_batch_size)
        & (train_state.timesteps > cfg.learning_starts)
        & (train_state.timesteps % cfg.training_interval == 0)
    )
    train_state, rng, loss = jax.lax.cond(
        is_learn_time, _learn, _no_learn, (train_state, rng)
    )

    # target network update
    train_state = jax.lax.cond(
        train_state.timesteps % cfg.target_update_interval == 0,
        lambda ts: ts.replace(
            target_network_params=optax.incremental_update(
                ts.params, ts.target_network_params, cfg.tau
            )
        ),
        lambda ts: ts,
        train_state,
    )

    new_agent_state = AgentState(
        q_train_state=train_state,
        buffer_state=buffer_state,
        agent_config=cfg,
        buffer=buf,
        init_params = init_params,
    )
    metrics = jnp.array([loss])
    return new_agent_state, metrics, rng


DQNAgent = Agent(init_agent_state, agent_step, update_step, METRIC_NAMES)
