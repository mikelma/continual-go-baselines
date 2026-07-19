
import datetime
import os
import pickle
import time

import jax
import jax.numpy as jnp
import tyro
import wandb

from continual_go import ContinualGo
from src.configs import PPOConfig
from src.algorithms.ppo import PPOAgent, Transition


def get_obs(state, k):
    return (state.turn * state.board / k)[..., None].astype(jnp.float32)


def main(opponent_path: str = "checkpoints/000025.ckpt", cfg: PPOConfig = PPOConfig()):
    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project="continual-go-ppo", config=cfg.model_dump())

    # env
    env = ContinualGo.create(
        size=cfg.board_size,
        k=cfg.max_stones,
        total_steps=cfg.total_steps,
        opponent_path=opponent_path,
    )
    action_dim = env.num_actions
    obs_shape = (cfg.board_size, cfg.board_size, 1)

    # agent
    agent_rng = jax.random.PRNGKey(cfg.seed)
    agent_state, agent_rng = PPOAgent.init_state(cfg, action_dim, obs_shape, agent_rng)

    env_rng = jax.random.PRNGKey(cfg.seed)
    # initial env state
    state = env.init()
    obs = get_obs(state, env.k)

    # checkpoint dir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("ppo/checkpoints", f"ppo_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    wall_start = time.time()

    # logging callbacks
    def _flush_log(mean_reward, metrics, env_steps):
        hours = (time.time() - wall_start) / 3600
        log = {
            "env_steps": int(env_steps),
            "avg_reward_per_step": float(mean_reward),
            "hours": hours,
        }
        log.update({name: float(v) for name, v in zip(PPOAgent.metric_names, metrics)})
        wandb.log(log)

    def _save_ckpt(actor_params, critic_params, env_steps):
        path = os.path.join(ckpt_dir, f"{int(env_steps):09d}.ckpt")
        with open(path, "wb") as f:
            pickle.dump(
                {"config": cfg.model_dump(),
                "opponent_path": opponent_path,
                "actor_params": actor_params,
                "critic_params": critic_params,
                "env_steps": int(env_steps)},
                f)

    # training step

    def rollout_step(carry, unused):
        state, obs, agent_state, agent_rng, env_rng = carry

        legal = env.legal_actions(state).reshape(-1)
        action, value, log_prob, agent_rng = PPOAgent.step(agent_state, obs, legal, agent_rng)

        env_rng, step_rng = jax.random.split(env_rng)
        next_state, reward = env.step(step_rng, state, action)
        #TODO: fix this in continual-go
        next_state = next_state.replace(ko=jnp.squeeze(next_state.ko))
        next_obs = get_obs(next_state, env.k)

        transition = Transition(
            obs, action, reward.astype(jnp.float32), jnp.bool_(False), value, log_prob, legal,
        )
        carry = (next_state, next_obs, agent_state, agent_rng, env_rng)
        return carry, transition

    def update_fn(carry, i):
        # Collect a trajectory
        carry, traj_batch = jax.lax.scan(rollout_step, carry, None, cfg.rollout_steps)
        state, obs, agent_state, agent_rng, env_rng = carry

        # Calculate last value
        critic_train_state = agent_state.critic_train_state
        last_val = critic_train_state.apply_fn(critic_train_state.params, obs)

        # Update step
        agent_state, metrics, agent_rng = PPOAgent.update(agent_state, traj_batch, last_val, agent_rng)

        # periodic log (one update = rollout_steps env steps)
        env_steps = (i + 1) * cfg.rollout_steps
        jax.debug.callback(
            _flush_log, traj_batch.reward.mean(), metrics, env_steps, ordered=True,
        )

        # TODO: save checkpoint

        carry = (state, obs, agent_state, agent_rng, env_rng)
        return carry, None

    init_carry = (state, obs, agent_state, agent_rng, env_rng)
    num_updates = cfg.total_steps // cfg.rollout_steps

    @jax.jit
    def run_training(init_carry):
        updates = jnp.arange(num_updates, dtype=jnp.int32)
        return jax.lax.scan(update_fn, init_carry, updates)

    final_carry, _ = run_training(init_carry)
    jax.block_until_ready(final_carry)


if __name__ == "__main__":
    tyro.cli(main)
