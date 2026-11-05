import datetime
import os
import pickle
import time

import jax
import jax.numpy as jnp
import tyro
import wandb

from continual_go import get_benchmark
from src.configs import PPOConfig
from src.algorithms.ppo import PPOAgent, Transition


def get_obs(state, k):
    return (state.turn * state.board / k)[..., None].astype(jnp.float32)


def main(
    benchmark: str = "9x9-k16-1",
    cache_dir: str = "cache_cgo",
    reps: int = 1,
    cfg: PPOConfig = PPOConfig(),
):
    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "disabled"

    # Initialize one W&B run per replica
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_runs = []

    if cfg.wandb:
        for rep in range(reps):
            run = wandb.init(
                project="continual-go-ppo",
                name=f"{benchmark}-seed{cfg.seed}-rep{rep}",
                group=f"ppo_{now}",
                config={**cfg.model_dump(), "rep": rep},
                reinit="create_new",
            )
            wandb_runs.append(run)

    env = get_benchmark(name=benchmark, opponent_cache_dir=cache_dir)

    def _gen_init_carry(seed):
        agent_rng, env_rng = jax.random.split(seed)

        # env
        env_rng, key_bench = jax.random.split(env_rng)
        action_dim = env.num_actions
        obs_shape = (env.size, env.size, 1)

        # agent
        agent_state, agent_rng = PPOAgent.init_state(
            cfg, action_dim, obs_shape, agent_rng
        )

        # initial env state
        state = env.init()
        obs = get_obs(state, env.k)

        return state, obs, agent_state, agent_rng, env_rng

    # checkpoint dir
    ckpt_dir = os.path.join("ppo/checkpoints", f"ppo_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    wall_start = time.time()

    # logging callback
    # Receives metrics for all replicas at once.
    def _flush_log(mean_rewards, metrics, env_steps):
        hours = (time.time() - wall_start) / 3600
        env_steps = int(env_steps)

        for rep in range(reps):
            log = {
                "env_steps": env_steps,
                "avg_reward_per_step": float(mean_rewards[rep]),
                "hours": hours,
            }

            log.update(
                {name: float(v) for name, v in zip(PPOAgent.metric_names, metrics[rep])}
            )

            print(f"rep={rep}", log)

            if cfg.wandb:
                wandb_runs[rep].log(log, step=env_steps)

    def _save_ckpt(actor_params, critic_params, env_steps):
        path = os.path.join(ckpt_dir, f"{int(env_steps):09d}.ckpt")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": cfg.model_dump(),
                    "benchmark": benchmark,
                    "actor_params": actor_params,
                    "critic_params": critic_params,
                    "env_steps": int(env_steps),
                },
                f,
            )

    # training step

    def rollout_step(carry, unused):
        state, obs, agent_state, agent_rng, env_rng = carry

        legal = env.legal_actions(state).reshape(-1)
        action, value, log_prob, agent_rng = PPOAgent.step(
            agent_state, obs, legal, agent_rng
        )

        env_rng, step_rng = jax.random.split(env_rng)
        next_state, reward = env.step(step_rng, state, action)
        # TODO: fix this in continual-go
        next_state = next_state.replace(ko=jnp.squeeze(next_state.ko))
        next_obs = get_obs(next_state, env.k)

        transition = Transition(
            obs,
            action,
            reward.astype(jnp.float32),
            jnp.bool_(False),
            value,
            log_prob,
            legal,
        )
        carry = (next_state, next_obs, agent_state, agent_rng, env_rng)
        return carry, transition

    def update_one(carry, i):
        # Collect a trajectory
        carry, traj_batch = jax.lax.scan(rollout_step, carry, None, cfg.rollout_steps)
        state, obs, agent_state, agent_rng, env_rng = carry

        # Calculate last value
        critic_train_state = agent_state.critic_train_state
        last_val = critic_train_state.apply_fn(critic_train_state.params, obs)

        # Update step
        agent_state, metrics, agent_rng = PPOAgent.update(
            agent_state, traj_batch, last_val, agent_rng
        )

        carry = (state, obs, agent_state, agent_rng, env_rng)

        # Return metrics instead of logging inside the replica-level update
        return carry, (traj_batch.reward.mean(), metrics)

    # Vectorized update + logging
    def update_fn(carries, i):
        # Execute all replicas in parallel
        carries, (mean_rewards, metrics) = jax.vmap(
            update_one,
            in_axes=(0, None),
        )(carries, i)

        # One logging callback per training iteration
        env_steps = (i + 1) * cfg.rollout_steps

        jax.experimental.io_callback(
            _flush_log,
            None,
            mean_rewards,
            metrics,
            env_steps,
            ordered=True,
        )

        return carries, None

    # training loop
    num_updates = cfg.total_steps // cfg.rollout_steps

    @jax.jit
    def run_training(init_carries):
        updates = jnp.arange(num_updates, dtype=jnp.int32)
        return jax.lax.scan(update_fn, init_carries, updates)

    key = jax.random.PRNGKey(cfg.seed)

    keys = jax.random.split(key, num=reps)
    init_carries = jax.vmap(_gen_init_carry)(keys)

    final_carries, _ = run_training(init_carries)
    jax.block_until_ready(final_carries)

    # Ensure all logging callbacks have completed and finish runs
    jax.effects_barrier()
    for run in wandb_runs:
        run.finish()


if __name__ == "__main__":
    tyro.cli(main)
