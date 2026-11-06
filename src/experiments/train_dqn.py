import datetime
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb

from continual_go import get_benchmark
from src.configs import DQNConfig
from src.algorithms.dqn import DQNAgent


def get_obs(state, k):
    return (state.turn * state.board / k)[..., None].astype(jnp.float32)


def main(
    benchmark: str = "9x9-k16-1",
    cache_dir: str = "cache_cgo",
    cfg: DQNConfig = DQNConfig(),
):
    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "disabled"
    # assert here to drop can_sample from the learn condition
    assert cfg.buffer_size >= cfg.buffer_batch_size

    num_seeds = cfg.num_seeds

    # Initialize one W&B run per seed
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_runs = []

    if cfg.wandb:
        for i in range(num_seeds):
            run = wandb.init(
                project="continual-go-dqn",
                name=f"{benchmark}-seed{cfg.seed + i}",
                group=f"dqn_{now}",
                config={**cfg.model_dump(), "seed": cfg.seed + i},
                reinit="create_new",
            )
            wandb_runs.append(run)

    # env (key_bench only seeds the skill scheduler's unused key, so the
    # schedule is identical across seeds and the env is built once)
    rng = jax.random.PRNGKey(cfg.seed)
    _, env_rng = jax.random.split(rng)
    env_rng, key_bench = jax.random.split(env_rng)
    env = get_benchmark(name=benchmark, key=key_bench, opponent_cache_dir=cache_dir)
    action_dim = env.num_actions
    obs_shape = (env.size, env.size, 1)

    # log accumulator: (sum_reward, sum_loss, count_loss, last_epsilon)
    init_log_acc = (jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0), jnp.float32(0.0))

    # per-seed init, vmapped over seeds. key splits match a single-seed run
    def init_seed(seed):
        rng = jax.random.PRNGKey(seed)
        agent_rng, env_rng = jax.random.split(rng)
        env_rng, _key_bench = jax.random.split(env_rng)

        agent_state, agent_rng = DQNAgent.init_state(cfg, action_dim, obs_shape, agent_rng)

        state = env.init()
        obs = get_obs(state, env.k)
        return (state, obs, agent_state, agent_rng, env_rng, init_log_acc)

    seeds = cfg.seed + jnp.arange(num_seeds, dtype=jnp.int32)
    init_carry = jax.jit(jax.vmap(init_seed))(seeds)

    # checkpoint dir
    ckpt_dir = os.path.join("dqn/checkpoints", f"dqn_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    wall_start = time.time()

    # performance stats to keep track of between logging callbacks
    perf = {"compile_s": None, "last_time": None, "last_steps": None}

    # logging callback
    # Receives metrics for all seeds at once ([num_seeds] arrays).
    def _flush_log(mean_rewards, mean_losses, epsilons, env_steps):
        now = time.perf_counter()
        hours = (time.time() - wall_start) / 3600
        env_steps = int(env_steps)
        mean_rewards = np.asarray(mean_rewards)
        mean_losses = np.asarray(mean_losses)
        epsilons = np.asarray(epsilons)

        perf_log = {}
        if perf["last_time"] is not None:
            sps_per_seed = (env_steps - perf["last_steps"]) / (now - perf["last_time"])
            perf_log["perf/sps_per_seed"] = sps_per_seed
        perf["last_time"], perf["last_steps"] = now, env_steps

        mem = jax.devices()[0].memory_stats() or {}
        if mem.get("peak_bytes_in_use") is not None:
            perf_log["perf/peak_mem_gb"] = mem["peak_bytes_in_use"] / 1e9

        if perf["compile_s"] is not None:
            perf_log["perf/compile_s"] = perf["compile_s"]
            perf["compile_s"] = None

        for i in range(num_seeds):
            log = {
                "env_steps": env_steps,
                "avg_reward_per_step": float(mean_rewards[i]),
                "loss": float(mean_losses[i]),
                "epsilon": float(epsilons[i]),
                "hours": hours,
                **perf_log,
            }

            print(f"seed={cfg.seed + i}", log)

            if cfg.wandb:
                wandb_runs[i].log(log, step=env_steps)

    def _save_ckpt(params, opt_state, env_steps):
        path = os.path.join(ckpt_dir, f"{int(env_steps):09d}.ckpt")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": cfg.model_dump(),
                    "benchmark": benchmark,
                    "params": params,
                    "opt_state": opt_state,
                    "env_steps": int(env_steps),
                },
                f,
            )

    # training step

    # per-seed step (vmapped over seed axis, t is shared)
    def seed_step(seed_carry, t):
        state, obs, agent_state, agent_rng, env_rng, log_acc = seed_carry

        legal = env.legal_actions(state).reshape(-1)
        action, agent_rng, aux = DQNAgent.step(agent_state, obs, legal, t, agent_rng)

        env_rng, step_rng = jax.random.split(env_rng)
        next_state, reward = env.step(step_rng, state, action)
        # TODO: fix this in continual-go
        next_state = next_state.replace(ko=jnp.squeeze(next_state.ko))
        next_obs = get_obs(next_state, env.k)

        transition = (
            obs,
            action,
            next_obs,
            reward,
            jnp.bool_(False),
            jnp.bool_(False),
            aux,
        )
        agent_state, metrics, agent_rng = DQNAgent.update(
            agent_state, transition, t, agent_rng
        )

        # log accumulation
        reward_f = reward.astype(jnp.float32)
        loss_val = metrics[0]
        eps_val = aux["epsilon"].astype(jnp.float32)

        sum_r, sum_l, count_l, _last_eps = log_acc
        sum_r = sum_r + reward_f
        is_real = ~jnp.isnan(loss_val)
        sum_l = sum_l + jnp.where(is_real, loss_val, jnp.float32(0.0))
        count_l = count_l + is_real.astype(jnp.int32)
        new_log_acc = (sum_r, sum_l, count_l, eps_val)

        return (next_state, next_obs, agent_state, agent_rng, env_rng, new_log_acc)

    def step_fn(carry, t):
        carry = jax.vmap(seed_step, in_axes=(0, None))(carry, t)
        state, obs, agent_state, agent_rng, env_rng, new_log_acc = carry

        # periodic log
        log_step = ((t + 1) % cfg.logging_freq) == 0

        def do_log(acc):
            sr, sl, cl, le = acc
            mean_r = sr / jnp.float32(cfg.logging_freq)
            mean_l = sl / jnp.maximum(cl, 1).astype(jnp.float32)
            jax.debug.callback(
                _flush_log,
                mean_r,
                mean_l,
                le,
                t + 1,
                ordered=True,
            )
            # reset sums; keep the last epsilon
            return (jnp.zeros_like(sr), jnp.zeros_like(sl), jnp.zeros_like(cl), le)

        def no_log(acc):
            return acc

        new_log_acc = jax.lax.cond(log_step, do_log, no_log, new_log_acc)

        # periodic checkpoint (params + opt_state only)
        save_step = ((t + 1) % cfg.save_interval) == 0

        # def do_save(_):
        #    _save_ckpt(agent_state.q_train_state.params, agent_state.q_train_state.opt_state, t + 1)
        #    return jnp.int32(0)
        # TODO: save checkpoint
        # _ = jax.lax.cond(save_step, do_save, lambda _: jnp.int32(0), None)

        carry = (state, obs, agent_state, agent_rng, env_rng, new_log_acc)
        return carry, None

    def run_training(init_carry):
        ts = jnp.arange(cfg.total_steps, dtype=jnp.int32)
        return jax.lax.scan(step_fn, init_carry, ts)

    # AOT compile (timed separately); the initial carry is donated
    compile_start = time.perf_counter()
    compiled = jax.jit(run_training, donate_argnums=0).lower(init_carry).compile()
    perf["compile_s"] = time.perf_counter() - compile_start

    final_carry, _ = compiled(init_carry)
    del init_carry  # donated, must not be reused
    jax.block_until_ready(final_carry)

    # Ensure all logging callbacks have completed and finish runs
    jax.effects_barrier()
    for run in wandb_runs:
        run.finish()


if __name__ == "__main__":
    tyro.cli(main)
