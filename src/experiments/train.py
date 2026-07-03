
import datetime
import os
import pickle
import time

import jax
import jax.numpy as jnp
import tyro
import wandb

from continual_go import ContinualGo
from src.configs import DQNConfig
from src.algorithms.dqn import DQNAgent


def get_obs(state, k):
    return (state.turn * state.board / k)[..., None].astype(jnp.float32)


def main(opponent_path: str = "checkpoints/000025.ckpt", cfg: DQNConfig = DQNConfig()):
    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project="continual-go-dqn", config=cfg.model_dump())

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
    agent_state, agent_rng = DQNAgent.init_state(cfg, action_dim, obs_shape, agent_rng)

    env_rng = jax.random.PRNGKey(cfg.seed)
    # initial env state
    state = env.init()
    obs = get_obs(state, env.k)

    # checkpoint dir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("dqn/checkpoints", f"dqn_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    wall_start = time.time()

    # logging callbacks
    def _flush_log(mean_reward, mean_loss, epsilon, env_steps):
        hours = (time.time() - wall_start) / 3600
        wandb.log({
            "env_steps":int(env_steps),
            "avg_reward_per_step": float(mean_reward),
            "loss":float(mean_loss),
            "epsilon":float(epsilon),
            "hours":hours,
        })

    def _save_ckpt(params, opt_state, env_steps):
        path = os.path.join(ckpt_dir, f"{int(env_steps):09d}.ckpt")
        with open(path, "wb") as f:
            pickle.dump(
                {"config": cfg.model_dump(), 
                "opponent_path": opponent_path, 
                "params": params, 
                "opt_state": opt_state, 
                "env_steps": int(env_steps)},
                f)

    # training step

    # log accumulator: (sum_reward, sum_loss, count_loss, last_epsilon)
    init_log_acc = (jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0), jnp.float32(0.0))

    def step_fn(carry, t):
        state, obs, agent_state, agent_rng, env_rng, log_acc = carry

        legal = env.legal_actions(state).reshape(-1)
        action, agent_rng, aux = DQNAgent.step(agent_state, obs, legal, t, agent_rng)

        env_rng, step_rng = jax.random.split(env_rng)
        next_state, reward = env.step(step_rng, state, action)
        #TODO: fix this in continual-go
        next_state = next_state.replace(ko=jnp.squeeze(next_state.ko))
        next_obs = get_obs(next_state, env.k)

        transition = (obs, action, next_obs, reward, jnp.bool_(False), jnp.bool_(False), aux)
        agent_state, metrics, agent_rng = DQNAgent.update(agent_state, transition, agent_rng)

        # log accumulation
        reward_f = reward.astype(jnp.float32)
        loss_val = metrics[0]
        eps_val  = aux["epsilon"].astype(jnp.float32)

        sum_r, sum_l, count_l, _last_eps = log_acc
        sum_r = sum_r + reward_f
        is_real = ~jnp.isnan(loss_val)
        sum_l = sum_l + jnp.where(is_real, loss_val, jnp.float32(0.0))
        count_l = count_l + is_real.astype(jnp.int32)
        new_log_acc = (sum_r, sum_l, count_l, eps_val)

        # periodic log
        log_step = ((t + 1) % cfg.logging_freq) == 0

        def do_log(acc):
            sr, sl, cl, le = acc
            mean_r = sr / jnp.float32(cfg.logging_freq)
            mean_l = sl / jnp.maximum(cl, 1).astype(jnp.float32)
            jax.debug.callback(
                _flush_log, mean_r, mean_l, le, t + 1, ordered=True,
            )
            # reset sums; keep the last epsilon
            return (jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0), le)

        def no_log(acc):
            return acc

        new_log_acc = jax.lax.cond(log_step, do_log, no_log, new_log_acc)

        # periodic checkpoint (params + opt_state only)
        save_step = ((t + 1) % cfg.save_interval) == 0

        #def do_save(_):
        #    _save_ckpt(agent_state.q_train_state.params, agent_state.q_train_state.opt_state, t + 1)
        #    return jnp.int32(0)
        # TODO: save checkpoint
        #_ = jax.lax.cond(save_step, do_save, lambda _: jnp.int32(0), None)

        carry = (next_state, next_obs, agent_state, agent_rng, env_rng, new_log_acc)
        return carry, None

    init_carry = (state, obs, agent_state, agent_rng, env_rng, init_log_acc)

    @jax.jit
    def run_training(init_carry):
        ts = jnp.arange(cfg.total_steps, dtype=jnp.int32)
        return jax.lax.scan(step_fn, init_carry, ts)

    final_carry, _ = run_training(init_carry)
    jax.block_until_ready(final_carry)


if __name__ == "__main__":
    tyro.cli(main)
