import wandb
from continual_go import get_benchmark
from functools import partial
import datetime
import jax
import jax.numpy as jnp
import os
import tyro


def act_randomly(key, mask):
    """Ignore observation and choose randomly from legal actions"""
    mask = mask.reshape(-1)
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(key, logits=logits, axis=-1)


def main(
    benchmark: str = "9x9-k16-A1M",
    cache_dir: str = "cache_cgo",
    reps: int = 1,
    seed: int = 42,
    window: int = 1000,
    total_steps: int = int(1e6),
    log_wandb: bool = True,
):
    if not wandb:
        os.environ["WANDB_MODE"] = "disabled"

    env = get_benchmark(name=benchmark, opponent_cache_dir=cache_dir)

    def _step(state, key):
        key_act, key_step = jax.random.split(key)
        action = act_randomly(key_act, env.legal_actions(state))
        state, reward = env.step(key_step, state, action)
        return (state, reward)

    @partial(jax.jit, static_argnames="num_steps")
    def _full_episode(key, num_steps):
        init_state = env.init()
        keys = jax.random.split(key, num=num_steps)
        _carry, rewards = jax.lax.scan(_step, init_state, keys)
        return rewards

    key = jax.random.key(seed)

    keys = jax.random.split(key, num=reps)
    batched_rwds = jax.vmap(_full_episode, in_axes=(0, None))(keys, total_steps)

    avg_rwds = batched_rwds.reshape(reps, -1, window).mean(axis=-1)

    if log_wandb:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_runs = []
        for rep in range(reps):
            run = wandb.init(
                project="continual-go-random",
                name=f"{benchmark}-seed{seed}-rep{rep}",
                group=f"random_{now}",
                config={"rep": rep, "benchmark": benchmark},
                reinit="create_new",
            )
            wandb_runs.append(run)

        avg_rwds_np = jax.device_get(avg_rwds)
        for rep, run in enumerate(wandb_runs):
            for i, rwd in enumerate(avg_rwds_np[rep]):
                run.log(
                    {"avg_reward_per_step": float(rwd), "env_steps": (i + 1) * window},
                    step=i,
                )
            run.finish()


if __name__ == "__main__":
    tyro.cli(main)
