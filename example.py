import jax
from continual_go import ContinualGo


def main():
    key = jax.random.key(42)

    env = ContinualGo.create(
        size=9,
        k=32,
        total_steps=int(1e6),
        opponent_path="../continual-go/alpha_zero/az_good.ckpt",  # NOTE replace with your own path
    )

    state = env.init()

    action = 0
    state, reward = env.step(key, state, action)

    print(state)


if __name__ == "__main__":
    main()
