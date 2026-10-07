import jax
from continual_go import get_benchmark


def main():
    key = jax.random.key(42)

    key_init, key_step = jax.random.split(key)

    env = get_benchmark("9x9-k16-1", key_init)

    state = env.init()

    action = 0
    state, reward = env.step(key_step, state, action)

    print(state)


if __name__ == "__main__":
    main()
