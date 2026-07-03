# based on https://github.com/kenjyoung/dreamerv2_JAX/blob/main/replay_buffer.py
from functools import partial

import jax
import jax.numpy as jnp


class ReplayBuffer:
    def __init__(self, buffer_size, dummy_items):
        self.buffer_size = buffer_size
        self.dummy_items = dummy_items

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self):
        location = 0
        full = False

        buffers = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(
                jnp.repeat(jnp.expand_dims(x, 0), self.buffer_size, axis=0)
            ),
            self.dummy_items,
        )
        return (location, full, buffers)

    @partial(jax.jit, static_argnums=(0,))
    def add(self, state, *args):
        location, full, buffers = state
        buffers = jax.tree_util.tree_map(
            lambda x, y: x.at[location].set(y), buffers, tuple(args)
        )
        full = jnp.where(location + 1 >= self.buffer_size, True, full)
        location = (location + 1) % self.buffer_size
        return (location, full, buffers)

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, state, batch_size, key):
        location, full, buffers = state
        indices = jax.random.randint(
            key,
            minval=0,
            maxval=jnp.where(full, self.buffer_size, location),
            shape=(batch_size,),
        )
        return jax.tree_util.tree_map(lambda x: x.take(indices, axis=0), buffers)

    @partial(jax.jit, static_argnums=(0,))
    def can_sample(self, state, batch_size):
        location, full, _ = state
        n_items = jnp.where(full, self.buffer_size, location)
        return n_items >= batch_size
