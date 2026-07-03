import flax.linen as nn
import jax.numpy as jnp


class ResBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        identity = x
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        return x + identity


class DQNResnetV2(nn.Module):
    action_dim: int
    num_channels: int = 128
    num_blocks: int = 6

    @nn.compact
    def __call__(self, x):
        no_batch = (x.ndim == 3)
        if no_batch:
            x = x[None]

        x = nn.Conv(self.num_channels, (3, 3), padding="SAME")(x)

        for _ in range(self.num_blocks):
            x = ResBlock(self.num_channels)(x)

        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # q-head
        x = nn.Conv(2, (1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.action_dim, name="final_layer")(x)

        if no_batch:
            x = jnp.squeeze(x, axis=0)
        return x
