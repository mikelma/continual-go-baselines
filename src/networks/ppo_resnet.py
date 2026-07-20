import flax.linen as nn
import jax.numpy as jnp

from src.networks.dqn_resnet import ResBlock


class ActorResnetV2(nn.Module):
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

        # policy head
        x = nn.Conv(2, (1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        logits = nn.Dense(self.action_dim, name="final_layer")(x)

        if no_batch:
            logits = jnp.squeeze(logits, axis=0)
        return logits


class CriticResnetV2(nn.Module):
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

        # value head
        x = nn.Conv(2, (1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        value = nn.Dense(1, name="final_layer")(x)
        value = jnp.squeeze(value, axis=-1)

        if no_batch:
            value = jnp.squeeze(value, axis=0)
        return value
