import flax.linen as nn
import jax.numpy as jnp

def crelu(x):
    return jnp.concatenate([nn.relu(x), nn.relu(-x)], axis=-1)

def fourier_features(x):
    return jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)

_ACTIVATIONS = {"relu": nn.relu, "crelu": crelu, "fourier": fourier_features}

"""
For ReDo : 

- Weights for both nn.Conv and nn.Dense are initialized according to LeCun Normal initialization.
- Biases are initialized by zeros.
- Using capture_intermediates won't work here, since the activation function is not a nn submodule.
"""

class ResBlock(nn.Module):
    channels: int
    activations : str = "relu"

    @nn.compact
    def __call__(self, x):
        identity = x
        x = nn.LayerNorm()(x)
        # x = nn.relu(x)
        x = _ACTIVATIONS[self.activations](x)
        self.sow('intermediates', 'resnet_act_0', x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        # x = nn.relu(x)
        x = _ACTIVATIONS[self.activations](x)
        self.sow('intermediates', 'resnet_act_1', x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        return x + identity


class DQNResnetV2(nn.Module):
    action_dim: int
    activations : str = "relu"
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
        # x = nn.relu(x)
        x = _ACTIVATIONS[self.activations](x)
        self.sow('intermediates', 'act_0', x)

        # q-head
        x = nn.Conv(2, (1, 1))(x)
        # x = nn.relu(x)
        x = _ACTIVATIONS[self.activations](x)
        self.sow('intermediates', 'act_1', x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.action_dim, name="final_layer")(x)

        if no_batch:
            x = jnp.squeeze(x, axis=0)
        return x
