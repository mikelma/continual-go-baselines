import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import unfreeze, freeze, FrozenDict


def ReDo(train_state, obs, num_blocks, threshold, rng):

    _, sow = train_state.apply_fn(train_state.params, obs, mutable=["intermediates"])
    intermediates = sow["intermediates"]
    # Using unfreeze for mutability (In place resetting)
    params = unfreeze(train_state.params)
    frozen = isinstance(train_state.params, FrozenDict)
    adam = train_state.opt_state[0]
    mu = unfreeze(adam.mu)
    nu = unfreeze(adam.nu)

    ingoing_weight_activation = nn.initializers.lecun_normal()

    for i in range(num_blocks):
        block = f"ResBlock_{i}"

        activations = intermediates[block]["resnet_act_1"][0]

        channel = jnp.mean(jnp.abs(activations), axis=(0, 1, 2))
        neuron_score = channel / (jnp.mean(channel))
        mask = neuron_score <= threshold

        conv_0_weights = params["params"][block]["Conv_0"]["kernel"]
        conv_0_bias = params["params"][block]["Conv_0"]["bias"]
        conv_1_weights = params["params"][block]["Conv_1"]["kernel"]

        rng, sub = jax.random.split(rng)
        reinit_weights = ingoing_weight_activation(
            sub, conv_0_weights.shape, conv_0_weights.dtype
        )

        params["params"][block]["Conv_0"]["kernel"] = jnp.where(
            mask[None, None, None, :], reinit_weights, conv_0_weights
        )
        params["params"][block]["Conv_0"]["bias"] = jnp.where(mask, 0.0, conv_0_bias)
        params["params"][block]["Conv_1"]["kernel"] = jnp.where(
            mask[None, None, :, None], 0.0, conv_1_weights
        )

        for M in (mu, nu):
            M["params"][block]["Conv_0"]["kernel"] = jnp.where(
                mask[None, None, None, :], 0.0, M["params"][block]["Conv_0"]["kernel"]
            )
            M["params"][block]["Conv_0"]["bias"] = jnp.where(
                mask, 0.0, M["params"][block]["Conv_0"]["bias"]
            )
            M["params"][block]["Conv_1"]["kernel"] = jnp.where(
                mask[None, None, :, None], 0.0, M["params"][block]["Conv_1"]["kernel"]
            )

    if frozen:
        # Make it FrozenDict again for jax.lax purposes
        params, nu, mu = freeze(params), freeze(mu), freeze(nu)

    new_opt_state = (adam._replace(mu=mu, nu=nu), *train_state.opt_state[1:]) # Check optax.adam 
    return train_state.replace(params=params, opt_state=new_opt_state)
