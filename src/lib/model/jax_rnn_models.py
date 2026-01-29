from typing import Any, Callable, Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen.initializers import zeros

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any


@struct.dataclass
class RNNConfig:
    tau: float = 1.0
    dt: float = 0.05
    hidden_size: int = 400
    input_size: int = 25
    output_size: int = 7
    Win_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    Wout_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    Wr_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    h0_trainable: bool = False


class SimpleJaxRNN(nn.Module):
    config: RNNConfig

    @nn.compact
    def __call__(self, carray, inputs):
        """
        inputs, shape is (batch, N)
        carray, external inputs, shape is (batch, N_hidden)

        mean = self.param('mean', lecun_normal(), (2, 2))
        """

        x = carray
        config = self.config

        Win = self.param(
            "Win", config.Win_init, (config.input_size, config.hidden_size), jnp.float32
        )
        Wout = self.param(
            "Wout",
            config.Wout_init,
            (config.hidden_size, config.output_size),
            jnp.float32,
        )
        Wr = self.param(
            "Wr", config.Wr_init, (config.hidden_size, config.hidden_size), jnp.float32
        )
        bias = self.param(
            "bias", config.bias_init, (1, config.hidden_size), jnp.float32
        )

        dx = -x + inputs @ Win + jnp.tanh(x) @ Wr + bias
        x = x + dx * config.dt / config.tau

        h = jnp.tanh(x)
        out = h @ Wout

        return x, (x, out)


class RNNNet(nn.Module):
    """
    carray, initial rnn state.
    """

    config: RNNConfig

    @nn.compact
    def __call__(self, carray, inputs):
        rnn = nn.scan(
            SimpleJaxRNN,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )

        return rnn(self.config)(carray, inputs)


class LeakyRNN(nn.Module):
    config: RNNConfig

    @nn.compact
    def __call__(self, carray, inputs):
        """
        inputs, shape is (batch, N)
        carray, external inputs, shape is (batch, N_hidden)

        mean = self.param('mean', lecun_normal(), (2, 2))
        """

        x = carray
        config = self.config

        Win = self.param(
            "Win", config.Win_init, (config.input_size, config.hidden_size), jnp.float32
        )
        Wout = self.param(
            "Wout",
            config.Wout_init,
            (config.hidden_size, config.output_size),
            jnp.float32,
        )
        Wr = self.param(
            "Wr", config.Wr_init, (config.hidden_size, config.hidden_size), jnp.float32
        )
        bias = self.param(
            "bias", config.bias_init, (1, config.hidden_size), jnp.float32
        )

        dx = -x + inputs @ Win + jnp.tanh(x) @ Wr + bias
        x = x + dx * config.dt / config.tau

        h = jnp.tanh(x)
        out = h @ Wout

        return x, (x, out)


class LeakyRNNPb(nn.Module):

    config: RNNConfig

    @nn.compact
    def __call__(self, carray, inputs):
        """
        inputs, shape is (batch, N)
        carray, external inputs, shape is (batch, N_hidden)


        mean = self.param('mean', lecun_normal(), (2, 2))
        """

        x = carray
        config = self.config

        inputs_x, pt = inputs[:, : config.input_size], inputs[:, config.input_size :]

        Win = self.param(
            "Win", config.Win_init, (config.input_size, config.hidden_size), jnp.float32
        )
        Wout = self.param(
            "Wout",
            config.Wout_init,
            (config.hidden_size, config.output_size),
            jnp.float32,
        )
        Wr = self.param(
            "Wr", config.Wr_init, (config.hidden_size, config.hidden_size), jnp.float32
        )
        bias = self.param(
            "bias", config.bias_init, (1, config.hidden_size), jnp.float32
        )

        dx = -x + inputs_x @ Win + jnp.tanh(x) @ Wr + bias
        x = x + dx * config.dt / config.tau + pt

        h = jnp.tanh(x)
        out = h @ Wout

        return x, (x, out)


class RNNNetPb(nn.Module):
    """
    carray, initial rnn state.
    """

    config: RNNConfig

    @nn.compact
    def __call__(self, carray, inputs):
        rnn = nn.scan(
            LeakyRNNPb,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )

        return rnn(self.config)(carray, inputs)
