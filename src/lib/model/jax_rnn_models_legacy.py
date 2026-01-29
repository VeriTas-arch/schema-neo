from flax import linen as nn
import jax.numpy as jnp

# from flax.linen.initializers import orthogonal
from flax.linen.initializers import zeros
from typing import Any, Callable, Tuple

from flax import struct


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any


@struct.dataclass
class LegacyRNNConfig:
    tau: float = 1.0
    dt: float = 0.05
    N_hid: int = 100
    N_in: int = 31
    N_out: int = 6
    Win_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    Wout_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    Wr_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros


class LegacySimpleRNN(nn.Module):
    config: LegacyRNNConfig

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
            "Win", config.Win_init, (config.N_in, config.N_hid), jnp.float32
        )
        Wout = self.param(
            "Wout", config.Wout_init, (config.N_hid, config.N_out), jnp.float32
        )
        Wr = self.param("Wr", config.Wr_init, (config.N_hid, config.N_hid), jnp.float32)
        bias = self.param("bias", config.bias_init, (1, config.N_hid), jnp.float32)

        dx = -x + inputs @ Win + jnp.tanh(x) @ Wr + bias
        x = x + dx * config.dt / config.tau

        h = jnp.tanh(x)
        out = h @ Wout

        return x, (x, out)


class LegacyRNNNet(nn.Module):
    """
    carray, initial rnn state.

    """

    config: LegacyRNNConfig

    @nn.compact
    def __call__(self, carray, inputs):
        rnn = nn.scan(
            LegacySimpleRNN,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )

        return rnn(self.config)(carray, inputs)
