import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

# typing imports not required here


# JIT-friendly RNN unroll: performs the recurrent update over time for a
# precomputed input projection. This is scripted to move the Python loop into
# TorchScript where it runs faster.
@torch.jit.script
def _rnn_unroll(
    input_proj: torch.Tensor,
    Wr: torch.Tensor,
    bias: torch.Tensor,
    h0: torch.Tensor,
    dt: float,
    tau: float,
    activation_type: int,
) -> torch.Tensor:
    # input_proj: (batch, seq_len, hidden)
    batch = input_proj.size(0)
    seq_len = input_proj.size(1)
    hidden = input_proj.size(2)

    h_t = h0
    hiddens = torch.zeros(
        batch, seq_len, hidden, device=input_proj.device, dtype=input_proj.dtype
    )
    t = 0
    while t < seq_len:
        if activation_type == 0:
            r = torch.tanh(h_t)
        else:
            r = torch.relu(h_t)
        r = r @ Wr
        # recurrence update
        h_t = (1.0 - dt / tau) * h_t + dt / tau * (r + input_proj[:, t] + bias)
        hiddens[:, t, :] = h_t
        t += 1

    return hiddens


class SimpleRNN(nn.Module):
    """
    Build the rnn model, for loop

    RNN dynamics:
    x(t+1) = x(t) + dt/tau * (-x(t) + Wr * tanh[x(t)] + Win * U + bias)

    g, modulate the distribution of the rnn weights.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_fn="tanh",
        g=0.9,
        dt=0.05,
        tau=1,
        h0_trainable=False,
        use_layernorm=False,
        **kwargs,
    ):
        super(SimpleRNN, self).__init__(**kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt
        self.tau = tau
        self.g = g

        # optional LayerNorm for hidden state (applied before activation)
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            # create LayerNorm after hidden_size is set
            self.ln_hidden = nn.LayerNorm(hidden_size)

        if activation_fn == "tanh":
            self.activation_fn = torch.tanh
        elif activation_fn == "relu":
            self.activation_fn = torch.relu
        else:
            raise ValueError("No such activation function !")

        self.Win = Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.Wr = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = Parameter(torch.Tensor(1, self.hidden_size))
        self.Wout = Parameter(torch.Tensor(self.hidden_size, self.output_size))

        ### for GRU
        if h0_trainable:
            self.h0 = Parameter(torch.Tensor(1, self.hidden_size))
            h_factor = 1.0 / np.sqrt(self.hidden_size)
            self.h0.data.uniform_(-h_factor, h_factor)

        self.init_weights()

    def init_hidden(self, batch):
        #! remember to use this function when analyzing the hidden state
        #! so that the initial hidden state is always zero
        # If the model defines a trainable h0, use it (expanded to batch).
        if hasattr(self, "h0"):
            # self.h0.shape ---> (1, hidden_size)
            return self.h0.expand(batch, -1).contiguous()

        # create zeros on the same device and dtype as model parameters
        param = next(self.parameters(), None)
        device = param.device if param is not None else None
        dtype = param.dtype if param is not None else None
        return torch.zeros((batch, self.hidden_size), device=device, dtype=dtype)

    def init_weights(self):
        ## init input and output weights
        self.Win.data.normal_(0, 1 / self.hidden_size)
        self.Wout.data.normal_(0, 1 / self.hidden_size)
        self.bias.data.zero_()

        ## init rnn connections
        self.Wr.data.normal_(0, self.g**2 / self.hidden_size)

    def forward(self, x, hidden_init=None):
        """
        x.shape ---> (tsq, batch, n_input)
        """
        # Prefer vectorized path: move time-major -> batch-major and precompute
        # input projection. For models with LayerNorm enabled we fall back to the
        # original path because LayerNorm + activation is a small extra cost and
        # complicates scripting.
        tsq = x.shape[0]
        batch = x.shape[1]

        # bring x to the right device/dtype once
        x = x.to(device=self.Win.device, dtype=self.Win.dtype)

        # batch-major for easier matmuls: (batch, tsq, input_size)
        x_b = x.transpose(0, 1)

        # precompute input projection for all time steps: (batch, tsq, hidden)
        input_proj = x_b @ self.Win

        # initial hidden
        if hidden_init is None:
            h0 = self.init_hidden(batch)
        else:
            param = next(self.parameters(), None)
            if param is not None:
                device = param.device
                dtype = param.dtype
                h0 = hidden_init.to(device=device, dtype=dtype)
            else:
                h0 = hidden_init

        # If LayerNorm is enabled, keep the current (Python) loop because applying
        # layer norm inside scripted loop is less straightforward and LayerNorm
        # cost is small compared to the recurrence.
        if self.use_layernorm:
            h_t = h0
            tot_input_list = []
            tot_rnnhid_list = []
            tot_output_list = []
            for t in range(tsq):
                linear_input_t = input_proj[:, t, :]
                h_t = (1 - self.dt / self.tau) * h_t + self.dt / self.tau * (
                    self.activation_fn(h_t) @ self.Wr + linear_input_t + self.bias
                )
                rnn_out_t = self.activation_fn(self.ln_hidden(h_t))
                linear_output_t = rnn_out_t @ self.Wout

                tot_input_list.append(linear_input_t)
                tot_rnnhid_list.append(h_t)
                tot_output_list.append(linear_output_t)

            tot_input_tensor = torch.stack(tot_input_list, dim=0).transpose(0, 1)
            tot_rnnhid_tensor = torch.stack(tot_rnnhid_list, dim=0).transpose(0, 1)
            tot_output_tensor = torch.stack(tot_output_list, dim=0).transpose(0, 1)
            output_tensor = torch.stack([t for t in tot_output_list], dim=0)

            return output_tensor, (
                tot_input_tensor,
                tot_rnnhid_tensor,
                tot_output_tensor,
            )

        # Use scripted unroll when no LayerNorm: faster loop in TorchScript
        activation_type = 0 if self.activation_fn is torch.tanh else 1
        tot_rnnhid_tensor = _rnn_unroll(
            input_proj,
            self.Wr,
            self.bias,
            h0,
            float(self.dt),
            float(self.tau),
            activation_type,
        )

        # apply activation if needed (rnn_unroll returns pre-activation hidden states)
        if self.use_layernorm:
            rnn_out = self.activation_fn(self.ln_hidden(tot_rnnhid_tensor))
        else:
            rnn_out = self.activation_fn(tot_rnnhid_tensor)

        tot_input_tensor = input_proj  # (batch, tsq, hidden)
        tot_output_tensor = rnn_out @ self.Wout  # (batch, tsq, out)

        # match original return shapes: output_tensor (tsq, batch, out)
        output_tensor = tot_output_tensor.transpose(0, 1)

        return output_tensor, (tot_input_tensor, tot_rnnhid_tensor, tot_output_tensor)
